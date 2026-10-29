use std::{
    ptr,
    sync::{
        Arc,
        atomic::{AtomicPtr, Ordering},
    },
};

use singe_cuda::context::Context;

use crate::{
    communicator::{Communicator, create},
    error::{Error, Result, Status},
    sys, try_ffi,
    types::{CommunicatorState, Config, UniqueId},
    utility::ensure_primary_context,
};

/// Communicator allocated GPU memory tracked by NCCL, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStatistics {
    /// Allocated GPU memory that can be suspended (bytes).
    pub suspendable_bytes: u64,
    /// GPU memory suspended?
    pub suspended: bool,
    /// Allocated GPU memory that cannot be suspended (bytes).
    pub persistent_bytes: u64,
    /// Total allocated GPU memory tracked by NCCL (bytes).
    pub total_bytes: u64,
}

/// A growing communicator that retains its parent until initialization completes.
/// Dropping it before completion aborts the new communicator.
#[derive(Debug)]
pub struct CommunicatorGrowth<'a> {
    communicator: Option<Communicator>,
    _parent: &'a Communicator,
}

impl CommunicatorGrowth<'_> {
    /// Returns the new communicator once asynchronous initialization completes.
    /// Returns `None` while initialization is in progress or after the result was taken.
    ///
    /// # Errors
    ///
    /// Returns an error if querying asynchronous initialization status fails or NCCL reports an initialization failure.
    pub fn poll(&mut self) -> Result<Option<Communicator>> {
        let Some(communicator) = self.communicator.as_ref() else {
            return Ok(None);
        };
        match communicator.async_error()? {
            None => Ok(self.communicator.take()),
            Some(Status::InProgress) => Ok(None),
            Some(status) => Err(status.into()),
        }
    }
}

impl Communicator {
    /// Suspend communicator operations to free dynamic GPU memory allocations.
    /// The communicator cannot be used for collective operations while suspended.
    /// This method synchronizes its CUDA context before releasing resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active or binding its CUDA context fails.
    /// Synchronizing pending CUDA work can also fail.
    /// NCCL suspend errors are propagated.
    pub fn suspend(&mut self) -> Result<()> {
        self.require_state(CommunicatorState::Active)?;
        self.cuda_context().synchronize()?;
        self.cuda_context().bind()?;
        // SAFETY: `ncclCommSuspend` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclCommSuspend(
                self.as_raw(),
                sys::NCCL_SUSPEND_MEM.cast_signed()
            ))?;
        }
        self.set_state(CommunicatorState::Suspended);
        Ok(())
    }

    /// Resume all previously suspended resources.
    /// After this call returns successfully, the communicator is fully operational.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not suspended or binding its CUDA context fails.
    /// NCCL resume errors are propagated.
    pub fn resume(&self) -> Result<()> {
        self.require_state(CommunicatorState::Suspended)?;
        self.cuda_context().bind()?;
        // SAFETY: `ncclCommResume` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclCommResume(self.as_raw()))?;
        }
        self.set_state(CommunicatorState::Active);
        Ok(())
    }

    /// Query communicator memory statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is neither active nor suspended, binding the CUDA context fails, or NCCL cannot query a memory statistic.
    pub fn memory_statistics(&self) -> Result<MemoryStatistics> {
        if !matches!(
            self.state(),
            CommunicatorState::Active | CommunicatorState::Suspended
        ) {
            return Err(Error::InvalidCommunicatorState {
                state: self.state().name().into(),
            });
        }
        self.cuda_context().bind()?;
        let query = |stat| -> Result<u64> {
            let mut value = 0;
            // SAFETY: `ncclCommMemStats` receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::ncclCommMemStats(self.as_raw(), stat, &raw mut value))?;
            }
            Ok(value)
        };
        Ok(MemoryStatistics {
            suspendable_bytes: query(sys::ncclCommMemStat_t::ncclStatGpuMemSuspend)?,
            suspended: query(sys::ncclCommMemStat_t::ncclStatGpuMemSuspended)? != 0,
            persistent_bytes: query(sys::ncclCommMemStat_t::ncclStatGpuMemPersist)?,
            total_bytes: query(sys::ncclCommMemStat_t::ncclStatGpuMemTotal)?,
        })
    }

    /// Generates a unique identifier for growing an existing communicator exactly once.
    /// Only the coordinator rank calls this before each grow operation.
    /// The coordinator must distribute the identifier to new ranks out of band.
    /// For a single-rank parent, also pass this identifier to [`Self::grow`].
    /// This method synchronizes the CUDA context before obtaining the identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state does not allow growth, pending CUDA work cannot be synchronized, context binding fails, or NCCL cannot generate the growth identifier.
    pub fn grow_unique_id(&self) -> Result<UniqueId> {
        self.require_state(CommunicatorState::Active)?;
        self.bind()?;
        self.cuda_context().synchronize()?;
        let mut id = sys::ncclUniqueId::default();
        // SAFETY: `ncclCommGetUniqueId` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommGetUniqueId(self.as_raw(), &raw mut id))?;
        }
        Ok(id.into())
    }

    /// Creates a new communicator by adding ranks to this communicator.
    /// All existing ranks must call this, and new ranks must call [`Self::join_growing`].
    /// Existing ranks retain their original rank in the new communicator.
    /// A single-rank parent must pass `Some(unique_id)` obtained from [`Self::grow_unique_id`].
    /// Larger parents may pass `None` because NCCL distributes the identifier internally.
    /// Destroy the parent communicator after the grow operation completes.
    /// Poll [`CommunicatorGrowth::poll`] to obtain the initialized communicator.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullPointer`] if a single-rank parent omits `unique_id`.
    /// Also returns an error if the communicator is inactive, the rank count does not
    /// increase, CUDA synchronization fails, or NCCL rejects initialization.
    pub fn grow(
        &self,
        rank_count: i32,
        unique_id: Option<UniqueId>,
        config: Option<Config<'_>>,
    ) -> Result<CommunicatorGrowth<'_>> {
        self.require_state(CommunicatorState::Active)?;
        let parent_rank_count = self.rank_count()?;
        if rank_count <= parent_rank_count {
            return Err(Error::out_of_range("rank_count"));
        }
        if parent_rank_count == 1 && unique_id.is_none() {
            return Err(Error::NullPointer);
        }
        let id = unique_id.map(UniqueId::into_raw);
        self.cuda_context().synchronize()?;
        let communicator = grow(
            self.cuda_context(),
            self.as_raw(),
            rank_count,
            id.as_ref().map_or(ptr::null(), |id| id),
            -1,
            config,
        )?;
        Ok(CommunicatorGrowth {
            communicator: Some(communicator),
            _parent: self,
        })
    }

    /// Joins a growing communicator as a new rank.
    /// `rank` must be at least the parent communicator size and less than `rank_count`.
    /// `unique_id` must come from the coordinator's [`Self::grow_unique_id`] call.
    /// Poll [`Self::async_error`] until it returns `None` before using the communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if `rank` is outside `0..rank_count`, the context is not the device primary CUDA context, context binding fails, or NCCL initialization fails.
    pub fn join_growing(
        context: &Arc<Context>,
        rank_count: i32,
        rank: i32,
        unique_id: UniqueId,
        config: Option<Config<'_>>,
    ) -> Result<Self> {
        if rank < 0 || rank >= rank_count {
            return Err(Error::out_of_range("rank"));
        }
        let id = unique_id.into_raw();
        grow(
            context,
            ptr::null_mut(),
            rank_count,
            &raw const id,
            rank,
            config,
        )
    }
}

fn grow(
    context: &Arc<Context>,
    parent: sys::ncclComm_t,
    rank_count: i32,
    id: *const sys::ncclUniqueId,
    rank: i32,
    config: Option<Config<'_>>,
) -> Result<Communicator> {
    ensure_primary_context(context)?;
    context.bind()?;
    // NCCL writes this output again from its asynchronous initialization job.
    // Keep its address stable until the owned communicator is destroyed or aborted.
    let output = Box::new(AtomicPtr::new(ptr::null_mut()));
    let mut config = config.map(Config::into_raw);
    // SAFETY: `ncclCommGrow` receives pointers into buffers borrowed for the call.
    let status = unsafe {
        sys::ncclCommGrow(
            parent,
            rank_count,
            id,
            rank,
            output.as_ptr(),
            config.as_mut().map_or(ptr::null_mut(), |config| config),
        )
    };
    let raw = output.load(Ordering::Acquire);
    let mut communicator =
        create::communicator_from_initialization(raw, Arc::clone(context), status)?;
    communicator.handle.initialization_output = Some(output);
    communicator.set_state(CommunicatorState::Initializing);
    Ok(communicator)
}
