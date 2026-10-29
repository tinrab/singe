use std::{
    cell::RefCell,
    ffi::CStr,
    marker::PhantomData,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicU8, Ordering},
    },
};

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    event::Event,
    stream::Stream,
    view::{DeviceRepr, DeviceViewMut},
};

use crate::{
    error::{Error, Result, Status},
    sys, try_ffi,
    types::{CommunicatorState, Config, ShrinkFlags, SplitColor, UniqueId},
    utility::{ensure_primary_context, to_i32, to_u64, to_usize},
};

/// A stateful NCCL communicator in a device primary CUDA context.
///
/// Use one communicator per host thread or concurrent task. The communicator
/// is movable between threads, but it is intentionally not `Clone` or `Sync`.
///
/// # Raw collectives
///
/// The public unsafe collective and point-to-point methods return after native
/// submission. All ranks must issue matching calls. Retain the communicator,
/// streams, backing allocations, and any referenced device scalars or windows
/// through completion, including errors and graph replay. Inputs must be
/// initialized and ordered before reading; outputs require exclusive access.
/// Separate input and output views must not overlap. Use explicit in-place
/// operations when reading and overwriting the same initialized allocation.
///
/// Poll nonblocking native submission through [`Self::async_error`] before
/// querying or waiting on its CUDA stream. Successful device completion and
/// asynchronous status establish the written output range; enqueue success
/// alone does not initialize it. Local completed helpers handle these waits.
#[derive(Debug)]
pub struct Communicator {
    pub(crate) handle: Handle,
}

/// A native communicator transferred with its actual context and retained resources.
/// Destroy or abort the native handle before releasing these owners, or return
/// the complete value through [`Communicator::from_raw`].
#[derive(Debug)]
pub struct RawCommunicator {
    /// The transferred native communicator handle.
    pub handle: sys::ncclComm_t,
    /// The device primary CUDA context associated with the communicator.
    pub cuda_context: Arc<CudaContext>,
    /// The lifecycle state transferred with the handle.
    pub state: CommunicatorState,
    /// Stable storage for a native initialization result that may still be written asynchronously.
    pub initialization_output: Option<Box<AtomicPtr<sys::ncclComm>>>,
    /// Events retained by queued operations or captured graphs.
    pub launch_events: Vec<Arc<Event>>,
}

/// A builder for one rank or a local clique of NCCL communicators.
#[derive(Debug)]
pub struct CommunicatorBuilder<'a> {
    mode: CommunicatorBuilderMode<'a>,
    config: Option<Config<'a>>,
}

#[derive(Debug)]
enum CommunicatorBuilderMode<'a> {
    Devices(Vec<i32>),
    Rank {
        cuda_context: &'a Arc<CudaContext>,
        rank: i32,
        rank_count: i32,
        unique_id: UniqueId,
    },
    ScalableRank {
        cuda_context: &'a Arc<CudaContext>,
        rank: i32,
        rank_count: i32,
        unique_ids: Vec<UniqueId>,
    },
}

#[derive(Debug)]
pub(crate) struct Handle {
    raw: sys::ncclComm_t,
    cuda_context: Arc<CudaContext>,
    state: AtomicU8,
    initialization_output: Option<Box<AtomicPtr<sys::ncclComm>>>,
    pub(crate) launch_events: RefCell<Vec<Arc<Event>>>,
}

/// A borrowed buffer registration tied to its communicator.
#[derive(Debug)]
pub struct RegisteredBuffer<'a, T: DeviceRepr> {
    communicator: &'a Communicator,
    raw: *mut (),
    buffer: DeviceViewMut<'a, T>,
}

/// A registered byte range used for one-sided or device-initiated communication.
#[derive(Debug)]
pub struct Window<'a> {
    communicator: &'a Communicator,
    raw: sys::ncclWindow_t,
    pointer: *mut u8,
    byte_len: u64,
    _buffer: PhantomData<&'a mut [u8]>,
}

/// A custom reduction operator owned by and borrowing its communicator.
#[derive(Debug)]
pub struct CustomReductionOperator<'a> {
    communicator: &'a Communicator,
    raw: i32,
}

pub mod device;
mod lifecycle;
mod one_sided;
mod resources;

pub use lifecycle::{CommunicatorGrowth, MemoryStatistics};
pub use one_sided::WaitSignal;

mod create;

mod operations;

// SAFETY: NCCL communicators are stateful collective handles. The owner may move
// between threads, but shared concurrent mutation is intentionally not exposed.
unsafe impl Send for Handle {}

impl Communicator {
    /// Wraps an existing NCCL communicator and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be uniquely owned and associated with
    /// `parts.cuda_context`, with the declared native lifecycle state. It must be
    /// the device primary context. Retain any native initialization output slot
    /// and launch events in `parts`; NCCL must not write through a different
    /// output slot or use released resources.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if the transferred handle is null, or an error if the retained context is not the device primary CUDA context.
    pub unsafe fn from_raw(parts: RawCommunicator) -> Result<Self> {
        if parts.handle.is_null() {
            return Err(Error::NullHandle);
        }
        ensure_primary_context(&parts.cuda_context)?;
        Ok(Self {
            handle: Handle {
                raw: parts.handle,
                cuda_context: parts.cuda_context,
                state: AtomicU8::new(parts.state.into_raw()),
                initialization_output: parts.initialization_output,
                launch_events: RefCell::new(parts.launch_events),
            },
        })
    }

    /// Creates a blocking communicator clique within a single process.
    ///
    /// `device_ids` defines the CUDA device assigned to each rank. Each rank
    /// retains its actual device primary context, as required by NCCL.
    /// Use [`Communicator::create_all_first_devices`] for the first `n` devices in order.
    ///
    /// # Errors
    ///
    /// Returns an error if a CUDA device cannot be selected, a CUDA context
    /// cannot be created for one of the ranks, the rank count cannot be
    /// represented for NCCL, NCCL rejects communicator initialization, or NCCL
    /// returns a null communicator handle.
    pub fn create_all(device_ids: &[i32]) -> Result<Vec<Self>> {
        CommunicatorBuilder::for_device_ids(device_ids).create_all()
    }

    /// Creates a local clique using device IDs from 0 through `device_count - 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if the device count cannot be represented, an NCCL group is already open, CUDA primary-context setup fails, or NCCL cannot initialize the clique.
    pub fn create_all_first_devices(device_count: i32) -> Result<Vec<Self>> {
        let device_count = to_usize(device_count, "device_count")?;
        let device_ids = (0..device_count)
            .map(|device_id| to_i32(device_id, "device_id"))
            .collect::<Result<Vec<i32>>>()?;
        Self::create_all(&device_ids)
    }

    /// Creates a local clique spanning all visible CUDA devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device count cannot be represented, an NCCL group is already open, CUDA primary-context setup fails, or NCCL cannot initialize the clique.
    pub fn create_all_current_devices() -> Result<Vec<Self>> {
        Self::create_all_first_devices(Device::count()?)
    }

    /// Creates a new communicator for the multi-threaded or multi-process case.
    /// `rank` must be between `0` and `rank_count - 1` and unique within the communicator clique.
    /// Each rank is associated with a CUDA device, which must be current before calling [`Communicator::create_rank`].
    /// [`Communicator::create_rank`] implicitly synchronizes with other ranks, so it must be called by different threads/processes or used within [`group_start`](crate::group::start)/[`group_end`](crate::group::end).
    ///
    /// # Errors
    ///
    /// Returns an error if `cuda_context` cannot be bound, `rank_count` or `rank`
    /// is invalid for the communicator clique, the unique ID does not match
    /// the other ranks, NCCL rejects initialization, or NCCL returns a null
    /// communicator handle.
    pub fn create_rank(
        cuda_context: &Arc<CudaContext>,
        rank_count: i32,
        unique_id: UniqueId,
        rank: i32,
    ) -> Result<Self> {
        CommunicatorBuilder::rank(cuda_context, rank, rank_count, unique_id).create()
    }

    /// Like [`Communicator::create_rank`], but accepts an explicit communicator configuration.
    /// For nonblocking initialization, poll [`Self::async_error`] until it returns `None`
    /// before using the communicator.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Communicator::create_rank`], and also
    /// returns an error if NCCL rejects `config`.
    pub fn create_rank_with_config(
        cuda_context: &Arc<CudaContext>,
        rank_count: i32,
        unique_id: UniqueId,
        rank: i32,
        config: Config<'_>,
    ) -> Result<Self> {
        CommunicatorBuilder::rank(cuda_context, rank, rank_count, unique_id)
            .with_config(config)
            .create()
    }

    /// Creates one rank using multiple distributed unique identifiers for scalable initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if rank arguments or identifier counts are invalid, the CUDA context is not the device primary context, CUDA setup fails, or NCCL cannot initialize the communicator.
    pub fn create_rank_scalable(
        cuda_context: &Arc<CudaContext>,
        rank_count: i32,
        rank: i32,
        unique_ids: &[UniqueId],
    ) -> Result<Self> {
        CommunicatorBuilder::scalable_rank(cuda_context, rank, rank_count, unique_ids).create()
    }

    /// Like [`Communicator::create_rank_with_config`], but accepts multiple [`UniqueId`] values for scalable initialization.
    ///
    /// Passing a single [`UniqueId`] is equivalent to
    /// [`Communicator::create_rank_with_config`].
    ///
    /// Use this when scalable initialization requires distributing more than one unique ID.
    ///
    /// # Errors
    ///
    /// Returns an error if `cuda_context` cannot be bound, `unique_ids.len()` cannot
    /// be represented for NCCL, `rank_count` or `rank` is invalid for the
    /// communicator clique, the scalable IDs do not match the other ranks, NCCL
    /// rejects initialization or `config`, or NCCL returns a null communicator
    /// handle.
    pub fn create_rank_scalable_with_config(
        cuda_context: &Arc<CudaContext>,
        rank_count: i32,
        rank: i32,
        unique_ids: &[UniqueId],
        config: Option<Config<'_>>,
    ) -> Result<Self> {
        let builder =
            CommunicatorBuilder::scalable_rank(cuda_context, rank, rank_count, unique_ids);
        match config {
            Some(config) => builder.with_config(config).create(),
            None => builder.create(),
        }
    }

    /// Returns the CUDA context retained by this communicator.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    /// Binds the CUDA context after checking that the communicator permits control operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    pub fn bind(&self) -> Result<()> {
        self.require_state_for_control()?;
        Ok(self.cuda_context().bind()?)
    }

    /// Returns the lifecycle state tracked by the wrapper.
    pub fn state(&self) -> CommunicatorState {
        CommunicatorState::from_raw(self.handle.state.load(Ordering::Relaxed))
    }

    pub(crate) fn ensure_buffer_context(&self, context: &Arc<CudaContext>) -> Result<()> {
        if context != self.cuda_context() {
            return Err(Error::ContextMismatch);
        }
        Ok(())
    }

    /// Checks communicator readiness and stream ownership, then binds the CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the stream belongs to another CUDA context, or binding the CUDA context fails.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.require_state_for_operations()?;
        self.bind()
    }

    /// Returns the number of ranks in this communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// report the communicator size.
    pub fn rank_count(&self) -> Result<i32> {
        self.bind()?;

        let mut value = 0;
        // SAFETY: `ncclCommCount` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommCount(self.as_raw(), &raw mut value))?;
        }
        Ok(value)
    }

    /// Returns the caller's rank within this communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// report this communicator's rank.
    pub fn rank(&self) -> Result<i32> {
        self.bind()?;

        let mut value = 0;
        // SAFETY: `ncclCommUserRank` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommUserRank(self.as_raw(), &raw mut value))?;
        }
        Ok(value)
    }

    /// Returns the CUDA device associated with this communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// report the communicator device.
    pub fn device(&self) -> Result<Device> {
        self.bind()?;

        let mut value = 0;
        // SAFETY: `ncclCommCuDevice` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommCuDevice(self.as_raw(), &raw mut value))?;
        }
        Ok(Device::new(value))
    }

    /// Returns a human-readable string corresponding to the last error that occurred in NCCL.
    /// Retrieving the string does not clear the error.
    /// The returned string may be unrelated to the current call and can reflect
    /// a previously launched asynchronous operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn last_error(&self) -> Result<String> {
        self.bind()?;

        // SAFETY: `ncclGetLastError` receives live handles owned by the wrappers.
        let value = unsafe { sys::ncclGetLastError(self.as_raw()) };
        if value.is_null() {
            return Ok(String::new());
        }

        // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
        Ok(unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned())
    }

    /// Queries the progress and potential errors of asynchronous NCCL operations.
    /// Operations that do not require a stream argument, such as [`Communicator::finalize`], are complete once this method reports [`Status::Success`].
    /// Operations with a stream argument, such as [`Communicator::all_reduce`], report [`Status::Success`] once they are posted to the stream, but may still surface failures through [`Communicator::async_error`] until execution completes.
    /// If any NCCL call reports [`Status::InProgress`], the operation is still being enqueued in the background. Poll communicator state until all involved communicators report [`Status::Success`] before issuing another NCCL call.
    /// Before the state changes to [`Status::Success`], do not launch CUDA kernels on streams currently being used by NCCL.
    /// If there has been an error on the communicator, abort it with [`Communicator::abort`].
    /// If an error occurs on the communicator, nothing can be assumed about the completion or correctness of operations enqueued on that communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// report the asynchronous state.
    pub fn async_error(&self) -> Result<Option<Status>> {
        if self.state() != CommunicatorState::Initializing {
            self.require_state_for_control()?;
        }
        self.cuda_context().bind()?;

        let mut value = sys::ncclResult_t::ncclSuccess;
        // SAFETY: `ncclCommGetAsyncError` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommGetAsyncError(self.as_raw(), &raw mut value))?;
        }
        if value == sys::ncclResult_t::ncclSuccess
            && self.state() == CommunicatorState::Initializing
        {
            self.set_state(CommunicatorState::Active);
        }
        Ok((value != sys::ncclResult_t::ncclSuccess).then(|| value.into()))
    }

    /// Finalizes this communicator.
    /// When the communicator is configured as nonblocking, [`Communicator::finalize`] is also nonblocking.
    /// A successful return may transition the communicator into an in-progress finalization state while uncompleted operations and network resources are flushed and released.
    /// Once all NCCL operations are complete, [`Communicator::async_error`] reports [`Status::Success`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if the communicator
    /// is not active, or if NCCL rejects finalization.
    pub fn finalize(&self) -> Result<()> {
        self.bind()?;
        self.require_state(CommunicatorState::Active)?;
        // SAFETY: `ncclCommFinalize` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclCommFinalize(self.as_raw()))?;
        }
        self.set_state(CommunicatorState::Finalizing);
        Ok(())
    }

    /// Aborts this communicator and frees its resources.
    /// All active ranks must call this to abort the NCCL communicator successfully.
    /// Recovery path once a communicator has entered an error state.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if the communicator
    /// is already destroyed or aborted, or if NCCL rejects the abort.
    pub fn abort(&mut self) -> Result<()> {
        self.cuda_context().bind()?;
        if matches!(
            self.state(),
            CommunicatorState::Destroyed | CommunicatorState::Aborted
        ) {
            return self.invalid_state_error();
        }
        // SAFETY: `ncclCommAbort` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclCommAbort(self.as_raw()))?;
        }
        self.set_state(CommunicatorState::Aborted);
        Ok(())
    }

    /// Revokes in-flight operations on a communicator without destroying resources.
    /// Successful return may be [`Status::InProgress`] while revocation completes asynchronously; applications can query [`Communicator::async_error`] until it reports `Ok`.
    ///
    /// Uses the default revoke flags expected by NCCL.
    ///
    /// After revoke completes, the communicator is quiesced and safe for destroy, split, and shrink.
    /// Launching new collectives on a revoked communicator returns [`Status::InvalidUsage`].
    /// Calling [`Communicator::finalize`] after revoke is not supported.
    /// Resource sharing for split and shrink operations is disabled once the parent communicator has been revoked.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if the communicator
    /// is not active, or if NCCL rejects revocation.
    pub fn revoke(&self) -> Result<()> {
        self.bind()?;
        self.require_state(CommunicatorState::Active)?;
        // SAFETY: `ncclCommRevoke` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclCommRevoke(
                self.as_raw(),
                sys::NCCL_REVOKE_DEFAULT.cast_signed()
            ))?;
        }
        self.set_state(CommunicatorState::Revoked);
        Ok(())
    }

    /// Destroys the native communicator and marks this wrapper as destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or NCCL rejects destruction, including destruction while initialization is still in progress.
    pub fn destroy(self) -> Result<()> {
        self.cuda_context().bind()?;
        self.handle.destroy_raw()
    }

    /// Splits the communicator by color and orders ranks within each color by `key`.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// NCCL split errors are propagated.
    pub fn split(&self, color: SplitColor, key: i32) -> Result<Option<Self>> {
        Self::split_with_config(self, color, key, None)
    }

    /// Collectively splits a communicator into one or more new communicators.
    ///
    /// Ranks that pass the same `color` join the same subgroup. Use
    /// [`SplitColor::Undefined`] to exclude a rank from all subgroups; in that
    /// case this returns `Ok(None)`.
    ///
    /// `key` determines the rank ordering inside the new communicator. If keys
    /// are equal, NCCL falls back to the original communicator rank order.
    ///
    /// Passing `None` for `config` makes the new communicator inherit the parent
    /// communicator configuration.
    ///
    /// Split only when no NCCL operations are outstanding on the parent communicator.
    /// Outstanding parent operations can deadlock the split.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// NCCL split errors are propagated.
    pub fn split_with_config(
        &self,
        color: SplitColor,
        key: i32,
        config: Option<Config<'_>>,
    ) -> Result<Option<Self>> {
        self.bind()?;

        let mut handle = ptr::null_mut();
        let mut config = config.map(Config::into_raw);
        // SAFETY: `ncclCommSplit` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommSplit(
                self.as_raw(),
                color.into_raw(),
                key,
                &raw mut handle,
                config.as_mut().map_or(ptr::null_mut(), ptr::from_mut),
            ))?;
        }

        if handle.is_null() {
            return Ok(None);
        }

        Ok(Some(Self {
            handle: Handle {
                raw: handle,
                cuda_context: Arc::clone(self.cuda_context()),
                state: AtomicU8::new(CommunicatorState::Active.into_raw()),
                initialization_output: None,
                launch_events: RefCell::new(Vec::new()),
            },
        }))
    }

    /// Creates a communicator excluding the specified ranks.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Also returns an error if the exclusion count exceeds the `i32` range, NCCL rejects shrinking, or the returned communicator handle is null.
    pub fn shrink(&self, excluded_ranks: &[i32], flags: ShrinkFlags) -> Result<Self> {
        Self::shrink_with_config(self, excluded_ranks, None, flags)
    }

    /// Creates a new communicator by removing selected ranks from an existing communicator.
    /// All ranks participating in the newly created communicator must call this collective operation.
    /// Ranks listed in `excluded_ranks` must not call the shrink operation.
    /// The original ranks listed in `excluded_ranks` are removed from the new communicator.
    /// Within the new communicator, ranks are updated to maintain a contiguous set of IDs.
    /// Passing `None` for `config` makes the new communicator inherit the parent
    /// communicator configuration.
    ///
    /// `flags` controls the behavior of the operation.
    /// Use [`ShrinkFlags::DEFAULT`] for normal operation, or
    /// [`ShrinkFlags::ABORT`] when shrinking after an error on the parent
    /// communicator.
    /// With [`ShrinkFlags::DEFAULT`], no NCCL operations may be outstanding on
    /// the parent communicator.
    /// If the parent communicator configuration enables shrink sharing, NCCL may reuse resources from the parent communicator.
    /// When using [`ShrinkFlags::ABORT`], NCCL aborts any outstanding operations
    /// on the parent communicator and does not share resources with the new
    /// communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Also returns an error if the exclusion count exceeds the `i32` range, NCCL rejects shrinking, or the returned communicator handle is null.
    pub fn shrink_with_config(
        &self,
        excluded_ranks: &[i32],
        config: Option<Config<'_>>,
        flags: ShrinkFlags,
    ) -> Result<Self> {
        self.bind()?;

        let mut excluded_ranks = excluded_ranks.to_vec();
        let excluded_count = to_i32(excluded_ranks.len(), "excluded_ranks")?;
        let mut handle = ptr::null_mut();
        let mut config = config.map(Config::into_raw);
        // SAFETY: `ncclCommShrink` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclCommShrink(
                self.as_raw(),
                excluded_ranks.as_mut_ptr(),
                excluded_count,
                &raw mut handle,
                config.as_mut().map_or(ptr::null_mut(), ptr::from_mut),
                flags.bits().cast_signed(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_context: Arc::clone(self.cuda_context()),
                state: AtomicU8::new(CommunicatorState::Active.into_raw()),
                initialization_output: None,
                launch_events: RefCell::new(Vec::new()),
            },
        })
    }

    /// Returns the raw NCCL communicator.
    ///
    /// The returned communicator is borrowed and remains valid only while this
    /// wrapper is alive.
    pub fn as_raw(&self) -> sys::ncclComm_t {
        self.handle.raw
    }

    /// Transfers the native handle, lifecycle state, and retained resource owners.
    ///
    /// # Safety
    /// Retain every returned owner and native output slot through background
    /// initialization, submission, execution, and replay. Destroy or abort the
    /// raw handle before releasing its owners. This includes unfinished work
    /// started by safe nonblocking initialization and lifecycle methods.
    pub unsafe fn into_raw(mut self) -> RawCommunicator {
        let state = self.state();
        self.set_state(CommunicatorState::Destroyed);
        RawCommunicator {
            handle: self.handle.raw,
            cuda_context: Arc::clone(&self.handle.cuda_context),
            state,
            initialization_output: self.handle.initialization_output.take(),
            launch_events: self.handle.launch_events.take(),
        }
    }

    pub(crate) fn set_state(&self, state: CommunicatorState) {
        self.handle.state.store(state.into_raw(), Ordering::Relaxed);
    }

    fn require_state(&self, expected: CommunicatorState) -> Result<()> {
        if self.state() != expected {
            return self.invalid_state_error();
        }
        Ok(())
    }

    fn require_state_for_control(&self) -> Result<()> {
        if matches!(
            self.state(),
            CommunicatorState::Destroyed
                | CommunicatorState::Aborted
                | CommunicatorState::Suspended
                | CommunicatorState::Initializing
        ) {
            return self.invalid_state_error();
        }
        Ok(())
    }

    fn require_state_for_operations(&self) -> Result<()> {
        if self.state() != CommunicatorState::Active {
            return self.invalid_state_error();
        }
        Ok(())
    }

    fn invalid_state_error(&self) -> Result<()> {
        Err(Error::InvalidCommunicatorState {
            state: self.state().name().into(),
        })
    }
}

fn buffer_byte_len(len: usize, name: &str, element_size: usize) -> Result<u64> {
    let byte_len = len
        .checked_mul(element_size)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))?;
    to_u64(byte_len, name)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::testing::{TestContext, setup_communicator};

    fn maybe_setup() -> Option<TestContext> {
        match setup_communicator() {
            Ok(context) => Some(context),
            Err(error) => {
                eprintln!("skipping nccl runtime test: {error}");
                None
            }
        }
    }

    #[test]
    fn test_single_rank_metadata_is_available() -> Result<()> {
        let Some(context) = maybe_setup() else {
            return Ok(());
        };

        assert_eq!(context.state(), CommunicatorState::Active);
        assert_eq!(context.rank_count()?, 1);
        assert_eq!(context.rank()?, 0);
        assert_eq!(context.device()?.id(), context.cuda_context().device().id());
        assert!(context.async_error()?.is_none());

        Ok(())
    }

    #[test]
    fn test_destroy_consumes_communicator() -> Result<()> {
        let Some(context) = maybe_setup() else {
            return Ok(());
        };

        context.into_communicator().destroy()?;

        Ok(())
    }

    #[test]
    fn test_finalize_blocks_new_operations_locally() -> Result<()> {
        let Some(context) = maybe_setup() else {
            return Ok(());
        };
        let mut buffer = DeviceMemory::<f32>::from_slice(context.cuda_context(), &[])?;

        context.finalize()?;

        assert_eq!(context.state(), CommunicatorState::Finalizing);
        // SAFETY: the finalized handle rejects this empty operation before submission.
        match unsafe { context.broadcast_in_place(buffer.view_mut(), 0, context.stream()) } {
            Err(Error::InvalidCommunicatorState { state }) => {
                assert_eq!(state, "finalizing")
            }
            other => panic!("expected finalizing communicator error, got {other:?}"),
        }

        Ok(())
    }
}
