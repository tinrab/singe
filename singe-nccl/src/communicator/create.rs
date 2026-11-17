use std::sync::{
    Mutex, PoisonError,
    atomic::{AtomicPtr, Ordering},
};

use crate::{
    communicator::{
        Arc, Communicator, CommunicatorState, Config, CudaContext, Device, Error, HandleState,
        Result, UniqueId, ensure_primary_context, ptr, sys, to_i32, try_ffi,
    },
    group::{is_active, with_group},
    types::RankId,
};

impl Communicator {
    /// Creates a blocking communicator clique within one process, with one rank per device in `devices`, in order.
    ///
    /// Each rank retains its device's primary CUDA context, as NCCL requires.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is already open, a primary CUDA context cannot be retained, the rank count cannot be represented for NCCL, NCCL rejects initialization, or NCCL returns a null communicator handle.
    pub fn create_all(devices: &[Device]) -> Result<Vec<Self>> {
        if is_active() {
            return Err(Error::GroupActive);
        }

        let contexts = devices
            .iter()
            .map(|&device| CudaContext::create_for_device(device).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let rank_count = to_i32(contexts.len(), "devices")?;
        let unique_id = crate::unique_id()?.into_raw();
        // Stable output slots stay live through the outermost group end.
        let mut handles = vec![ptr::null_mut(); contexts.len()];
        let result = with_group(|| {
            for (rank, (context, handle)) in contexts.iter().zip(&mut handles).enumerate() {
                context.bind()?;
                // SAFETY: `handle` is a valid output slot that outlives the group, and each rank initializes in its own retained primary context.
                unsafe {
                    try_ffi!(sys::ncclCommInitRank(
                        handle,
                        rank_count,
                        unique_id,
                        to_i32(rank, "rank")?
                    ))?;
                }
            }
            Ok(())
        });
        if let Err(error) = result {
            for (context, handle) in contexts.iter().zip(&handles) {
                if !handle.is_null() && context.bind().is_ok() {
                    // SAFETY: a failed grouped initialization still transfers ownership of each non-null handle, which nothing else owns.
                    unsafe {
                        sys::ncclCommAbort(*handle);
                    }
                }
            }
            return Err(error);
        }
        handles
            .into_iter()
            .zip(contexts)
            .map(|(handle, context)| {
                // SAFETY: each handle comes from the successful `ncclCommInitRank` above in its own retained context, and nothing else owns it.
                unsafe { communicator_from_raw(handle, context) }
            })
            .collect()
    }

    /// Creates the communicator for `rank` in a clique of `rank_count` ranks.
    ///
    /// Every rank passes the same `unique_ids`, generated once with [`unique_id`](crate::unique_id) and distributed to all ranks.
    /// Passing more than one identifier selects NCCL's scalable initialization.
    /// `cuda_context` must be the primary context of the rank's device.
    /// The call synchronizes with the other ranks, so each rank calls it from its own thread or process, or all ranks call it inside one [`Group`](crate::group::Group).
    /// With a nonblocking `config`, poll [`Self::async_error`] until it returns `None` before using the communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if `unique_ids` is empty or its length cannot be represented, `cuda_context` is not the device primary context or cannot be bound, `rank` or `rank_count` is invalid for the clique, NCCL rejects initialization or `config`, or NCCL returns a null communicator handle.
    pub fn create_rank(
        cuda_context: &Arc<CudaContext>,
        rank: RankId,
        rank_count: i32,
        unique_ids: &[UniqueId],
        config: Option<Config<'_>>,
    ) -> Result<Self> {
        if unique_ids.is_empty() {
            return Err(Error::Empty {
                name: "unique_ids".into(),
            });
        }

        ensure_primary_context(cuda_context)?;
        cuda_context.bind()?;

        let mut raw_ids: Vec<_> = unique_ids.iter().copied().map(UniqueId::into_raw).collect();
        let raw_id_count = to_i32(raw_ids.len(), "unique_ids")?;
        let mut config = config.map(Config::into_raw);
        // A nonblocking or grouped initialization writes this slot again from its job, so it must not move or be freed while that job may run.
        let output = Box::new(AtomicPtr::new(ptr::null_mut()));
        // SAFETY: `output` is boxed so it stays put for an asynchronous write, the identifier array is valid for the call, the configuration is null or a valid native configuration, and the primary context is bound.
        let status = unsafe {
            sys::ncclCommInitRankScalable(
                output.as_ptr(),
                rank_count,
                rank.get(),
                raw_id_count,
                raw_ids.as_mut_ptr(),
                config.as_mut().map_or(ptr::null_mut(), ptr::from_mut),
            )
        };
        let handle = output.load(Ordering::Acquire);
        // SAFETY: `handle` and `status` come from `ncclCommInitRankScalable` above in `cuda_context`, nothing else owns the handle, and `output` is the slot its initialization writes.
        unsafe {
            communicator_from_initialization(handle, Arc::clone(cuda_context), status, Some(output))
        }
    }
}

// Initialization output slots that a pending NCCL job may still write after initialization returned no handle.
// No communicator owns them, so they stay allocated for the life of the process.
#[expect(
    clippy::vec_box,
    reason = "NCCL writes through each slot's address, which must not move when the vector grows"
)]
static ORPHANED_OUTPUTS: Mutex<Vec<Box<AtomicPtr<sys::ncclComm>>>> = Mutex::new(Vec::new());

// Callers must pass a null handle or one NCCL created in `cuda_context` that no other owner destroys or aborts.
pub(crate) unsafe fn communicator_from_raw(
    raw: sys::ncclComm_t,
    cuda_context: Arc<CudaContext>,
) -> Result<Communicator> {
    let state = HandleState::new(CommunicatorState::Active, None, Vec::new(), Vec::new());
    // SAFETY: the caller's contract, with an active lifecycle state and no output slot.
    unsafe { Communicator::adopt(raw, cuda_context, state) }
}

// Nonblocking initialization transfers ownership even while work is in progress.
// Callers must pass the handle and status of one NCCL initialization call made in `cuda_context`, whose handle no other owner destroys or aborts, and the output slot that call may still write.
pub(crate) unsafe fn communicator_from_initialization(
    raw: sys::ncclComm_t,
    cuda_context: Arc<CudaContext>,
    status: sys::ncclResult_t,
    initialization_output: Option<Box<AtomicPtr<sys::ncclComm>>>,
) -> Result<Communicator> {
    if !matches!(
        status,
        sys::ncclResult_t::ncclSuccess | sys::ncclResult_t::ncclInProgress
    ) {
        if !raw.is_null() {
            // NCCL can return an initialization failure with an owned handle.
            // SAFETY: the failed initialization transferred this handle, which nothing else owns.
            unsafe { sys::ncclCommAbort(raw) };
        }
        return Err(status.into());
    }

    if raw.is_null() {
        if let Some(output) = initialization_output
            && (status == sys::ncclResult_t::ncclInProgress || is_active())
        {
            // NCCL stores the handle before it returns, so this path only follows a contract violation; a pending job may still write the slot, and no communicator owns it.
            ORPHANED_OUTPUTS
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(output);
        }
        return Err(Error::NullHandle);
    }

    let lifecycle = if status == sys::ncclResult_t::ncclInProgress {
        CommunicatorState::Initializing
    } else {
        CommunicatorState::Active
    };
    let state = HandleState::new(lifecycle, initialization_output, Vec::new(), Vec::new());
    // SAFETY: the caller passes a handle from a successful or in-progress initialization in `cuda_context` with no other owner, and the output slot that initialization writes.
    unsafe { Communicator::adopt(raw, cuda_context, state) }
}
