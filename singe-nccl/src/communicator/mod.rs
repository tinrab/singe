pub mod device;

mod create;
mod lifecycle;
mod one_sided;
mod operations;
mod resources;

use std::{
    cell::RefCell,
    convert::Infallible,
    ffi::CStr,
    marker::PhantomData,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicU8, Ordering},
    },
    thread,
};

use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    error::Error as CudaError,
    event::Event,
    handle::{Handle, RawHandle},
    stream::{Stream, StreamScope},
    submission::Access,
    view::{DeviceRepr, DeviceViewMut},
};

use crate::{
    error::{Error, Result, Status},
    group, sys, try_ffi,
    types::{CommunicatorState, Config, ReductionOperator, ShrinkFlags, SplitColor, UniqueId},
    utility::{ensure_primary_context, to_i32},
};

pub use self::{
    lifecycle::{CommunicatorGrowth, MemoryStatistics},
    one_sided::WaitSignal,
};

/// A stateful NCCL communicator in a device primary CUDA context.
///
/// Use one communicator per host thread or concurrent task.
/// The communicator is movable between threads, but it is intentionally not `Clone` or `Sync`.
///
/// Its collective and point-to-point methods are `unsafe` `*_async` functions that follow the crate's [raw execution](crate#raw-execution) contract, and safe `*_scoped` forms whose [`StreamScope`] waits for the work.
/// [`LocalCommunicatorGroup`](crate::local::LocalCommunicatorGroup) runs them for all local ranks and waits for completion.
#[derive(Debug)]
pub struct Communicator {
    pub(crate) handle: Handle<NcclCommunicator>,
}

/// A native communicator transferred with its actual context and retained resources.
/// Destroy or abort the native handle before releasing these owners, or return the complete value through [`Communicator::from_raw`].
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
    /// Stable storage for split or shrink results that an unfinished job of this communicator may still write.
    pub child_outputs: Vec<Box<AtomicPtr<sys::ncclComm>>>,
    /// Events retained by queued operations or captured graphs.
    pub launch_events: Vec<Arc<Event>>,
}

#[derive(Debug)]
pub(crate) struct NcclCommunicator(Infallible);

// The state `Handle` keeps with a native communicator and passes to its teardown.
#[derive(Debug)]
pub(crate) struct HandleState {
    lifecycle: AtomicU8,
    // NCCL writes this slot from its asynchronous initialization job, so it lives as long as the native communicator.
    initialization_output: Option<Box<AtomicPtr<sys::ncclComm>>>,
    // A split or shrink job whose completion could not be observed may still write these slots, so they live as long as the native communicator.
    #[expect(
        clippy::vec_box,
        reason = "NCCL writes through each slot's address, which must not move when the vector grows"
    )]
    child_outputs: RefCell<Vec<Box<AtomicPtr<sys::ncclComm>>>>,
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

/// A custom reduction operator for elements of type `T`, owned by and borrowing its communicator.
#[derive(Debug)]
pub struct CustomReductionOperator<'a, T> {
    communicator: &'a Communicator,
    raw: sys::ncclRedOp_t,
    _element: PhantomData<fn() -> T>,
}

/// The reduction a collective applies to elements of type `T`.
#[derive(Debug)]
pub enum Reduction<'a, T> {
    /// A built-in NCCL reduction operator.
    BuiltIn(ReductionOperator),
    /// A custom operator created on the communicator that runs the collective.
    Custom(&'a CustomReductionOperator<'a, T>),
}

impl<T> Clone for Reduction<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Reduction<'_, T> {}

impl<T> From<ReductionOperator> for Reduction<'_, T> {
    fn from(operator: ReductionOperator) -> Self {
        Self::BuiltIn(operator)
    }
}

impl<'a, 'c: 'a, T> From<&'a CustomReductionOperator<'c, T>> for Reduction<'a, T> {
    fn from(operator: &'a CustomReductionOperator<'c, T>) -> Self {
        Self::Custom(operator)
    }
}

impl HandleState {
    #[expect(
        clippy::vec_box,
        reason = "NCCL writes through each slot's address, which must not move when the vector grows"
    )]
    pub(crate) fn new(
        lifecycle: CommunicatorState,
        initialization_output: Option<Box<AtomicPtr<sys::ncclComm>>>,
        child_outputs: Vec<Box<AtomicPtr<sys::ncclComm>>>,
        launch_events: Vec<Arc<Event>>,
    ) -> Self {
        Self {
            lifecycle: AtomicU8::new(lifecycle.into_raw()),
            initialization_output,
            child_outputs: RefCell::new(child_outputs),
            launch_events: RefCell::new(launch_events),
        }
    }
}

impl Communicator {
    /// Wraps an existing NCCL communicator and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be uniquely owned and associated with `raw.cuda_context`, with the declared native lifecycle state.
    /// It must be the device primary context.
    /// Retain any native initialization output slot and launch events in `raw`; NCCL must not write through a different output slot or use released resources.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null or the retained context is not the device primary CUDA context.
    pub unsafe fn from_raw(raw: RawCommunicator) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        ensure_primary_context(&raw.cuda_context)?;
        let state = HandleState::new(
            raw.state,
            raw.initialization_output,
            raw.child_outputs,
            raw.launch_events,
        );
        // SAFETY: the caller passes a uniquely owned handle created in `raw.cuda_context`, with its lifecycle state, output slot, and launch events.
        unsafe { Self::adopt(raw.handle, raw.cuda_context, state) }
    }

    // Callers must pass a handle NCCL created in `cuda_context`, described by `state`, that no other owner destroys or aborts.
    pub(crate) unsafe fn adopt(
        raw: sys::ncclComm_t,
        cuda_context: Arc<CudaContext>,
        state: HandleState,
    ) -> Result<Self> {
        // SAFETY: the caller's contract; NCCL communicators install no stream.
        let handle = unsafe {
            Handle::from_raw(RawHandle {
                handle: raw,
                cuda_context,
                stream: None,
                state,
            })?
        };
        Ok(Self { handle })
    }

    /// Returns the CUDA context retained by this communicator.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
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
        CommunicatorState::from_raw(self.handle.state().lifecycle.load(Ordering::Relaxed))
    }

    pub(crate) fn ensure_buffer_context(&self, context: &Arc<CudaContext>) -> Result<()> {
        if context != self.cuda_context() {
            return Err(Error::Cuda(CudaError::MemoryContextMismatch));
        }
        Ok(())
    }

    pub(crate) fn ensure_stream(&self, stream: &Stream) -> Result<()> {
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
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot report the communicator size.
    pub fn rank_count(&self) -> Result<i32> {
        self.bind()?;

        let mut value = 0;
        // SAFETY: `self` owns the live communicator handle, its context is bound, and `value` is a local output slot.
        unsafe {
            try_ffi!(sys::ncclCommCount(self.as_raw(), &raw mut value))?;
        }
        Ok(value)
    }

    /// Returns the caller's rank within this communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot report this communicator's rank.
    pub fn rank(&self) -> Result<i32> {
        self.bind()?;

        let mut value = 0;
        // SAFETY: `self` owns the live communicator handle, its context is bound, and `value` is a local output slot.
        unsafe {
            try_ffi!(sys::ncclCommUserRank(self.as_raw(), &raw mut value))?;
        }
        Ok(value)
    }

    /// Returns the CUDA device associated with this communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot report the communicator device.
    pub fn device(&self) -> Result<Device> {
        self.bind()?;

        let mut value = 0;
        // SAFETY: `self` owns the live communicator handle, its context is bound, and `value` is a local output slot.
        unsafe {
            try_ffi!(sys::ncclCommCuDevice(self.as_raw(), &raw mut value))?;
        }
        Ok(Device::new(value))
    }

    /// Returns a human-readable string corresponding to the last error that occurred in NCCL.
    /// Retrieving the string does not clear the error.
    /// The returned string may be unrelated to the current call and can reflect a previously launched asynchronous operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn last_error(&self) -> Result<String> {
        self.bind()?;

        // SAFETY: `self` owns the live communicator handle and its context is bound.
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
    /// Operations with a stream argument, such as [`Communicator::all_reduce_async`], report [`Status::Success`] once they are posted to the stream, but may still surface failures through [`Communicator::async_error`] until execution completes.
    /// If any NCCL call reports [`Status::InProgress`], the operation is still being enqueued in the background.
    /// Poll communicator state until all involved communicators report [`Status::Success`] before issuing another NCCL call.
    /// Before the state changes to [`Status::Success`], do not launch CUDA kernels on streams currently being used by NCCL.
    /// If there has been an error on the communicator, abort it with [`Communicator::abort`].
    /// If an error occurs on the communicator, nothing can be assumed about the completion or correctness of operations enqueued on that communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot report the asynchronous state.
    pub fn async_error(&self) -> Result<Option<Status>> {
        if self.state() != CommunicatorState::Initializing {
            self.require_state_for_control()?;
        }

        self.cuda_context().bind()?;

        let mut value = sys::ncclResult_t::ncclSuccess;
        // SAFETY: `self` owns the communicator handle, live even while initializing, its context is bound, and `value` is a local output slot.
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
    /// Returns an error if the CUDA context cannot be bound, if the communicator is not active, or if NCCL rejects finalization.
    pub fn finalize(&self) -> Result<()> {
        self.bind()?;
        self.require_state(CommunicatorState::Active)?;
        // SAFETY: `self` owns the handle, checked active above, and its context is bound.
        let status = unsafe { sys::ncclCommFinalize(self.as_raw()) };
        check_nonblocking(status)?;
        self.set_state(CommunicatorState::Finalizing);
        Ok(())
    }

    /// Aborts this communicator and frees its resources.
    /// All active ranks must call this to abort the NCCL communicator successfully.
    /// Recovery path once a communicator has entered an error state.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if the communicator is already destroyed or aborted, or if NCCL rejects the abort.
    pub fn abort(&mut self) -> Result<()> {
        self.cuda_context().bind()?;
        if matches!(
            self.state(),
            CommunicatorState::Destroyed | CommunicatorState::Aborted
        ) {
            return self.invalid_state_error();
        }

        // SAFETY: `self` owns the handle, checked neither destroyed nor aborted above, and its context is bound.
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
    /// Returns an error if the CUDA context cannot be bound, if the communicator is not active, or if NCCL rejects revocation.
    pub fn revoke(&self) -> Result<()> {
        self.bind()?;
        self.require_state(CommunicatorState::Active)?;
        // SAFETY: `self` owns the handle, checked active above, and its context is bound.
        let status =
            unsafe { sys::ncclCommRevoke(self.as_raw(), sys::NCCL_REVOKE_DEFAULT.cast_signed()) };
        check_nonblocking(status)?;
        self.set_state(CommunicatorState::Revoked);
        Ok(())
    }

    /// Destroys the native communicator, or aborts it if its initialization is in progress or failed.
    ///
    /// Does nothing native for an aborted communicator.
    /// If the CUDA context cannot be bound, the communicator and its retained resources leak.
    /// If teardown fails while initialization, a split, or a shrink may still write an output slot, the process aborts.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails or NCCL rejects destruction or abort.
    pub fn destroy(self) -> Result<()> {
        self.handle.destroy()
    }

    /// Collectively splits a communicator into one or more new communicators.
    ///
    /// Ranks that pass the same `color` join the same subgroup.
    /// Use [`SplitColor::Undefined`] to exclude a rank from all subgroups; in that case this returns `Ok(None)`.
    ///
    /// `key` determines the rank ordering inside the new communicator.
    /// If keys are equal, NCCL falls back to the original communicator rank order.
    ///
    /// Passing `None` for `config` makes the new communicator inherit the parent communicator configuration.
    /// With a nonblocking parent, this waits until the parent reports that NCCL finished creating the new communicator.
    ///
    /// Split only when no NCCL operations are outstanding on the parent communicator.
    /// Outstanding parent operations can deadlock the split.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, the communicator state disallows control operations, or binding its CUDA context fails.
    /// NCCL split errors are propagated.
    pub fn split(
        &self,
        color: SplitColor,
        key: i32,
        config: Option<Config<'_>>,
    ) -> Result<Option<Self>> {
        reject_group()?;
        self.bind()?;

        // The split's job writes the new communicator into this slot when it completes, so the slot must not move.
        let output = Box::new(AtomicPtr::new(ptr::null_mut()));
        let mut config = config.map(Config::into_raw);
        // SAFETY: `self` owns the live parent handle with its context bound, `output` is boxed so it stays put for an asynchronous write, and `config` is null or a local native configuration.
        let status = unsafe {
            sys::ncclCommSplit(
                self.as_raw(),
                color.into_raw(),
                key,
                output.as_ptr(),
                config.as_mut().map_or(ptr::null_mut(), ptr::from_mut),
            )
        };
        self.finish_child(output, status)
    }

    /// Creates a new communicator by removing selected ranks from an existing communicator.
    /// All ranks participating in the newly created communicator must call this collective operation.
    /// Ranks listed in `excluded_ranks` must not call the shrink operation.
    /// The original ranks listed in `excluded_ranks` are removed from the new communicator.
    /// Within the new communicator, ranks are updated to maintain a contiguous set of IDs.
    /// Passing `None` for `config` makes the new communicator inherit the parent communicator configuration.
    /// With a nonblocking parent, this waits until the parent reports that NCCL finished creating the new communicator.
    ///
    /// `flags` controls the behavior of the operation.
    /// Use [`ShrinkFlags::DEFAULT`] for normal operation, or [`ShrinkFlags::ABORT`] when shrinking after an error on the parent communicator.
    /// With [`ShrinkFlags::DEFAULT`], no NCCL operations may be outstanding on the parent communicator.
    /// If the parent communicator configuration enables shrink sharing, NCCL may reuse resources from the parent communicator.
    /// When using [`ShrinkFlags::ABORT`], NCCL aborts any outstanding operations on the parent communicator and does not share resources with the new communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if an NCCL group is open on this thread, the communicator state disallows control operations, or binding its CUDA context fails.
    /// Also returns an error if the exclusion count exceeds the `i32` range, NCCL rejects shrinking, or the returned communicator handle is null.
    pub fn shrink(
        &self,
        excluded_ranks: &[i32],
        flags: ShrinkFlags,
        config: Option<Config<'_>>,
    ) -> Result<Self> {
        reject_group()?;
        self.bind()?;

        let mut excluded_ranks = excluded_ranks.to_vec();
        let excluded_count = to_i32(excluded_ranks.len(), "excluded_ranks")?;
        // The shrink's job writes the new communicator into this slot when it completes, so the slot must not move.
        let output = Box::new(AtomicPtr::new(ptr::null_mut()));
        let mut config = config.map(Config::into_raw);
        // SAFETY: `self` owns the live parent handle with its context bound, `excluded_ranks` holds `excluded_count` ranks, `output` is boxed so it stays put for an asynchronous write, and `config` is null or a local native configuration.
        let status = unsafe {
            sys::ncclCommShrink(
                self.as_raw(),
                excluded_ranks.as_mut_ptr(),
                excluded_count,
                output.as_ptr(),
                config.as_mut().map_or(ptr::null_mut(), ptr::from_mut),
                flags.bits().cast_signed(),
            )
        };
        self.finish_child(output, status)?.ok_or(Error::NullHandle)
    }

    /// Returns the raw NCCL communicator.
    ///
    /// The returned communicator is borrowed and remains valid only while this wrapper is alive.
    pub fn as_raw(&self) -> sys::ncclComm_t {
        self.handle.as_raw()
    }

    /// Transfers the native handle, lifecycle state, and retained resource owners.
    ///
    /// # Safety
    ///
    /// Retain every returned owner and native output slot through background initialization, submission, execution, and replay.
    /// Destroy or abort the raw handle before releasing its owners.
    /// This includes unfinished work started by safe nonblocking initialization and lifecycle methods.
    pub unsafe fn into_raw(self) -> RawCommunicator {
        let raw = self.handle.into_raw();
        let HandleState {
            lifecycle,
            initialization_output,
            child_outputs,
            launch_events,
        } = raw.state;
        RawCommunicator {
            handle: raw.handle,
            cuda_context: raw.cuda_context,
            state: CommunicatorState::from_raw(lifecycle.into_inner()),
            initialization_output,
            child_outputs: child_outputs.into_inner(),
            launch_events: launch_events.into_inner(),
        }
    }

    pub(crate) fn set_state(&self, state: CommunicatorState) {
        self.handle
            .state()
            .lifecycle
            .store(state.into_raw(), Ordering::Relaxed);
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

    // Finishes a split or shrink, whose job stores the new communicator, or null for an excluded rank, into `output` only when it completes.
    // A nonblocking parent returns `InProgress` with the slot still null and tracks the job in its asynchronous state, so this waits for that state before reading the slot.
    fn finish_child(
        &self,
        output: Box<AtomicPtr<sys::ncclComm>>,
        status: sys::ncclResult_t,
    ) -> Result<Option<Self>> {
        let status = if status == sys::ncclResult_t::ncclInProgress {
            match self.wait_for_job() {
                Ok(status) => status,
                Err(error) => {
                    // The job's completion could not be observed, so it may still write the slot, and the parent keeps it until its native release ends the job.
                    self.handle.state().child_outputs.borrow_mut().push(output);
                    return Err(error);
                }
            }
        } else {
            status
        };
        let raw = output.load(Ordering::Acquire);
        if raw.is_null() && status == sys::ncclResult_t::ncclSuccess {
            return Ok(None);
        }

        // SAFETY: `raw` and `status` come from this communicator's finished split or shrink in its context, nothing else owns the new handle, and `output` is the slot that job wrote.
        unsafe {
            create::communicator_from_initialization(
                raw,
                Arc::clone(self.cuda_context()),
                status,
                Some(output),
            )
        }
        .map(Some)
    }

    // Polls this communicator's asynchronous state until its pending job finishes, and returns the job's status.
    fn wait_for_job(&self) -> Result<sys::ncclResult_t> {
        loop {
            let mut value = sys::ncclResult_t::ncclSuccess;
            // SAFETY: `self` owns the live communicator handle, its context is bound, and `value` is a local output slot.
            unsafe {
                try_ffi!(sys::ncclCommGetAsyncError(self.as_raw(), &raw mut value))?;
            }

            if value != sys::ncclResult_t::ncclInProgress {
                return Ok(value);
            }

            thread::yield_now();
        }
    }

    pub(crate) fn ensure_rank(&self, rank: i32, name: &'static str) -> Result<()> {
        if rank < 0 || rank >= self.rank_count()? {
            return Err(Error::out_of_range(name));
        }
        Ok(())
    }

    pub(crate) fn reduction<T>(&self, op: Reduction<'_, T>) -> Result<sys::ncclRedOp_t> {
        match op {
            Reduction::BuiltIn(op) => Ok(op.into()),
            Reduction::Custom(op) => {
                if !ptr::eq(op.communicator, self) {
                    return Err(Error::ContextMismatch {
                        name: "custom reduction operator".into(),
                    });
                }
                Ok(op.raw)
            }
        }
    }

    // Enqueues one operation in `scope` after declaring its accesses.
    // Nonblocking submission finishes inside the enqueue, so the scope's completion event and drain follow the native work.
    // An open group would defer the native enqueue past the scope, so it is rejected.
    pub(crate) fn enqueue_scoped(
        &self,
        accesses: &[Access],
        scope: &StreamScope<'_, '_>,
        enqueue: impl FnOnce(&Stream) -> Result<()>,
    ) -> Result<()> {
        if group::is_active() {
            return Err(Error::GroupActive);
        }

        let mut outcome = Ok(());
        // SAFETY: callers pass an `enqueue` that issues one NCCL operation on the scope's stream touching only the declared accesses, with operands borrowed for the scope's `'env`, and no group can defer that operation past this call.
        unsafe {
            scope.enqueue(accesses, || {
                outcome = self.finish_submission(enqueue(scope.stream()));
                Ok(())
            })?;
        }
        outcome
    }

    // Waits until NCCL has finished submitting a nonblocking operation to its stream.
    fn finish_submission(&self, result: Result<()>) -> Result<()> {
        match result {
            Ok(())
            | Err(Error::Nccl {
                code: Status::InProgress,
            }) => {}
            Err(error) => return Err(error),
        }

        loop {
            match self.async_error()? {
                None => return Ok(()),
                Some(Status::InProgress) => thread::yield_now(),
                Some(status) => return Err(status.into()),
            }
        }
    }
}

// Nonblocking communicators report `InProgress` while NCCL completes a call in the background.
fn check_nonblocking(status: sys::ncclResult_t) -> Result<()> {
    if status == sys::ncclResult_t::ncclInProgress {
        return Ok(());
    }
    Error::check(status)
}

// A grouped split or shrink stores the new communicator only at the outermost group end, after the call has returned.
fn reject_group() -> Result<()> {
    if group::is_active() {
        return Err(Error::GroupActive);
    }
    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::{collective::CollectiveOptions, testing::setup_communicator};

    #[test]
    fn test_single_rank_metadata_is_available() -> Result<()> {
        let test = setup_communicator()?;
        let context = test.communicator();

        assert_eq!(context.state(), CommunicatorState::Active);
        assert_eq!(context.rank_count()?, 1);
        assert_eq!(context.rank()?, 0);
        assert_eq!(context.device()?.id(), context.cuda_context().device().id());
        assert!(context.async_error()?.is_none());

        Ok(())
    }

    #[test]
    fn test_destroy_consumes_communicator() -> Result<()> {
        let (communicator, _device) = setup_communicator()?.into_parts();

        communicator.destroy()?;

        Ok(())
    }

    #[test]
    fn test_finalize_blocks_new_operations_locally() -> Result<()> {
        let test = setup_communicator()?;
        let context = test.communicator();
        let mut buffer = DeviceMemory::<f32>::from_slice(context.cuda_context(), &[])?;

        context.finalize()?;

        assert_eq!(context.state(), CommunicatorState::Finalizing);
        // SAFETY: the finalized handle rejects this empty operation before submission.
        match unsafe {
            context.broadcast_in_place_async(
                buffer.view_mut(),
                0,
                &CollectiveOptions::default(),
                test.stream(),
            )
        } {
            Err(Error::InvalidCommunicatorState { state }) => {
                assert_eq!(state, "finalizing");
            }
            other => panic!("expected finalizing communicator error, got {other:?}"),
        }

        Ok(())
    }
}
