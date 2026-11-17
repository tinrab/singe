use std::mem;

use singe_core::{abort_after_cleanup_failure, report_cleanup_failure};
use singe_cuda::{handle::NativeHandle, view::DeviceView};

use crate::communicator::{
    CommunicatorState, CustomReductionOperator, DeviceRepr, DeviceViewMut, Error, HandleState,
    NcclCommunicator, Ordering, RegisteredBuffer, Result, Window, sys, try_ffi,
};

impl<T: DeviceRepr> RegisteredBuffer<'_, T> {
    /// Borrows the registered allocation for a collective input.
    #[must_use]
    pub fn buffer(&self) -> DeviceView<'_, T> {
        self.buffer.as_view()
    }

    /// Borrows the registered allocation for a collective output.
    pub fn buffer_mut(&mut self) -> DeviceViewMut<'_, T> {
        self.buffer.reborrow()
    }

    /// Returns the borrowed native registration or operator handle.
    #[must_use]
    pub const fn as_raw(&self) -> *mut () {
        self.raw
    }

    /// Deregisters this buffer from its communicator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot deregister the buffer.
    pub fn deregister(self) -> Result<()> {
        self.communicator.bind()?;
        self.communicator.cuda_context().synchronize()?;
        // SAFETY: `self.raw` is this live registration on the borrowed communicator, the context is bound and synchronized, and `deregister` consumes `self`, so it is released once.
        unsafe {
            try_ffi!(sys::ncclCommDeregister(
                self.communicator.as_raw(),
                self.raw.cast(),
            ))?;
        }

        self.communicator.set_state(self.communicator.state());
        mem::forget(self);
        Ok(())
    }
}

impl<T: DeviceRepr> Drop for RegisteredBuffer<'_, T> {
    fn drop(&mut self) {
        if matches!(
            self.communicator.state(),
            CommunicatorState::Destroyed | CommunicatorState::Aborted
        ) {
            return;
        }

        if let Err(error) = self.communicator.bind() {
            report_cleanup_failure("bind the cuda context to deregister", "nccl buffer", &error);
            return;
        }

        self.communicator.cuda_context().synchronize_or_abort();
        // SAFETY: `self.raw` is this live registration on the borrowed communicator, which is neither destroyed nor aborted, and its context is bound and synchronized.
        unsafe {
            if let Err(error) = try_ffi!(sys::ncclCommDeregister(
                self.communicator.as_raw(),
                self.raw.cast(),
            )) {
                report_cleanup_failure("deregister", "nccl buffer", &error);
            }
        }
    }
}

impl Window<'_> {
    /// Returns the borrowed native registration or operator handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::ncclWindow_t {
        self.raw
    }

    /// Deregisters this NCCL window from its communicator.
    /// Deregistration is local to the rank.
    /// The corresponding window buffer must not be in use by any NCCL operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot deregister the window.
    pub fn deregister(self) -> Result<()> {
        self.communicator.bind()?;
        self.communicator.cuda_context().synchronize()?;
        // SAFETY: `self.raw` is this live window on the borrowed communicator, the context is bound and synchronized, and `deregister` consumes `self`, so it is released once.
        unsafe {
            try_ffi!(sys::ncclCommWindowDeregister(
                self.communicator.as_raw(),
                self.raw,
            ))?;
        }

        mem::forget(self);
        Ok(())
    }
}

impl Drop for Window<'_> {
    fn drop(&mut self) {
        if matches!(
            self.communicator.state(),
            CommunicatorState::Destroyed | CommunicatorState::Aborted
        ) {
            return;
        }

        if let Err(error) = self.communicator.bind() {
            report_cleanup_failure("bind the cuda context to deregister", "nccl window", &error);
            return;
        }

        self.communicator.cuda_context().synchronize_or_abort();
        // SAFETY: `self.raw` is this live window on the borrowed communicator, which is neither destroyed nor aborted, and its context is bound and synchronized.
        unsafe {
            if let Err(error) = try_ffi!(sys::ncclCommWindowDeregister(
                self.communicator.as_raw(),
                self.raw,
            )) {
                report_cleanup_failure("deregister", "nccl window", &error);
            }
        }
    }
}

impl<T> CustomReductionOperator<'_, T> {
    /// Returns the native operator value, valid while this operator lives.
    #[must_use]
    pub const fn as_raw(&self) -> sys::ncclRedOp_t {
        self.raw
    }

    /// Destroys this custom reduction operator.
    /// The operator must have been created by [`Communicator::create_custom_pre_mul_sum`](crate::communicator::Communicator::create_custom_pre_mul_sum) on the same communicator.
    /// It may be destroyed as soon as the last NCCL call using it has returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot destroy the operator.
    pub fn destroy(self) -> Result<()> {
        self.communicator.bind()?;
        // SAFETY: `self.raw` is the live operator registered on this communicator; `destroy` consumes `self`, so it is destroyed once.
        unsafe {
            try_ffi!(sys::ncclRedOpDestroy(self.raw, self.communicator.as_raw()))?;
        }

        mem::forget(self);
        Ok(())
    }
}

impl<T> Drop for CustomReductionOperator<'_, T> {
    fn drop(&mut self) {
        if matches!(
            self.communicator.state(),
            CommunicatorState::Destroyed | CommunicatorState::Aborted
        ) {
            return;
        }

        if let Err(error) = self.communicator.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nccl reduction operator",
                &error,
            );
            return;
        }

        // SAFETY: `self.raw` is the live operator on the borrowed communicator, which is neither destroyed nor aborted; drop runs once.
        unsafe {
            if let Err(error) =
                try_ffi!(sys::ncclRedOpDestroy(self.raw, self.communicator.as_raw()))
            {
                report_cleanup_failure("destroy", "nccl reduction operator", &error);
            }
        }
    }
}

// SAFETY: NCCL lets a communicator be used and destroyed from any thread, one call at a time, and `destroy` aborts the process instead of releasing an output slot that initialization, a split, or a shrink may still write.
unsafe impl NativeHandle for NcclCommunicator {
    type Raw = sys::ncclComm_t;

    type State = HandleState;

    type Error = Error;

    const OBJECT: &'static str = "nccl communicator";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, state: HandleState) -> Result<()> {
        // SAFETY: the caller passes the live, uniquely owned communicator described by `state`, with its context bound.
        let result = unsafe { release(raw, &state) };
        if let Err(error) = &result
            && (state.initialization_output.is_some() || !state.child_outputs.borrow().is_empty())
        {
            abort_after_cleanup_failure("destroy", Self::OBJECT, error);
        }
        result
    }
}

// An aborted or destroyed communicator is already released.
// Initialization that is still in progress or failed must be aborted, since NCCL rejects destroying it.
// Callers must pass a live, uniquely owned communicator described by `state`, with its context bound.
unsafe fn release(raw: sys::ncclComm_t, state: &HandleState) -> Result<()> {
    let lifecycle = CommunicatorState::from_raw(state.lifecycle.load(Ordering::Relaxed));
    // SAFETY: the caller passes a live communicator with its context bound, and `status` is a local output slot.
    unsafe {
        match lifecycle {
            CommunicatorState::Destroyed | CommunicatorState::Aborted => Ok(()),
            CommunicatorState::Initializing => {
                let mut status = sys::ncclResult_t::ncclInProgress;
                let initialized = try_ffi!(sys::ncclCommGetAsyncError(raw, &raw mut status))
                    .is_ok()
                    && status == sys::ncclResult_t::ncclSuccess;
                if initialized {
                    try_ffi!(sys::ncclCommDestroy(raw))
                } else {
                    try_ffi!(sys::ncclCommAbort(raw))
                }
            }
            _ => try_ffi!(sys::ncclCommDestroy(raw)),
        }
    }
}
