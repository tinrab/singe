use std::{mem, process};

use singe_cuda::view::DeviceView;

use crate::communicator::{
    CommunicatorState, CustomReductionOperator, DeviceRepr, DeviceViewMut, Handle, Ordering,
    RegisteredBuffer, Result, Window, operations, sys, try_ffi,
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
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// deregister the buffer.
    pub fn deregister(self) -> Result<()> {
        self.communicator.bind()?;
        self.communicator.cuda_context().synchronize()?;
        // SAFETY: `ncclCommDeregister` receives live handles owned by the wrappers.
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
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before deregistering nccl buffer: {error}");
            return;
        }

        if self.communicator.cuda_context().synchronize().is_err() {
            return;
        }
        // SAFETY: `ncclCommDeregister` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::ncclCommDeregister(
                self.communicator.as_raw(),
                self.raw.cast(),
            )) {
                #[cfg(debug_assertions)]
                eprintln!("failed to deregister nccl buffer: {error}");
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
    /// Deregistration is local to the rank. The corresponding window buffer must not be in use by any NCCL operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// deregister the window.
    pub fn deregister(self) -> Result<()> {
        self.communicator.bind()?;
        self.communicator.cuda_context().synchronize()?;
        // SAFETY: `ncclCommWindowDeregister` receives live handles owned by the wrappers.
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
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before deregistering nccl window: {error}");
            return;
        }

        if self.communicator.cuda_context().synchronize().is_err() {
            return;
        }
        // SAFETY: `ncclCommWindowDeregister` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::ncclCommWindowDeregister(
                self.communicator.as_raw(),
                self.raw,
            )) {
                #[cfg(debug_assertions)]
                eprintln!("failed to deregister nccl window: {error}");
            }
        }
    }
}

impl CustomReductionOperator<'_> {
    /// Returns the borrowed native registration or operator handle.
    #[must_use]
    pub const fn as_raw(&self) -> i32 {
        self.raw
    }

    /// Destroys this custom reduction operator.
    /// The operator must have been created by [`Communicator::create_custom_pre_mul_sum`](crate::communicator::Communicator::create_custom_pre_mul_sum) on the same communicator.
    /// It may be destroyed as soon as the last NCCL call using it has returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// destroy the operator.
    pub fn destroy(self) -> Result<()> {
        self.communicator.bind()?;
        // SAFETY: `self.raw` is the live operator registered on this communicator; `destroy` consumes `self`, so it is destroyed once.
        unsafe {
            try_ffi!(operations::custom_ops::destroy_dynamic_operator(
                self.raw,
                self.communicator.as_raw()
            ))?;
        }
        mem::forget(self);
        Ok(())
    }
}

impl Drop for CustomReductionOperator<'_> {
    fn drop(&mut self) {
        if matches!(
            self.communicator.state(),
            CommunicatorState::Destroyed | CommunicatorState::Aborted
        ) {
            return;
        }

        if let Err(error) = self.communicator.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying nccl reduction op: {error}");
            return;
        }

        // SAFETY: this owner destroys its live operator exactly once.
        unsafe {
            if let Err(error) = try_ffi!(operations::custom_ops::destroy_dynamic_operator(
                self.raw,
                self.communicator.as_raw()
            )) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy nccl reduction op: {error}");
            }
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.state.load(Ordering::Relaxed) == CommunicatorState::Destroyed.into_raw() {
            return;
        }

        if let Err(error) = self.cuda_context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying nccl communicator: {error}");
        }

        if let Err(error) = self.destroy_raw() {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy nccl communicator: {error}");
            // Native grow may still write this slot. It cannot be released if
            // destruction or cancellation failed to establish completion.
            if self.initialization_output.is_some() {
                process::abort();
            }
        }
    }
}

impl Handle {
    pub(crate) fn destroy_raw(&self) -> Result<()> {
        let previous = self
            .state
            .swap(CommunicatorState::Destroyed.into_raw(), Ordering::AcqRel);
        if previous == CommunicatorState::Destroyed.into_raw()
            || previous == CommunicatorState::Aborted.into_raw()
        {
            return Ok(());
        }

        // SAFETY: `ncclCommAbort`, `ncclCommDestroy` receive live handles owned by the wrappers.
        unsafe {
            if previous == CommunicatorState::Initializing.into_raw() {
                try_ffi!(sys::ncclCommAbort(self.raw))?;
            } else {
                try_ffi!(sys::ncclCommDestroy(self.raw))?;
            }
        }
        Ok(())
    }
}
