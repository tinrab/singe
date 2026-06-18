use super::*;

impl RegisteredBuffer<'_> {
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
        unsafe {
            try_ffi!(sys::ncclCommDeregister(
                self.communicator.as_raw(),
                self.raw.cast(),
            ))?;
        }
        self.communicator.set_state(self.communicator.state());
        std::mem::forget(self);
        Ok(())
    }
}

impl Drop for RegisteredBuffer<'_> {
    fn drop(&mut self) {
        if self.communicator.state() == CommunicatorState::Destroyed {
            return;
        }

        if let Err(err) = self.communicator.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before deregistering nccl buffer: {err}");
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::ncclCommDeregister(
                self.communicator.as_raw(),
                self.raw.cast(),
            )) {
                #[cfg(debug_assertions)]
                eprintln!("failed to deregister nccl buffer: {err}");
            }
        }
    }
}

impl Window<'_> {
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
        unsafe {
            try_ffi!(sys::ncclCommWindowDeregister(
                self.communicator.as_raw(),
                self.raw,
            ))?;
        }
        std::mem::forget(self);
        Ok(())
    }
}

impl Drop for Window<'_> {
    fn drop(&mut self) {
        if self.communicator.state() == CommunicatorState::Destroyed {
            return;
        }

        if let Err(err) = self.communicator.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before deregistering nccl window: {err}");
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::ncclCommWindowDeregister(
                self.communicator.as_raw(),
                self.raw,
            )) {
                #[cfg(debug_assertions)]
                eprintln!("failed to deregister nccl window: {err}");
            }
        }
    }
}

impl CustomReductionOperator<'_> {
    pub const fn as_raw(&self) -> sys::ncclRedOp_t {
        self.raw
    }

    /// Destroys this custom reduction operator.
    /// The operator must have been created by [`Communicator::create_custom_pre_mul_sum`] on the same communicator.
    /// It may be destroyed as soon as the last NCCL call using it has returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if NCCL cannot
    /// destroy the operator.
    pub fn destroy(self) -> Result<()> {
        self.communicator.bind()?;
        unsafe {
            try_ffi!(sys::ncclRedOpDestroy(self.raw, self.communicator.as_raw()))?;
        }
        std::mem::forget(self);
        Ok(())
    }
}

impl Drop for CustomReductionOperator<'_> {
    fn drop(&mut self) {
        if self.communicator.state() == CommunicatorState::Destroyed {
            return;
        }

        if let Err(err) = self.communicator.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying nccl reduction op: {err}");
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::ncclRedOpDestroy(self.raw, self.communicator.as_raw()))
            {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy nccl reduction op: {err}");
            }
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.state.load(Ordering::Relaxed) == CommunicatorState::Destroyed.into_raw() {
            return;
        }

        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying nccl communicator: {err}");
        }

        if let Err(err) = self.destroy_raw() {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy nccl communicator: {err}");
        }
    }
}

impl Handle {
    pub(super) fn destroy_raw(&self) -> Result<()> {
        let previous = self
            .state
            .swap(CommunicatorState::Destroyed.into_raw(), Ordering::AcqRel);
        if previous == CommunicatorState::Destroyed.into_raw() {
            return Ok(());
        }

        unsafe {
            try_ffi!(sys::ncclCommDestroy(self.raw))?;
        }
        Ok(())
    }
}
