use std::{marker::PhantomData, mem::size_of, ptr};

use singe_cuda::view::{DeviceRepr, DeviceViewMut};

use crate::{
    communicator::{Communicator, RegisteredBuffer, Window, buffer_byte_len},
    error::Result,
    sys, try_ffi,
    types::WindowFlags,
};

impl Communicator {
    /// Registers `buffer` with this communicator for zero-copy communication.
    /// Use the returned [`RegisteredBuffer`] to deregister the mapping.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if the buffer byte
    /// size overflows, or if NCCL cannot register the buffer.
    pub fn register_buffer<'a, T: DeviceRepr>(
        &'a self,
        buffer: DeviceViewMut<'a, T>,
    ) -> Result<RegisteredBuffer<'a, T>> {
        self.ensure_buffer_context(buffer.context())?;
        self.require_state_for_operations()?;
        self.bind()?;

        let mut handle = ptr::null_mut();
        let size = buffer_byte_len(buffer.len(), "registered buffer size", size_of::<T>())?;
        // SAFETY: `ncclCommRegister` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclCommRegister(
                self.as_raw(),
                buffer.as_ptr().cast_mut().cast(),
                size,
                &raw mut handle,
            ))?;
        }

        Ok(RegisteredBuffer {
            communicator: self,
            raw: handle.cast(),
            buffer,
        })
    }

    /// Collectively registers `buffer` for this rank as part of an NCCL window.
    /// Every rank in the communicator must participate in the registration, and
    /// by default the registered sizes must match across ranks.
    /// Use the returned [`Window`] to deregister the mapping. If this is called
    /// inside a group, the returned window may not be fully initialized until
    /// [`group_end`](crate::group::end) completes.
    /// `flags` controls NCCL window registration behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator is not active, the buffer belongs to another CUDA context, its byte length exceeds the native range, or CUDA context binding fails.
    /// NCCL window-registration errors are propagated.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "registration takes ownership of the exclusive device borrow for the window lifetime"
    )]
    pub fn register_window<'a, T: DeviceRepr>(
        &'a self,
        buffer: DeviceViewMut<'a, T>,
        flags: WindowFlags,
    ) -> Result<Window<'a>> {
        self.ensure_buffer_context(buffer.context())?;
        self.require_state_for_operations()?;
        self.bind()?;

        let mut handle = ptr::null_mut();
        let size = buffer_byte_len(buffer.len(), "window size", size_of::<T>())?;
        // SAFETY: `ncclCommWindowRegister` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::ncclCommWindowRegister(
                self.as_raw(),
                buffer.as_ptr().cast_mut().cast(),
                size,
                &raw mut handle,
                flags.bits().cast_signed(),
            ))?;
        }

        Ok(Window {
            communicator: self,
            raw: handle,
            pointer: buffer.as_ptr().cast_mut().cast(),
            byte_len: size,
            _buffer: PhantomData,
        })
    }
}
