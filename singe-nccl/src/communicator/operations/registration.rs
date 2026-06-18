use std::{mem::size_of, ptr};

use singe_cuda::memory::DeviceMemory;

use super::super::buffer_byte_len;
use crate::{
    buffer::BufferMut,
    communicator::{Communicator, RegisteredBuffer, Window},
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
    pub fn register_buffer<T>(&self, buffer: BufferMut<'_, T>) -> Result<RegisteredBuffer<'_>> {
        self.bind()?;

        let mut handle = ptr::null_mut();
        let size = buffer_byte_len(buffer.len(), "registered buffer size", size_of::<T>())?;
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
        })
    }

    pub fn register_buffer_memory<T>(
        &self,
        buffer: &mut DeviceMemory<T>,
    ) -> Result<RegisteredBuffer<'_>> {
        self.register_buffer(BufferMut::from_device_memory(buffer)?)
    }

    /// Collectively registers `buffer` for this rank as part of an NCCL window.
    /// Every rank in the communicator must participate in the registration, and
    /// by default the registered sizes must match across ranks.
    /// Use the returned [`Window`] to deregister the mapping. If this is called
    /// inside a group, the returned window may not be fully initialized until
    /// [`group_end`](crate::group_end) completes.
    /// `flags` controls NCCL window registration behavior.
    pub fn register_window<T>(
        &self,
        buffer: BufferMut<'_, T>,
        flags: WindowFlags,
    ) -> Result<Window<'_>> {
        self.bind()?;

        let mut handle = ptr::null_mut();
        let size = buffer_byte_len(buffer.len(), "window size", size_of::<T>())?;
        unsafe {
            try_ffi!(sys::ncclCommWindowRegister(
                self.as_raw(),
                buffer.as_ptr().cast_mut().cast(),
                size,
                &raw mut handle,
                flags.bits() as i32,
            ))?;
        }

        Ok(Window {
            communicator: self,
            raw: handle,
        })
    }

    pub fn register_window_memory<T>(
        &self,
        buffer: &mut DeviceMemory<T>,
        flags: WindowFlags,
    ) -> Result<Window<'_>> {
        self.register_window(BufferMut::from_device_memory(buffer)?, flags)
    }
}
