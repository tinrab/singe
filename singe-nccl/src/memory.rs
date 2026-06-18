use std::{marker::PhantomData, mem, mem::size_of, ptr};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_u64,
};

#[derive(Debug)]
pub struct Memory<T> {
    ptr: *mut T,
    len: usize,
    _t: PhantomData<T>,
}

unsafe impl<T: Send> Send for Memory<T> {}
unsafe impl<T: Sync> Sync for Memory<T> {}

impl<T> Memory<T> {
    /// Allocates GPU memory for `len` elements.
    /// The allocation is owned by the returned [`Memory`] and released with
    /// `ncclMemFree` when it is dropped.
    /// The actual allocation may be larger than requested because NCCL can
    /// round up to satisfy its internal granularity requirements.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows, cannot be represented for
    /// NCCL, or if NCCL cannot allocate memory.
    pub fn create(len: usize) -> Result<Self> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or(Error::OutOfRange {
            name: "nccl memory size".into(),
        })?;
        let bytes = to_u64(bytes, "nccl memory size")?;
        if bytes == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len,
                _t: PhantomData,
            });
        }

        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::ncclMemAlloc(&raw mut raw, bytes))?;
        }

        if raw.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(Self {
            ptr: raw.cast(),
            len,
            _t: PhantomData,
        })
    }

    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _t: PhantomData,
        }
    }

    pub fn into_raw_parts(self) -> (*mut T, usize) {
        let ptr = self.ptr;
        let len = self.len;
        mem::forget(self);
        (ptr, len)
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub const fn byte_len(&self) -> usize {
        self.len.saturating_mul(size_of::<T>())
    }
}

impl<T> Drop for Memory<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::ncclMemFree(self.ptr.cast())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to free nccl memory: {err}");
            }
        }
    }
}
