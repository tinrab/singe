use std::{marker::PhantomData, ptr::NonNull};

use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceBuffer, DeviceBufferMut, DeviceRepr},
};

use crate::{
    error::{Error, Result},
    types::DataTypeLike,
};

#[derive(Debug, Clone, Copy)]
pub struct BufferRef<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    _t: PhantomData<&'a T>,
}

#[derive(Debug)]
pub struct BufferMut<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    _t: PhantomData<&'a mut T>,
}

impl<'a, T> BufferRef<'a, T> {
    pub fn from_device_buffer(buffer: &'a impl DeviceBuffer<T>) -> Result<Self>
    where
        T: DeviceRepr,
    {
        Self::new(buffer.as_device_ptr().cast_mut(), buffer.len())
    }

    pub fn from_device_memory(memory: &'a DeviceMemory<T>) -> Result<Self> {
        Self::new(memory.as_ptr().cast_mut(), memory.len())
    }

    pub fn from_slice(memory: &'a DeviceMemory<T>, offset: usize, len: usize) -> Result<Self> {
        let end = offset.checked_add(len).ok_or(Error::OutOfRange {
            name: "buffer slice end".into(),
        })?;
        if end > memory.len() {
            return Err(Error::LengthMismatch {
                name: "buffer slice".into(),
            });
        }
        let ptr = memory.as_ptr().wrapping_add(offset).cast_mut();
        Self::new(ptr, len)
    }

    /// Creates a borrowed NCCL buffer view from a raw pointer and element count.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `len` contiguous device elements for the returned
    /// lifetime. If `len` is zero, `ptr` may be null and is normalized to
    /// `NonNull::dangling()` because NCCL will not touch the pointer for an
    /// empty buffer.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Result<Self> {
        Self::new(ptr, len)
    }

    fn new(ptr: *mut T, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                _t: PhantomData,
            });
        }

        let ptr = NonNull::new(ptr).ok_or(Error::NullPointer)?;
        Ok(Self {
            ptr,
            len,
            _t: PhantomData,
        })
    }

    pub const fn len(self) -> usize {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub const fn as_ptr(self) -> *const T {
        self.ptr.as_ptr()
    }
}

impl<'a, T> BufferMut<'a, T> {
    pub fn from_device_buffer(buffer: &'a mut impl DeviceBufferMut<T>) -> Result<Self>
    where
        T: DeviceRepr,
    {
        Self::new(buffer.as_device_mut_ptr(), buffer.len())
    }

    pub fn from_device_memory(memory: &'a mut DeviceMemory<T>) -> Result<Self> {
        Self::new(memory.as_mut_ptr(), memory.len())
    }

    pub fn from_slice(memory: &'a mut DeviceMemory<T>, offset: usize, len: usize) -> Result<Self> {
        let end = offset.checked_add(len).ok_or(Error::OutOfRange {
            name: "buffer slice end".into(),
        })?;
        if end > memory.len() {
            return Err(Error::LengthMismatch {
                name: "buffer slice".into(),
            });
        }
        let ptr = memory.as_mut_ptr().wrapping_add(offset);
        Self::new(ptr, len)
    }

    /// Creates a borrowed mutable NCCL buffer view from a raw pointer and
    /// element count.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `len` contiguous mutable device elements for the
    /// returned lifetime. If `len` is zero, `ptr` may be null and is normalized
    /// to `NonNull::dangling()` because NCCL will not touch the pointer for an
    /// empty buffer. The caller must guarantee unique access to the buffer.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Result<Self> {
        Self::new(ptr, len)
    }

    fn new(ptr: *mut T, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                _t: PhantomData,
            });
        }

        let ptr = NonNull::new(ptr).ok_or(Error::NullPointer)?;
        Ok(Self {
            ptr,
            len,
            _t: PhantomData,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub const fn as_ref(&self) -> BufferRef<'_, T> {
        BufferRef {
            ptr: self.ptr,
            len: self.len,
            _t: PhantomData,
        }
    }
}

pub trait BufferElement: DataTypeLike + DeviceRepr {}

impl<T: DataTypeLike + DeviceRepr> BufferElement for T {}
