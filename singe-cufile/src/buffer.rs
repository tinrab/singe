use std::{marker::PhantomData, ptr::NonNull};

use singe_cuda::{
    external_memory::MappedBuffer,
    memory::{DeviceMemory, ManagedMemory},
    view::{
        ByteBuffer, ByteBufferMut, DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView,
        DeviceViewMut,
    },
};
use singe_cufile_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::BufferRegisterFlags,
    utility::{checked_byte_len, to_u64},
};

#[derive(Debug)]
struct RegisteredBufferInner {
    ptr: NonNull<()>,
    size: usize,
}

#[derive(Debug)]
pub struct RegisteredBuffer<'a, T: ?Sized> {
    inner: RegisteredBufferInner,
    _buffer: PhantomData<&'a T>,
}

#[derive(Debug)]
pub struct RegisteredBufferMut<'a, T: ?Sized> {
    inner: RegisteredBufferInner,
    _t: PhantomData<&'a mut T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferPointer {
    ptr: *mut (),
}

impl BufferPointer {
    pub(crate) const fn from_const(ptr: *const ()) -> Self {
        Self {
            ptr: ptr.cast_mut(),
        }
    }

    pub(crate) const fn from_mut(ptr: *mut ()) -> Self {
        Self { ptr }
    }

    pub(crate) const fn as_const(self) -> *const () {
        self.ptr
    }

    pub(crate) const fn as_mut(self) -> *mut () {
        self.ptr
    }
}

pub trait IoBuffer {
    fn ptr(&self) -> BufferPointer;

    fn byte_len(&self) -> Result<usize>;
}

pub trait IoBufferMut: IoBuffer {
    fn ptr_mut(&mut self) -> BufferPointer;
}

impl<T> IoBuffer for [T] {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.as_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        checked_byte_len::<T>(self.len(), "buffer")
    }
}

impl<T> IoBufferMut for [T] {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.as_mut_ptr().cast())
    }
}

impl<T> IoBuffer for Vec<T> {
    fn ptr(&self) -> BufferPointer {
        self.as_slice().ptr()
    }

    fn byte_len(&self) -> Result<usize> {
        self.as_slice().byte_len()
    }
}

impl<T> IoBufferMut for Vec<T> {
    fn ptr_mut(&mut self) -> BufferPointer {
        self.as_mut_slice().ptr_mut()
    }
}

impl<T> IoBuffer for DeviceMemory<T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.as_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        Ok(DeviceMemory::byte_len(self))
    }
}

impl<T> IoBufferMut for DeviceMemory<T> {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.as_mut_ptr().cast())
    }
}

impl<T: DeviceRepr> IoBuffer for ManagedMemory<T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.as_device_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        checked_byte_len::<T>(self.len(), "buffer")
    }
}

impl<T: DeviceRepr> IoBufferMut for ManagedMemory<T> {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.as_device_mut_ptr().cast())
    }
}

impl<T: DeviceRepr> IoBuffer for DeviceView<'_, T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.as_device_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        checked_byte_len::<T>(self.len(), "buffer")
    }
}

impl<T: DeviceRepr> IoBuffer for DeviceViewMut<'_, T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.as_device_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        checked_byte_len::<T>(self.len(), "buffer")
    }
}

impl<T: DeviceRepr> IoBufferMut for DeviceViewMut<'_, T> {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.as_device_mut_ptr().cast())
    }
}

impl<T: DeviceRepr> IoBuffer for MappedBuffer<'_, T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.as_device_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        checked_byte_len::<T>(self.len(), "buffer")
    }
}

impl<T: DeviceRepr> IoBufferMut for MappedBuffer<'_, T> {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.as_device_mut_ptr().cast())
    }
}

impl<'a, B: IoBuffer + ?Sized> RegisteredBuffer<'a, B> {
    /// Registers any stable host or device buffer supported by cuFile.
    ///
    /// The returned value borrows `buffer`, so the buffer cannot be dropped
    /// while cuFile has the pointer registered.
    pub fn register(buffer: &'a B, flags: BufferRegisterFlags) -> Result<Self> {
        let inner = unsafe {
            RegisteredBufferInner::register(buffer.ptr().as_const(), buffer.byte_len()?, flags)
        }?;
        Ok(Self {
            inner,
            _buffer: PhantomData,
        })
    }
}

impl<'a, B: ByteBuffer + ?Sized> RegisteredBuffer<'a, B> {
    pub fn register_byte_buffer(buffer: &'a B, flags: BufferRegisterFlags) -> Result<Self> {
        let inner = unsafe {
            RegisteredBufferInner::register(buffer.as_byte_ptr().cast(), buffer.byte_len(), flags)
        }?;
        Ok(Self {
            inner,
            _buffer: PhantomData,
        })
    }
}

impl<'a, B: IoBufferMut + ?Sized> RegisteredBufferMut<'a, B> {
    /// Registers a mutable buffer and keeps exclusive access for the
    /// registration lifetime.
    pub fn register(buffer: &'a mut B, flags: BufferRegisterFlags) -> Result<Self> {
        let size = buffer.byte_len()?;
        let inner =
            unsafe { RegisteredBufferInner::register(buffer.ptr_mut().as_mut(), size, flags) }?;
        Ok(Self {
            inner,
            _t: PhantomData,
        })
    }
}

impl<'a, B: ByteBufferMut + ?Sized> RegisteredBufferMut<'a, B> {
    pub fn register_byte_buffer(buffer: &'a mut B, flags: BufferRegisterFlags) -> Result<Self> {
        let inner = unsafe {
            RegisteredBufferInner::register(
                buffer.as_byte_mut_ptr().cast(),
                buffer.byte_len(),
                flags,
            )
        }?;
        Ok(Self {
            inner,
            _t: PhantomData,
        })
    }
}

impl<'a, T> RegisteredBuffer<'a, [T]> {
    pub fn register_slice(slice: &'a [T], flags: BufferRegisterFlags) -> Result<Self> {
        let size = checked_byte_len::<T>(slice.len(), "slice")?;
        let inner = unsafe { RegisteredBufferInner::register(slice.as_ptr().cast(), size, flags) }?;
        Ok(Self {
            inner,
            _buffer: PhantomData,
        })
    }
}

impl<'a, T> RegisteredBuffer<'a, DeviceMemory<T>> {
    pub fn register_memory(
        memory: &'a DeviceMemory<T>,
        flags: BufferRegisterFlags,
    ) -> Result<Self> {
        let inner = unsafe {
            RegisteredBufferInner::register(memory.as_ptr().cast(), memory.byte_len(), flags)
        }?;
        Ok(Self {
            inner,
            _buffer: PhantomData,
        })
    }
}

impl<'a, T> RegisteredBufferMut<'a, [T]> {
    pub fn register_slice(slice: &'a mut [T], flags: BufferRegisterFlags) -> Result<Self> {
        let size = checked_byte_len::<T>(slice.len(), "slice")?;
        let inner =
            unsafe { RegisteredBufferInner::register(slice.as_mut_ptr().cast(), size, flags) }?;
        Ok(Self {
            inner,
            _t: PhantomData,
        })
    }
}

impl<'a, T> RegisteredBufferMut<'a, DeviceMemory<T>> {
    pub fn register_memory(
        memory: &'a mut DeviceMemory<T>,
        flags: BufferRegisterFlags,
    ) -> Result<Self> {
        let inner = unsafe {
            RegisteredBufferInner::register(memory.as_mut_ptr().cast(), memory.byte_len(), flags)
        }?;
        Ok(Self {
            inner,
            _t: PhantomData,
        })
    }
}

impl<T: ?Sized> RegisteredBuffer<'_, T> {
    pub fn byte_len(&self) -> usize {
        self.inner.byte_len()
    }
}

impl<T: ?Sized> RegisteredBufferMut<'_, T> {
    pub fn byte_len(&self) -> usize {
        self.inner.byte_len()
    }
}

impl RegisteredBufferInner {
    unsafe fn register(ptr: *const (), size: usize, flags: BufferRegisterFlags) -> Result<Self> {
        let ptr = NonNull::new(ptr.cast_mut()).ok_or(Error::NullHandle)?;
        unsafe {
            try_ffi!(sys::cuFileBufRegister(
                ptr.as_ptr() as _,
                to_u64(size, "size")?,
                flags.bits(),
            ))?;
        }

        Ok(Self { ptr, size })
    }

    fn as_ptr(&self) -> *const () {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut () {
        self.ptr.as_ptr()
    }

    fn byte_len(&self) -> usize {
        self.size
    }
}

impl<T: ?Sized> IoBuffer for RegisteredBuffer<'_, T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.inner.as_ptr())
    }

    fn byte_len(&self) -> Result<usize> {
        Ok(RegisteredBuffer::byte_len(self))
    }
}

impl<T: ?Sized> IoBuffer for RegisteredBufferMut<'_, T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.inner.as_ptr())
    }

    fn byte_len(&self) -> Result<usize> {
        Ok(RegisteredBufferMut::byte_len(self))
    }
}

impl<T: ?Sized> IoBufferMut for RegisteredBufferMut<'_, T> {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.inner.as_mut_ptr())
    }
}

impl Drop for RegisteredBufferInner {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuFileBufDeregister(self.ptr.as_ptr() as _);
        }
    }
}
