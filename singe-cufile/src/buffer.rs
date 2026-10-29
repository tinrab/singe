//! Buffer addressability, initialized inputs, and arbitrary-byte outputs for cuFile.
use std::{marker::PhantomData, process, ptr::NonNull, sync::Arc};

use singe_cuda::{
    context::Context,
    external_memory::MappedBuffer,
    graph::GraphBuffer,
    memory::{
        AsyncDeviceMemory, DeviceMemory, ManagedMemory, PinnedHostMemory, RegisteredHostMemory,
        RegisteredHostSlice, RegisteredHostVec,
    },
    view::{
        AnyBitPatternDeviceRepr, DeviceOutputView, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceView, DeviceViewMut, NoUninitDeviceRepr,
    },
};
use singe_cufile_sys as sys;

use crate::{
    driver::Driver,
    error::{Error, Result},
    file::{checked_io_size, prepare_buffer},
    try_ffi,
    types::BufferRegisterFlags,
    utility::{checked_byte_len, to_u64},
};

/// A live contiguous host or device storage range, without an initialization claim.
///
/// # Safety
/// The pointer and byte length must describe the same live range for the shared borrow.
/// Device storage must provide a device region with actual context, allocation identity, bounds, and readiness.
/// CUDA-associated host storage must report its retained context even when it has no device region.
pub unsafe trait IoBuffer {
    /// Returns the address of the storage range.
    fn ptr(&self) -> BufferPointer;

    /// Returns the length of the storage range in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer size in bytes cannot be represented as `usize`.
    fn byte_len(&self) -> Result<usize>;

    /// Returns device allocation metadata when the storage resides on a GPU.
    fn device_region(&self) -> Option<DeviceRegion<'_>> {
        None
    }

    /// Returns the CUDA context retained by this storage, if any.
    fn context(&self) -> Option<&Arc<Context>> {
        self.device_region().map(DeviceRegion::context)
    }

    /// Returns whether the storage already has a cuFile buffer registration.
    fn is_registered(&self) -> bool {
        false
    }
}

/// Exclusively writable storage, without a guarantee that arbitrary file bytes form valid values.
///
/// # Safety
/// The mutable pointer must describe the same range as `IoBuffer::ptr` and permit writes for the exclusive borrow.
pub unsafe trait IoBufferMut: IoBuffer {
    /// Returns the address of the exclusively writable storage range.
    fn ptr_mut(&mut self) -> BufferPointer;
}

/// Storage whose entire representation is initialized and may be read as bytes.
///
/// # Safety
/// Every exposed byte must be initialized before I/O reads it, after any declared readiness dependency completes.
/// Host storage without a readiness dependency must remain initialized throughout the shared borrow.
/// This guarantee includes representation padding.
pub unsafe trait IoInput: IoBuffer {}

/// Exclusive storage that may receive arbitrary file bytes, including partial writes.
///
/// # Safety
/// Every initialized byte pattern must remain valid for its host or device element type.
/// Uninitialized storage must remain represented as uninitialized until its produced range is established.
pub unsafe trait IoOutput: IoBufferMut {}

/// A raw address without ownership, initialization, or access permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferPointer {
    ptr: *mut (),
}

impl BufferPointer {
    /// Stores a const address without taking ownership or establishing access permissions.
    #[must_use]
    pub const fn from_const(ptr: *const ()) -> Self {
        Self {
            ptr: ptr.cast_mut(),
        }
    }

    /// Stores a mutable address without taking ownership or establishing access permissions.
    pub const fn from_mut(ptr: *mut ()) -> Self {
        Self { ptr }
    }

    /// Returns the stored address as a const pointer.
    #[must_use]
    pub const fn as_const(self) -> *const () {
        self.ptr.cast_const()
    }

    /// Returns the stored address as a mutable pointer; this does not establish write permission.
    #[must_use]
    pub const fn as_mut(self) -> *mut () {
        self.ptr
    }
}

macro_rules! host_buffer {
    ($ty:ty) => {
        unsafe impl<T: DeviceRepr> IoBuffer for $ty {
            fn ptr(&self) -> BufferPointer {
                BufferPointer::from_const(self.as_ptr().cast())
            }
            fn byte_len(&self) -> Result<usize> {
                let () = T::LAYOUT;
                checked_byte_len::<T>(self.len(), "buffer")
            }
        }
        unsafe impl<T: NoUninitDeviceRepr> IoInput for $ty {}
        unsafe impl<T: DeviceRepr> IoBufferMut for $ty {
            fn ptr_mut(&mut self) -> BufferPointer {
                BufferPointer::from_mut(self.as_mut_ptr().cast())
            }
        }
        unsafe impl<T: AnyBitPatternDeviceRepr> IoOutput for $ty {}
    };
}

host_buffer!([T]);

host_buffer!(Vec<T>);

macro_rules! device_buffer {
    ($ty:ty) => {
        unsafe impl<T: DeviceRepr> IoBuffer for $ty {
            fn ptr(&self) -> BufferPointer {
                BufferPointer::from_const(self.region().as_ptr().cast())
            }
            fn byte_len(&self) -> Result<usize> {
                let () = T::LAYOUT;
                checked_byte_len::<T>(self.len(), "buffer")
            }
            fn device_region(&self) -> Option<DeviceRegion<'_>> {
                Some(self.region())
            }
        }
        unsafe impl<T: NoUninitDeviceRepr> IoInput for $ty {}
    };
}

macro_rules! device_output {
    ($ty:ty) => {
        unsafe impl<T: DeviceRepr> IoBufferMut for $ty {
            fn ptr_mut(&mut self) -> BufferPointer {
                BufferPointer::from_mut(self.region().as_ptr().cast())
            }
        }
        unsafe impl<T: AnyBitPatternDeviceRepr> IoOutput for $ty {}
    };
}

device_buffer!(DeviceMemory<T>);

device_output!(DeviceMemory<T>);

device_buffer!(AsyncDeviceMemory<T>);

device_output!(AsyncDeviceMemory<T>);

device_buffer!(ManagedMemory<T>);

device_output!(ManagedMemory<T>);

device_buffer!(MappedBuffer<T>);

device_output!(MappedBuffer<T>);

device_buffer!(GraphBuffer<T>);

device_output!(GraphBuffer<T>);

device_buffer!(DeviceView<'_, T>);

device_buffer!(DeviceViewMut<'_, T>);

device_output!(DeviceViewMut<'_, T>);

// Output-only views never claim initialized input bytes.
unsafe impl<T: DeviceRepr> IoBuffer for DeviceOutputView<'_, T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_mut(self.region().as_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        let () = T::LAYOUT;
        Ok(self.region().byte_len())
    }

    fn device_region(&self) -> Option<DeviceRegion<'_>> {
        Some(self.region())
    }
}

device_output!(DeviceOutputView<'_, T>);

macro_rules! registered_host_buffer {
    ($ty:ty) => {
        unsafe impl<T: DeviceRepr> IoBuffer for $ty {
            fn ptr(&self) -> BufferPointer {
                BufferPointer::from_const(self.as_ptr().cast())
            }
            fn byte_len(&self) -> Result<usize> {
                let () = T::LAYOUT;
                checked_byte_len::<T>(self.len(), "buffer")
            }
            fn context(&self) -> Option<&Arc<Context>> {
                Some(self.context())
            }
        }
        unsafe impl<T: NoUninitDeviceRepr> IoInput for $ty {}
    };
}

macro_rules! host_output {
    ($ty:ty) => {
        unsafe impl<T: DeviceRepr> IoBufferMut for $ty {
            fn ptr_mut(&mut self) -> BufferPointer {
                BufferPointer::from_mut(self.as_mut_ptr().cast())
            }
        }
        unsafe impl<T: AnyBitPatternDeviceRepr> IoOutput for $ty {}
    };
}

registered_host_buffer!(PinnedHostMemory<T>);

host_output!(PinnedHostMemory<T>);

registered_host_buffer!(RegisteredHostVec<T>);

host_output!(RegisteredHostVec<T>);

registered_host_buffer!(RegisteredHostMemory<'_, T>);

host_output!(RegisteredHostMemory<'_, T>);

registered_host_buffer!(RegisteredHostSlice<'_, T>);

/// An initialized input byte range retaining its buffer borrow and actual device metadata.
#[derive(Debug)]
pub struct IoSlice<'a> {
    pub(crate) raw: sys::CUfileIOVec_t,
    pub(crate) region: Option<DeviceRegion<'a>>,
    pub(crate) context: Option<&'a Arc<Context>>,
    _borrow: PhantomData<&'a ()>,
}

impl<'a> IoSlice<'a> {
    /// Borrows a byte range starting at `offset` with length `size`.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the selected range exceeds the buffer, or an offset or size exceeds its required integer range.
    ///
    /// # Panics
    ///
    /// Panics if an offset accepted by range validation cannot be represented as `usize`.
    pub fn create<B: IoInput + ?Sized>(buffer: &'a B, offset: u64, size: usize) -> Result<Self> {
        checked_io_size(buffer.byte_len()?, size, offset)?;
        Ok(Self {
            raw: sys::CUfileIOVec_t {
                base: buffer
                    .ptr()
                    .as_mut()
                    .cast::<u8>()
                    .wrapping_add(
                        usize::try_from(offset)
                            .expect("buffer offset is validated against its host byte length"),
                    )
                    .cast(),
                len: to_u64(size, "size")?,
            },
            region: buffer.device_region(),
            context: buffer.context(),
            _borrow: PhantomData,
        })
    }
}

/// An exclusively borrowed byte range accepting arbitrary file bytes.
#[derive(Debug)]
pub struct IoSliceMut<'a> {
    pub(crate) raw: sys::CUfileIOVec_t,
    pub(crate) region: Option<DeviceRegion<'a>>,
    pub(crate) context: Option<&'a Arc<Context>>,
    _borrow: PhantomData<&'a mut ()>,
}

impl<'a> IoSliceMut<'a> {
    /// Borrows a byte range starting at `offset` with length `size`.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the selected range exceeds the buffer, or an offset or size exceeds its required integer range.
    ///
    /// # Panics
    ///
    /// Panics if an offset accepted by range validation cannot be represented as `usize`.
    pub fn create<B: IoOutput + ?Sized>(
        buffer: &'a mut B,
        offset: u64,
        size: usize,
    ) -> Result<Self> {
        checked_io_size(buffer.byte_len()?, size, offset)?;
        let pointer = buffer.ptr_mut().as_mut();
        Ok(Self {
            raw: sys::CUfileIOVec_t {
                base: pointer
                    .cast::<u8>()
                    .wrapping_add(
                        usize::try_from(offset)
                            .expect("buffer offset is validated against its host byte length"),
                    )
                    .cast(),
                len: to_u64(size, "size")?,
            },
            region: buffer.device_region(),
            context: buffer.context(),
            _borrow: PhantomData,
        })
    }
}

#[derive(Debug)]
struct RegisteredBufferInner {
    ptr: NonNull<()>,
    size: usize,
    context: Option<Arc<Context>>,
    _driver: Arc<Driver>,
}

/// A borrowed native registration. Forgetting it requires retaining backing storage until native deregistration.
#[derive(Debug)]
pub struct RegisteredBuffer<'a, B: IoBuffer + ?Sized> {
    inner: RegisteredBufferInner,
    buffer: &'a B,
}

/// An exclusively borrowed native buffer registration retaining its driver session.
///
/// Forgetting the guard requires retaining backing storage until native deregistration.
#[derive(Debug)]
pub struct RegisteredBufferMut<'a, B: IoBufferMut + ?Sized> {
    inner: RegisteredBufferInner,
    buffer: &'a mut B,
}

impl<'a, B: IoBuffer + ?Sized> RegisteredBuffer<'a, B> {
    /// Registers storage without asserting initialized input bytes.
    ///
    /// # Safety
    /// Deregister before releasing backing storage, even if this guard is forgotten.
    /// Complete all I/O and coordinate prior/future CUDA accesses before deregistration.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer address is null, the byte length cannot be computed or represented, CUDA readiness or capture checks fail, or context binding fails.
    /// Also returns an error if cuFile rejects the address range or registration flags.
    pub unsafe fn register(
        driver: &Arc<Driver>,
        buffer: &'a B,
        flags: BufferRegisterFlags,
    ) -> Result<Self> {
        prepare_buffer(buffer, None)?;
        // SAFETY: the caller upholds this function's safety contract.
        let inner = unsafe {
            RegisteredBufferInner::register(
                driver,
                buffer.ptr().as_const(),
                buffer.byte_len()?,
                buffer.context(),
                flags,
            )?
        };
        Ok(Self { inner, buffer })
    }
}

impl<'a, B: IoBufferMut + ?Sized> RegisteredBufferMut<'a, B> {
    /// Registers exclusively borrowed storage without asserting element validity.
    ///
    /// # Safety
    /// Deregister before releasing backing storage, even if this guard is forgotten.
    /// Complete all I/O and coordinate CUDA accesses before deregistration.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer address is null, the byte length cannot be computed or represented, CUDA readiness or capture checks fail, or context binding fails.
    /// Also returns an error if cuFile rejects the address range or registration flags.
    pub unsafe fn register(
        driver: &Arc<Driver>,
        buffer: &'a mut B,
        flags: BufferRegisterFlags,
    ) -> Result<Self> {
        prepare_buffer(buffer, None)?;
        let size = buffer.byte_len()?;
        let pointer = buffer.ptr_mut().as_mut();
        // SAFETY: the caller upholds this function's safety contract.
        let inner = unsafe {
            RegisteredBufferInner::register(driver, pointer, size, buffer.context(), flags)?
        };
        Ok(Self { inner, buffer })
    }
}

impl<B: IoBuffer + ?Sized> RegisteredBuffer<'_, B> {
    /// Returns the registered storage length, in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.inner.size
    }
}

impl<B: IoBufferMut + ?Sized> RegisteredBufferMut<'_, B> {
    /// Returns the registered storage length, in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.inner.size
    }
}

unsafe impl<B: IoBuffer + ?Sized> IoBuffer for RegisteredBuffer<'_, B> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_mut(self.inner.ptr.as_ptr())
    }

    fn byte_len(&self) -> Result<usize> {
        Ok(self.inner.size)
    }

    fn context(&self) -> Option<&Arc<Context>> {
        self.inner.context.as_ref()
    }

    fn device_region(&self) -> Option<DeviceRegion<'_>> {
        self.buffer.device_region()
    }

    fn is_registered(&self) -> bool {
        true
    }
}

unsafe impl<B: IoInput + ?Sized> IoInput for RegisteredBuffer<'_, B> {}

unsafe impl<B: IoBufferMut + ?Sized> IoBuffer for RegisteredBufferMut<'_, B> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_mut(self.inner.ptr.as_ptr())
    }

    fn byte_len(&self) -> Result<usize> {
        Ok(self.inner.size)
    }

    fn context(&self) -> Option<&Arc<Context>> {
        self.inner.context.as_ref()
    }

    fn device_region(&self) -> Option<DeviceRegion<'_>> {
        self.buffer.device_region()
    }

    fn is_registered(&self) -> bool {
        true
    }
}

unsafe impl<B: IoBufferMut + IoInput + ?Sized> IoInput for RegisteredBufferMut<'_, B> {}

unsafe impl<B: IoBufferMut + ?Sized> IoBufferMut for RegisteredBufferMut<'_, B> {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.inner.ptr.as_ptr())
    }
}

unsafe impl<B: IoOutput + ?Sized> IoOutput for RegisteredBufferMut<'_, B> {}

impl RegisteredBufferInner {
    unsafe fn register(
        driver: &Arc<Driver>,
        pointer: *const (),
        size: usize,
        context: Option<&Arc<Context>>,
        flags: BufferRegisterFlags,
    ) -> Result<Self> {
        let ptr = NonNull::new(pointer.cast_mut()).ok_or(Error::NullHandle)?;
        if let Some(context) = context {
            context.bind()?;
        }
        // SAFETY: `cuFileBufRegister` receives pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuFileBufRegister(
                ptr.as_ptr().cast(),
                to_u64(size, "size")?,
                flags.bits()
            ))?;
        }
        Ok(Self {
            ptr,
            size,
            context: context.cloned(),
            _driver: Arc::clone(driver),
        })
    }
}

impl Drop for RegisteredBufferInner {
    fn drop(&mut self) {
        if let Some(context) = &self.context
            && context.bind().is_err()
        {
            process::abort();
        }
        // Borrowed backing storage cannot be retained if native deregistration fails.
        // SAFETY: `cuFileBufDeregister` receives pointers into buffers borrowed for the call.
        if unsafe { try_ffi!(sys::cuFileBufDeregister(self.ptr.as_ptr().cast())) }.is_err() {
            process::abort();
        }
    }
}
