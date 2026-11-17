//! Buffer addressability, initialized inputs, and arbitrary-byte outputs for cuFile.
use std::{
    ffi::c_void,
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    ptr::NonNull,
    sync::Arc,
};

use singe_core::abort_after_cleanup_failure;
use singe_cuda::{
    context::Context,
    external_memory::MappedBuffer,
    graph::GraphBuffer,
    memory::{DeviceMemory, ManagedMemory, PinnedVec, RegisteredHostMemory, RegisteredHostSlice},
    view::{
        AnyBitPatternDeviceRepr, DeviceOutputView, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceView, DeviceViewMut, NoUninitDeviceRepr,
    },
};
use singe_cufile_sys as sys;

use crate::{
    driver::Driver,
    error::{Error, Result},
    file::prepare_host_access,
    try_ffi,
    types::BufferRegisterFlags,
    utility::{checked_byte_len, to_i64, to_u64},
};

/// A live contiguous host or device storage range, without an initialization claim.
///
/// # Safety
///
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
///
/// The mutable pointer must describe the same range as `IoBuffer::ptr` and permit writes for the exclusive borrow.
pub unsafe trait IoBufferMut: IoBuffer {
    /// Returns the address of the exclusively writable storage range.
    fn ptr_mut(&mut self) -> BufferPointer;
}

/// Storage whose entire representation is initialized and may be read as bytes.
///
/// # Safety
///
/// Every exposed byte must be initialized before I/O reads it, after any declared readiness dependency completes.
/// Host storage without a readiness dependency must remain initialized throughout the shared borrow.
/// This guarantee includes representation padding.
pub unsafe trait IoInput: IoBuffer {}

/// Exclusive storage that may receive arbitrary file bytes, including partial writes.
///
/// # Safety
///
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
        // SAFETY: the pointer and byte length come from the same live slice.
        unsafe impl<T: DeviceRepr> IoBuffer for $ty {
            fn ptr(&self) -> BufferPointer {
                BufferPointer::from_const(self.as_ptr().cast())
            }
            fn byte_len(&self) -> Result<usize> {
                let () = T::LAYOUT;
                checked_byte_len::<T>(self.len(), "buffer")
            }
        }
        // SAFETY: the slice holds initialized `NoUninitDeviceRepr` elements, which have no padding bytes.
        unsafe impl<T: NoUninitDeviceRepr> IoInput for $ty {}
        // SAFETY: the mutable pointer addresses the same live slice as `ptr` through the exclusive borrow.
        unsafe impl<T: DeviceRepr> IoBufferMut for $ty {
            fn ptr_mut(&mut self) -> BufferPointer {
                BufferPointer::from_mut(self.as_mut_ptr().cast())
            }
        }
        // SAFETY: every byte pattern is a valid `AnyBitPatternDeviceRepr` element.
        unsafe impl<T: AnyBitPatternDeviceRepr> IoOutput for $ty {}
    };
}

host_buffer!([T]);

host_buffer!(Vec<T>);

macro_rules! device_buffer {
    ($ty:ty) => {
        // SAFETY: the pointer and byte length come from the storage's own device region, which is reported with its context.
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
        // SAFETY: the storage holds initialized `NoUninitDeviceRepr` elements, which have no padding bytes.
        unsafe impl<T: NoUninitDeviceRepr> IoInput for $ty {}
    };
}

macro_rules! device_output {
    ($ty:ty) => {
        // SAFETY: the mutable pointer addresses the same device region as `ptr` through the exclusive borrow.
        unsafe impl<T: DeviceRepr> IoBufferMut for $ty {
            fn ptr_mut(&mut self) -> BufferPointer {
                BufferPointer::from_mut(self.region().as_ptr().cast())
            }
        }
        // SAFETY: every byte pattern is a valid `AnyBitPatternDeviceRepr` element.
        unsafe impl<T: AnyBitPatternDeviceRepr> IoOutput for $ty {}
    };
}

device_buffer!(DeviceMemory<T>);

device_output!(DeviceMemory<T>);

device_buffer!(ManagedMemory<T>);

device_output!(ManagedMemory<T>);

device_buffer!(MappedBuffer<T>);

device_output!(MappedBuffer<T>);

device_buffer!(GraphBuffer<T>);

device_output!(GraphBuffer<T>);

device_buffer!(DeviceView<'_, T>);

device_buffer!(DeviceViewMut<'_, T>);

device_output!(DeviceViewMut<'_, T>);

// SAFETY: the pointer and byte length come from the view's own device region, which is reported with its context; an output-only view implements no `IoInput`, so it never claims initialized bytes.
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
        // SAFETY: the pointer and byte length come from the live registered host storage, and the context is the one it is registered in.
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
        // SAFETY: the storage holds initialized `NoUninitDeviceRepr` elements, which have no padding bytes.
        unsafe impl<T: NoUninitDeviceRepr> IoInput for $ty {}
    };
}

macro_rules! host_output {
    ($ty:ty) => {
        // SAFETY: the mutable pointer addresses the same live range as `ptr` through the exclusive borrow.
        unsafe impl<T: DeviceRepr> IoBufferMut for $ty {
            fn ptr_mut(&mut self) -> BufferPointer {
                BufferPointer::from_mut(self.as_mut_ptr().cast())
            }
        }
        // SAFETY: every byte pattern is a valid `AnyBitPatternDeviceRepr` element.
        unsafe impl<T: AnyBitPatternDeviceRepr> IoOutput for $ty {}
    };
}

// SAFETY: the pointer and byte length come from the live vector, and the context is its allocator's.
unsafe impl<T: DeviceRepr> IoBuffer for PinnedVec<T> {
    fn ptr(&self) -> BufferPointer {
        BufferPointer::from_const(self.as_ptr().cast())
    }

    fn byte_len(&self) -> Result<usize> {
        let () = T::LAYOUT;
        checked_byte_len::<T>(self.len(), "buffer")
    }

    fn context(&self) -> Option<&Arc<Context>> {
        Some(self.allocator().context())
    }
}

// SAFETY: the vector holds initialized `NoUninitDeviceRepr` elements, which have no padding bytes.
unsafe impl<T: NoUninitDeviceRepr> IoInput for PinnedVec<T> {}

host_output!(PinnedVec<T>);

registered_host_buffer!(RegisteredHostMemory<'_, T>);

host_output!(RegisteredHostMemory<'_, T>);

registered_host_buffer!(RegisteredHostSlice<'_, T>);

// A byte range of one buffer, with the device metadata cuFile calls need.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IoSpan<'a> {
    base: BufferPointer,
    byte_offset: usize,
    byte_len: usize,
    registered: bool,
    pub(crate) region: Option<DeviceRegion<'a>>,
    pub(crate) context: Option<&'a Arc<Context>>,
}

impl<'a> IoSpan<'a> {
    fn create<B: IoBuffer + ?Sized>(
        buffer: &'a B,
        base: BufferPointer,
        range: impl RangeBounds<usize>,
    ) -> Result<Self> {
        let buffer_len = buffer.byte_len()?;
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start
                .checked_add(1)
                .ok_or_else(|| Error::out_of_range("byte range start"))?,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end
                .checked_add(1)
                .ok_or_else(|| Error::out_of_range("byte range end"))?,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => buffer_len,
        };
        if end > buffer_len {
            return Err(Error::out_of_range("byte range end"));
        }

        let byte_len = end
            .checked_sub(start)
            .ok_or_else(|| Error::out_of_range("byte range start"))?;
        Ok(Self {
            base,
            byte_offset: start,
            byte_len,
            registered: buffer.is_registered(),
            region: buffer.device_region(),
            context: buffer.context(),
        })
    }

    pub(crate) const fn byte_len(&self) -> usize {
        self.byte_len
    }

    // A registered buffer is addressed by its registered base and an offset; any other buffer by the range's own address.
    pub(crate) fn native_address(&self) -> Result<(*mut c_void, i64)> {
        if self.registered {
            Ok((
                self.base.as_mut().cast(),
                to_i64(self.byte_offset, "byte offset")?,
            ))
        } else {
            Ok((self.range_start(), 0))
        }
    }

    pub(crate) fn io_vector(&self) -> Result<sys::CUfileIOVec_t> {
        Ok(sys::CUfileIOVec_t {
            base: self.range_start(),
            len: to_u64(self.byte_len, "byte length")?,
        })
    }

    fn range_start(&self) -> *mut c_void {
        self.base
            .as_mut()
            .cast::<u8>()
            .wrapping_add(self.byte_offset)
            .cast()
    }
}

/// An initialized byte range of a buffer that cuFile reads from, retaining the buffer's shared borrow.
#[derive(Debug, Clone, Copy)]
pub struct IoSlice<'a> {
    pub(crate) span: IoSpan<'a>,
    _borrow: PhantomData<&'a ()>,
}

impl<'a> IoSlice<'a> {
    /// Borrows the bytes of `buffer` in `range`, such as `..` for every byte or `0..len` for a prefix.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed or `range` does not lie within the buffer.
    pub fn create<B: IoInput + ?Sized>(
        buffer: &'a B,
        range: impl RangeBounds<usize>,
    ) -> Result<Self> {
        Ok(Self {
            span: IoSpan::create(buffer, buffer.ptr(), range)?,
            _borrow: PhantomData,
        })
    }

    /// Borrows the bytes of `buffer` in `range` without requiring that they are initialized.
    ///
    /// # Safety
    ///
    /// Every byte in `range`, including padding, must be initialized when cuFile reads it.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed or `range` does not lie within the buffer.
    pub unsafe fn create_unchecked<B: IoBuffer + ?Sized>(
        buffer: &'a B,
        range: impl RangeBounds<usize>,
    ) -> Result<Self> {
        Ok(Self {
            span: IoSpan::create(buffer, buffer.ptr(), range)?,
            _borrow: PhantomData,
        })
    }

    /// Returns the length of the range, in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.span.byte_len
    }
}

/// A byte range of a buffer that cuFile writes file bytes into, retaining the buffer's exclusive borrow.
#[derive(Debug)]
pub struct IoSliceMut<'a> {
    pub(crate) span: IoSpan<'a>,
    _borrow: PhantomData<&'a mut ()>,
}

impl<'a> IoSliceMut<'a> {
    /// Borrows the bytes of `buffer` in `range`, such as `..` for every byte or `0..len` for a prefix.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed or `range` does not lie within the buffer.
    pub fn create<B: IoOutput + ?Sized>(
        buffer: &'a mut B,
        range: impl RangeBounds<usize>,
    ) -> Result<Self> {
        let base = buffer.ptr_mut();
        let buffer: &'a B = buffer;
        Ok(Self {
            span: IoSpan::create(buffer, base, range)?,
            _borrow: PhantomData,
        })
    }

    /// Borrows the bytes of `buffer` in `range` without requiring that arbitrary file bytes form valid values.
    ///
    /// # Safety
    ///
    /// The bytes cuFile writes, including partial writes, must leave valid values or storage the caller treats as uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed or `range` does not lie within the buffer.
    pub unsafe fn create_unchecked<B: IoBufferMut + ?Sized>(
        buffer: &'a mut B,
        range: impl RangeBounds<usize>,
    ) -> Result<Self> {
        let base = buffer.ptr_mut();
        let buffer: &'a B = buffer;
        Ok(Self {
            span: IoSpan::create(buffer, base, range)?,
            _borrow: PhantomData,
        })
    }

    /// Returns the length of the range, in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.span.byte_len
    }

    pub(crate) const fn into_span(self) -> IoSpan<'a> {
        self.span
    }
}

#[derive(Debug)]
struct RegisteredBufferInner {
    ptr: NonNull<()>,
    size: usize,
    context: Option<Arc<Context>>,
    _driver: Arc<Driver>,
}

/// A borrowed native registration.
/// Forgetting it requires retaining backing storage until native deregistration.
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
    ///
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
        prepare_host_access(buffer.device_region(), buffer.context())?;
        // SAFETY: `buffer` supplies its own address, length, and context, and the caller guarantees deregistration and I/O completion before the storage is released.
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
    ///
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
        prepare_host_access(buffer.device_region(), buffer.context())?;
        let size = buffer.byte_len()?;
        let pointer = buffer.ptr_mut().as_mut();
        // SAFETY: `pointer` and `size` come from the exclusively borrowed `buffer`, and the caller guarantees deregistration and I/O completion before the storage is released.
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

// SAFETY: the registered pointer and length are the borrowed buffer's own, and the region and context are the buffer's.
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

// SAFETY: the registration only reads the borrowed `IoInput` buffer, whose bytes stay initialized.
unsafe impl<B: IoInput + ?Sized> IoInput for RegisteredBuffer<'_, B> {}

// SAFETY: the registered pointer and length are the borrowed buffer's own, and the region and context are the buffer's.
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

// SAFETY: the registration only reads the borrowed `IoInput` buffer, whose bytes stay initialized.
unsafe impl<B: IoBufferMut + IoInput + ?Sized> IoInput for RegisteredBufferMut<'_, B> {}

// SAFETY: the registration holds the buffer's exclusive borrow, which permits writes through the registered pointer.
unsafe impl<B: IoBufferMut + ?Sized> IoBufferMut for RegisteredBufferMut<'_, B> {
    fn ptr_mut(&mut self) -> BufferPointer {
        BufferPointer::from_mut(self.inner.ptr.as_ptr())
    }
}

// SAFETY: the borrowed buffer is `IoOutput`, so arbitrary file bytes leave valid values.
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

        // SAFETY: `ptr` is non-null and addresses `size` bytes of storage that the caller of this function keeps allocated until deregistration.
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
        // Borrowed backing storage cannot be retained if native deregistration fails.
        if let Some(context) = &self.context
            && let Err(error) = context.bind()
        {
            abort_after_cleanup_failure(
                "bind the cuda context to deregister",
                "cufile buffer",
                &error,
            );
        }

        // SAFETY: `self.ptr` is the address this value registered, and the guard that owns it still borrows the backing storage.
        if let Err(error) = unsafe { try_ffi!(sys::cuFileBufDeregister(self.ptr.as_ptr().cast())) }
        {
            abort_after_cleanup_failure("deregister", "cufile buffer", &error);
        }
    }
}
