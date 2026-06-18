//! Borrowed typed views over CUDA-accessible memory.

use std::{
    marker::PhantomData,
    mem::size_of,
    ops::{Bound, RangeBounds},
    ptr::NonNull,
};

use crate::{
    error::{Error, Result},
    memory::DeviceMemory,
    types::{Complex32, Complex64, bf16, f4e2m1, f6e2m3, f6e3m2, f8e4m3, f8e5m2, f8ue8m0, f16},
};

/// A Rust type that can be represented as plain CUDA device memory.
///
/// # Safety
///
/// Implementors must have a stable bit representation for device memory. Values
/// may be copied byte-for-byte between host and device memory without running
/// Rust destructors or relying on host-only pointer validity.
pub unsafe trait DeviceRepr: Copy + 'static {}

/// A [`DeviceRepr`] whose all-zero byte pattern is a valid value.
///
/// # Safety
///
/// Implementors must accept an all-zero byte pattern as a valid instance.
pub unsafe trait ZeroableDeviceRepr: DeviceRepr {}

macro_rules! impl_device_repr {
    ($($ty:ty),* $(,)?) => {
        $(
            unsafe impl DeviceRepr for $ty {}
            unsafe impl ZeroableDeviceRepr for $ty {}
        )*
    };
}

impl_device_repr!(
    bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, f16, bf16,
    Complex32, Complex64, f8e4m3, f8e5m2, f8ue8m0, f6e2m3, f6e3m2, f4e2m1,
);

/// A typed contiguous range of CUDA-accessible device memory.
pub trait DeviceSlice<T: DeviceRepr> {
    fn as_device_ptr(&self) -> *const T;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn byte_len(&self) -> Result<usize> {
        self.len()
            .checked_mul(size_of::<T>())
            .ok_or(Error::InvalidMemoryAllocationRequest)
    }
}

/// A mutable typed contiguous range of CUDA-accessible device memory.
pub trait DeviceSliceMut<T: DeviceRepr>: DeviceSlice<T> {
    fn as_device_mut_ptr(&mut self) -> *mut T;
}

/// Shared abstraction for read-only typed device buffers.
pub trait DeviceBuffer<T: DeviceRepr>: DeviceSlice<T> {}

impl<T, B> DeviceBuffer<T> for B
where
    T: DeviceRepr,
    B: DeviceSlice<T> + ?Sized,
{
}

/// Shared abstraction for mutable typed device buffers.
pub trait DeviceBufferMut<T: DeviceRepr>: DeviceBuffer<T> + DeviceSliceMut<T> {}

impl<T, B> DeviceBufferMut<T> for B
where
    T: DeviceRepr,
    B: DeviceBuffer<T> + DeviceSliceMut<T> + ?Sized,
{
}

/// A typed contiguous host-memory range that can be copied to CUDA memory.
pub trait HostSlice<T: DeviceRepr> {
    fn as_host_ptr(&self) -> *const T;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A mutable typed contiguous host-memory range that can be copied from CUDA memory.
pub trait HostSliceMut<T: DeviceRepr>: HostSlice<T> {
    fn as_host_mut_ptr(&mut self) -> *mut T;
}

/// Shared abstraction for read-only host buffers.
pub trait HostBuffer<T: DeviceRepr>: HostSlice<T> {}

impl<T, B> HostBuffer<T> for B
where
    T: DeviceRepr,
    B: HostSlice<T> + ?Sized,
{
}

/// Shared abstraction for mutable host buffers.
pub trait HostBufferMut<T: DeviceRepr>: HostBuffer<T> + HostSliceMut<T> {}

impl<T, B> HostBufferMut<T> for B
where
    T: DeviceRepr,
    B: HostBuffer<T> + HostSliceMut<T> + ?Sized,
{
}

/// Shared abstraction for read-only byte buffers.
pub trait ByteBuffer {
    fn as_byte_ptr(&self) -> *const u8;

    fn byte_len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.byte_len() == 0
    }
}

/// Shared abstraction for mutable byte buffers.
pub trait ByteBufferMut: ByteBuffer {
    fn as_byte_mut_ptr(&mut self) -> *mut u8;
}

impl<B> ByteBuffer for B
where
    B: DeviceSlice<u8> + ?Sized,
{
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_device_ptr()
    }

    fn byte_len(&self) -> usize {
        self.len()
    }
}

impl<B> ByteBufferMut for B
where
    B: DeviceSliceMut<u8> + ?Sized,
{
    fn as_byte_mut_ptr(&mut self) -> *mut u8 {
        self.as_device_mut_ptr()
    }
}

#[derive(Debug, Clone, Copy)]
/// Non-owning immutable view over CUDA-accessible device memory.
///
/// This type is `Copy` because it models a shared immutable borrow: duplicating
/// the view duplicates only the pointer/length pair and does not create or free
/// device memory. The lifetime ties the view to the allocation or owner that
/// created it, but CUDA kernels may still observe mutations performed through
/// other aliases according to CUDA stream ordering.
pub struct DeviceView<'a, T: DeviceRepr> {
    ptr: *const T,
    length: usize,
    _t: PhantomData<&'a T>,
}

#[derive(Debug)]
/// Non-owning mutable view over CUDA-accessible device memory.
///
/// This type is intentionally not `Clone` or `Copy` because it models a unique
/// mutable borrow of a device-memory range for the lifetime `'a`.
pub struct DeviceViewMut<'a, T: DeviceRepr> {
    ptr: *mut T,
    length: usize,
    _t: PhantomData<&'a mut T>,
}

impl<'a, T: DeviceRepr> DeviceView<'a, T> {
    /// Creates a borrowed immutable device view from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `length` contiguous elements of `T` for the
    /// returned lifetime. If `length` is zero, `ptr` may be null; the stored
    /// view pointer is normalized to `NonNull::dangling()` because safe
    /// borrowed views should not expose null unless a vendor API explicitly
    /// requires it. The memory must remain alive and CUDA-accessible while the
    /// view is used.
    pub const unsafe fn from_raw_parts(ptr: *const T, length: usize) -> Self {
        let ptr = if length == 0 {
            NonNull::<T>::dangling().as_ptr() as *const T
        } else {
            ptr
        };
        Self {
            ptr,
            length,
            _t: PhantomData,
        }
    }

    pub fn from_memory(memory: &'a DeviceMemory<T>) -> Self {
        Self {
            ptr: memory.as_ptr(),
            length: memory.len(),
            _t: PhantomData,
        }
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Result<Self> {
        let (start, end) = bounds_to_range(range, self.length)?;
        // Empty device allocations are represented by null pointers. Use
        // wrapping_add so slicing an empty view with a zero offset stays defined.
        let ptr = self.ptr.wrapping_add(start);
        Ok(Self {
            ptr,
            length: end - start,
            _t: PhantomData,
        })
    }
}

impl<'a, T: DeviceRepr> DeviceViewMut<'a, T> {
    /// Creates a borrowed mutable device view from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `length` contiguous mutable elements of `T` for
    /// the returned lifetime. If `length` is zero, `ptr` may be null; the
    /// stored view pointer is normalized to `NonNull::dangling()` because safe
    /// borrowed views should not expose null unless a vendor API explicitly
    /// requires it. The caller must guarantee unique access to the memory
    /// represented by the view.
    pub const unsafe fn from_raw_parts(ptr: *mut T, length: usize) -> Self {
        let ptr = if length == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            ptr
        };
        Self {
            ptr,
            length,
            _t: PhantomData,
        }
    }

    pub fn from_memory(memory: &'a mut DeviceMemory<T>) -> Self {
        Self {
            ptr: memory.as_mut_ptr(),
            length: memory.len(),
            _t: PhantomData,
        }
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn as_view(&self) -> DeviceView<'_, T> {
        DeviceView {
            ptr: self.ptr,
            length: self.length,
            _t: PhantomData,
        }
    }

    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> Result<DeviceView<'_, T>> {
        self.as_view().slice(range)
    }

    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, range: R) -> Result<DeviceViewMut<'_, T>> {
        let (start, end) = bounds_to_range(range, self.length)?;
        // Empty device allocations are represented by null pointers. Use
        // wrapping_add so slicing an empty view with a zero offset stays defined.
        let ptr = self.ptr.wrapping_add(start);
        Ok(DeviceViewMut {
            ptr,
            length: end - start,
            _t: PhantomData,
        })
    }

    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> Result<(DeviceViewMut<'_, T>, DeviceViewMut<'_, T>)> {
        if mid > self.length {
            return Err(Error::InvalidMemoryAccess);
        }

        // See slice_mut: split_at_mut(0) must be valid for empty null views.
        let right = self.ptr.wrapping_add(mid);
        Ok((
            DeviceViewMut {
                ptr: self.ptr,
                length: mid,
                _t: PhantomData,
            },
            DeviceViewMut {
                ptr: right,
                length: self.length - mid,
                _t: PhantomData,
            },
        ))
    }
}

impl<T: DeviceRepr> DeviceMemory<T> {
    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView::from_memory(self)
    }

    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::from_memory(self)
    }
}

impl<T: DeviceRepr> DeviceSlice<T> for DeviceMemory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: DeviceRepr> DeviceSliceMut<T> for DeviceMemory<T> {
    fn as_device_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

impl<T: DeviceRepr> DeviceSlice<T> for DeviceView<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl<T: DeviceRepr> DeviceSlice<T> for DeviceViewMut<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl<T: DeviceRepr> DeviceSliceMut<T> for DeviceViewMut<'_, T> {
    fn as_device_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: DeviceRepr> HostSlice<T> for [T] {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: DeviceRepr> HostSliceMut<T> for [T] {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

impl<T: DeviceRepr, const N: usize> HostSlice<T> for [T; N] {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        N
    }
}

impl<T: DeviceRepr, const N: usize> HostSliceMut<T> for [T; N] {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

impl<T: DeviceRepr> HostSlice<T> for Vec<T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: DeviceRepr> HostSliceMut<T> for Vec<T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

fn bounds_to_range<R: RangeBounds<usize>>(range: R, length: usize) -> Result<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&value) => value,
        Bound::Excluded(&value) => value.checked_add(1).ok_or(Error::InvalidMemoryAccess)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&value) => value.checked_add(1).ok_or(Error::InvalidMemoryAccess)?,
        Bound::Excluded(&value) => value,
        Bound::Unbounded => length,
    };

    if start > end || end > length {
        return Err(Error::InvalidMemoryAccess);
    }

    Ok((start, end))
}
