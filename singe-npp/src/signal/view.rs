use std::{marker::PhantomData, ptr::NonNull};

use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceRepr, DeviceSlice, DeviceSliceMut},
};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct SignalView<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    _t: PhantomData<&'a T>,
}

impl<'a, T> SignalView<'a, T> {
    pub fn from_memory(memory: &'a DeviceMemory<T>, len: usize) -> Result<Self> {
        validate_signal_length(len)?;
        if len > memory.len() {
            return Err(Error::LengthMismatch {
                name: "signal memory".into(),
                expected: len,
                actual: memory.len(),
            });
        }
        Ok(Self {
            ptr: NonNull::new(memory.as_ptr().cast_mut()).ok_or(Error::NullHandle)?,
            len,
            _t: PhantomData,
        })
    }

    /// # Safety
    ///
    /// `ptr` must be non-null CUDA device memory aligned for `T` and containing
    /// at least `len` contiguous initialized `T` elements. The allocation must
    /// remain valid for `'a`, and the memory must be readable by NPP on the
    /// stream used with this view.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` cannot be represented by NPP or `ptr` is null.
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Result<Self> {
        validate_signal_length(len)?;
        Ok(Self {
            ptr: NonNull::new(ptr.cast_mut()).ok_or(Error::NullHandle)?,
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
}

#[derive(Debug)]
pub struct SignalViewMut<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    _t: PhantomData<&'a mut T>,
}

impl<'a, T> SignalViewMut<'a, T> {
    pub fn from_memory(memory: &'a mut DeviceMemory<T>, len: usize) -> Result<Self> {
        validate_signal_length(len)?;
        if len > memory.len() {
            return Err(Error::LengthMismatch {
                name: "signal memory".into(),
                expected: len,
                actual: memory.len(),
            });
        }
        Ok(Self {
            ptr: NonNull::new(memory.as_mut_ptr()).ok_or(Error::NullHandle)?,
            len,
            _t: PhantomData,
        })
    }

    /// # Safety
    ///
    /// `ptr` must be non-null CUDA device memory aligned for `T` and containing
    /// at least `len` contiguous initialized `T` elements. The allocation must
    /// remain valid for `'a`, be writable by NPP on the stream used with this
    /// view, and be uniquely writable for the returned view lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` cannot be represented by NPP or `ptr` is null.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Result<Self> {
        validate_signal_length(len)?;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(Error::NullHandle)?,
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

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T: DeviceRepr> DeviceSlice<T> for SignalView<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: DeviceRepr> DeviceSlice<T> for SignalViewMut<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: DeviceRepr> DeviceSliceMut<T> for SignalViewMut<'_, T> {
    fn as_device_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

fn validate_signal_length(len: usize) -> Result<()> {
    if len == 0 {
        return Err(Error::OutOfRange {
            name: "signal length".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use singe_cuda::memory::DeviceMemory;

    use super::*;

    #[test]
    fn signal_views_reject_zero_length_before_null_pointer() -> Result<()> {
        let mut memory = DeviceMemory::<u8>::create(0)?;

        assert!(matches!(
            SignalView::from_memory(&memory, 0),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            SignalViewMut::from_memory(&mut memory, 0),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            unsafe { SignalView::<u8>::from_raw_parts(ptr::null(), 0) },
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            unsafe { SignalViewMut::<u8>::from_raw_parts(ptr::null_mut(), 0) },
            Err(Error::OutOfRange { .. })
        ));

        Ok(())
    }

    #[test]
    fn signal_views_reject_lengths_larger_than_device_memory() -> Result<()> {
        let mut memory = DeviceMemory::<u8>::create(0)?;

        assert!(matches!(
            SignalView::from_memory(&memory, 1),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            SignalViewMut::from_memory(&mut memory, 1),
            Err(Error::LengthMismatch { .. })
        ));

        Ok(())
    }

    #[test]
    fn signal_views_store_checked_raw_length_as_npp_size_t() {
        let ptr = NonNull::<u8>::dangling().as_ptr();
        let source = unsafe { SignalView::from_raw_parts(ptr, 17) }.unwrap();
        let destination = unsafe { SignalViewMut::from_raw_parts(ptr, 23) }.unwrap();

        assert_eq!(source.len(), 17);
        assert_eq!(destination.len(), 23);
    }
}
