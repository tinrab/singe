use std::{marker::PhantomData, ptr};

use singe_cuda::types::DevicePtr;

use crate::types::PointerMode;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Scalar<'a, T> {
    Host(&'a T),
    Device(DeviceScalar<T>),
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ScalarMut<'a, T> {
    Host(&'a mut T),
    Device(DeviceScalarMut<T>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceScalar<T> {
    ptr: DevicePtr,
    _t: PhantomData<*const T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceScalarMut<T> {
    ptr: DevicePtr,
    _t: PhantomData<*mut T>,
}

impl<'a, T> Scalar<'a, T> {
    pub const fn host(value: &'a T) -> Self {
        Self::Host(value)
    }

    /// Creates a device-resident scalar from a raw device pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid device allocation containing a `T` for every
    /// cuSPARSE operation using this scalar.
    pub const unsafe fn device(ptr: DevicePtr) -> Self {
        Self::Device(unsafe { DeviceScalar::new(ptr) })
    }

    pub(crate) const fn pointer_mode(self) -> PointerMode {
        match self {
            Self::Host(_) => PointerMode::Host,
            Self::Device(_) => PointerMode::Device,
        }
    }

    pub(crate) const fn ptr(self) -> *const T {
        match self {
            Self::Host(value) => ptr::from_ref(value),
            Self::Device(value) => value.as_ptr(),
        }
    }
}

impl<'a, T> ScalarMut<'a, T> {
    pub const fn host(value: &'a mut T) -> Self {
        Self::Host(value)
    }

    /// Creates a mutable device-resident scalar from a raw device pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid writable device allocation containing a `T`
    /// for every cuSPARSE operation using this scalar.
    pub const unsafe fn device(ptr: DevicePtr) -> Self {
        Self::Device(unsafe { DeviceScalarMut::new(ptr) })
    }

    pub(crate) const fn pointer_mode(&self) -> PointerMode {
        match self {
            Self::Host(_) => PointerMode::Host,
            Self::Device(_) => PointerMode::Device,
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Self::Host(value) => ptr::from_mut(*value),
            Self::Device(value) => value.as_mut_ptr(),
        }
    }
}

impl<'a, T> From<&'a T> for Scalar<'a, T> {
    fn from(value: &'a T) -> Self {
        Self::host(value)
    }
}

impl<T> From<DeviceScalar<T>> for Scalar<'_, T> {
    fn from(value: DeviceScalar<T>) -> Self {
        Self::Device(value)
    }
}

impl<T> DeviceScalar<T> {
    /// Creates a device scalar from a raw device pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid device allocation containing a `T` for every
    /// cuSPARSE operation using this scalar.
    pub const unsafe fn new(ptr: DevicePtr) -> Self {
        Self {
            ptr,
            _t: PhantomData,
        }
    }

    pub const fn as_device_ptr(self) -> DevicePtr {
        self.ptr
    }

    pub const fn as_ptr(self) -> *const T {
        self.ptr.cast()
    }
}

impl<T> DeviceScalarMut<T> {
    /// Creates a mutable device scalar from a raw device pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid writable device allocation containing a `T`
    /// for every cuSPARSE operation using this scalar.
    pub const unsafe fn new(ptr: DevicePtr) -> Self {
        Self {
            ptr,
            _t: PhantomData,
        }
    }

    pub const fn as_device_ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr.cast()
    }
}

impl<'a, T> From<&'a mut T> for ScalarMut<'a, T> {
    fn from(value: &'a mut T) -> Self {
        Self::host(value)
    }
}

impl<T> From<DeviceScalarMut<T>> for ScalarMut<'_, T> {
    fn from(value: DeviceScalarMut<T>) -> Self {
        Self::Device(value)
    }
}
