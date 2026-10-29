use singe_cuda::memory::DeviceMemory;

use crate::{
    error::{Error, Result},
    types::PointerMode,
};

/// A borrowed scalar in host memory.
#[derive(Debug, Clone, Copy)]
pub struct HostScalar<'a, T> {
    value: &'a T,
}

impl<'a, T> HostScalar<'a, T> {
    /// Borrows a host scalar.
    pub const fn new(value: &'a T) -> Self {
        Self { value }
    }

    /// Returns a borrowed pointer to the host scalar.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.value
    }
}

/// A borrowed scalar stored in the first element of a device allocation.
#[derive(Debug)]
pub struct DeviceScalar<'a, T> {
    value: &'a DeviceMemory<T>,
}

impl<'a, T> DeviceScalar<'a, T> {
    /// Borrows the first element of `value` as a device scalar.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVectorShape`] if `value` is empty.
    pub fn create(value: &'a DeviceMemory<T>) -> Result<Self> {
        if value.is_empty() {
            return Err(Error::InvalidVectorShape);
        }
        Ok(Self { value })
    }

    /// Returns a borrowed device pointer to the scalar.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.value.as_ptr()
    }
}

/// A borrowed scalar and the pointer mode needed to read it.
#[derive(Debug)]
#[non_exhaustive]
pub enum Scalar<'a, T> {
    /// A scalar in host memory.
    Host(HostScalar<'a, T>),
    /// A scalar in device memory.
    Device(DeviceScalar<'a, T>),
}

impl<'a, T> Scalar<'a, T> {
    /// Borrows a scalar in host memory.
    pub const fn host(value: &'a T) -> Self {
        Self::Host(HostScalar::new(value))
    }

    /// Borrows the first element of `value` as a device scalar.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidVectorShape`] if `value` is empty.
    pub fn device(value: &'a DeviceMemory<T>) -> Result<Self> {
        Ok(Self::Device(DeviceScalar::create(value)?))
    }

    /// Returns the cuBLAS pointer mode for this scalar.
    #[must_use]
    pub const fn pointer_mode(&self) -> PointerMode {
        match self {
            Self::Host(_) => PointerMode::Host,
            Self::Device(_) => PointerMode::Device,
        }
    }

    /// Returns a borrowed pointer in the address space selected by [`Self::pointer_mode`].
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        match self {
            Self::Host(value) => value.as_ptr(),
            Self::Device(value) => value.as_ptr(),
        }
    }
}

impl<'a, T> From<&'a T> for Scalar<'a, T> {
    fn from(value: &'a T) -> Self {
        Self::host(value)
    }
}
