use singe_cuda::memory::DeviceMemory;

use crate::{
    error::{Error, Result},
    types::PointerMode,
};

#[derive(Debug, Clone, Copy)]
pub struct HostScalar<'a, T> {
    value: &'a T,
}

impl<'a, T> HostScalar<'a, T> {
    pub const fn new(value: &'a T) -> Self {
        Self { value }
    }

    pub const fn as_ptr(&self) -> *const T {
        self.value
    }
}

#[derive(Debug)]
pub struct DeviceScalar<'a, T> {
    value: &'a DeviceMemory<T>,
}

impl<'a, T> DeviceScalar<'a, T> {
    pub fn create(value: &'a DeviceMemory<T>) -> Result<Self> {
        if value.is_empty() {
            return Err(Error::InvalidVectorShape);
        }
        Ok(Self { value })
    }

    pub fn as_ptr(&self) -> *const T {
        self.value.as_ptr()
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Scalar<'a, T> {
    Host(HostScalar<'a, T>),
    Device(DeviceScalar<'a, T>),
}

impl<'a, T> Scalar<'a, T> {
    pub const fn host(value: &'a T) -> Self {
        Self::Host(HostScalar::new(value))
    }

    pub fn device(value: &'a DeviceMemory<T>) -> Result<Self> {
        Ok(Self::Device(DeviceScalar::create(value)?))
    }

    pub const fn pointer_mode(&self) -> PointerMode {
        match self {
            Self::Host(_) => PointerMode::Host,
            Self::Device(_) => PointerMode::Device,
        }
    }

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
