use singe_cuda::memory::DeviceMemory;

use crate::{
    error::{Error, Result},
    types::Size,
};

pub(super) fn validate_same_size(expected: Size, actual: Size) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected,
        actual,
    })
}

pub(super) fn validate_device_constant_len(actual: usize, expected: usize) -> Result<()> {
    if actual >= expected {
        return Ok(());
    }
    Err(Error::LengthMismatch {
        name: "device constants".into(),
        expected,
        actual,
    })
}

pub(super) fn device_constant_ptr<T>(
    constants: &DeviceMemory<T>,
    expected: usize,
) -> Result<*const T> {
    validate_device_constant_len(constants.len(), expected)?;
    Ok(constants.as_ptr())
}

pub(super) fn device_constant_mut_ptr<T>(
    constants: &DeviceMemory<T>,
    expected: usize,
) -> Result<*mut T> {
    validate_device_constant_len(constants.len(), expected)?;
    Ok(constants.as_ptr().cast_mut())
}
