use singe_core::{checked_arithmetic, checked_conversions};
use singe_cuda::{error::Error as CudaError, memory::DeviceMemory, types::DevicePtr};

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_u64 -> u64,
    to_i32 -> i32,
    to_usize -> usize,
});

checked_arithmetic!(Error { validate_same_len });

pub(crate) fn validate_reserved<T, const N: usize>(reserved: &[T; N]) -> Result<()>
where
    T: Copy + Default + PartialEq,
{
    if reserved.iter().any(|&value| value != T::default()) {
        return Err(Error::out_of_range("reserved option bytes"));
    }
    Ok(())
}

pub(crate) fn bool_from_i32(value: i32, name: &str) -> Result<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Error::UnknownEnumValue {
            name: name.to_owned().into(),
            value: value.into(),
        }),
    }
}

pub(crate) const fn is_alignment_satisfied(
    len: usize,
    ptr_alignment: usize,
    alignment: usize,
) -> bool {
    len == 0 || alignment <= 1 || ptr_alignment.is_multiple_of(alignment)
}

pub(crate) fn validate_chunk_size_limit(name: &str, actual: usize, max: usize) -> Result<()> {
    if actual > max {
        return Err(Error::SizeLimitExceeded {
            name: name.to_owned().into(),
            max,
            actual,
        });
    }

    Ok(())
}

pub(crate) fn validate_alignment(name: &str, aligned: bool, alignment: usize) -> Result<()> {
    if !aligned {
        return Err(CudaError::Misaligned {
            name: name.to_owned().into(),
            alignment,
        }
        .into());
    }

    Ok(())
}

pub(crate) const fn codec_size(value: u64) -> usize {
    assert!(
        value <= usize::MAX as u64,
        "native codec size exceeds the host address space"
    );
    value as usize
}

// Returns null for empty memory, which nvCOMP accepts where an array has no entries.
pub(crate) fn nonempty_ptr<T>(memory: &DeviceMemory<T>) -> DevicePtr {
    if memory.is_empty() {
        DevicePtr::null()
    } else {
        memory.as_device_ptr()
    }
}
