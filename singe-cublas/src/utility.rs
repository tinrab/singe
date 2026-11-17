use singe_core::{checked_arithmetic, checked_conversions};
use singe_cublas_sys as sys;
use singe_cuda::data_type::DataType;

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_i32 -> i32,
    to_u32 -> u32,
    to_i64 -> i64,
    to_u64 -> u64,
    to_usize -> usize,
    to_size_t -> sys::size_t,
});

checked_arithmetic!(Error {
    checked_byte_len,
    checked_pitched_extent,
    validate_min_len,
    validate_same_len,
});

#[cfg(feature = "lt")]
pub(crate) fn ensure_exact_size(name: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::SizeMismatch {
            name: name.into(),
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn ensure_data_type(
    name: &'static str,
    expected: DataType,
    actual: DataType,
) -> Result<()> {
    if actual != expected {
        return Err(Error::DataTypeMismatch {
            name: name.into(),
            expected,
            actual,
        });
    }
    Ok(())
}
