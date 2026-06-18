use crate::error::{Error, Result};
use singe_core::checked_int;

pub fn required_vector_len(n: usize, inc: usize) -> Result<usize> {
    if inc == 0 {
        return Err(Error::InvalidIncrement);
    }

    n.checked_sub(1)
        .and_then(|count| count.checked_mul(inc))
        .and_then(|count| count.checked_add(1))
        .ok_or(Error::OutOfRange {
            name: "vector length".into(),
        })
}

pub fn required_matrix_len(leading_dimension: usize, cols: usize) -> Result<usize> {
    leading_dimension
        .checked_mul(cols)
        .ok_or(Error::OutOfRange {
            name: "matrix length".into(),
        })
}

pub fn to_i32(value: impl TryInto<i32>, name: &str) -> Result<i32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_u64(value: impl TryInto<u64>, name: &str) -> Result<u64> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn to_usize(value: impl TryInto<usize>, name: &str) -> Result<usize> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

pub fn ensure_exact_size(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::AttributeSizeMismatch { expected, actual });
    }
    Ok(())
}
