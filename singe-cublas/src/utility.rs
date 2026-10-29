use singe_core::checked_conversions;

use crate::error::{Error, Result};

pub fn required_vector_len(n: usize, inc: usize) -> Result<usize> {
    if inc == 0 {
        return Err(Error::InvalidIncrement);
    }

    n.checked_sub(1)
        .and_then(|count| count.checked_mul(inc))
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| Error::out_of_range("vector length"))
}

pub fn required_matrix_len(leading_dimension: usize, cols: usize) -> Result<usize> {
    leading_dimension
        .checked_mul(cols)
        .ok_or_else(|| Error::out_of_range("matrix length"))
}

checked_conversions!(Error {
    to_i32 -> i32,
    to_i64 -> i64,
    to_u64 -> u64,
    to_usize -> usize,
});

#[cfg(feature = "lt")]
pub fn ensure_exact_size(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::AttributeSizeMismatch { expected, actual });
    }
    Ok(())
}
