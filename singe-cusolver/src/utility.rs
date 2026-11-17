use singe_core::checked_conversions;

use crate::{
    error::{Error, Result},
    sys,
};

checked_conversions!(Error {
    to_i32 -> i32,
    to_u64 -> u64,
    to_i64 -> i64,
    to_usize -> usize,
    to_size_t -> sys::size_t,
});

pub(crate) fn require_workspace_bytes(
    name: &'static str,
    actual: usize,
    required: usize,
) -> Result<()> {
    if actual < required {
        return Err(Error::BufferTooSmall {
            name: name.into(),
            required,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn validate_matrix(rows: usize, cols: usize, len: usize, lda: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidMatrixShape);
    }

    if lda < rows {
        return Err(Error::InvalidLeadingDimension);
    }

    let required = lda.checked_mul(cols).ok_or(Error::InvalidMatrixShape)?;
    if len < required {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}
