use singe_core::checked_conversions;
use singe_cudss_sys as sys;

use crate::{
    error::{Error, Result},
    types::Layout,
};

checked_conversions!(Error {
    to_i32 -> i32,
    to_size_t -> sys::size_t,
    to_usize -> usize,
    to_i64 -> i64,
});

pub fn ensure_exact_size(actual: usize, expected: usize, name: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::SizeMismatch {
            name: name.to_owned().into(),
            expected,
            actual,
        })
    }
}

pub fn ensure_at_most_size(actual: usize, expected: usize, name: &str) -> Result<()> {
    if actual <= expected {
        Ok(())
    } else {
        Err(Error::SizeMismatch {
            name: name.to_owned().into(),
            expected,
            actual,
        })
    }
}

pub fn validate_nonnegative(value: i64, name: &str) -> Result<()> {
    if value >= 0 {
        Ok(())
    } else {
        Err(Error::InvalidMatrixDimension {
            name: name.to_owned().into(),
        })
    }
}

pub fn validate_dense_shape(
    rows: i64,
    columns: i64,
    leading_dim: i64,
    layout: Layout,
) -> Result<()> {
    validate_nonnegative(rows, "rows")?;
    validate_nonnegative(columns, "columns")?;
    validate_nonnegative(leading_dim, "leading_dim")?;

    let minimum_leading_dim = match layout {
        Layout::ColumnMajor => rows,
        Layout::RowMajor => columns,
    };
    if leading_dim < minimum_leading_dim {
        return Err(Error::InvalidLeadingDimension);
    }

    Ok(())
}

pub fn checked_element_count(dimensions: &[i64], name: &str) -> Result<usize> {
    let mut count = 1_usize;
    for dimension in dimensions {
        validate_nonnegative(*dimension, name)?;
        let dimension = to_usize(*dimension, name)?;
        count = count
            .checked_mul(dimension)
            .ok_or_else(|| Error::OutOfRange {
                name: name.to_owned().into(),
            })?;
    }
    Ok(count)
}
