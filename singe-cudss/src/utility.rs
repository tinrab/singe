use std::{ffi::c_void, slice};

use singe_core::{checked_arithmetic, checked_conversions};
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

checked_arithmetic!(Error {
    checked_element_count,
    validate_min_len,
    validate_same_len,
});

pub(crate) fn ensure_exact_size(actual: usize, expected: usize, name: &str) -> Result<()> {
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

pub(crate) fn validate_nonnegative(value: i64, name: &str) -> Result<()> {
    if value >= 0 {
        Ok(())
    } else {
        Err(Error::out_of_range(name.to_owned()))
    }
}

pub(crate) fn validate_dense_shape(
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
        return Err(Error::out_of_range("leading_dim"));
    }

    Ok(())
}

// Copies `len` values from a native array, treating a null array as empty.
// The caller passes null or a pointer to `len` readable `i64` values.
pub(crate) unsafe fn copy_i64_batch(ptr: *mut c_void, len: usize) -> Vec<i64> {
    if ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        // SAFETY: the caller guarantees a non-null `ptr` points to `len` readable `i64` values.
        unsafe { slice::from_raw_parts(ptr.cast::<i64>(), len).to_vec() }
    }
}
