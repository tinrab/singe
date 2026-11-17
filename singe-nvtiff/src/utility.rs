use std::ptr;

use singe_core::{checked_arithmetic, checked_conversions};
use singe_nvtiff_sys as sys;

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_size_t -> sys::size_t,
    to_usize -> usize,
    to_u32 -> u32,
    to_i32 -> i32,
});

checked_arithmetic!(Error {
    checked_element_count,
    checked_pitched_extent,
});

pub(crate) fn checked_pixels(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(Error::OutOfRange {
            name: "image dimensions".into(),
        });
    }
    checked_element_count(&[width, height], "image dimensions")
}

// Reinterprets native-endian bytes as values of `T`, which callers only instantiate with `u16`, `u32`, and `f64`.
pub(crate) fn bytes_to_values<T: Copy>(bytes: &[u8]) -> Result<Vec<T>> {
    if !bytes.len().is_multiple_of(size_of::<T>()) {
        return Err(Error::LengthMismatch {
            name: "tag value".into(),
            expected: bytes.len().next_multiple_of(size_of::<T>()),
            actual: bytes.len(),
        });
    }

    let count = bytes.len() / size_of::<T>();
    let mut values = Vec::<T>::with_capacity(count);
    // SAFETY: `values` has capacity for `count` elements, `bytes` holds exactly that many, and every caller's `T` (`u16`, `u32`, `f64`) accepts any bit pattern.
    unsafe {
        ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            values.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
        values.set_len(count);
    }
    Ok(values)
}
