use singe_core::{checked_arithmetic, checked_conversions};

use crate::{
    error::{Error, Result},
    types::Size,
};

checked_conversions!(Error {
    to_i32 -> i32,
    to_u16 -> u16,
    to_u32 -> u32,
    to_usize -> usize,
    to_u64 -> u64,
});

checked_arithmetic!(Error {
    checked_byte_len,
    validate_same_len,
});

pub fn checked_into<T, U: TryInto<T>>(value: U, name: &'static str) -> Result<T> {
    value.try_into().map_err(|_| Error::out_of_range(name))
}

pub fn validate_same_size(expected: Size, actual: Size) -> Result<()> {
    if expected == actual {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected,
        actual,
    })
}

pub fn ensure_positive(value: i32, name: &str) -> Result<()> {
    if value <= 0 {
        return Err(Error::NotPositive {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}

pub fn batch_size_i32(left_len: usize, right_len: usize) -> Result<i32> {
    validate_batch_lengths(left_len, right_len)?;
    to_i32(left_len, "batch size")
}

pub fn batch_size_u32(left_len: usize, right_len: usize) -> Result<u32> {
    validate_batch_lengths(left_len, right_len)?;
    to_u32(left_len, "batch size")
}

fn validate_batch_lengths(left_len: usize, right_len: usize) -> Result<()> {
    if left_len != right_len {
        return Err(Error::LengthMismatch {
            name: "batch".into(),
            expected: left_len,
            actual: right_len,
        });
    }

    if left_len <= 1 {
        return Err(Error::out_of_range("batch size"));
    }
    Ok(())
}

pub fn checked_len(size: Size, channels: usize) -> Result<usize> {
    size.validate()?;
    (size.width as usize)
        .checked_mul(size.height as usize)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| Error::out_of_range("len"))
}

pub fn checked_step(width: i32, channels: usize, element_size: usize) -> Result<i32> {
    let bytes = (width as usize)
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| Error::out_of_range("step"))?;
    to_i32(bytes, "step")
}

pub fn smallest_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.min(right.width),
            height: left.height.min(right.height),
        })
        .ok_or_else(|| Error::out_of_range("batch size"))
}

pub fn largest_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.max(right.width),
            height: left.height.max(right.height),
        })
        .ok_or_else(|| Error::out_of_range("batch size"))
}
