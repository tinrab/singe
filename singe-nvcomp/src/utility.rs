use singe_core::checked_conversions;

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_u64 -> u64,
    to_i32 -> i32,
    to_usize -> usize,
});

pub fn validate_reserved<T, const N: usize>(reserved: &[T; N]) -> Result<()>
where
    T: Copy + Default + PartialEq,
{
    if reserved.iter().any(|&value| value != T::default()) {
        return Err(Error::InvalidValue {
            name: "reserved option bytes".into(),
        });
    }
    Ok(())
}

pub fn bool_from_i32(value: i32, name: &str) -> Result<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Error::InvalidValue {
            name: name.to_owned().into(),
        }),
    }
}

pub const fn is_alignment_satisfied(len: usize, ptr_alignment: usize, alignment: usize) -> bool {
    len == 0 || alignment <= 1 || ptr_alignment.is_multiple_of(alignment)
}

pub fn validate_same_len(name: &str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected,
            actual,
        });
    }

    Ok(())
}

pub fn validate_chunk_size_limit(name: &str, actual: usize, max: usize) -> Result<()> {
    if actual > max {
        return Err(Error::SizeLimitExceeded {
            name: name.to_owned().into(),
            max,
            actual,
        });
    }

    Ok(())
}

pub fn validate_alignment(name: &str, aligned: bool, alignment: usize) -> Result<()> {
    if !aligned {
        return Err(Error::AlignmentMismatch {
            name: name.to_owned().into(),
            alignment,
        });
    }

    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "native codec constants are asserted to fit the host size before conversion"
)]
pub const fn codec_size(value: u64) -> usize {
    assert!(
        value <= usize::MAX as u64,
        "native codec size exceeds the host address space"
    );
    value as usize
}
