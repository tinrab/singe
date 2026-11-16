use singe_core::{checked_arithmetic, checked_conversions};

use crate::{
    error::{Error, Result},
    types::Mode,
};

checked_conversions!(Error {
    to_i32 -> i32,
    to_u32 -> u32,
    to_usize -> usize,
    to_i64 -> i64,
});

checked_arithmetic!(Error { checked_byte_len });

pub fn to_i64_vec<T: TryInto<i64> + Copy>(values: &[T], name: &str) -> Result<Vec<i64>> {
    values
        .iter()
        .copied()
        .map(|value| to_i64(value, name))
        .collect()
}

pub fn modes_to_i32_vec(modes: &[Mode]) -> Vec<i32> {
    modes.iter().map(|mode| mode.as_raw()).collect()
}

pub fn collect_optional_mode_values<M>(
    modes: &[M],
    name: &str,
    value: impl Fn(&M) -> Option<u64>,
) -> Result<Option<Vec<u64>>> {
    let values = modes.iter().filter_map(value).collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(None);
    }

    if values.len() != modes.len() {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected: modes.len(),
            actual: values.len(),
        });
    }
    Ok(Some(values))
}
