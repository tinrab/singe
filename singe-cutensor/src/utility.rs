use singe_core::checked_conversions;

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
