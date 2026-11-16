use singe_core::{checked_arithmetic, checked_conversions};

use crate::error::Error;

checked_conversions!(Error {
    to_i64 -> i64,
    to_usize -> usize,
    to_i32 -> i32,
    to_u64 -> u64,
});

checked_arithmetic!(Error {
    checked_add,
    checked_mul
});
