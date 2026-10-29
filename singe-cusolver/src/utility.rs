use singe_core::checked_conversions;

use crate::error::Error;

checked_conversions!(Error {
    to_i32 -> i32,
    to_u64 -> u64,
    to_i64 -> i64,
    to_usize -> usize,
});
