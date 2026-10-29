use singe_core::checked_conversions;

use crate::error::Error;

checked_conversions!(Error {
    to_i64 -> i64,
    to_usize -> usize,
    to_i32 -> i32,
});
