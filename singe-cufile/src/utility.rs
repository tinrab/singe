use singe_core::{checked_arithmetic, checked_conversions};

use crate::{error::Error, types::StatisticsLevel};

checked_conversions!(Error {
    to_i64 -> i64,
    to_u64 -> u64,
    to_i32 -> i32,
    to_u32 -> u32,
    to_usize -> usize,
    to_statistics_level -> StatisticsLevel,
});

checked_arithmetic!(Error { checked_byte_len });
