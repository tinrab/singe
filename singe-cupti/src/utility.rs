use singe_core::{checked_arithmetic, checked_conversions};

use crate::error::Error;

checked_conversions!(Error {
    to_u64 -> u64,
    to_u32 -> u32,
    to_usize -> usize,
});

checked_arithmetic!(Error { checked_byte_len });
