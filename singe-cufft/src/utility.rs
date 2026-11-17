use singe_core::{checked_arithmetic, checked_conversions};
use singe_cufft_sys::xt::size_t;

use crate::error::Error;

checked_conversions!(Error {
    to_i32 -> i32,
    to_i64 -> i64,
    to_usize -> usize,
    to_size_t -> size_t,
});

checked_arithmetic!(Error { checked_byte_len });
