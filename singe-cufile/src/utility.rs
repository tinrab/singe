use std::ptr;

use singe_core::{checked_arithmetic, checked_conversions};
use singe_cufile_sys as sys;

use crate::error::Error;

checked_conversions!(Error {
    to_i64 -> i64,
    to_u64 -> u64,
    to_i32 -> i32,
    to_u32 -> u32,
    to_usize -> usize,
});

checked_arithmetic!(Error { checked_byte_len });

pub fn null_fs_ops() -> *const sys::CUfileFSOps_t {
    ptr::null()
}
