pub mod blas;
pub mod context;
pub mod types;

use crate::{
    error::Result,
    try_ffi,
    utility::{to_i32, to_usize},
};
use singe_cuda::device::Device;

/// Returns the maximum number of GPU boards supported by cuBLASXt.
///
/// # Errors
///
/// Returns an error if cuBLASXt cannot report the value.
pub fn get_max_boards() -> Result<usize> {
    let mut boards = 0;
    unsafe {
        try_ffi!(singe_cublas_sys::cublasXtMaxBoards(&raw mut boards))?;
    }
    to_usize(boards, "maximum GPU boards")
}

/// Returns the number of GPU boards represented by `devices`.
///
/// # Errors
///
/// Returns an error if `devices` is too long for cuBLASXt or cuBLASXt cannot query the board count.
pub fn get_board_count(devices: &[Device]) -> Result<usize> {
    let mut device_ids = devices
        .iter()
        .map(|device| to_i32(device.id(), "device id"))
        .collect::<Result<Vec<i32>>>()?;
    let mut boards = 0;
    unsafe {
        try_ffi!(singe_cublas_sys::cublasXtGetNumBoards(
            to_i32(device_ids.len(), "device count")?,
            device_ids.as_mut_ptr(),
            &raw mut boards,
        ))?;
    }
    to_usize(boards, "GPU boards")
}
