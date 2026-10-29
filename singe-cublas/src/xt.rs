/// Host-memory BLAS operations distributed across cuBLASXt devices.
pub mod blas;
/// cuBLASXt handle ownership, device selection, and hybrid execution settings.
pub mod context;
/// Typed cuBLASXt operation and host-memory pinning settings.
pub mod types;

use singe_cuda::device::Device;

use crate::{
    error::Result,
    try_ffi,
    utility::{to_i32, to_usize},
};

/// Returns the maximum number of GPU boards supported by cuBLASXt.
///
/// # Errors
///
/// Returns an error if cuBLASXt cannot report the value.
pub fn max_boards() -> Result<usize> {
    let mut boards = 0;
    // SAFETY: `cublasXtMaxBoards` receives valid output locations.
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
pub fn board_count(devices: &[Device]) -> Result<usize> {
    let mut device_ids = devices
        .iter()
        .map(|device| to_i32(device.id(), "device id"))
        .collect::<Result<Vec<i32>>>()?;
    let mut boards = 0;
    // SAFETY: `cublasXtGetNumBoards` receives valid output locations.
    unsafe {
        try_ffi!(singe_cublas_sys::cublasXtGetNumBoards(
            to_i32(device_ids.len(), "device count")?,
            device_ids.as_mut_ptr(),
            &raw mut boards,
        ))?;
    }
    to_usize(boards, "GPU boards")
}
