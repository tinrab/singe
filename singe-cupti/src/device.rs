use singe_cuda::device::Device;
use singe_cupti_sys as sys;

use crate::{error::Result, try_ffi, types::DeviceVirtualizationMode};

/// Returns whether this CUPTI runtime supports a CUDA compute capability.
///
/// Prefer [`supports_device`] when a concrete CUDA device is available, because support can vary between devices with the same compute capability.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn supports_compute_capability(major: i32, minor: i32) -> Result<bool> {
    let mut support = 0;
    unsafe {
        try_ffi!(sys::cuptiComputeCapabilitySupported(
            major,
            minor,
            &mut support,
        ))?;
    }
    Ok(support != 0)
}

/// Returns whether this CUPTI runtime supports profiling the CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid or CUPTI cannot resolve it.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn supports_device(device: Device) -> Result<bool> {
    let mut support = 0;
    unsafe {
        try_ffi!(sys::cuptiDeviceSupported(
            device.id() as sys::CUdevice,
            &mut support,
        ))?;
    }
    Ok(support != 0)
}

/// Returns the virtualization mode reported for the CUDA device.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid or CUPTI cannot resolve it.
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn device_virtualization_mode(device: Device) -> Result<DeviceVirtualizationMode> {
    let mut mode = sys::CUpti_DeviceVirtualizationMode::CUPTI_DEVICE_VIRTUALIZATION_MODE_NONE;
    unsafe {
        try_ffi!(sys::cuptiDeviceVirtualizationMode(
            device.id() as sys::CUdevice,
            &mut mode,
        ))?;
    }
    Ok(DeviceVirtualizationMode::from(mode))
}
