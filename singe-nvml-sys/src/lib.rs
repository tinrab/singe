//! Raw FFI bindings for NVML.
//!
//! Prefer the safe `singe-nvml` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "nvml_13_2")]
include!("nvml_sys_13.rs");

#[cfg(feature = "nvml_13_2")]
pub use self::{
    nvmlDeviceGetComputeRunningProcesses_v3 as nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetGraphicsRunningProcesses_v3 as nvmlDeviceGetGraphicsRunningProcesses,
    nvmlDeviceGetMPSComputeRunningProcesses_v3 as nvmlDeviceGetMPSComputeRunningProcesses,
    nvmlDeviceGetMemoryInfo_v2 as nvmlDeviceGetMemoryInfoV,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        unsafe {
            let version = nvmlSystemGetNVMLVersion as usize;
            assert_ne!(version, 0);
        }
    }
}
