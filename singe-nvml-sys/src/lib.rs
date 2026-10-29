//! Raw FFI bindings for NVML.
//!
//! Prefer the safe `singe-nvml` crate unless direct NVIDIA ABI access is required.

#![allow(
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    non_upper_case_globals,
    clippy::all,
    clippy::correctness,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::absolute_paths,
    clippy::pedantic,
    clippy::nursery,
    reason = "generated bindings preserve upstream ABI conventions and documentation"
)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "nvml_615")]
include!("nvml_sys_13.rs");

#[cfg(feature = "nvml_615")]
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
    fn test_it_works() {
        let version = nvmlSystemGetNVMLVersion as usize;
        assert_ne!(version, 0);
    }
}
