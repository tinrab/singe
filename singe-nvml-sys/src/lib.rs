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
pub use singe_loader::LoadError;

#[cfg(feature = "nvml_615")]
include!("nvml_sys_13.rs");

static LIBRARY: singe_loader::Library = singe_loader::Library::nvml();

// Loads NVML, which every function otherwise does on first call, returning `NVML_ERROR_LIBRARY_NOT_FOUND` when it cannot.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "nvml_615")]
singe_loader::missing_status!(
    nvmlReturn_t,
    library: nvmlReturn_t::NVML_ERROR_LIBRARY_NOT_FOUND,
    symbol: nvmlReturn_t::NVML_ERROR_FUNCTION_NOT_FOUND,
);

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
