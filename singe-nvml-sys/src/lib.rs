//! Low-level FFI bindings for NVML 13, the NVIDIA Management Library, generated from `nvml.h`.
//! [`singe-loader`](singe_loader) loads the NVML library that ships with the NVIDIA driver at run time, on the first call or through [`load`], so building links no NVIDIA library.
//! Most users want the safe `singe-nvml` crate.
#![allow(
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    non_upper_case_globals,
    missing_debug_implementations,
    unnameable_types,
    unreachable_pub,
    clippy::correctness,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::absolute_paths,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks,
    reason = "generated bindings preserve upstream ABI conventions and documentation"
)]

#[cfg(not(feature = "nvml_13"))]
compile_error!("singe-nvml-sys requires a version feature, such as `nvml_13`");

pub use singe_loader::{Absence, LoadError, take_absence};

#[cfg(feature = "nvml_13")]
include!("nvml_sys_13.rs");

static LIBRARY: singe_loader::Library = singe_loader::Library::nvml();

/// Loads NVML now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "nvml_13")]
singe_loader::missing_status!(
    nvmlReturn_t,
    library: nvmlReturn_t::NVML_ERROR_LIBRARY_NOT_FOUND,
    symbol: nvmlReturn_t::NVML_ERROR_FUNCTION_NOT_FOUND,
);

#[cfg(feature = "nvml_13")]
pub use self::{
    nvmlDeviceGetComputeRunningProcesses_v3 as nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetGraphicsRunningProcesses_v3 as nvmlDeviceGetGraphicsRunningProcesses,
    nvmlDeviceGetMPSComputeRunningProcesses_v3 as nvmlDeviceGetMPSComputeRunningProcesses,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        load().unwrap();
    }
}
