//! Safe Rust wrappers for NVML, the NVIDIA Management Library, for monitoring and managing GPUs.
//! NVML ships with the NVIDIA driver and needs no CUDA Toolkit or CUDA runtime.
//!
//! This crate provides:
//!
//! - **Library:** [`Library`](library::Library) owns an NVML initialization and reports driver and CUDA driver versions, enumerates devices and units, and creates event sets and GPM samples.
//! - **Devices:** [`Device`](device::Device) queries memory, utilization, temperature, fans, clocks, power, ECC, PCI, topology, and processes, and changes the settings NVML exposes for them.
//! - **MIG:** GPU instances and compute instances on partitioned GPUs.
//! - **vGPU:** vGPU types and instances.
//! - **Operational events:** subscriptions to GPU operational events.
//!
//! # Example
//!
//! Print each GPU's name and free memory, then the NVML version:
//!
//! ```rust,no_run
//! use singe_nvml::{library::Library, types::InitFlags};
//!
//! fn main() -> anyhow::Result<()> {
//!     let nvml = Library::create(InitFlags::empty())?;
//!     for index in 0..nvml.device_count()? {
//!         let device = nvml.device(index)?;
//!         let memory = device.memory_info()?;
//!         println!("{}: {} MiB free", device.name()?, memory.free / 1_048_576);
//!     }
//!
//!     println!("NVML version: {}", singe_nvml::version()?);
//!     Ok(())
//! }
//! ```
//!
//! [`Library::create`](library::Library::create) initializes NVML without initializing any GPU; NVML initializes a GPU when a handle to it is first requested.
//! Devices and other objects keep their [`Library`](library::Library) alive, and NVML shuts down after the last [`Library`](library::Library) clone and every object that holds one are dropped.
//! The queries and settings available depend on the GPU and driver, and changing a setting needs root or administrator permissions.
//!
//! # Thread safety
//!
//! NVML is thread-safe, so handles such as [`device::Device`] are [`Send`] and [`Sync`].
//! APIs that write into caller-owned objects, such as GPM samples, take `&mut`.

#[cfg(not(feature = "nvml_13"))]
compile_error!("singe-nvml requires a version feature, such as `nvml_13`");

/// MIG compute-instance handles, profiles, and placement queries.
pub mod compute_instance;

/// GPU device handles and device monitoring queries.
pub mod device;

/// Typed NVML statuses and wrapper error values.
pub mod error;

/// MIG GPU-instance handles, profiles, and placement queries.
pub mod gpu_instance;

/// NVML initialization, reference-counted shutdown, and system queries.
pub mod library;

/// Owned event records and structured GPU Operational Event subscriptions.
pub mod operational_event;

/// Typed device properties, query records, identifiers, and control settings.
pub mod types;

/// Multi-GPU unit monitoring, fan, power-supply, and enclosure information.
pub mod unit;

/// Queries and controls for virtual GPU instances.
pub mod vgpu_instance;

/// Virtual GPU type capabilities, geometry, and resource limits.
pub mod vgpu_type;

pub(crate) mod utility;

pub use singe_core::CudaVersion;
use singe_core::string_from_c_chars;
pub use singe_nvml_sys as sys;

use crate::{
    error::{Result, try_ffi},
    utility::to_u32,
};

/// Returns the version of the NVML library.
///
/// The version is an alphanumeric string, and the query does not need an initialized [`library::Library`].
///
/// # Errors
///
/// Returns an error if the internal version buffer is too small or if NVML rejects the output argument.
pub fn version() -> Result<String> {
    let mut buffer = [0; sys::NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE as usize];
    // SAFETY: `buffer` holds the `buffer.len()` writable bytes passed as its length.
    unsafe {
        try_ffi!(sys::nvmlSystemGetNVMLVersion(
            buffer.as_mut_ptr(),
            to_u32(buffer.len(), "NVML buffer length")?,
        ))?;
    }
    Ok(string_from_c_chars(&buffer))
}

#[cfg(test)]
mod tests {
    use crate::{error::Result, library::Library, types::InitFlags};

    #[test]
    fn test_system_info_is_available_when_nvml_initializes() -> Result<()> {
        let nvml = Library::create(InitFlags::NO_GPUS)?;
        assert!(!crate::version()?.is_empty());
        assert!(!nvml.driver_version()?.is_empty());
        assert_ne!(nvml.cuda_driver_version()?.major(), 0);
        Ok(())
    }
}
