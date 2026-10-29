//! Safe NVIDIA Management Library wrappers for monitoring, device management,
//! MIG, and vGPU APIs.
//!
//! This crate wraps NVML initialization and shutdown, system queries, device and
//! unit handles, MIG GPU and compute instances, vGPU types and instances, event
//! sets, clocks, power, thermals, memory, utilization, and other management
//! interfaces exposed by NVIDIA drivers.
//!
//! # Setup
//!
//! NVML is supplied by the NVIDIA driver rather than the CUDA Toolkit. The build
//! and runtime environment must be able to locate the driver NVML library, for
//! example through the platform library path.
//!
//! # Thread safety
//!
//! NVML is thread-safe, so handles such as [`device::Device`] are `Send` and `Sync`.
//! APIs that write into caller-owned objects, such as GPM samples, take `&mut`.
//!
//! # Testing
//!
//! Tests query the installed NVIDIA driver and any devices it reports.
//! This crate has no `testing` feature.
//!
//! # Example
//!
//! ```
//! use singe_nvml::library::Library;
//!
//! let nvml = Library::create()?;
//! let count = nvml.device_count()?;
//!
//! for index in 0..count {
//!     let device = nvml.device(index)?;
//!     let memory = device.memory_info()?;
//!     println!("{}: {} MiB free", device.name()?, memory.free / 1_048_576);
//! }
//!
//! println!("NVML version: {}", singe_nvml::version()?);
//! # Ok::<(), singe_nvml::error::Error>(())
//! ```

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

/// Device power, memory partitioning, and telemetry controls.
pub mod management;

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

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{error::Result, utility::to_u32};

/// Returns the version of the NVML library.
///
/// The version identifier is an alphanumeric string and does not require an
/// initialized [`library::Library`] handle.
///
/// # Errors
///
/// Returns an error if the internal version buffer is too small or if NVML
/// rejects the output argument.
pub fn version() -> Result<String> {
    let mut buffer = [0i8; sys::NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE as usize];
    // SAFETY: `nvmlSystemGetNVMLVersion` receives valid output locations.
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
    use crate::{
        error::{Error, Status},
        library::Library,
        types::{InitFlags, SystemEventTypes},
    };

    #[test]
    fn test_system_info_is_available_when_nvml_initializes() {
        match Library::create_with_flags(InitFlags::NO_GPUS) {
            Ok(nvml) => {
                assert!(!crate::version().unwrap().is_empty());
                assert!(!nvml.driver_version().unwrap().is_empty());
                assert_ne!(nvml.cuda_driver_version().unwrap().raw, 0);
                assert_ne!(nvml.cuda_driver_version_fallback().unwrap().raw, 0);
                let _ = nvml.vgpu_version();
                let _ = nvml.excluded_devices();
                let _ = nvml.system_event_set().and_then(|set| {
                    let _ = set.register_events(
                        SystemEventTypes::GPU_DRIVER_BIND | SystemEventTypes::GPU_DRIVER_UNBIND,
                    );
                    match set.wait(0, 4) {
                        Ok(_) => Ok(()),
                        Err(Error::Nvml {
                            code: Status::Timeout,
                            ..
                        }) => Ok(()),
                        Err(error) => Err(error),
                    }
                });
            }
            Err(error) => eprintln!("error initializing nvml: {error:?}"),
        }
    }
}

mod optional;
