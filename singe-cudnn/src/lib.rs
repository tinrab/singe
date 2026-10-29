//! Safe cuDNN graph/frontend, backend descriptor, and execution wrappers.
//!
//! This crate exposes cuDNN frontend graph construction, graph lowering,
//! execution plan selection, variant packs, tensor descriptors, operation
//! descriptors, and lower-level backend APIs. It is designed for
//! inference-oriented graph execution while still leaving access to cuDNN
//! descriptor and engine details when required.

#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

pub mod backend;
pub mod causal_convolution;
/// Native cuDNN handle ownership and CUDA context and stream binding.
pub mod context;
/// Tensor storage data types, packed encodings, and scalar type associations.
pub mod data_type;
pub mod diagnostic;
/// Native status codes and checked cuDNN wrapper errors.
pub mod error;
pub mod frontend;
pub mod gnn;
pub mod legacy;
/// Math modes, NaN propagation, determinism, and rational parameters.
pub mod math;
mod metadata;
pub mod operation;
/// Scalar values and aligned host storage for by-value tensor bindings.
pub mod scalar;

pub(crate) mod utility;

#[cfg(feature = "testing")]
/// Serialized GPU test fixtures with an installed cuDNN stream.
pub mod testing;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cuda::types::LibraryProperty;
use singe_cudnn_sys as sys;

use crate::{error::Result, utility::to_u64};

/// Returns the version number of the cuDNN library.
/// It returns the version number reported by the linked cuDNN library.
/// This can identify the cuDNN library dynamically loaded by the current process.
/// This distinguishes cuDNN libraries when the same application may be linked against different cuDNN versions.
///
/// # Errors
///
/// Returns an error if the native version value cannot be represented as `u64`.
pub fn version() -> Result<LibraryVersion> {
    // SAFETY: `cudnnGetVersion` takes no arguments.
    let version = to_u64(unsafe { sys::cudnnGetVersion() }, "version")?;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// The same version of a given cuDNN library can be compiled against different CUDA toolkit versions.
/// Returns the CUDA toolkit version that the currently used cuDNN library was compiled against.
///
/// # Errors
///
/// Returns an error if the native CUDA runtime version cannot be represented as `u64`.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    // SAFETY: `cudnnGetCudartVersion` takes no arguments.
    to_u64(unsafe { sys::cudnnGetCudartVersion() }, "cudart version").map(CudaRuntimeVersion::from)
}

/// Returns the maximum SM version that the cuDNN library is aware of and supports natively.
/// Any SM version higher than this would be supported in forward compatibility mode.
/// For more information about forward compatibility, refer to the hardware forward compatibility
/// section in the cuDNN frontend developer guide.
///
/// A value indicating the latest known SM number for the current version of the library.
/// For example, if NVIDIA Hopper (GH100) is the latest known SM that the library is aware of, the value returned would be `900`.
#[must_use]
pub fn max_device_version() -> u64 {
    // SAFETY: `cudnnGetMaxDeviceVersion` takes no arguments.
    unsafe { sys::cudnnGetMaxDeviceVersion() }
}

/// Cross-library version checker.
/// Each sub-library has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent graph sublibrary versions.
pub fn graph_version_check() -> Result<()> {
    // SAFETY: `cudnnGraphVersionCheck` takes no arguments.
    unsafe { try_ffi!(sys::cudnnGraphVersionCheck()) }
}

/// Cross-library version checker.
/// Each sublibrary has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent operations sublibrary versions.
pub fn ops_version_check() -> Result<()> {
    // SAFETY: `cudnnOpsVersionCheck` takes no arguments.
    unsafe { try_ffi!(sys::cudnnOpsVersionCheck()) }
}

/// Cross-library version checker.
/// Each sublibrary has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent advanced sublibrary versions.
pub fn adv_version_check() -> Result<()> {
    // SAFETY: `cudnnAdvVersionCheck` takes no arguments.
    unsafe { try_ffi!(sys::cudnnAdvVersionCheck()) }
}

/// Cross-library version checker.
/// Each sublibrary has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent CNN sublibrary versions.
pub fn cnn_version_check() -> Result<()> {
    // SAFETY: `cudnnCnnVersionCheck` takes no arguments.
    unsafe { try_ffi!(sys::cudnnCnnVersionCheck()) }
}

/// Returns the requested cuDNN library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuDNN cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `cudnnGetProperty` receives valid output locations.
    unsafe {
        try_ffi!(sys::cudnnGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        assert_ne!(version()?, 0);
        Ok(())
    }
}
