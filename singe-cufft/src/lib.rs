//! Safe cuFFT plan and execution wrappers for GPU Fourier transforms.
//!
//! This crate wraps cuFFT plan creation, stream binding, workspace management, and
//! transform execution over the raw
//! `singe-cufft-sys` bindings.

/// cuFFT status codes and errors from planning, execution, and CUDA operations.
pub mod error;
/// Single-GPU FFT planning, workspace configuration, and transform submission.
pub mod plan;
/// Transform directions, element types, and native plan properties.
pub mod types;
pub(crate) mod utility;

/// Serialized test setup for cuFFT on CUDA device 0.
#[cfg(feature = "testing")]
pub mod testing;

use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
use singe_cufft_sys as sys;

use crate::error::Result;

/// Returns the loaded cuFFT library version.
///
/// # Errors
///
/// Returns an error if cuFFT cannot report the loaded library version.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    // SAFETY: `cufftGetVersion` receives valid output locations.
    unsafe {
        try_ffi!(sys::cufftGetVersion(&raw mut version))?;
    }
    let version = version as u64;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Returns the requested cuFFT library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuFFT cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `cufftGetProperty` receives valid output locations.
    unsafe {
        try_ffi!(sys::cufftGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_it_works() -> Result<()> {
        let _context = setup_context()?;
        assert_ne!(version()?, 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}

pub mod device;

pub mod xt;
