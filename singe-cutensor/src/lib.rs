//! Safe cuTENSOR wrappers for tensor descriptors, operations, plans, and execution.
//!
//! Optional `mg` and `mp` features expose multi-GPU and multi-process cuTENSOR modules.

/// Single-GPU library context ownership and native plan or kernel caches.
pub mod context;
/// Native statuses and typed wrapper validation errors.
pub mod error;
pub mod execution;
pub mod logger;
/// Tensor operation descriptors, signatures, and typed scalar parameters.
pub mod operation;
/// Kernel-selection preferences, workspace estimates, and native execution plans.
pub mod plan;
/// Dense and block-sparse tensor layout descriptors.
pub mod tensor;
/// Typed native algorithms, operators, attributes, logging selectors, and mode labels.
pub mod types;

#[cfg(feature = "mg")]
pub mod mg;

#[cfg(feature = "mp")]
pub mod mp;

pub(crate) mod utility;

#[cfg(feature = "testing")]
/// Serialized GPU test setup for cuTENSOR.
pub mod testing;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cutensor_sys as sys;

use crate::error::{Error, Result};

/// Returns the loaded cuTENSOR library version.
///
/// # Errors
///
/// Returns an error if cuTENSOR reports version zero.
pub fn version() -> Result<LibraryVersion> {
    // SAFETY: `cutensorGetVersion` takes no arguments.
    let version = nonzero_version("library", unsafe { sys::cutensorGetVersion() })?;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Returns the CUDA runtime version that cuTENSOR was compiled against.
///
/// Can be compared against the CUDA runtime version reported by the CUDA wrapper crate.
///
/// # Errors
///
/// Returns an error if cuTENSOR reports CUDA runtime version zero.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    // SAFETY: `cutensorGetCudartVersion` takes no arguments.
    nonzero_version("cuda runtime", unsafe { sys::cutensorGetCudartVersion() })
        .map(CudaRuntimeVersion::from)
}

fn nonzero_version(name: &'static str, version: u64) -> Result<u64> {
    if version == 0 {
        return Err(Error::InvalidVersion { name });
    }
    Ok(version)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::{error::Result, testing::setup_context};

    #[test]
    fn test_it_works() -> Result<()> {
        let _context = setup_context()?;
        assert_ne!(version()?, 0);
        assert_ne!(cudart_version()?, 0);
        Ok(())
    }
}
