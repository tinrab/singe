//! Safe cuTENSOR wrappers for tensor descriptors, operations, plans, and execution.
//!
//! Optional `mg` and `mp` features expose multi-GPU and multi-process cuTENSOR modules.

pub mod context;
pub mod error;
pub mod execution;
pub mod operation;
pub mod plan;
pub mod tensor;
pub mod types;

#[cfg(feature = "mg")]
pub mod mg;

#[cfg(feature = "mp")]
pub mod mp;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cutensor_sys as sys;

use crate::error::Error;

pub fn version() -> error::Result<LibraryVersion> {
    let version = nonzero_version("library", unsafe { sys::cutensorGetVersion() as _ })? as u64;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Returns the CUDA runtime version that cuTENSOR was compiled against.
///
/// Can be compared against the CUDA runtime version reported by the CUDA wrapper crate.
pub fn cudart_version() -> error::Result<CudaRuntimeVersion> {
    nonzero_version("cuda runtime", unsafe {
        sys::cutensorGetCudartVersion() as _
    })
    .map(CudaRuntimeVersion::from)
}

fn nonzero_version(name: &'static str, version: usize) -> error::Result<usize> {
    if version == 0 {
        return Err(Error::InvalidVersion { name });
    }
    Ok(version)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::testing::setup_context;

    #[test]
    fn it_works() -> Result<()> {
        let _context = setup_context()?;
        assert_ne!(version()?, 0);
        assert_ne!(cudart_version()?, 0);
        Ok(())
    }
}
