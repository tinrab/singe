//! Low-level FFI bindings for the dense cuSOLVER 12.3 API (`cusolverDn.h`).
//! The library loads at run time through [`singe-loader`](singe_loader), on the first call of one of its functions or up front with [`load`], so building links no NVIDIA library.
//! If the library or one of its symbols is missing, a function that returns a status returns a failure status, and [`take_absence`] reports what was missing.
//! Most users want the safe `singe-cusolver` crate.
#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
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

#[cfg(not(feature = "cusolver_12_3"))]
compile_error!("singe-cusolver-sys requires a version feature, such as `cusolver_12_3`");

use singe_cuda_sys::{
    library_types::{
        cudaDataType, cudaDataType_t, cudaEmulationMantissaControl_t,
        cudaEmulationSpecialValuesSupport_t, cudaEmulationStrategy_t, libraryPropertyType,
    },
    runtime::cudaStream_t,
};

#[cfg(feature = "cusolver_12_3")]
include!("sys_12304.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cusolver();

/// Loads cuSOLVER now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cusolver_12_3")]
singe_loader::missing_status!(
    cusolverStatus_t,
    library: cusolverStatus_t::CUSOLVER_STATUS_NOT_INITIALIZED,
    symbol: cusolverStatus_t::CUSOLVER_STATUS_NOT_SUPPORTED,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(
                cusolverGetVersion(&mut version),
                cusolverStatus_t::CUSOLVER_STATUS_SUCCESS,
            );
        }
        println!("cuSOLVER version: {}", version);
        assert_ne!(version, 0);
    }
}
