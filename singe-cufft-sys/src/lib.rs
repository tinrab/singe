//! Low-level FFI bindings for cuFFT 12.4 (`cufft.h`).
//! The [`xt`] module binds the extended API from `cufftXt.h`, and the [`device`] module binds the device-code generation API from `cufft_device.h`.
//! The library loads at run time through [`singe-loader`](singe_loader), on the first call of one of its functions or up front with [`load`], so building links no NVIDIA library.
//! If the library or one of its symbols is missing, a function that returns a status returns a failure status, and [`take_absence`] reports what was missing.
//! Most users want the safe `singe-cufft` crate.
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

#[cfg(not(feature = "cufft_12_4"))]
compile_error!("singe-cufft-sys requires a version feature, such as `cufft_12_4`");

#[cfg(feature = "cufft_12_4")]
use singe_cuda_sys::{
    library_types::{cudaDataType, cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cufft_12_4")]
include!("sys_12400.rs");

/// Experimental device FFT description and code-generation interface.
#[cfg(feature = "cufft_12_4")]
pub mod device {
    use crate::{cufftResult, cufftResult_t};

    include!("device_12400.rs");
}

/// Extended FFT planning, distributed memory and callback interface.
#[cfg(feature = "cufft_12_4")]
pub mod xt {
    use singe_cuda_sys::{
        library_types::{cudaDataType, cudaDataType_t},
        runtime::cudaStream_t,
    };

    use crate::{cufftHandle, cufftResult, cufftResult_t};

    include!("xt_12400.rs");
}

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cufft();

/// Loads cuFFT now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cufft_12_4")]
singe_loader::missing_status!(
    cufftResult,
    library: cufftResult::CUFFT_SETUP_FAILED,
    symbol: cufftResult::CUFFT_NOT_IMPLEMENTED,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(cufftGetVersion(&mut version), cufftResult::CUFFT_SUCCESS,);
        }
        println!("cuFFT version: {}", version);
        assert_ne!(version, 0);
    }
}
