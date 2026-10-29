//! Raw FFI bindings for cuFFT.
//!
//! Prefer the safe `singe-cufft` crate unless direct NVIDIA ABI access is required.

#![allow(
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    clippy::all,
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
#[cfg(feature = "cufft_12_4")]
use singe_cuda_sys::{
    library_types::{cudaDataType, cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cufft_12_4")]
include!("sys_12400.rs");

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

/// Experimental device FFT description and code-generation interface.
#[cfg(feature = "cufft_12_4")]
pub mod device {
    use super::{cufftResult, cufftResult_t};

    include!("device_12400.rs");
}

/// Extended FFT planning, distributed memory and callback interface.
#[cfg(feature = "cufft_12_4")]
pub mod xt {
    use singe_cuda_sys::{
        library_types::{cudaDataType, cudaDataType_t},
        runtime::cudaStream_t,
    };

    use super::{cufftHandle, cufftResult, cufftResult_t};

    include!("xt_12400.rs");
}
