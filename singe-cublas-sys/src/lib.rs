//! Raw FFI bindings for cuBLAS and cuBLASLt.
//!
//! Prefer the safe `singe-cublas` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "cublas_13_4")]
use singe_cuda_sys::{
    library_types::{
        cudaDataType, cudaDataType_t, cudaEmulationMantissaControl,
        cudaEmulationSpecialValuesSupport, libraryPropertyType,
    },
    runtime::cudaStream_t,
};

#[cfg(feature = "cublas_13_4")]
include!("sys_130400.rs");

#[cfg(feature = "cublas_13_4")]
mod lt_bindings {
    use super::{
        cublasComputeType_t, cublasSetWorkspace_v2, cublasStatus_t, cudaDataType, cudaDataType_t,
        cudaStream_t, libraryPropertyType,
    };
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    include!("sys_lt_130400.rs");
}

#[cfg(feature = "cublas_13_4")]
pub use self::lt_bindings::*;

#[cfg(feature = "cublas_13_4")]
pub use self::{
    cublasCreate_v2 as cublasCreate, cublasDestroy_v2 as cublasDestroy,
    cublasGetPointerMode_v2 as cublasGetPointerMode, cublasGetStream_v2 as cublasGetStream,
    cublasSetPointerMode_v2 as cublasSetPointerMode, cublasSetStream_v2 as cublasSetStream,
    cublasSetWorkspace_v2 as cublasSetWorkspace,
};

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[cfg(feature = "cublas_13_4")]
    #[test]
    fn it_works() {
        let mut version = 0;
        unsafe {
            let mut ctx = ptr::null_mut();
            assert_eq!(
                cublasCreate_v2(&mut ctx),
                cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            );
            cublasGetVersion_v2(ctx, &mut version);
        }
        println!("cuBLAS version: {}", version);
        assert_ne!(version, 0);
    }
}
