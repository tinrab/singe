//! Raw FFI bindings for cuBLAS, cuBLASLt, and cuBLASXt.
//!
//! Prefer the safe `singe-cublas` crate unless direct NVIDIA ABI access is required.

#![allow(
    deprecated,
    warnings,
    dead_code,
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
#[cfg(feature = "cublas_13_8")]
use singe_cuda_sys::{
    library_types::{
        cudaDataType, cudaDataType_t, cudaEmulationMantissaControl,
        cudaEmulationSpecialValuesSupport, libraryPropertyType,
    },
    runtime::cudaStream_t,
};

#[cfg(feature = "cublas_13_8")]
include!("sys_130800.rs");

#[cfg(all(feature = "cublas_13_8", feature = "lt"))]
mod lt_bindings {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    use super::{
        cublasComputeType_t, cublasSetWorkspace_v2, cublasStatus_t, cudaDataType, cudaDataType_t,
        cudaStream_t, libraryPropertyType,
    };

    include!("sys_lt_130800.rs");
}

#[cfg(all(feature = "cublas_13_8", feature = "lt"))]
pub use self::lt_bindings::*;

#[cfg(all(feature = "cublas_13_8", feature = "xt"))]
mod xt_bindings {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    use super::{
        cuComplex, cuDoubleComplex, cublasDiagType_t, cublasFillMode_t, cublasOperation_t,
        cublasSideMode_t, cublasStatus_t,
    };

    include!("sys_xt_130800.rs");
}

#[cfg(all(feature = "cublas_13_8", feature = "xt"))]
pub use self::xt_bindings::*;
#[cfg(feature = "cublas_13_8")]
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

    #[cfg(feature = "cublas_13_8")]
    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the handle is created, used and destroyed within this block.
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
