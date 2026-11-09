//! Raw FFI bindings for cuTENSOR and cuTENSORMg.
//!
//! Prefer the safe `singe-cutensor` crate unless direct NVIDIA ABI access is required.

#![allow(
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
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
use singe_cuda_sys::driver::CUstream as cudaStream_t;
pub use singe_cuda_sys::library_types::cudaDataType as cudaDataType_t;
use singe_nccl_sys::ncclComm_t;

#[cfg(feature = "cutensor_2_8")]
include!("sys_20800.rs");

#[cfg(all(feature = "cutensor_2_8", feature = "mg"))]
include!("mg_20800.rs");

#[cfg(all(feature = "cutensor_2_8", feature = "mp"))]
include!("mp_20800.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::cutensor();

// Loads cuTENSOR, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

static MG: singe_loader::Library = singe_loader::Library::cutensor_mg();

// Loads cuTENSORMg, which each of its functions otherwise does on first call.
pub fn load_mg() -> Result<(), LoadError> {
    MG.load()
}

static MP: singe_loader::Library = singe_loader::Library::cutensor_mp();

// Loads cuTENSORMp, which each of its functions otherwise does on first call.
pub fn load_mp() -> Result<(), LoadError> {
    MP.load()
}

#[cfg(feature = "cutensor_2_8")]
singe_loader::missing_status!(
    cutensorStatus_t,
    library: cutensorStatus_t::CUTENSOR_STATUS_NOT_INITIALIZED,
    symbol: cutensorStatus_t::CUTENSOR_STATUS_NOT_SUPPORTED,
);

// Internal-linkage constants of `cutensor.h`, which no library exports.
#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_16F: cutensorDataType_t = cutensorDataType_t::CUDA_R_16F;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_16F: cutensorDataType_t = cutensorDataType_t::CUDA_C_16F;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_16BF: cutensorDataType_t = cutensorDataType_t::CUDA_R_16BF;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_16BF: cutensorDataType_t = cutensorDataType_t::CUDA_C_16BF;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_32F: cutensorDataType_t = cutensorDataType_t::CUDA_R_32F;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_32F: cutensorDataType_t = cutensorDataType_t::CUDA_C_32F;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_64F: cutensorDataType_t = cutensorDataType_t::CUDA_R_64F;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_64F: cutensorDataType_t = cutensorDataType_t::CUDA_C_64F;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_4I: cutensorDataType_t = cutensorDataType_t::CUDA_R_4I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_4I: cutensorDataType_t = cutensorDataType_t::CUDA_C_4I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_4U: cutensorDataType_t = cutensorDataType_t::CUDA_R_4U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_4U: cutensorDataType_t = cutensorDataType_t::CUDA_C_4U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_8I: cutensorDataType_t = cutensorDataType_t::CUDA_R_8I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_8I: cutensorDataType_t = cutensorDataType_t::CUDA_C_8I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_8U: cutensorDataType_t = cutensorDataType_t::CUDA_R_8U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_8U: cutensorDataType_t = cutensorDataType_t::CUDA_C_8U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_16I: cutensorDataType_t = cutensorDataType_t::CUDA_R_16I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_16I: cutensorDataType_t = cutensorDataType_t::CUDA_C_16I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_16U: cutensorDataType_t = cutensorDataType_t::CUDA_R_16U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_16U: cutensorDataType_t = cutensorDataType_t::CUDA_C_16U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_32I: cutensorDataType_t = cutensorDataType_t::CUDA_R_32I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_32I: cutensorDataType_t = cutensorDataType_t::CUDA_C_32I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_32U: cutensorDataType_t = cutensorDataType_t::CUDA_R_32U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_32U: cutensorDataType_t = cutensorDataType_t::CUDA_C_32U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_64I: cutensorDataType_t = cutensorDataType_t::CUDA_R_64I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_64I: cutensorDataType_t = cutensorDataType_t::CUDA_C_64I;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_R_64U: cutensorDataType_t = cutensorDataType_t::CUDA_R_64U;

#[cfg(feature = "cutensor_2_8")]
pub const CUTENSOR_C_64U: cutensorDataType_t = cutensorDataType_t::CUDA_C_64U;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        // SAFETY: the version query writes only into the local output locations.
        let version = unsafe { cutensorGetVersion() };
        println!("cuTENSOR version: {}", version);
        assert_ne!(version, 0);
    }
}
