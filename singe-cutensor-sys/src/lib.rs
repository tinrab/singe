//! Low-level FFI bindings for NVIDIA cuTENSOR, cuTENSORMg, and cuTENSORMp.
//!
//! Most users want the safe crate, `singe-cutensor`.
//!
//! # Bindings
//!
//! | Binds                                                                 | Header         | Module     | Feature        |
//! | --------------------------------------------------------------------- | -------------- | ---------- | -------------- |
//! | cuTENSOR, tensor contractions, reductions, and elementwise operations | `cutensor.h`   | crate root | `cutensor_2_8` |
//! | cuTENSORMg, for tensor operations across several GPUs in one process  | `cutensorMg.h` | crate root | `mg`           |
//! | cuTENSORMp, for tensor operations across processes over NCCL          | `cutensorMp.h` | crate root | `mp`           |
//!
//! The `cutensor_2_8` feature is on by default, and the others are off; `mp` adds a dependency on `singe-nccl-sys`.
//! The bindings are generated with bindgen from the cuTENSOR 2.8 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads each library on the first call of one of its functions, or up front with [`load`] for cuTENSOR, `load_mg` for cuTENSORMg, and `load_mp` for cuTENSORMp.
//! When a library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
//! A function without a status returns zero, null, or nothing instead.
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

#[cfg(not(feature = "cutensor_2_8"))]
compile_error!("singe-cutensor-sys requires a version feature, such as `cutensor_2_8`");

#[cfg(feature = "cutensor_2_8")]
use singe_cuda_sys::driver::CUstream as cudaStream_t;
#[cfg(feature = "cutensor_2_8")]
pub use singe_cuda_sys::library_types::cudaDataType as cudaDataType_t;
#[cfg(feature = "mp")]
use singe_nccl_sys::ncclComm_t;

#[cfg(feature = "cutensor_2_8")]
include!("sys_20800.rs");

#[cfg(all(feature = "cutensor_2_8", feature = "mg"))]
include!("mg_20800.rs");

#[cfg(all(feature = "cutensor_2_8", feature = "mp"))]
include!("mp_20800.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cutensor();

/// Loads cuTENSOR now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "mg")]
static MG: singe_loader::Library = singe_loader::Library::cutensor_mg();

/// Loads cuTENSORMg now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
#[cfg(feature = "mg")]
pub fn load_mg() -> Result<(), LoadError> {
    MG.load()
}

#[cfg(feature = "mp")]
static MP: singe_loader::Library = singe_loader::Library::cutensor_mp();

/// Loads cuTENSORMp now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
#[cfg(feature = "mp")]
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
