//! CUDA Toolkit and NVIDIA library discovery helpers for Singe build scripts and
//! wrapper crates.
//!
//! This crate centralizes the path probing used by the Singe CUDA ecosystem. It
//! looks for headers, libraries, and tools through explicit environment
//! variables, the active `PATH` or dynamic library path, and common CUDA
//! installation roots. The result is a small [`Dependency`] value containing
//! include and library directories plus a detected version when the header
//! exposes one.
//!
//! # Discovery
//!
//! Toolkit components prefer component-specific roots such as `CUBLAS_PATH` or
//! `CUSOLVER_ROOT`, then fall back to `CUDA_PATH` or `CUDA_HOME`, then common
//! platform locations. Driver-owned libraries such as CUDA Driver and NVML are
//! searched in driver library paths because they are not necessarily installed
//! inside the CUDA Toolkit.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cuda = singe_cuda_find::find_cuda_runtime()?
//!     .expect("CUDA runtime headers and library were not found");
//!
//! println!("cargo:include={}", cuda.include_path.display());
//! println!("cargo:rustc-link-search=native={}", cuda.library_path.display());
//!
//! if let Some(version) = cuda.version {
//!     println!("cargo:warning=found CUDA runtime {version}");
//! }
//! # Ok(())
//! # }
//! ```

mod dependency;
mod environment;
mod error;
mod patterns;
mod search;
mod version;

pub use crate::{
    dependency::{
        Dependency, emit_dependency_metadata, emit_library_path_metadata, emit_link_libraries,
        emit_static_link_libraries, find_cublas, find_cuda_driver_path, find_cuda_runtime,
        find_cudnn, find_cudss, find_cufile, find_cupti, find_curand, find_cusolver, find_cusparse,
        find_cutensor, find_nccl, find_npp, find_nvcc, find_nvcomp, find_nvimagecodec,
        find_nvjitlink, find_nvjpeg, find_nvjpeg2k, find_nvml, find_nvrtc, find_nvtiff, find_nvvm,
    },
    environment::emit_rerun_if_env_changed,
    error::{FindError, Result},
};
