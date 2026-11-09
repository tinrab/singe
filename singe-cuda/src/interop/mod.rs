//! Owner-retaining adapters for foreign CUDA runtimes.

#[cfg(feature = "interop_cuda_core")]
pub mod cuda_core;
#[cfg(feature = "interop_cudarc")]
pub mod cudarc;
