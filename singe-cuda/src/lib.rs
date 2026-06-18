//! Safe CUDA wrappers.
//!
//! This crate wraps CUDA Driver, CUDA Runtime, NVRTC, NVVM, NVTX, contexts, streams,
//! events, device memory, modules, kernels, graphs, IPC, and JIT helpers. It
//! keeps unsafe FFI calls at the boundary while still exposing enough low-level
//! control for inference runtimes and NVIDIA library wrappers.
//!
//! # Examples
//!
//! ```no_run
//! use singe_cuda::{
//!     context::Context,
//!     memory::DeviceMemory,
//! };
//!
//! let ctx = Context::create()?;
//! let stream = ctx.create_stream()?;
//!
//! let mut memory = DeviceMemory::<f32>::create(1024)?;
//! memory.copy_from_host(&vec![1.0_f32; 1024])?;
//! stream.synchronize()?;
//!
//! println!("CUDA driver v{}", singe_cuda::driver::version()?);
//! # Ok::<(), singe_cuda::error::Error>(())
//! ```

pub mod driver;
pub mod runtime;

pub mod architecture;
pub mod checkpoint;
pub mod context;
pub mod data_type;
pub mod device;
pub mod dim;
pub mod error;
pub mod event;
pub mod external_memory;
pub mod graph;
pub mod ipc;
pub mod jit;
pub mod kernel;
pub mod library;
pub mod memory;
pub mod module;
pub mod nvrtc;
pub mod nvtx;
pub mod nvvm;
pub mod profile;
pub mod stream;
pub mod types;
pub mod view;

#[cfg(feature = "macros")]
pub use singe_cuda_macros::cuda_module;

#[cfg(feature = "testing")]
pub mod testing;
