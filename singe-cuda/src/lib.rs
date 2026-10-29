//! CUDA resources, typed memory, and public raw submission APIs.
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
//! let context = Context::create()?;
//! let memory = DeviceMemory::from_slice(&context, &[1.0_f32; 1024])?;
//! assert_eq!(memory.copy_to_host_vec()?, vec![1.0_f32; 1024]);
//!
//! println!("CUDA runtime v{}", singe_cuda::version()?);
//! # Ok::<(), singe_cuda::error::Error>(())
//! ```

/// CUDA GPU architecture names and target parsing.
pub mod architecture;
/// CUDA process checkpoint, restore, and checkpoint storage APIs.
pub mod checkpoint;
/// Shared CUDA contexts, primary-context retention, and code loading.
pub mod context;
/// CUDA scalar data types and native type mappings.
pub mod data_type;
/// Device enumeration, properties, attributes, and peer capabilities.
pub mod device;
/// Three-dimensional CUDA grid, block, and cluster dimensions.
pub mod dim;
/// CUDA driver version queries.
pub mod driver;
/// Typed CUDA errors, native status values, and FFI status conversion.
pub mod error;
/// CUDA events, timing, and stream ordering.
pub mod event;
pub mod external_memory;
pub mod fabric;
/// Asynchronous stream, event, and graph completion futures.
pub mod future;
/// CUDA graph construction, instantiation, update, and execution.
pub mod graph;
pub mod green;
/// CUDA runtime version queries.
pub mod runtime;

#[cfg(any(feature = "interop_cuda_core", feature = "interop_cudarc"))]
pub mod interop;
/// Interprocess CUDA memory and event handle exchange.
pub mod ipc;
/// CUDA driver JIT options and device-code linking.
pub mod jit;
/// Raw kernel handle adapters and typed launch preparation.
pub mod kernel;
/// CUDA libraries, kernel lookup, and library-owned globals.
pub mod library;
/// Device, managed, pinned host, and CUDA array memory.
pub mod memory;
/// CUDA modules, device-code images, kernel arguments, and launch configuration.
pub mod module;
#[cfg(feature = "nvjitlink_13_4")]
pub mod nvjitlink;
/// Runtime compilation of CUDA source and retrieval of compilation artifacts.
pub mod nvrtc;
/// NVTX annotations, ranges, markers, and profiling resources.
pub mod nvtx;
/// NVVM IR compilation and compiler options.
pub mod nvvm;
pub mod profile;
pub mod semaphore;
/// CUDA streams, capture scopes, synchronization, and ordered operations.
pub mod stream;
pub mod submission;
pub mod tensor_map;
pub mod texture;
pub mod tile_ir;
/// CUDA handle, pointer, scalar-storage, and configuration types.
pub mod types;
pub mod view;
pub mod virtual_memory;

pub use view::{AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr};

pub(crate) mod utility;

#[cfg(feature = "macros")]
pub use singe_cuda_macros::cuda_module;

#[cfg(feature = "testing")]
/// Device locks and context setup for CUDA tests.
pub mod testing;

use singe_core::CudaRuntimeVersion;

use crate::error::Result;

/// Returns the CUDA Runtime version.
///
/// # Errors
///
/// Returns an error if the CUDA Runtime cannot report its version.
pub fn version() -> Result<CudaRuntimeVersion> {
    runtime::version()
}
