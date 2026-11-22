//! Safe Rust wrappers for cuDNN, NVIDIA's deep neural network library, including a Rust port of NVIDIA's cuDNN Frontend graph API.
//!
//! This crate provides:
//!
//! - **Frontend graphs:** [`frontend::graph::Graph`] for matmul, convolution, normalization, pointwise, reduction, resampling, `RoPE`, and mixture-of-experts operations, with shape inference.
//! - **Attention:** [`Graph::sdpa`](crate::frontend::graph::Graph::sdpa) and [`Graph::sdpa_backward`](crate::frontend::graph::Graph::sdpa_backward) in floating point, FP8, or MXFP8, with causal, sliding-window, `ALiBi`, paged-cache, and block masks.
//! - **Plans:** heuristics, candidate filtering, autotuning, plan caches, and serialization of compiled plans in [`frontend::plan`].
//! - **Owned execution:** [`operation::FrontendExecution`] keeps a plan, its workspace, and its buffers alive, so it can be submitted on a stream or captured into a CUDA graph.
//! - **Backend and legacy APIs:** the backend descriptor and engine API in [`backend`], causal depthwise convolutions, graph neural network aggregation, and the legacy CTC loss, dropout, LRN, softmax, and spatial transformer APIs.
//!
//! # Example
//!
//! Multiply two bf16 matrices with a frontend graph:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
//! use singe_cudnn::{
//!     context::Context,
//!     data_type::{DataType, bf16},
//!     frontend::{
//!         graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
//!         operation::MatmulConfig,
//!         plan::{CompileConfig, ExecuteArgs, HeuristicMode},
//!     },
//!     tensor::{Shape, TensorId, TensorSpec},
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let cuda = CudaContext::create()?;
//!     let cudnn = Context::create(&cuda)?;
//!     let (m, n, k): (i64, i64, i64) = (32, 64, 128);
//!
//!     let mut graph = Graph::with_config(
//!         GraphConfig::new().with_data_type_policy(DataTypePolicy::new().with_compute(DataType::F32)),
//!     );
//!     let a = graph.insert_tensor(
//!         TensorId::new(1),
//!         TensorSpec::new(DataType::Bf16, Shape::contiguous([1, m, k])?),
//!     )?;
//!     let b = graph.insert_tensor(
//!         TensorId::new(2),
//!         TensorSpec::new(DataType::Bf16, Shape::contiguous([1, k, n])?),
//!     )?;
//!     let c = graph.matmul(
//!         a,
//!         b,
//!         MatmulConfig::new(),
//!         TensorOutputConfig::new()
//!             .with_data_type(DataType::F32)
//!             .output_tensor(),
//!     )?;
//!     let plan = graph.compile(
//!         Some(&cudnn),
//!         &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
//!     )?;
//!
//!     let a_device = DeviceMemory::from_slice(&cuda, &vec![bf16::ONE; usize::try_from(m * k)?])?;
//!     let b_device = DeviceMemory::from_slice(&cuda, &vec![bf16::ONE; usize::try_from(k * n)?])?;
//!     let mut c_device = DeviceMemory::<f32>::zeroed(&cuda, usize::try_from(m * n)?)?;
//!     let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, plan.workspace_size())?;
//!
//!     let mut bindings = plan.bindings()?;
//!     bindings.set_input(a, &a_device.view())?;
//!     bindings.set_input(b, &b_device.view())?;
//!     bindings.set(c, &mut c_device)?;
//!
//!     // SAFETY: the inputs, the output, and the workspace stay alive and unchanged
//!     // until `cuda.synchronize` returns.
//!     unsafe {
//!         plan.execute_async(
//!             &cudnn,
//!             ExecuteArgs::new(&bindings).with_workspace(workspace.output()),
//!         )?;
//!     }
//!     cuda.synchronize()?;
//!
//!     assert!(c_device.copy_to_host_vec()?.iter().all(|&value| value == 128.0));
//!     Ok(())
//! }
//! ```
//!
//! A [`Context`](crate::context::Context) owns a cuDNN handle in a [`singe-cuda`](singe_cuda) context.
//! Building a graph and compiling a plan are safe; running a plan is `unsafe`, because its bindings record device addresses without borrowing the buffers.
//!
//! [`Graph::compile`](crate::frontend::graph::Graph::compile) asks cuDNN's heuristics for engine configurations and builds the first one that supports the graph on this GPU.
//! The handle starts on the legacy default stream; [`Context::set_stream`](crate::context::Context::set_stream) installs another one.
//!
//! # Executing plans
//!
//! Plan execution has no safe counterpart; [`operation::FrontendExecution`] owns the resources it needs instead.
//! [`FrontendExecution::create`](crate::operation::FrontendExecution::create) and [`TunedPlan::prepare_owned`](crate::frontend::plan::TunedPlan::prepare_owned), both `unsafe`, allocate the workspace and build the variant pack up front.
//! The execution retains the cuDNN context, the compiled graph, the workspace, and the buffers registered with their owners through [`Bindings::set_owned_region`](crate::frontend::bindings::Bindings::set_owned_region).
//! Submit it with [`Stream::submit`](singe_cuda::stream::Stream::submit), or record it with [`Stream::capture_owned`](singe_cuda::stream::Stream::capture_owned).
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` functions check contexts, sizes, and descriptors, order the work after the producers that their device regions record, enqueue it on the handle's stream or the stream they take, and return without waiting.
//! They are allowed during stream capture.
//! The caller must:
//!
//! - keep every buffer, descriptor, plan, and host argument alive until the work completes, including after an error that followed a partial submission, and through every replay of a captured graph;
//! - initialize every input before the device reads it, including a destination read with a nonzero `beta`;
//! - give the work exclusive access to its outputs, and read them only after it completes successfully.
//!
//! Each safe counterpart without the suffix runs the `_async` function inside [`sync_scope`](crate::context::Context::sync_scope), so it rejects stream capture and waits even when the call fails.
//! Frontend and backend plan execution has no safe counterpart and does not order itself after the producers of its storage, because the plan's bindings record device addresses without borrowing the storage; the caller orders them, as [`Bindings`](crate::frontend::bindings::Bindings) describes.

#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

#[cfg(not(feature = "cudnn_9_27"))]
compile_error!("singe-cudnn requires a version feature, such as `cudnn_9_27`");

pub mod backend;
pub mod causal_convolution;
pub mod context;
pub mod data_type;
pub mod diagnostic;
pub mod error;
pub mod frontend;
pub mod gnn;
pub mod legacy;
pub mod math;
mod metadata;
pub mod operation;
pub mod scalar;
pub mod tensor;
#[cfg(feature = "testing")]
pub mod testing;
pub(crate) mod utility;

use singe_core::PackedVersion;
pub use singe_core::{CudaVersion, LibraryVersion};
use singe_cuda::types::LibraryProperty;
pub use singe_cudnn_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the major, minor and patch version of the cuDNN library loaded by this process as a [`LibraryVersion`].
///
/// # Errors
///
/// Returns an error if cuDNN cannot report its version properties or a property is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("cudnn version"))
}

pub(crate) fn pack_cudnn_version(version: LibraryVersion) -> u64 {
    // cuDNN 9 packs its version as `major * 10000 + minor * 100 + patch`.
    // cuDNN 8 and earlier packed `major * 1000 + minor * 100 + patch`, which C++ frontend errata files still use.
    let major_scale = if version.major() >= 9 { 10_000 } else { 1000 };
    u64::from(version.major()) * major_scale
        + u64::from(version.minor()) * 100
        + u64::from(version.patch())
}

pub(crate) fn unpack_cudnn_version(value: u64) -> Option<LibraryVersion> {
    let scheme = if value >= 10_000 {
        PackedVersion::Major10000Minor100
    } else {
        PackedVersion::Major1000Minor100
    };
    LibraryVersion::from_packed(value, scheme)
}

/// Returns the CUDA toolkit version that the loaded cuDNN library was built against, as a [`CudaVersion`].
/// One cuDNN release ships builds for several CUDA toolkits.
///
/// # Errors
///
/// Returns an error if the native CUDA runtime version does not fit in `u32`.
pub fn cudart_version() -> Result<CudaVersion> {
    // SAFETY: `cudnnGetCudartVersion` takes no arguments.
    CudaVersion::from_raw(unsafe { sys::cudnnGetCudartVersion() })
        .ok_or_else(|| Error::out_of_range("cudnn cudart version"))
}

/// Returns the newest SM version that the loaded cuDNN library supports natively, such as `900` for Hopper.
/// cuDNN runs newer devices in forward-compatibility mode.
#[must_use]
pub fn max_device_version() -> u64 {
    // SAFETY: `cudnnGetMaxDeviceVersion` takes no arguments.
    unsafe { sys::cudnnGetMaxDeviceVersion() }
}

/// A cuDNN sub-library with its own version check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Sublibrary {
    /// The graph sub-library.
    Graph,
    /// The operations sub-library.
    Ops,
    /// The advanced operations sub-library.
    Adv,
    /// The convolution sub-library.
    Cnn,
    /// The subquadratic operations sub-library used by causal convolution.
    SubquadraticOps,
}

/// Checks that `sublibrary` has the same version as the sub-libraries it depends on.
///
/// # Errors
///
/// Returns an error if cuDNN reports that the sub-library versions are inconsistent.
pub fn version_check(sublibrary: Sublibrary) -> Result<()> {
    // SAFETY: the version checks take no arguments.
    unsafe {
        try_ffi!(match sublibrary {
            Sublibrary::Graph => sys::cudnnGraphVersionCheck(),
            Sublibrary::Ops => sys::cudnnOpsVersionCheck(),
            Sublibrary::Adv => sys::cudnnAdvVersionCheck(),
            Sublibrary::Cnn => sys::cudnnCnnVersionCheck(),
            Sublibrary::SubquadraticOps => sys::cudnnSubquadraticOpsVersionCheck(),
        })
    }
}

/// Returns the requested cuDNN library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuDNN cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a live local `i32` that cuDNN writes the property into.
    unsafe {
        try_ffi!(sys::cudnnGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        assert_ne!(version()?.major(), 0);
        Ok(())
    }
}
