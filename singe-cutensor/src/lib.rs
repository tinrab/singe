//! Safe Rust wrappers for NVIDIA cuTENSOR, for tensor contractions, reductions, permutations, and elementwise operations on the GPU.
//!
//! This crate provides:
//!
//! - **Tensor descriptors:** dense tensors with explicit extents, strides, and alignment, and block-sparse tensors.
//! - **Operations:** contractions of two or three tensors, reductions, permutations, elementwise binary and trinary operations, and block-sparse contractions, with modes labelled by characters.
//! - **Planning:** kernel-selection preferences, JIT-compiled kernels, workspace estimates, and plan and kernel caches that can be saved to files.
//! - **Multi-GPU and multi-process:** cuTENSORMg and cuTENSORMp, behind the `mg` and `mp` features.
//!
//! # Example
//!
//! Contract `D[m, n] = A[m, k] * B[k, n]`, a matrix product written as a tensor contraction:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::StreamOptions};
//! use singe_cutensor::{
//!     context::Context,
//!     operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
//!     plan::{Plan, PlanPreference},
//!     tensor::TensorDescriptor,
//!     types::{Mode, WorkspacePreference},
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let cuda = CudaContext::create()?;
//!     let context = Context::create(&cuda)?;
//!     let stream = cuda.create_stream(StreamOptions::default())?;
//!
//!     let modes_a = [Mode::from('m'), Mode::from('k')];
//!     let modes_b = [Mode::from('k'), Mode::from('n')];
//!     let modes_d = [Mode::from('m'), Mode::from('n')];
//!     let descriptor_a = TensorDescriptor::create_for::<f32>(&context, &[2, 4], 128)?;
//!     let descriptor_b = TensorDescriptor::create_for::<f32>(&context, &[4, 3], 128)?;
//!     let descriptor_d = TensorDescriptor::create_for::<f32>(&context, &[2, 3], 128)?;
//!
//!     let contraction = OperationDescriptor::contraction(
//!         &context,
//!         TensorOperand::identity(&descriptor_a, &modes_a),
//!         TensorOperand::identity(&descriptor_b, &modes_b),
//!         TensorOperand::identity(&descriptor_d, &modes_d),
//!         TensorOperand::identity(&descriptor_d, &modes_d),
//!         ComputeDescriptor::f32(),
//!     )?;
//!     let preference = PlanPreference::create_default(&context)?;
//!     let workspace_limit = Plan::estimate_workspace_size(
//!         &context,
//!         &contraction,
//!         &preference,
//!         WorkspacePreference::Default,
//!     )?;
//!     let plan = Plan::create(&context, &contraction, &preference, workspace_limit)?;
//!     let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, plan.workspace_size())?;
//!
//!     let a = DeviceMemory::from_slice(&cuda, &[1.0_f32; 8])?;
//!     let b = DeviceMemory::from_slice(&cuda, &[0.5_f32; 12])?;
//!     let c = DeviceMemory::<f32>::zeroed(&cuda, 6)?;
//!     let mut d = DeviceMemory::<f32>::zeroed(&cuda, 6)?;
//!
//!     // D = 1 * A * B + 0 * C
//!     plan.contract(
//!         &1.0_f32,
//!         &a,
//!         &b,
//!         &0.0_f32,
//!         &c,
//!         &mut d,
//!         Some(workspace.output()),
//!         &stream,
//!     )?;
//!
//!     assert_eq!(d.copy_to_host_vec()?, [2.0; 6]);
//!     Ok(())
//! }
//! ```
//!
//! An operation is described by [`TensorDescriptor`](crate::tensor::TensorDescriptor)s and an [`OperationDescriptor`](crate::operation::OperationDescriptor), planned into a [`Plan`](crate::plan::Plan), and run through methods on the plan.
//!
//! [`Plan::contract`](crate::plan::Plan::contract) checks every operand against the plan, rejects stream capture, and waits for the stream before it returns.
//! Each operation also has an `unsafe` `*_async` form that runs the same checks and only enqueues the work; its caller keeps the plan, operands, workspace, and scalars alive until the work finishes.
//!
//! # Features
//!
//! - `mg`: the `mg` module, for cuTENSORMg, which splits tensors across several GPUs in one process.
//! - `mp`: the `mp` module, for cuTENSORMp, which distributes tensors across processes and communicates through `singe-nccl`.
//!
//! # Raw execution
//!
//! Each operation's `unsafe` `*_async` form checks its operands' contexts, data types, and storage spans against the plan, orders the stream after its operands' producers, enqueues the work, and returns without waiting.
//! It is allowed during stream capture.
//! The caller must:
//!
//! - keep the plan, every operand, the workspace, and the host scalars and pointer tables alive until the work completes, and through every replay of a captured graph;
//! - initialize inputs and accumulators before the work reads them;
//! - give the work exclusive access to outputs and the workspace until it completes;
//! - for block-sparse, multi-GPU, and multi-process operations, order producers on other devices or streams before the work, since only producers in the stream's own context are joined.
//!
//! The work initializes only the output elements a descriptor reaches, so padding and stride gaps in uninitialized storage stay uninitialized unless the operation writes them.
//! The safe form of the same name runs the same checks, rejects capture, and waits for every stream it used; the dense single-GPU operations also have a `*_scoped` form that enqueues on a [`StreamScope`](singe_cuda::stream::StreamScope).

#[cfg(not(feature = "cutensor_2_8"))]
compile_error!("singe-cutensor requires a version feature, such as `cutensor_2_8`");

/// Single-GPU library context ownership and native plan or kernel caches.
pub mod context;

/// Native statuses and typed wrapper validation errors.
pub mod error;

pub mod execution;
pub mod logger;

/// Tensor operation descriptors, signatures, and typed scalar parameters.
pub mod operation;

/// Kernel-selection preferences, workspace estimates, and native execution plans.
pub mod plan;

/// Dense and block-sparse tensor layout descriptors.
pub mod tensor;

/// Typed native algorithms, operators, attributes, logging selectors, and mode labels.
pub mod types;

#[cfg(feature = "mg")]
pub mod mg;

#[cfg(feature = "mp")]
pub mod mp;

#[cfg(feature = "mg")]
pub(crate) mod streams;

pub(crate) mod utility;

#[cfg(feature = "testing")]
/// Serialized GPU test setup for cuTENSOR.
pub mod testing;

use singe_core::PackedVersion;
pub use singe_core::{CudaVersion, LibraryVersion};
pub use singe_cutensor_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the loaded cuTENSOR library version.
///
/// # Errors
///
/// Returns an error if cuTENSOR reports version zero or a version that does not fit `u32`.
pub fn version() -> Result<LibraryVersion> {
    // SAFETY: `cutensorGetVersion` takes no arguments.
    let version = nonzero_version("library", unsafe { sys::cutensorGetVersion() })?;
    LibraryVersion::from_packed(version, PackedVersion::Major10000Minor100)
        .ok_or_else(|| Error::out_of_range("cutensor version"))
}

/// Returns the CUDA runtime version that cuTENSOR was compiled against.
///
/// Can be compared against the CUDA runtime version reported by the CUDA wrapper crate.
///
/// # Errors
///
/// Returns an error if cuTENSOR reports CUDA runtime version zero or a version that does not fit `u32`.
pub fn cudart_version() -> Result<CudaVersion> {
    // SAFETY: `cutensorGetCudartVersion` takes no arguments.
    let version = nonzero_version("cuda runtime", unsafe { sys::cutensorGetCudartVersion() })?;
    CudaVersion::from_raw(version).ok_or_else(|| Error::out_of_range("cutensor cudart version"))
}

fn nonzero_version(name: &'static str, version: u64) -> Result<u64> {
    if version == 0 {
        return Err(Error::InvalidVersion { name });
    }
    Ok(version)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::{error::Result, testing::setup_context};

    #[test]
    fn test_it_works() -> Result<()> {
        let _test = setup_context()?;
        assert_ne!(version()?.major(), 0);
        assert_ne!(cudart_version()?.major(), 0);
        Ok(())
    }
}
