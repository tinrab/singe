//! Typed kernel signatures and context-bound launch preparation.
use std::{
    fmt,
    marker::PhantomData,
    mem::{align_of, size_of},
    sync::Arc,
};

use singe_cuda_sys::driver;
use thiserror::Error;

use crate::{
    context::Context,
    device::DeviceAttribute,
    dim::Dim3,
    error::{Error as CudaError, Result, Status},
    graph::{
        ExecutableGraph, Graph, GraphNode,
        owned::{Recordable, Recording},
    },
    green::SmResource,
    module::{KernelFunction, KernelParameters, LaunchAttribute, LaunchConfig, LaunchGeometry},
    stream::{Stream, StreamCaptureScope},
    submission::{DeviceOperation, Execution, Operation, Resources},
    try_ffi,
    types::{FunctionAttribute, SharedMemoryMode},
    utility::{to_i32, to_usize},
    view::NoUninitDeviceRepr,
};

/// Access to the storage addressed by a kernel parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterAccess {
    /// A copied value, with no implied device memory access.
    Value,
    /// Reads initialized elements.
    Read,
    /// Writes elements without reading their previous values.
    Write,
    /// Reads and writes initialized elements.
    ReadWrite,
    /// A pointer whose access contract is supplied by the caller.
    OpaquePointer,
}

/// Host representation of one CUDA parameter, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterLayout {
    size: usize,
    alignment: usize,
    access: ParameterAccess,
}

impl ParameterLayout {
    /// Describes a by-value parameter with initialized representation bytes.
    #[must_use]
    pub const fn value<T: NoUninitDeviceRepr>() -> Self {
        let () = T::LAYOUT;
        Self {
            size: size_of::<T>(),
            alignment: align_of::<T>(),
            access: ParameterAccess::Value,
        }
    }

    /// Describes a CUDA device pointer and its declared memory access.
    #[must_use]
    pub const fn pointer(access: ParameterAccess) -> Self {
        Self {
            size: size_of::<*const ()>(),
            alignment: align_of::<*const ()>(),
            access,
        }
    }

    /// Returns the parameter representation size in bytes.
    #[must_use]
    pub const fn size(self) -> usize {
        self.size
    }

    /// Returns the parameter representation alignment in bytes.
    #[must_use]
    pub const fn alignment(self) -> usize {
        self.alignment
    }

    /// Returns the declared access to storage addressed by this parameter.
    #[must_use]
    pub const fn access(self) -> ParameterAccess {
        self.access
    }
}

/// Number of dimensions used by the kernel's indexing scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LaunchRank {
    /// Uses only the x axis; grid and block y and z dimensions must all be one.
    One,
    /// Uses the x and y axes; grid and block z dimensions must be one.
    Two,
    /// Permits all three grid and block axes.
    #[default]
    Three,
}

/// Integer representation used for thread coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateWidth {
    /// Nonnegative coordinates representable by a unsigned 32-bit integer.
    U32,
    /// Nonnegative coordinates representable by a signed 32-bit integer.
    I32,
    /// Nonnegative coordinates representable by a unsigned 64-bit integer.
    U64,
    /// Nonnegative coordinates representable by a signed 64-bit integer.
    I64,
}

impl CoordinateWidth {
    const fn max(self) -> u128 {
        match self {
            Self::U32 => u32::MAX as u128,
            Self::I32 => i32::MAX as u128,
            Self::U64 => u64::MAX as u128,
            Self::I64 => i64::MAX as u128,
        }
    }
}

/// Block dimensions accepted by the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockRequirement {
    /// Adds no block-shape restriction beyond CUDA function and device limits.
    #[default]
    Any,
    /// Requires these exact thread dimensions for each block.
    Exact(Dim3),
    /// The total number of threads must be a nonzero multiple of this value.
    MultipleOf(u32),
    /// Maximum total number of threads in a block.
    MaxThreads(u32),
}

/// Dynamic shared memory required by each block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SharedMemoryRequirement {
    /// Adds no dynamic shared-memory requirement beyond CUDA function and device limits.
    #[default]
    Any,
    /// Requires exactly this many dynamic shared-memory bytes per block.
    Exact(usize),
    /// Requires at least this many dynamic shared-memory bytes per block.
    AtLeast(usize),
    /// Inclusive kernel-specific bounds on dynamic shared memory.
    Range {
        /// Minimum permitted dynamic shared-memory bytes per block, inclusive.
        min: usize,
        /// Maximum permitted dynamic shared-memory bytes per block, inclusive.
        max: usize,
    },
}

/// Kernel requirements checked against a configuration and its actual context.
///
/// Coordinate requirements check the launched extent, including padded threads.
/// Leave them unspecified only when the kernel has no corresponding restriction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchRequirements {
    /// Axes used by the kernel's indexing scheme. Unused grid and block axes must equal one.
    pub rank: LaunchRank,
    /// Optional integer width bounding the largest launched coordinate on every axis, including padded threads.
    pub axis_coordinates: Option<CoordinateWidth>,
    /// Optional integer width bounding the largest flattened coordinate across the complete launched grid, including padded threads.
    pub linear_coordinates: Option<CoordinateWidth>,
    /// Kernel-specific constraints on the block shape or thread count.
    pub block: BlockRequirement,
    /// Kernel-specific dynamic shared-memory byte requirements per block.
    pub shared_memory: SharedMemoryRequirement,
    /// Minimum device compute capability as `(major, minor)`, or `None` for no additional minimum.
    pub min_compute_capability: Option<(u32, u32)>,
    /// Required cluster dimensions in blocks, or `None` for no signature-specific cluster shape. Native kernel requirements still apply.
    pub cluster: Option<Dim3>,
    /// Whether the launch must explicitly enable cooperative execution. A false value also permits cooperative launches when the device supports them.
    pub cooperative: bool,
}

impl Default for LaunchRequirements {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl LaunchRequirements {
    /// No restrictions beyond the CUDA function and device limits.
    pub const DEFAULT: Self = Self {
        rank: LaunchRank::Three,
        axis_coordinates: None,
        linear_coordinates: None,
        block: BlockRequirement::Any,
        shared_memory: SharedMemoryRequirement::Any,
        min_compute_capability: None,
        cluster: None,
        cooperative: false,
    };
}

/// A kernel ABI and its launch requirements.
///
/// # Safety
///
/// `encode` must encode exactly `PARAMETERS`, in order, using the declared CUDA
/// representation and access modes. `raw_arguments` must preserve that order and
/// check every typed view's actual context. By-value representations must contain no
/// uninitialized bytes. The requirements must describe the kernel being bound.
/// This contract does not certify argument-dependent memory bounds or output coverage.
pub unsafe trait KernelSignature: 'static {
    /// Typed values and borrowed device views in declaration order.
    type Arguments<'a>;

    /// CUDA parameter values, including raw device addresses, in declaration order.
    type RawArguments: Copy + 'static;

    /// Host ABI layouts of the CUDA parameters, in declaration order.
    const PARAMETERS: &'static [ParameterLayout];
    /// Indexing and execution requirements checked when binding a launch configuration.
    const REQUIREMENTS: LaunchRequirements;

    /// Checks device-view contexts and converts typed arguments into CUDA parameter values.
    ///
    /// # Errors
    ///
    /// Returns an error if a typed device view belongs to another context or the implementation cannot convert the supplied values to its raw argument representation. Implementations may impose additional argument-specific checks.
    fn raw_arguments(
        arguments: Self::Arguments<'_>,
        context: &Arc<Context>,
    ) -> Result<Self::RawArguments>;

    /// Copies raw argument values into storage valid for the duration of `f`.
    fn encode<R>(arguments: Self::RawArguments, f: impl FnOnce(KernelParameters<'_>) -> R) -> R;
}

/// Failure to match a kernel's ABI, indexing or execution requirements.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LaunchContractError {
    /// The native function parameter count differs from the signature.
    #[error("kernel parameter count mismatch (expected {expected}, got {actual})")]
    ParameterCount {
        /// Number of parameters declared by the signature.
        expected: usize,
        /// Number of parameters reported by CUDA.
        actual: usize,
    },
    /// A native parameter offset or size differs from the signature's host ABI layout.
    #[error(
        "kernel parameter {index} layout mismatch (expected offset {expected_offset}, size {expected_size}; got offset {actual_offset}, size {actual_size})"
    )]
    ParameterLayout {
        /// Zero-based parameter index.
        index: usize,
        /// Byte offset computed from the signature's preceding parameter sizes and alignments.
        expected_offset: usize,
        /// Parameter size in bytes declared by the signature.
        expected_size: usize,
        /// Parameter byte offset reported by CUDA.
        actual_offset: usize,
        /// Parameter size in bytes reported by CUDA.
        actual_size: usize,
    },
    /// Nonunit dimensions occur on axes outside the kernel's indexing rank.
    #[error("launch dimensions exceed the kernel's {rank:?} dimensional indexing")]
    Rank {
        /// Indexing rank required by the kernel.
        rank: LaunchRank,
    },
    /// The configured block violates a kernel-specific block requirement.
    #[error("block dimensions {actual:?} do not satisfy {required:?}")]
    Block {
        /// Kernel-specific block requirement.
        required: BlockRequirement,
        /// Configured block dimensions in threads.
        actual: Dim3,
    },
    /// A requested launch quantity exceeds its device, function, or coordinate limit.
    #[error("`{name}` exceeds its limit (requested {requested}, limit {limit})")]
    Limit {
        /// Name of the checked launch quantity.
        name: &'static str,
        /// Requested quantity in the units identified by `name`.
        requested: u128,
        /// Maximum permitted quantity in the same units as `requested`.
        limit: u128,
    },
    /// Dynamic shared memory violates the kernel's byte requirement.
    #[error("dynamic shared memory {actual} does not satisfy {required:?}")]
    SharedMemory {
        /// Kernel-specific dynamic shared-memory requirement.
        required: SharedMemoryRequirement,
        /// Configured dynamic shared-memory bytes per block.
        actual: usize,
    },
    /// The device's compute capability is below the kernel's required minimum.
    #[error("compute capability {actual:?} is below {required:?}")]
    Architecture {
        /// Minimum compute capability as `(major, minor)`.
        required: (u32, u32),
        /// Device compute capability as `(major, minor)`.
        actual: (u32, u32),
    },
    /// Configured cluster dimensions disagree with the signature or native kernel requirement.
    #[error("cluster dimensions {actual:?} do not satisfy {required:?}")]
    Cluster {
        /// Required cluster dimensions in blocks, when present.
        required: Option<Dim3>,
        /// Effective launch cluster dimensions in blocks, when present.
        actual: Option<Dim3>,
    },
    /// The native function requires an explicit cluster size, but none is available.
    #[error("kernel requires explicit cluster dimensions")]
    ClusterRequired,
    /// A cluster launch was requested on a device without cluster-launch support.
    #[error("device does not support cluster launches")]
    ClusterUnsupported,
    /// The cluster dimensions are zero or do not divide the grid dimensions.
    #[error("grid dimensions {grid:?} are incompatible with cluster dimensions {cluster:?}")]
    ClusterGrid {
        /// Configured grid dimensions in blocks.
        grid: Dim3,
        /// Effective cluster dimensions in blocks.
        cluster: Dim3,
    },
    /// A required cooperative attribute is absent, or the device does not support the requested cooperative launch.
    #[error("cooperative launch is required or unsupported on this device")]
    Cooperative,
    /// An explicit shared-memory mode was requested on a driver that cannot query or support shared-memory modes.
    #[error("shared-memory mode {mode:?} is unsupported by the current driver")]
    SharedMemoryModeUnsupported {
        /// Requested per-launch shared-memory mode.
        mode: SharedMemoryMode,
    },
    /// The product of grid and block volumes cannot be represented in 128 bits.
    #[error("linear launch coordinates exceed 128 bits")]
    CoordinateOverflow,
    /// Computing a parameter offset or end overflows the host address space.
    #[error("kernel parameter layout overflows the host address space")]
    ParameterLayoutOverflow,
}

/// A typed kernel and checked configuration retaining the native code owner.
///
/// Preparation checks the signature and launch requirements. It does not launch
/// work, initialize output storage, or prove argument-dependent accesses valid.
/// Immediate submissions, capture and graph-node creation share this configuration.
/// Executable-node updates reuse its geometry and preserve the executable's existing attributes.
pub struct PreparedLaunch<K: KernelSignature> {
    function: KernelFunction,
    config: LaunchConfig,
    _signature: PhantomData<fn() -> K>,
}

impl<K: KernelSignature> fmt::Debug for PreparedLaunch<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedLaunch")
            .field("function", &self.function)
            .field("config", &self.config)
            .finish()
    }
}

impl<K: KernelSignature> Clone for PreparedLaunch<K> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            config: self.config.clone(),
            _signature: PhantomData,
        }
    }
}

impl<K: KernelSignature> PreparedLaunch<K> {
    /// Binds this signature to a kernel and checks its ABI and launch requirements.
    ///
    /// # Safety
    /// `K`'s access modes and launch requirements must describe this function.
    /// Parameter count and representation mismatches are rejected.
    /// CUDA metadata does not verify struct field layouts or semantic identity.
    /// Explicit struct mappings must match the device field layout.
    ///
    /// # Errors
    ///
    /// Returns a [`LaunchContractError`] for parameter count or layout mismatches, layout overflow, incompatible launch requirements, or exceeded function or device limits. Also propagates context binding, native parameter or attribute query, occupancy, resource query, and integer conversion errors. No work is launched on success.
    pub unsafe fn create(function: &KernelFunction, config: LaunchConfig) -> Result<Self> {
        validate_parameters::<K>(function)?;
        validate_requirements(function, &config, K::REQUIREMENTS)?;
        Ok(Self {
            function: function.clone(),
            config,
            _signature: PhantomData,
        })
    }

    /// Returns the retained context-specific kernel function.
    #[must_use]
    pub fn function(&self) -> &KernelFunction {
        &self.function
    }

    /// Returns the checked launch configuration.
    #[must_use]
    pub fn config(&self) -> &LaunchConfig {
        &self.config
    }

    /// Returns the context in which the function's module was loaded.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.function.module().context()
    }

    /// Checks that `stream` belongs to the prepared kernel's context.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::StreamContextMismatch` if `stream` belongs to a different context.
    pub fn validate_stream(&self, stream: &Stream) -> Result<()> {
        if stream.context() != self.context() {
            return Err(CudaError::StreamContextMismatch);
        }
        Ok(())
    }

    /// Submits typed arguments on the context's default stream.
    ///
    /// # Safety
    /// Values and views must satisfy the kernel's bounds, initialization and semantic requirements.
    /// Preserve storage lifetimes and exclusive writable access until execution completes, including every graph replay.
    /// Keep this prepared launch or another owner of its kernel code alive through completion.
    ///
    /// # Errors
    ///
    /// Propagates typed argument conversion and view-context errors from [`KernelSignature::raw_arguments`], followed by errors from [`Self::launch_raw`].
    pub unsafe fn launch(&self, arguments: K::Arguments<'_>) -> Result<()> {
        let arguments = K::raw_arguments(arguments, self.context())?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.launch_raw(arguments) }
    }

    /// Submits typed arguments on an explicit stream in this kernel's context.
    ///
    /// # Safety
    /// Values and views must satisfy the kernel's bounds, initialization and semantic requirements.
    /// Preserve storage lifetimes and exclusive writable access until execution completes, including every graph replay.
    /// Keep this prepared launch or another owner of its kernel code alive through completion.
    ///
    /// # Errors
    ///
    /// Propagates typed argument conversion and view-context errors from [`KernelSignature::raw_arguments`], followed by errors from [`Self::launch_on_raw`].
    pub unsafe fn launch_on(&self, arguments: K::Arguments<'_>, stream: &Stream) -> Result<()> {
        let arguments = K::raw_arguments(arguments, self.context())?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.launch_on_raw(arguments, stream) }
    }

    /// Records typed arguments and retains the kernel's code for replay.
    ///
    /// # Safety
    /// Values and views must satisfy the kernel's bounds, initialization and semantic requirements for every replay.
    /// Preserve storage lifetimes and exclusive writable access until all graph executions complete.
    /// Keep a graph, executable or another code owner alive until queued executions complete.
    ///
    /// # Errors
    ///
    /// Propagates typed argument conversion and view-context errors from [`KernelSignature::raw_arguments`], followed by errors from [`Self::record_raw`].
    pub unsafe fn record(
        &self,
        arguments: K::Arguments<'_>,
        scope: &StreamCaptureScope<'_>,
    ) -> Result<()> {
        let arguments = K::raw_arguments(arguments, self.context())?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.record_raw(arguments, scope) }
    }

    /// Adds a graph node with typed arguments and retains the kernel's code.
    ///
    /// # Safety
    /// Values and views must satisfy the kernel's bounds, initialization and semantic requirements for every replay.
    /// Preserve storage lifetimes and exclusive writable access until all graph executions complete.
    /// Keep a graph, executable or another code owner alive until queued executions complete.
    ///
    /// # Errors
    ///
    /// Propagates typed argument conversion and view-context errors from [`KernelSignature::raw_arguments`], followed by errors from [`Self::add_to_graph_raw`].
    pub unsafe fn add_to_graph(
        &self,
        arguments: K::Arguments<'_>,
        graph: &mut Graph,
        dependencies: &[GraphNode],
    ) -> Result<GraphNode> {
        let arguments = K::raw_arguments(arguments, self.context())?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.add_to_graph_raw(arguments, graph, dependencies) }
    }

    /// Updates an executable node with this launch's geometry and typed arguments, retaining the kernel's code.
    ///
    /// # Safety
    /// Values and views must satisfy the kernel's bounds, initialization and semantic requirements for all future executions.
    /// Preserve storage lifetimes and exclusive writable access until all executions using them complete.
    /// Keep the executable or another code owner alive until queued executions complete.
    /// Only this launch's geometry is applied; executable attributes remain unchanged and must satisfy `K`.
    ///
    /// # Errors
    ///
    /// Propagates typed argument conversion and view-context errors from [`KernelSignature::raw_arguments`], followed by errors from [`Self::set_graph_node_params_raw`].
    pub unsafe fn set_graph_node_params(
        &self,
        arguments: K::Arguments<'_>,
        executable: &mut ExecutableGraph,
        node: &GraphNode,
    ) -> Result<()> {
        let arguments = K::raw_arguments(arguments, self.context())?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.set_graph_node_params_raw(arguments, executable, node) }
    }

    /// Submits work on the context's default stream.
    ///
    /// # Safety
    /// Arguments must satisfy the kernel's value, range, initialization and aliasing requirements.
    /// Keep pointed-to storage and this kernel's code alive until completion.
    /// During capture, preserve those guarantees for every replay.
    ///
    /// # Errors
    ///
    /// Propagates context binding and native kernel launch errors from [`KernelFunction::launch`], including invalid images, rejected configuration or arguments, and prior asynchronous CUDA errors.
    pub unsafe fn launch_raw(&self, arguments: K::RawArguments) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        K::encode(arguments, |params| unsafe {
            self.function.launch(&self.config, params)
        })
    }

    /// Submits work on an explicit stream.
    ///
    /// # Safety
    /// Arguments must satisfy the kernel's value, range, initialization and aliasing requirements.
    /// Keep pointed-to storage and this kernel's code alive until completion.
    /// During capture, preserve those guarantees for every replay.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::StreamContextMismatch` if `stream` belongs to another context. Also propagates context binding and native kernel launch errors from [`KernelFunction::launch_on`].
    pub unsafe fn launch_on_raw(&self, arguments: K::RawArguments, stream: &Stream) -> Result<()> {
        self.validate_stream(stream)?;
        // SAFETY: the caller upholds this function's safety contract.
        K::encode(arguments, |params| unsafe {
            self.function.launch_on(&self.config, params, stream)
        })
    }

    /// Records this prepared launch and retains its code in the captured graph.
    ///
    /// # Safety
    /// Arguments must satisfy the kernel's semantic and access requirements for every replay.
    /// Keep pointed-to storage alive and order conflicting accesses until all replays complete.
    /// Keep a graph, executable or another code owner alive until queued executions complete.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::StreamContextMismatch` if the capture stream belongs to another context. Also propagates context binding and native launch errors from [`Self::launch_on_raw`].
    pub unsafe fn record_raw(
        &self,
        arguments: K::RawArguments,
        scope: &StreamCaptureScope<'_>,
    ) -> Result<()> {
        self.validate_stream(scope.stream())?;
        scope.retain_module(self.function.module());
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.launch_on_raw(arguments, scope.stream()) }
    }

    /// Adds a node using the same signature and configuration as immediate launches.
    ///
    /// # Safety
    /// Arguments must satisfy the kernel's semantic and access requirements for every replay.
    /// Keep pointed-to storage alive until all graph executions using this node complete.
    /// Keep a graph, executable or another code owner alive until queued executions complete.
    ///
    /// # Errors
    ///
    /// Propagates graph-context, dependency ownership, context binding, and native node creation or launch-attribute errors from [`KernelFunction::add_to_graph`].
    pub unsafe fn add_to_graph_raw(
        &self,
        arguments: K::RawArguments,
        graph: &mut Graph,
        dependencies: &[GraphNode],
    ) -> Result<GraphNode> {
        // SAFETY: the caller upholds this function's safety contract.
        K::encode(arguments, |params| unsafe {
            self.function
                .add_to_graph(graph, dependencies, &self.config, params)
        })
    }

    /// Replaces an executable node's geometry and raw arguments, preserving its launch attributes.
    ///
    /// # Safety
    /// Arguments must satisfy the kernel's semantic and access requirements for all future executions.
    /// Keep pointed-to storage alive until all executions using these arguments complete.
    /// Keep the executable or another code owner alive until queued executions complete.
    /// Only this launch's geometry is applied; executable attributes remain unchanged and must satisfy `K`.
    ///
    /// # Errors
    ///
    /// Propagates executable-context, node ownership, context binding, and native update errors from [`KernelFunction::set_graph_node_params`].
    pub unsafe fn set_graph_node_params_raw(
        &self,
        arguments: K::RawArguments,
        executable: &mut ExecutableGraph,
        node: &GraphNode,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        K::encode(arguments, |params| unsafe {
            self.function
                .set_graph_node_params(executable, node, self.config.geometry(), params)
        })
    }
}

impl KernelFunction {
    /// Updates an executable node's geometry and typed arguments, retaining this kernel's code.
    ///
    /// Launch attributes remain unchanged.
    ///
    /// # Errors
    /// Returns an error for ABI, view-context or geometry mismatches, or if CUDA rejects the update.
    ///
    /// # Safety
    /// `K` must describe this function and the executable's existing attributes must satisfy it.
    /// Arguments must satisfy the kernel's bounds, initialization and aliasing requirements for every execution.
    /// Keep storage and the executable or another code owner alive until queued executions complete.
    pub unsafe fn set_typed_graph_node_params<K: KernelSignature>(
        &self,
        executable: &mut ExecutableGraph,
        node: &GraphNode,
        geometry: &LaunchGeometry,
        arguments: K::Arguments<'_>,
    ) -> Result<()> {
        let arguments = K::raw_arguments(arguments, self.module().context())?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.set_typed_graph_node_params_raw::<K>(executable, node, geometry, arguments) }
    }

    /// Updates an executable node using raw values in `K`'s declaration order.
    ///
    /// Launch attributes remain unchanged.
    ///
    /// # Errors
    /// Returns an error for ABI or geometry mismatches, or if CUDA rejects the update.
    ///
    /// # Safety
    /// `K` must describe this function and the executable's existing attributes must satisfy it.
    /// Values and pointers must satisfy the kernel's bounds, initialization and aliasing requirements for every execution.
    /// Keep storage and the executable or another code owner alive until queued executions complete.
    pub unsafe fn set_typed_graph_node_params_raw<K: KernelSignature>(
        &self,
        executable: &mut ExecutableGraph,
        node: &GraphNode,
        geometry: &LaunchGeometry,
        arguments: K::RawArguments,
    ) -> Result<()> {
        validate_parameters::<K>(self)?;
        validate_kernel_geometry(self, geometry, K::REQUIREMENTS)?;
        // SAFETY: the caller upholds this function's safety contract.
        K::encode(arguments, |params| unsafe {
            self.set_graph_node_params(executable, node, geometry, params)
        })
    }
}

fn validate_parameters<K: KernelSignature>(function: &KernelFunction) -> Result<()> {
    function.module().context().bind()?;
    let mut count = 0;
    // SAFETY: `cuFuncGetParamCount` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(driver::cuFuncGetParamCount(
            function.as_raw().as_raw(),
            &raw mut count
        ))?;
    }
    if to_usize(count, "kernel parameter count")? != K::PARAMETERS.len() {
        return Err(LaunchContractError::ParameterCount {
            expected: K::PARAMETERS.len(),
            actual: to_usize(count, "kernel parameter count")?,
        }
        .into());
    }
    let mut expected_offset = 0_usize;
    for (index, parameter) in K::PARAMETERS.iter().enumerate() {
        let padding = expected_offset.wrapping_neg() & (parameter.alignment - 1);
        expected_offset = expected_offset
            .checked_add(padding)
            .ok_or(LaunchContractError::ParameterLayoutOverflow)?;
        let mut offset = 0;
        let mut size = 0;
        // SAFETY: `cuFuncGetParamInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(driver::cuFuncGetParamInfo(
                function.as_raw().as_raw(),
                index as _,
                &raw mut offset,
                &raw mut size
            ))?;
        }
        if to_usize(offset, "kernel parameter offset")? != expected_offset
            || to_usize(size, "kernel parameter size")? != parameter.size
        {
            return Err(LaunchContractError::ParameterLayout {
                index,
                expected_offset,
                expected_size: parameter.size,
                actual_offset: to_usize(offset, "kernel parameter offset")?,
                actual_size: to_usize(size, "kernel parameter size")?,
            }
            .into());
        }
        expected_offset = expected_offset
            .checked_add(parameter.size)
            .ok_or(LaunchContractError::ParameterLayoutOverflow)?;
    }
    Ok(())
}

fn check_limit(name: &'static str, requested: u128, limit: u128) -> Result<()> {
    if requested > limit {
        return Err(LaunchContractError::Limit {
            name,
            requested,
            limit,
        }
        .into());
    }
    Ok(())
}

fn volume(dim: Dim3) -> u128 {
    u128::from(dim.x) * u128::from(dim.y) * u128::from(dim.z)
}

#[expect(
    clippy::too_many_lines,
    reason = "launch geometry is checked against the complete kernel and device contract"
)]
fn validate_geometry(
    function: &KernelFunction,
    geometry: &LaunchGeometry,
    required: LaunchRequirements,
) -> Result<()> {
    let grid = geometry.grid_dim();
    let block = geometry.block_dim();
    match required.rank {
        LaunchRank::One if grid.y != 1 || grid.z != 1 || block.y != 1 || block.z != 1 => {
            return Err(LaunchContractError::Rank {
                rank: required.rank,
            }
            .into());
        }
        LaunchRank::Two if grid.z != 1 || block.z != 1 => {
            return Err(LaunchContractError::Rank {
                rank: required.rank,
            }
            .into());
        }
        _ => {}
    }
    match required.block {
        BlockRequirement::MaxThreads(n) if volume(block) > u128::from(n) => {
            return Err(LaunchContractError::Block {
                required: required.block,
                actual: block,
            }
            .into());
        }
        BlockRequirement::Exact(dim) if dim != block => {
            return Err(LaunchContractError::Block {
                required: required.block,
                actual: block,
            }
            .into());
        }
        BlockRequirement::MultipleOf(n)
            if n == 0 || !volume(block).is_multiple_of(u128::from(n)) =>
        {
            return Err(LaunchContractError::Block {
                required: required.block,
                actual: block,
            }
            .into());
        }
        _ => {}
    }
    let context = function.module().context();
    let device = context.device();
    for (g, b, grid_attr, block_attr) in [
        (
            grid.x,
            block.x,
            DeviceAttribute::MaxGridDimX,
            DeviceAttribute::MaxBlockDimX,
        ),
        (
            grid.y,
            block.y,
            DeviceAttribute::MaxGridDimY,
            DeviceAttribute::MaxBlockDimY,
        ),
        (
            grid.z,
            block.z,
            DeviceAttribute::MaxGridDimZ,
            DeviceAttribute::MaxBlockDimZ,
        ),
    ] {
        check_limit(
            "grid dimension",
            u128::from(g),
            device.attribute(grid_attr)? as u128,
        )?;
        check_limit(
            "block dimension",
            u128::from(b),
            device.attribute(block_attr)? as u128,
        )?;
        if let Some(width) = required.axis_coordinates {
            check_limit(
                "axis coordinate",
                u128::from(g) * u128::from(b) - 1,
                width.max(),
            )?;
        }
    }
    if let Some(width) = required.linear_coordinates {
        check_limit(
            "linear coordinate",
            volume(grid)
                .checked_mul(volume(block))
                .ok_or(LaunchContractError::CoordinateOverflow)?
                - 1,
            width.max(),
        )?;
    }
    let threads = volume(block);
    check_limit(
        "threads per block",
        threads,
        device.attribute(DeviceAttribute::MaxThreadsPerBlock)? as u128,
    )?;
    check_limit(
        "kernel threads per block",
        threads,
        function.attribute(FunctionAttribute::MaxThreadsPerBlock)? as u128,
    )?;
    Ok(())
}

fn validate_kernel_geometry(
    function: &KernelFunction,
    geometry: &LaunchGeometry,
    required: LaunchRequirements,
) -> Result<()> {
    validate_geometry(function, geometry, required)?;
    let device = function.module().context().device();
    let shared = geometry.shared_memory_bytes();
    match required.shared_memory {
        SharedMemoryRequirement::Range { min, max } if shared < min || shared > max => {
            return Err(LaunchContractError::SharedMemory {
                required: required.shared_memory,
                actual: shared,
            }
            .into());
        }
        SharedMemoryRequirement::Exact(n) if shared != n => {
            return Err(LaunchContractError::SharedMemory {
                required: required.shared_memory,
                actual: shared,
            }
            .into());
        }
        SharedMemoryRequirement::AtLeast(n) if shared < n => {
            return Err(LaunchContractError::SharedMemory {
                required: required.shared_memory,
                actual: shared,
            }
            .into());
        }
        _ => {}
    }
    if let Some(minimum) = required.min_compute_capability {
        let actual = (
            device.attribute(DeviceAttribute::ComputeCapabilityMajor)? as u32,
            device.attribute(DeviceAttribute::ComputeCapabilityMinor)? as u32,
        );
        if actual < minimum {
            return Err(LaunchContractError::Architecture {
                required: minimum,
                actual,
            }
            .into());
        }
    }
    Ok(())
}

fn validate_requirements(
    function: &KernelFunction,
    config: &LaunchConfig,
    required: LaunchRequirements,
) -> Result<()> {
    validate_kernel_geometry(function, config.geometry(), required)?;
    let context = function.module().context();
    let device = context.device();
    let grid = config.grid_dim();
    let threads = volume(config.block_dim());
    let shared = config.shared_memory_bytes();
    check_limit(
        "dynamic shared memory",
        shared as u128,
        dynamic_shared_memory_limit(function, config)?,
    )?;
    let mut cluster = None;
    let mut cooperative = false;
    for attribute in config.attributes() {
        match attribute {
            LaunchAttribute::ClusterDimension(dim) => cluster = Some(*dim),
            LaunchAttribute::Cooperative(value) => cooperative = *value,
            _ => {}
        }
    }
    let native_cluster = Dim3::new(
        function.attribute(FunctionAttribute::RequiredClusterWidth)? as u32,
        function.attribute(FunctionAttribute::RequiredClusterHeight)? as u32,
        function.attribute(FunctionAttribute::RequiredClusterDepth)? as u32,
    );
    let native_cluster = (native_cluster.x != 0 || native_cluster.y != 0 || native_cluster.z != 0)
        .then_some(native_cluster);
    if let (Some(actual), Some(native)) = (cluster, native_cluster)
        && actual != native
    {
        return Err(LaunchContractError::Cluster {
            required: Some(native),
            actual: Some(actual),
        }
        .into());
    }
    let cluster = cluster.or(native_cluster);
    if required.cluster.is_some() && cluster != required.cluster {
        return Err(LaunchContractError::Cluster {
            required: required.cluster,
            actual: cluster,
        }
        .into());
    }
    if function.attribute(FunctionAttribute::ClusterSizeMustBeSet)? != 0 && cluster.is_none() {
        return Err(LaunchContractError::ClusterRequired.into());
    }
    if let Some(dim) = cluster {
        if device.attribute(DeviceAttribute::ClusterLaunch)? == 0 {
            return Err(LaunchContractError::ClusterUnsupported.into());
        }
        if dim.x == 0
            || dim.y == 0
            || dim.z == 0
            || !grid.x.is_multiple_of(dim.x)
            || !grid.y.is_multiple_of(dim.y)
            || !grid.z.is_multiple_of(dim.z)
        {
            return Err(LaunchContractError::ClusterGrid { grid, cluster: dim }.into());
        }
        let limit = function.occupancy_max_potential_cluster_size(config)?;
        check_limit("cluster blocks", volume(dim), limit as u128)?;
    }
    if required.cooperative && !cooperative
        || cooperative && device.attribute(DeviceAttribute::CooperativeLaunch)? == 0
    {
        return Err(LaunchContractError::Cooperative.into());
    }
    if let (true, Some(cluster)) = (cooperative, cluster) {
        let active = function.occupancy_max_active_clusters(config)?;
        check_limit(
            "cooperative grid clusters",
            volume(grid) / volume(cluster),
            active as u128,
        )?;
    } else if cooperative {
        let active = function.occupancy_max_active_blocks_per_multiprocessor(
            to_i32(threads, "block thread count")?,
            shared,
        )?;
        let sms = SmResource::for_context(context)?.sm_count();
        check_limit(
            "cooperative grid blocks",
            volume(grid),
            active as u128 * u128::from(sms),
        )?;
    }
    Ok(())
}

fn dynamic_shared_memory_limit(function: &KernelFunction, config: &LaunchConfig) -> Result<u128> {
    let requested = config.attributes().iter().find_map(|attribute| {
        if let LaunchAttribute::SharedMemoryMode(mode) = attribute {
            Some(*mode)
        } else {
            None
        }
    });
    let function_mode = match function.shared_memory_mode() {
        Ok(mode) => mode,
        // A live function on a driver without shared-memory modes rejects this attribute.
        // Ordinary launches still use the function's dynamic-memory limit.
        Err(CudaError::Cuda {
            code: Status::InvalidValue | Status::NotSupported,
            ..
        }) => match requested {
            None => SharedMemoryMode::Default,
            Some(mode) => {
                return Err(LaunchContractError::SharedMemoryModeUnsupported { mode }.into());
            }
        },
        Err(error) => return Err(error),
    };
    let mode = requested.unwrap_or(function_mode);
    let device = function.module().context().device();
    let capacity = match mode {
        SharedMemoryMode::Default => {
            return Ok(function.attribute(FunctionAttribute::MaxDynamicSharedSizeBytes)? as u128);
        }
        SharedMemoryMode::RequirePortable => {
            device.attribute(DeviceAttribute::MaxSharedMemoryPerBlock)? as u128
        }
        SharedMemoryMode::AllowNonPortable => {
            device.attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)? as u128
        }
        SharedMemoryMode::AllowOversizedSharedMemory
        | SharedMemoryMode::PreferOversizedSharedMemory => {
            let optin = device.attribute(DeviceAttribute::MaxSharedMemoryPerBlockOptin)?;
            let oversized = device.attribute(DeviceAttribute::MaxOversizedSharedMemoryPerBlock)?;
            optin.max(oversized) as u128
        }
    };
    let static_bytes = function.attribute(FunctionAttribute::SharedSizeBytes)? as u128;
    check_limit("static shared memory", static_bytes, capacity)?;
    Ok(capacity - static_bytes)
}

/// Prepared owned kernel work with an explicit resource and access contract.
#[derive(Debug)]
pub struct KernelOperation<K: KernelSignature, T> {
    launch: PreparedLaunch<K>,
    arguments: K::RawArguments,
    resources: Resources,
    output: T,
}

impl<K: KernelSignature> PreparedLaunch<K> {
    /// Binds owned resources to this prepared launch without enqueueing work.
    ///
    /// # Safety
    /// Arguments must satisfy the kernel's bounds, value, aliasing and output coverage contracts.
    /// Retain every pointed-to allocation and declare all accesses in `resources`.
    /// `output` must preserve pending initialization until successful completion, including capture.
    pub unsafe fn into_operation<T: 'static>(
        self,
        arguments: K::RawArguments,
        mut resources: Resources,
        output: T,
    ) -> KernelOperation<K, T> {
        resources.retain(Arc::new(self.function.module().clone()));
        KernelOperation {
            launch: self,
            arguments,
            resources,
            output,
        }
    }
}

unsafe impl<K: KernelSignature, T: 'static> Operation for KernelOperation<K, T> {
    type Pending = Self::Output;

    type Output = T;

    fn enqueue(self, execution: &mut Execution) -> Result<T> {
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(self.resources, |stream| {
                self.launch.launch_on_raw(self.arguments, stream)
            })?;
        }
        Ok(self.output)
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: enqueue returns pending device metadata and resolution is the identity function.
unsafe impl<K: KernelSignature, T: 'static> DeviceOperation for KernelOperation<K, T> {}

unsafe impl<K: KernelSignature, T: 'static> Recordable for KernelOperation<K, T> {
    type Output = T;

    fn record(self, recording: &mut Recording) -> Result<T> {
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            recording.enqueue(self.resources, |stream| {
                self.launch.launch_on_raw(self.arguments, stream)
            })?;
        }
        Ok(self.output)
    }
}
