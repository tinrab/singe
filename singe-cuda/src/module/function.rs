use std::{
    fmt::{self, Display, Formatter},
    ptr,
};

use bitflags::parser;
use singe_cuda_sys::driver;

use super::{
    KernelLaunchArgs, LaunchConfig, LaunchGeometry, Module,
    config::validate_dynamic_shared_memory_bytes,
};
use crate::{
    error::{Error, Result},
    graph::{ExecutableGraph, Graph, GraphNode},
    green::GreenStream,
    kernel::{self, ModuleKernelHandle},
    stream::Stream,
    try_ffi,
    types::{DeviceFunction, FunctionAttribute, SharedMemoryCarveout, SharedMemoryMode},
    utility::{to_i32, to_usize},
};

bitflags::bitflags! {
    /// Occupancy calculator flag
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct OccupancyFlags: u32 {
        /// Default behavior.
        const DEFAULT = driver::CUoccupancy_flags::CU_OCCUPANCY_DEFAULT.0;
        /// Assume global caching is enabled and cannot be automatically turned off.
        const DISABLE_CACHING_OVERRIDE = driver::CUoccupancy_flags::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE.0;
    }
}

impl Display for OccupancyFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Function resource usage, compilation versions, and launch constraints queried from CUDA.
///
/// These values describe the function in its module context; device and compiler choices can affect them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionAttributes {
    /// Statically allocated shared memory per block in bytes, excluding dynamic launch-time shared memory.
    pub shared_size_bytes: usize,
    /// User-allocated constant memory required by the function in bytes.
    pub const_size_bytes: usize,
    /// Local memory used per thread in bytes.
    pub local_size_bytes: usize,
    /// Maximum number of threads per block permitted for this function on the device.
    pub max_threads_per_block: i32,
    /// Number of registers used per thread.
    pub num_regs: i32,
    /// PTX virtual architecture version, encoded as `major * 10 + minor`.
    /// May be zero for cubins compiled before CUDA 3.0.
    pub ptx_version: i32,
    /// Binary architecture version, encoded as `major * 10 + minor`.
    /// Legacy cubins without an encoded version report 10.
    pub binary_version: i32,
    /// Whether the function was compiled with the `-Xptxas --dlcm=ca` cache setting.
    pub cache_mode_ca: bool,
    /// Maximum dynamic shared memory permitted per block for this function, in bytes.
    pub max_dynamic_shared_size_bytes: i32,
    /// Preferred percentage of shared L1/shared-memory resources assigned to shared memory.
    /// This is a hint; CUDA may choose another ratio.
    pub preferred_shared_memory_carveout: i32,
    /// Whether launches must specify a valid cluster size.
    pub cluster_dim_must_be_set: bool,
    /// Required cluster width in blocks; zero means no required width is set.
    pub required_cluster_width: i32,
    /// Required cluster height in blocks; zero means no required height is set.
    pub required_cluster_height: i32,
    /// Required cluster depth in blocks; zero means no required depth is set.
    pub required_cluster_depth: i32,
    /// Native cluster scheduling policy preference value.
    pub cluster_scheduling_policy_preference: i32,
    /// Whether the function permits cluster sizes beyond the portable limit.
    pub non_portable_cluster_size_allowed: bool,
}

/// A block size selected to maximize potential occupancy and the minimum grid size needed to achieve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyMaxPotentialBlockSize {
    /// Minimum number of blocks in the grid needed to achieve the calculated maximum occupancy.
    pub min_grid_size: i32,
    /// Number of threads per block selected by the occupancy calculator.
    pub block_size: i32,
}

/// A context-specific CUDA kernel function that retains its module and loading context.
#[derive(Debug, Clone)]
pub struct KernelFunction {
    pub(crate) handle: DeviceFunction,
    pub(crate) module: Module,
}

// SAFETY: a CUDA function handle is usable from any thread while its module is loaded, which the retained `module` guarantees.
unsafe impl Send for KernelFunction {}

// SAFETY: `&self` methods only pass the immutable function handle to the driver, whose attribute and launch calls are thread-safe.
unsafe impl Sync for KernelFunction {}

impl KernelFunction {
    /// Adopts a function handle while retaining its module.
    ///
    /// # Safety
    ///
    /// `handle` must identify a live function in `module` and its actual context.
    #[must_use]
    pub unsafe fn from_raw(handle: DeviceFunction, module: &Module) -> Self {
        Self {
            handle,
            module: module.clone(),
        }
    }

    fn check_graph_context(&self, graph: &Graph) -> Result<()> {
        if graph.context() != &self.module.inner.context {
            return Err(Error::context_mismatch("graph"));
        }
        Ok(())
    }

    fn check_executable_graph_context(&self, executable: &ExecutableGraph) -> Result<()> {
        if executable.context() != &self.module.inner.context {
            return Err(Error::context_mismatch("executable graph"));
        }
        Ok(())
    }

    /// Invokes this kernel function on a grid of blocks using the given stream.
    /// Each block contains the threads specified by [`LaunchConfig::block_dim`].
    ///
    /// [`LaunchConfig::shared_memory_bytes`] sets the amount of dynamic shared memory available to each thread block.
    ///
    /// Kernel parameters are passed with [`KernelParameters`](super::KernelParameters) or tuples of shared or mutable references.
    ///
    /// Launching the kernel invalidates the persistent function state set through the following deprecated APIs: [`sys::cuFuncSetBlockShape`](singe_cuda_sys::driver::cuFuncSetBlockShape), [`sys::cuFuncSetSharedSize`](singe_cuda_sys::driver::cuFuncSetSharedSize), [`sys::cuParamSetSize`](singe_cuda_sys::driver::cuParamSetSize), [`sys::cuParamSeti`](singe_cuda_sys::driver::cuParamSeti), [`sys::cuParamSetf`](singe_cuda_sys::driver::cuParamSetf), [`sys::cuParamSetv`](singe_cuda_sys::driver::cuParamSetv).
    ///
    /// The kernel must either have been compiled with toolchain version 3.2 or later so that it contains kernel parameter information, or have no kernel parameters.
    /// If either of these conditions is not met, the launch returns [`crate::error::Status::InvalidImage`].
    ///
    /// # Safety
    ///
    /// Arguments must match the kernel ABI and satisfy its access and value requirements.
    /// Keep the function and all pointed-to storage alive until execution completes.
    /// Order conflicting accesses and preserve exclusive writable access during execution.
    /// Successful submission does not establish output initialization or completion.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Status::InvalidImage`] if the kernel parameter metadata requirements above are not met.
    /// Also returns an error if `stream` belongs to a different context, the module context cannot be bound, CUDA rejects the launch, or a previous asynchronous launch reports an error.
    pub unsafe fn launch_async<'a, P>(
        &self,
        config: &LaunchConfig,
        params: P,
        stream: &Stream,
    ) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        if stream.context() != &self.module.inner.context {
            return Err(Error::StreamContextMismatch);
        }

        self.module.inner.context.bind()?;
        // SAFETY: the function and stream share the context bound above, the encoded arguments live through the call, and the caller guarantees their ABI, storage lifetime, and access ordering.
        params.with_encoded_arguments(|mut arguments| unsafe {
            config.launch_raw(
                self.handle.as_raw(),
                stream.as_raw(),
                arguments.as_mut_ptr().cast(),
            )?;
            Ok(())
        })
    }

    /// Adds this kernel to `graph` as a kernel node.
    ///
    /// # Safety
    ///
    /// CUDA copies each kernel argument value during this call.
    /// Non-pointer argument values may be borrowed from stack or temporary storage that outlives this call.
    /// If an argument value is a pointer, CUDA stores only the pointer address.
    /// The caller must ensure every copied pointer value remains valid for every graph instantiation, update, and launch that can execute the created node.
    /// Mutable pointer arguments must remain exclusive for the work ordered by those launches.
    /// Arguments must match the function ABI and satisfy its value and access requirements.
    /// Keep the graph, executable or another code owner alive until queued executions complete.
    ///
    /// # Errors
    ///
    /// Returns `Error::ContextMismatch` if the graph is associated with another context.
    /// Also propagates context binding errors, dependency ownership errors, and CUDA failures when creating the kernel node or setting its launch attributes.
    pub unsafe fn add_to_graph<'a, P>(
        &self,
        graph: &mut Graph,
        dependencies: &[GraphNode],
        config: &LaunchConfig,
        params: P,
    ) -> Result<GraphNode>
    where
        P: KernelLaunchArgs<'a>,
    {
        self.check_graph_context(graph)?;
        self.module.inner.context.bind()?;
        // SAFETY: `self.handle` is a live function of the module retained below, and the caller guarantees the argument ABI, pointer validity, and exclusivity that `add_kernel_node` requires.
        let node = unsafe { graph.add_kernel_node(dependencies, self.handle, config, params) }?;
        graph.retain_module(&self.module)?;
        Ok(node)
    }

    /// Updates this kernel's parameters in an executable graph node.
    ///
    /// # Safety
    ///
    /// CUDA copies each kernel argument value during this call.
    /// Non-pointer argument values may be borrowed from stack or temporary storage that outlives this call.
    /// If an argument value is a pointer, CUDA stores only the pointer address.
    /// The caller must ensure every copied pointer value remains valid for every future launch that can execute `node`.
    /// Mutable pointer arguments must remain exclusive for the work ordered by those launches.
    /// Arguments must match the function ABI and satisfy its value and access requirements.
    /// Keep the graph, executable or another code owner alive until queued executions complete.
    /// Launch attributes are preserved; they must remain compatible with this kernel and geometry.
    ///
    /// # Errors
    ///
    /// Returns `Error::ContextMismatch` if the executable is associated with another context.
    /// Also propagates context binding errors, node ownership errors, and CUDA failures when updating the node.
    pub unsafe fn set_graph_node_params<'a, P>(
        &self,
        executable: &mut ExecutableGraph,
        node: &GraphNode,
        geometry: &LaunchGeometry,
        params: P,
    ) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        self.check_executable_graph_context(executable)?;
        self.module.inner.context.bind()?;
        // SAFETY: `self.handle` is a live function of the module retained below, and the caller guarantees the argument ABI, pointer validity, exclusivity, and attribute compatibility that `set_kernel_node_params` requires.
        unsafe { executable.set_kernel_node_params(node, self.handle, geometry, params) }?;
        executable.retain_module(&self.module);
        Ok(())
    }

    /// Returns the module retained by this function.
    #[must_use]
    pub const fn module(&self) -> &Module {
        &self.module
    }

    /// Returns the native function name, replacing invalid UTF-8 sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, CUDA cannot query the name, or CUDA returns a null name pointer.
    pub fn name(&self) -> Result<String> {
        // SAFETY: the kernel handle belongs to the module or library retained by `self`.
        unsafe {
            kernel::name::<ModuleKernelHandle>(
                self.module.inner.context.as_ref(),
                self.handle.as_raw(),
            )
        }
    }

    /// Returns one function attribute for this module context.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the requested function attribute.
    pub fn attribute(&self, attribute: FunctionAttribute) -> Result<i32> {
        // SAFETY: the kernel handle belongs to the module or library retained by `self`.
        unsafe {
            kernel::attribute::<ModuleKernelHandle>(
                self.module.inner.context.as_ref(),
                self.handle.as_raw(),
                attribute,
            )
        }
    }

    /// Sets a mutable function attribute.
    ///
    /// See [`FunctionAttribute`] for the supported attributes and their value constraints.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the selected attribute or value.
    pub fn set_attribute(&self, attribute: FunctionAttribute, value: i32) -> Result<()> {
        // SAFETY: the kernel handle belongs to the module or library retained by `self`.
        unsafe {
            kernel::set_attribute::<ModuleKernelHandle>(
                self.module.inner.context.as_ref(),
                self.handle.as_raw(),
                attribute,
                value,
            )
        }
    }

    /// Sets the maximum dynamic shared memory permitted per block, in bytes.
    ///
    /// The requested size plus static shared memory must fit the device's opt-in per-block limit.
    /// This attribute does not enable oversized shared memory; use [`Self::set_shared_memory_mode`] for that.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the selected attribute or value.
    pub fn set_max_dynamic_shared_memory_bytes(&self, bytes: i32) -> Result<()> {
        self.set_attribute(FunctionAttribute::MaxDynamicSharedSizeBytes, bytes)
    }

    /// Sets the preferred shared-memory carveout for this function.
    ///
    /// On devices sharing hardware resources between L1 cache and shared memory, the driver treats this as a hint and may choose another ratio.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the selected attribute or value.
    pub fn set_preferred_shared_memory_carveout(
        &self,
        carveout: SharedMemoryCarveout,
    ) -> Result<()> {
        self.set_attribute(
            FunctionAttribute::PreferredSharedMemoryCarveout,
            i32::from(carveout),
        )
    }

    /// Queries the function's resource usage, compilation versions, and launch constraints.
    ///
    /// Each field is queried separately; the result is not an atomic snapshot of concurrent attribute changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or any constituent attribute query fails.
    /// No partial result is returned.
    pub fn attributes(&self) -> Result<FunctionAttributes> {
        Ok(FunctionAttributes {
            shared_size_bytes: to_usize(
                self.attribute(FunctionAttribute::SharedSizeBytes)?,
                "shared size bytes",
            )?,
            const_size_bytes: to_usize(
                self.attribute(FunctionAttribute::ConstSizeBytes)?,
                "const size bytes",
            )?,
            local_size_bytes: to_usize(
                self.attribute(FunctionAttribute::LocalSizeBytes)?,
                "local size bytes",
            )?,
            max_threads_per_block: self.attribute(FunctionAttribute::MaxThreadsPerBlock)?,
            num_regs: self.attribute(FunctionAttribute::NumRegs)?,
            ptx_version: self.attribute(FunctionAttribute::PtxVersion)?,
            binary_version: self.attribute(FunctionAttribute::BinaryVersion)?,
            cache_mode_ca: self.attribute(FunctionAttribute::CacheModeCa)? != 0,
            max_dynamic_shared_size_bytes: self
                .attribute(FunctionAttribute::MaxDynamicSharedSizeBytes)?,
            preferred_shared_memory_carveout: self
                .attribute(FunctionAttribute::PreferredSharedMemoryCarveout)?,
            cluster_dim_must_be_set: self.attribute(FunctionAttribute::ClusterSizeMustBeSet)? != 0,
            required_cluster_width: self.attribute(FunctionAttribute::RequiredClusterWidth)?,
            required_cluster_height: self.attribute(FunctionAttribute::RequiredClusterHeight)?,
            required_cluster_depth: self.attribute(FunctionAttribute::RequiredClusterDepth)?,
            cluster_scheduling_policy_preference: self
                .attribute(FunctionAttribute::ClusterSchedulingPolicyPreference)?,
            non_portable_cluster_size_allowed: self
                .attribute(FunctionAttribute::NonPortableClusterSizeAllowed)?
                != 0,
        })
    }

    /// Returns the maximum number of active blocks per streaming multiprocessor for `block_size` threads and the specified dynamic shared-memory bytes per block.
    ///
    /// `flags` controls how special cases are handled.
    /// The valid flags are:
    ///
    /// * [`OccupancyFlags::DEFAULT`], which maintains the default behavior as [`sys::cuOccupancyMaxActiveBlocksPerMultiprocessor`](singe_cuda_sys::driver::cuOccupancyMaxActiveBlocksPerMultiprocessor);
    ///
    /// * [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`], which suppresses the default behavior on platforms where global caching affects occupancy.
    ///   On such platforms, if caching is enabled, but per-block SM resource usage would result in zero occupancy, the occupancy calculator will calculate the occupancy as if caching is disabled.
    ///   Setting [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`] makes the occupancy calculator return 0 in such cases.
    ///   More information can be found about this feature in the "Unified L1/Texture Cache" section of the Maxwell tuning guide.
    ///
    /// For context-less kernels queried via [`Library::kernel`](crate::library::Library::kernel).
    /// Here, this wrapper uses the current context for calculations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, CUDA rejects the occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: i32,
        dynamic_shared_memory_bytes: usize,
        flags: OccupancyFlags,
    ) -> Result<i32> {
        self.module.inner.context.bind()?;
        let dynamic_shared_memory_bytes =
            validate_dynamic_shared_memory_bytes(dynamic_shared_memory_bytes)?;
        let mut blocks = 0;
        // SAFETY: `blocks` is a writable local, and `self.handle` is a live function of the module `self` retains, with its context bound above.
        unsafe {
            try_ffi!(
                driver::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &raw mut blocks,
                    self.handle.as_raw(),
                    block_size,
                    dynamic_shared_memory_bytes,
                    flags.bits(),
                )
            )?;
        }
        Ok(blocks)
    }

    /// Returns dynamic shared memory available per block when launching `num_blocks` blocks on a streaming multiprocessor.
    ///
    /// The returned value is the maximum size of dynamic shared memory that allows `num_blocks` blocks per streaming multiprocessor.
    ///
    /// For context-less kernels queried via [`Library::kernel`](crate::library::Library::kernel).
    /// Here, this wrapper uses the current context for calculations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, CUDA rejects the occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_available_dynamic_shared_memory_per_block(
        &self,
        num_blocks: i32,
        block_size: i32,
    ) -> Result<usize> {
        self.module.inner.context.bind()?;
        let mut bytes = 0;
        // SAFETY: `bytes` is a writable local, and `self.handle` is a live function of the module `self` retains, with its context bound above.
        unsafe {
            try_ffi!(driver::cuOccupancyAvailableDynamicSMemPerBlock(
                &raw mut bytes,
                self.handle.as_raw(),
                num_blocks,
                block_size,
            ))?;
        }
        to_usize(bytes, "occupancy byte count")
    }

    /// Selects a block size with maximum potential occupancy for the supplied dynamic shared-memory bytes per block.
    ///
    /// `block_size_limit` restricts the block size; zero adds no limit.
    ///
    /// `flags` controls how special cases are handled.
    /// The valid flags are:
    ///
    /// * [`OccupancyFlags::DEFAULT`], which maintains the default behavior as [`sys::cuOccupancyMaxPotentialBlockSize`](singe_cuda_sys::driver::cuOccupancyMaxPotentialBlockSize);
    ///
    /// * [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`], which suppresses the default behavior on platforms where global caching affects occupancy.
    ///   On such platforms, the launch configurations that produce maximal occupancy might not support global caching.
    ///   Setting [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`] guarantees that the produced launch configuration is global caching compatible at a potential cost of occupancy.
    ///   More information can be found about this feature in the "Unified L1/Texture Cache" section of the Maxwell tuning guide.
    ///
    /// For context-less kernels queried via [`Library::kernel`](crate::library::Library::kernel).
    /// Here, this wrapper uses the current context for calculations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, CUDA rejects the occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_potential_block_size(
        &self,
        dynamic_shared_memory_bytes: usize,
        block_size_limit: i32,
        flags: OccupancyFlags,
    ) -> Result<OccupancyMaxPotentialBlockSize> {
        self.module.inner.context.bind()?;
        let dynamic_shared_memory_bytes =
            validate_dynamic_shared_memory_bytes(dynamic_shared_memory_bytes)?;
        let mut min_grid_size = 0;
        let mut block_size = 0;
        // SAFETY: the outputs are writable locals, `self.handle` is a live function of the module `self` retains, and no block-size callback is passed.
        unsafe {
            try_ffi!(driver::cuOccupancyMaxPotentialBlockSizeWithFlags(
                &raw mut min_grid_size,
                &raw mut block_size,
                self.handle.as_raw(),
                None,
                dynamic_shared_memory_bytes,
                block_size_limit,
                flags.bits(),
            ))?;
        }
        Ok(OccupancyMaxPotentialBlockSize {
            min_grid_size,
            block_size,
        })
    }

    /// Given this kernel and launch configuration, returns the maximum cluster size.
    ///
    /// The cluster dimensions in `config` are ignored.
    /// If the kernel has a required cluster size set, the returned value reflects the required cluster size.
    ///
    /// By default this returns a value that is portable on future hardware.
    /// A higher value may be returned if the kernel function allows non-portable cluster sizes.
    ///
    /// Respects the compile-time launch bounds.
    ///
    /// For context-less kernels queried via [`Library::kernel`](crate::library::Library::kernel).
    /// Here, this wrapper uses the current context for calculations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, CUDA rejects the occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_potential_cluster_size(&self, config: &LaunchConfig) -> Result<i32> {
        self.module.inner.context.bind()?;
        let mut cluster_size = 0;
        // SAFETY: `cluster_size` is a writable local, `self.handle` is a live function of the module `self` retains, and `config` lives through the closure.
        config.with_raw_config(ptr::null_mut(), |config| unsafe {
            try_ffi!(driver::cuOccupancyMaxPotentialClusterSize(
                &raw mut cluster_size,
                self.handle.as_raw(),
                config,
            ))?;
            Ok(cluster_size)
        })
    }

    /// Given this kernel and launch configuration, returns the maximum number of clusters that could co-exist on the target device.
    ///
    /// If the kernel already has a required cluster size set, the cluster size from `config` must either be unspecified or match the required size.
    /// Without required sizes, the cluster size must be specified in `config`; otherwise this method returns an error.
    ///
    /// Various kernel function attributes may affect occupancy calculation.
    /// Runtime environment may affect how the hardware schedules the clusters, so the calculated occupancy is not guaranteed to be achievable.
    ///
    /// For context-less kernels queried via [`Library::kernel`](crate::library::Library::kernel).
    /// Here, this wrapper uses the current context for calculations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, `config` does not specify a valid cluster size for this kernel, CUDA rejects the occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_active_clusters(&self, config: &LaunchConfig) -> Result<i32> {
        self.module.inner.context.bind()?;
        let mut clusters = 0;
        // SAFETY: `clusters` is a writable local, `self.handle` is a live function of the module `self` retains, and `config` lives through the closure.
        config.with_raw_config(ptr::null_mut(), |config| unsafe {
            try_ffi!(driver::cuOccupancyMaxActiveClusters(
                &raw mut clusters,
                self.handle.as_raw(),
                config,
            ))?;
            Ok(clusters)
        })
    }

    /// Returns the borrowed device function handle without retaining another module owner.
    #[must_use]
    pub const fn as_raw(&self) -> DeviceFunction {
        self.handle
    }
}

impl KernelFunction {
    /// Specifies whether the function may use oversized shared memory configurations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the selected attribute or value.
    pub fn set_shared_memory_mode(&self, mode: SharedMemoryMode) -> Result<()> {
        self.set_attribute(
            FunctionAttribute::SharedMemoryMode,
            to_i32(u32::from(mode), "shared memory mode")?,
        )
    }

    /// Returns the shared memory mode configured for this function.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the requested function attribute.
    /// Returns `Error::UnknownEnumValue` if CUDA returns an unrecognized shared-memory mode.
    pub fn shared_memory_mode(&self) -> Result<SharedMemoryMode> {
        let value = self.attribute(FunctionAttribute::SharedMemoryMode)?;
        u32::try_from(value)
            .ok()
            .and_then(|raw| SharedMemoryMode::try_from(raw).ok())
            .ok_or_else(|| Error::UnknownEnumValue {
                name: "shared memory mode".into(),
                value: i64::from(value),
            })
    }

    /// Invokes this kernel on a stream provisioned by a green context.
    ///
    /// # Safety
    ///
    /// Arguments must match the kernel ABI and satisfy its access and value requirements.
    /// Keep the function and all pointed-to storage alive until execution completes.
    /// Order conflicting accesses and preserve exclusive writable access during execution.
    /// Successful submission does not establish output initialization or completion.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` unless the function was loaded in the primary context of the green context's device.
    /// Also returns an error if the module context cannot be bound, argument encoding fails, CUDA rejects the launch, or previous asynchronous work reports an error.
    pub unsafe fn launch_green_async<'a, P: KernelLaunchArgs<'a>>(
        &self,
        config: &LaunchConfig,
        params: P,
        stream: &GreenStream,
    ) -> Result<()> {
        if !self
            .module
            .inner
            .context
            .is_primary_for(stream.context().device())?
        {
            return Err(Error::StreamContextMismatch);
        }

        self.module.inner.context.bind()?;
        // SAFETY: the function's context is bound above, the encoded arguments live through the call, and the caller guarantees their ABI, storage lifetime, and access ordering.
        params.with_encoded_arguments(|mut arguments| unsafe {
            config.launch_raw(
                self.handle.as_raw(),
                stream.as_raw(),
                arguments.as_mut_ptr().cast(),
            )
        })
    }
}
