use std::{
    borrow::Cow,
    ffi::CString,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit, align_of, size_of},
    ptr,
    sync::Arc,
};

use singe_core::checked_int;
use singe_cuda_sys::driver;

use crate::{
    context::Context,
    dim::Dim3,
    error::{Error, Result},
    graph::{ExecutableGraph, Graph, GraphNode},
    kernel::{self, ModuleKernelHandle},
    memory::{DeviceMemory, ManagedMemory},
    stream::Stream,
    try_ffi,
    types::{DeviceFunction, FunctionAttribute, SharedMemoryCarveout},
    view::{DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut},
};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct OccupancyFlags: u32 {
        const DEFAULT = driver::CUoccupancy_flags::CU_OCCUPANCY_DEFAULT as _;
        const DISABLE_CACHING_OVERRIDE = driver::CUoccupancy_flags::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE as _;
    }
}

impl Display for OccupancyFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return Ok(());
        }
        let mut first = true;
        let write_sep = |f: &mut Formatter<'_>, first: &mut bool, name: &str| -> fmt::Result {
            if *first {
                *first = false;
            } else {
                f.write_str(" | ")?;
            }
            f.write_str(name)
        };

        if self.contains(Self::DEFAULT) {
            write_sep(f, &mut first, "CU_OCCUPANCY_DEFAULT")?;
        }
        if self.contains(Self::DISABLE_CACHING_OVERRIDE) {
            write_sep(f, &mut first, "CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE")?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionAttributes {
    pub shared_size_bytes: usize,
    pub const_size_bytes: usize,
    pub local_size_bytes: usize,
    pub max_threads_per_block: i32,
    pub num_regs: i32,
    pub ptx_version: i32,
    pub binary_version: i32,
    pub cache_mode_ca: bool,
    pub max_dynamic_shared_size_bytes: i32,
    pub preferred_shared_memory_carveout: i32,
    pub cluster_dim_must_be_set: bool,
    pub required_cluster_width: i32,
    pub required_cluster_height: i32,
    pub required_cluster_depth: i32,
    pub cluster_scheduling_policy_preference: i32,
    pub non_portable_cluster_size_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyMaxPotentialBlockSize {
    pub min_grid_size: i32,
    pub block_size: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterLaunchConfig {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub shared_memory_bytes: usize,
}

#[derive(Debug)]
pub struct Module {
    handle: driver::CUmodule,
    ctx: Arc<Context>,
    owns_handle: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Global<'a> {
    ptr: *mut (),
    size: usize,
    _module: &'a Module,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureReference<'a> {
    handle: driver::CUtexref,
    _module: &'a Module,
}

#[derive(Debug, Clone, Copy)]
pub struct SurfaceReference<'a> {
    handle: driver::CUsurfref,
    _module: &'a Module,
}

#[derive(Debug, Clone)]
pub struct ModuleImage<'a> {
    data: Cow<'a, [u8]>,
}

#[derive(Debug)]
pub struct KernelFunction<'a> {
    handle: DeviceFunction,
    module: &'a Module,
}

#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub shared_memory_bytes: usize,
}

/// Dynamically built CUDA kernel argument list.
///
/// Use this builder when the argument list depends on runtime conditions:
///
/// ```ignore
/// let mut params = KernelParameters::new();
/// params.arg(&input_ptr).arg(&output_ptr).push(len);
/// function.launch(&config, params)?;
/// ```
///
/// For fixed argument lists, passing a tuple of references directly to
/// [`KernelFunction::launch`] avoids building a dynamic list:
///
/// ```ignore
/// function.launch(&config, (&input_ptr, &mut output_ptr, &len))?;
/// ```
///
/// Borrowed arguments are tied to the lifetime of this list. Owned scalar and
/// pointer arguments pushed with [`Self::push`] or [`Self::owned_arg`] are stored
/// inline when they fit, so ordinary kernel launches do not allocate per
/// argument.
pub struct KernelParameters<'a> {
    arguments: Vec<KernelParameter<'a>>,
}

const INLINE_KERNEL_ARGUMENTS: usize = 16;
const INLINE_KERNEL_ARGUMENT_BYTES: usize = 16;

mod private {
    pub trait Sealed {}
}

/// Appends a value to a CUDA kernel parameter list.
///
/// Implementations convert Rust wrapper types into the value a CUDA kernel sees
/// at the ABI boundary, such as a scalar or device pointer.
pub trait PushKernelArg {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>);
}

/// Kernel launch arguments accepted by launch and graph-node APIs.
///
/// This sealed trait is implemented for [`KernelParameters`], `()`, and tuples
/// of shared or mutable references up to 16 elements.
pub trait KernelLaunchArgs<'a>: private::Sealed {
    #[doc(hidden)]
    fn with_raw_pointers<R>(self, f: impl FnOnce(&mut [*mut ()]) -> R) -> R;
}

trait KernelTupleArgument<'a> {
    fn into_kernel_argument_ptr(self) -> *mut ();
}

enum KernelParameter<'a> {
    Borrowed {
        ptr: *mut (),
        _marker: PhantomData<&'a ()>,
    },
    Owned(OwnedKernelArgument),
}

impl fmt::Debug for KernelParameter<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Borrowed { ptr, .. } => f.debug_tuple("Borrowed").field(ptr).finish(),
            Self::Owned(value) => f.debug_tuple("Owned").field(value).finish(),
        }
    }
}

enum OwnedKernelArgument {
    Inline(InlineKernelArgument),
    Boxed(Box<dyn KernelArgumentStorage>),
}

impl fmt::Debug for OwnedKernelArgument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline(value) => f.debug_tuple("Inline").field(value).finish(),
            Self::Boxed(_) => f.debug_tuple("Boxed").finish_non_exhaustive(),
        }
    }
}

trait KernelArgumentStorage {
    fn as_mut_ptr(&mut self) -> *mut ();
}

impl<T> KernelArgumentStorage for T {
    fn as_mut_ptr(&mut self) -> *mut () {
        ptr::from_mut(self).cast()
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct InlineKernelArgument {
    bytes: [MaybeUninit<u8>; INLINE_KERNEL_ARGUMENT_BYTES],
}

impl fmt::Debug for InlineKernelArgument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("InlineKernelArgument")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for KernelParameters<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelParameters")
            .field("arguments", &self.arguments.len())
            .finish()
    }
}

impl Module {
    pub unsafe fn from_raw(handle: driver::CUmodule, ctx: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            ctx,
            owns_handle: true,
        })
    }

    pub const unsafe fn from_borrowed_raw(handle: driver::CUmodule, ctx: Arc<Context>) -> Self {
        Self {
            handle,
            ctx,
            owns_handle: false,
        }
    }

    /// Returns the kernel function with the given name from the module.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Status::NotFound`] if the module has no kernel function
    /// named `name`. Also returns an error if `name` contains an interior NUL
    /// byte, the module context cannot be bound, CUDA rejects the lookup, or a
    /// previous asynchronous launch reports an error.
    pub fn function(&self, name: &str) -> Result<KernelFunction<'_>> {
        unsafe {
            let c_name = CString::new(name)?;
            let mut function_handle = ptr::null_mut();
            try_ffi!(driver::cuModuleGetFunction(
                &raw mut function_handle,
                self.handle,
                c_name.as_ptr(),
            ))?;
            if function_handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(KernelFunction::from_raw(function_handle.into(), self))
        }
    }

    /// Returns the number of functions in this module.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Driver cannot report the function count.
    pub fn function_count(&self) -> Result<usize> {
        unsafe {
            let mut count = 0;
            try_ffi!(driver::cuModuleGetFunctionCount(
                &raw mut count,
                self.handle
            ))?;
            Ok(count as usize)
        }
    }

    pub const fn as_raw(&self) -> driver::CUmodule {
        self.handle
    }

    /// Consumes the module and returns the raw CUDA module handle without
    /// unloading it.
    ///
    /// The caller becomes responsible for eventually unloading the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> driver::CUmodule {
        let module = ManuallyDrop::new(self);
        module.handle
    }

    /// Returns the base pointer and size of the global with the given name located in the module.
    ///
    /// The returned [`Global`] borrows this module, so the module remains loaded
    /// for at least as long as the global reference is usable.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Status::NotFound`] if the module has no global variable
    /// named `name`. Also returns an error if `name` contains an interior NUL
    /// byte, the module context cannot be bound, CUDA rejects the lookup, or a
    /// previous asynchronous launch reports an error.
    pub fn global(&self, name: &str) -> Result<Global<'_>> {
        let c_name = CString::new(name)?;
        let mut ptr = 0;
        let mut size = 0;
        self.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuModuleGetGlobal_v2(
                &raw mut ptr,
                &raw mut size,
                self.handle,
                c_name.as_ptr(),
            ))?;
        }
        Ok(Global {
            ptr: ptr as _,
            size: size as _,
            _module: self,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.owns_handle {
            return;
        }

        if let Err(err) = self.ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind context before unloading module: {err}");
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(driver::cuModuleUnload(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to unload cuda module: {err}");
            }
        }
    }
}

// CUDA modules are immutable after loading in this wrapper. Kernel/function
// lookups use shared references and CUDA owns internal synchronization.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl<'a> ModuleImage<'a> {
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data: Cow::Borrowed(data),
        }
    }

    pub fn from_vec(data: Vec<u8>) -> Self {
        Self {
            data: Cow::Owned(data),
        }
    }

    pub fn from_string(data: String) -> Self {
        Self::from_vec(data.into_bytes())
    }

    pub fn as_ptr(&self) -> *const () {
        self.data.as_ptr().cast()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_ref()
    }
}

impl Global<'_> {
    pub const fn as_ptr(&self) -> *mut () {
        self.ptr
    }

    pub const fn byte_len(&self) -> usize {
        self.size
    }
}

impl TextureReference<'_> {
    pub const fn as_raw(&self) -> driver::CUtexref {
        self.handle
    }
}

impl SurfaceReference<'_> {
    pub const fn as_raw(&self) -> driver::CUsurfref {
        self.handle
    }
}

impl KernelFunction<'_> {
    pub const unsafe fn from_raw(handle: DeviceFunction, module: &Module) -> KernelFunction<'_> {
        KernelFunction { handle, module }
    }

    /// Invokes this kernel function on a grid of blocks.
    /// Each block contains the threads specified by [`LaunchConfig::block_dim`].
    ///
    /// [`LaunchConfig::shared_memory_bytes`] sets the amount of dynamic shared memory available to each thread block.
    ///
    /// Kernel parameters are passed with [`KernelParameters`] or tuples of shared or mutable references.
    ///
    /// Launching the kernel invalidates the persistent function state set through the following deprecated APIs: [`sys::cuFuncSetBlockShape`](singe_cuda_sys::driver::cuFuncSetBlockShape), [`sys::cuFuncSetSharedSize`](singe_cuda_sys::driver::cuFuncSetSharedSize), [`sys::cuParamSetSize`](singe_cuda_sys::driver::cuParamSetSize), [`sys::cuParamSeti`](singe_cuda_sys::driver::cuParamSeti), [`sys::cuParamSetf`](singe_cuda_sys::driver::cuParamSetf), [`sys::cuParamSetv`](singe_cuda_sys::driver::cuParamSetv).
    ///
    /// The kernel must either have been compiled with toolchain version 3.2 or later so that it contains kernel parameter information, or have no kernel parameters.
    /// If either of these conditions is not met, the launch returns [`crate::error::Status::InvalidImage`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Status::InvalidImage`] if the kernel parameter metadata
    /// requirements above are not met. Also returns an error if the module
    /// context cannot be bound, CUDA rejects the launch, or a previous
    /// asynchronous launch reports an error.
    pub fn launch<'a, P>(&self, config: &LaunchConfig, params: P) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        self.module.ctx.bind()?;
        params.with_raw_pointers(|arguments| unsafe {
            try_ffi!(driver::cuLaunchKernel(
                self.handle.as_raw(),
                config.grid_dim.x,
                config.grid_dim.y,
                config.grid_dim.z,
                config.block_dim.x,
                config.block_dim.y,
                config.block_dim.z,
                config.shared_memory_bytes as _,
                ptr::null_mut(),
                arguments.as_mut_ptr().cast(),
                ptr::null_mut(),
            ))?;
            Ok(())
        })
    }

    /// Invokes this kernel function on a grid of blocks using the given stream.
    /// Each block contains the threads specified by [`LaunchConfig::block_dim`].
    ///
    /// [`LaunchConfig::shared_memory_bytes`] sets the amount of dynamic shared memory available to each thread block.
    ///
    /// Kernel parameters are passed with [`KernelParameters`] or tuples of shared or mutable references.
    ///
    /// Launching the kernel invalidates the persistent function state set through the following deprecated APIs: [`sys::cuFuncSetBlockShape`](singe_cuda_sys::driver::cuFuncSetBlockShape), [`sys::cuFuncSetSharedSize`](singe_cuda_sys::driver::cuFuncSetSharedSize), [`sys::cuParamSetSize`](singe_cuda_sys::driver::cuParamSetSize), [`sys::cuParamSeti`](singe_cuda_sys::driver::cuParamSeti), [`sys::cuParamSetf`](singe_cuda_sys::driver::cuParamSetf), [`sys::cuParamSetv`](singe_cuda_sys::driver::cuParamSetv).
    ///
    /// The kernel must either have been compiled with toolchain version 3.2 or later so that it contains kernel parameter information, or have no kernel parameters.
    /// If either of these conditions is not met, the launch returns [`crate::error::Status::InvalidImage`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Status::InvalidImage`] if the kernel parameter metadata
    /// requirements above are not met. Also returns an error if `stream` belongs
    /// to a different context, the module context cannot be bound, CUDA rejects
    /// the launch, or a previous asynchronous launch reports an error.
    pub fn launch_on<'a, P>(&self, config: &LaunchConfig, params: P, stream: &Stream) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        if stream.context() != self.module.ctx.as_ref() {
            return Err(driver::CUresult::CUDA_ERROR_INVALID_CONTEXT.into());
        }

        self.module.ctx.bind()?;
        params.with_raw_pointers(|arguments| unsafe {
            try_ffi!(driver::cuLaunchKernel(
                self.handle.as_raw(),
                config.grid_dim.x,
                config.grid_dim.y,
                config.grid_dim.z,
                config.block_dim.x,
                config.block_dim.y,
                config.block_dim.z,
                config.shared_memory_bytes as _,
                stream.as_raw(),
                arguments.as_mut_ptr().cast(),
                ptr::null_mut(),
            ))?;
            Ok(())
        })
    }

    pub fn add_to_graph<'a, P>(
        &self,
        graph: &mut Graph,
        dependencies: &[GraphNode],
        config: &LaunchConfig,
        params: P,
    ) -> Result<GraphNode>
    where
        P: KernelLaunchArgs<'a>,
    {
        graph.add_kernel_node(dependencies, self.handle, config, params)
    }

    pub fn set_graph_node_params<'a, P>(
        &self,
        executable: &mut ExecutableGraph,
        node: GraphNode,
        config: &LaunchConfig,
        params: P,
    ) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        executable.set_kernel_node_params(node, self.handle, config, params)
    }

    pub const fn module(&self) -> &Module {
        self.module
    }

    pub fn name(&self) -> Result<String> {
        kernel::name::<ModuleKernelHandle>(self.module.ctx.as_ref(), self.handle.as_raw())
    }

    pub fn attribute(&self, attribute: FunctionAttribute) -> Result<i32> {
        kernel::attribute::<ModuleKernelHandle>(
            self.module.ctx.as_ref(),
            self.handle.as_raw(),
            attribute,
        )
    }

    pub fn set_attribute(&self, attribute: FunctionAttribute, value: i32) -> Result<()> {
        kernel::set_attribute::<ModuleKernelHandle>(
            self.module.ctx.as_ref(),
            self.handle.as_raw(),
            attribute,
            value,
        )
    }

    pub fn set_max_dynamic_shared_memory_bytes(&self, bytes: i32) -> Result<()> {
        self.set_attribute(FunctionAttribute::MaxDynamicSharedSizeBytes, bytes)
    }

    pub fn set_preferred_shared_memory_carveout(
        &self,
        carveout: SharedMemoryCarveout,
    ) -> Result<()> {
        self.set_attribute(
            FunctionAttribute::PreferredSharedMemoryCarveout,
            i32::from(carveout),
        )
    }

    pub fn attributes(&self) -> Result<FunctionAttributes> {
        Ok(FunctionAttributes {
            shared_size_bytes: self.attribute(FunctionAttribute::SharedSizeBytes)? as usize,
            const_size_bytes: self.attribute(FunctionAttribute::ConstSizeBytes)? as usize,
            local_size_bytes: self.attribute(FunctionAttribute::LocalSizeBytes)? as usize,
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

    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: i32,
        dynamic_shared_memory_bytes: usize,
    ) -> Result<i32> {
        self.occupancy_max_active_blocks_per_multiprocessor_with_flags(
            block_size,
            dynamic_shared_memory_bytes,
            OccupancyFlags::DEFAULT,
        )
    }

    /// Returns the maximum number of active blocks per streaming multiprocessor.
    ///
    /// `flags` controls how special cases are handled.
    /// The valid flags are:
    ///
    /// * [`OccupancyFlags::DEFAULT`], which maintains the default behavior as [`sys::cuOccupancyMaxActiveBlocksPerMultiprocessor`](singe_cuda_sys::driver::cuOccupancyMaxActiveBlocksPerMultiprocessor);
    ///
    /// * [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`], which suppresses the default behavior on platforms where global caching affects occupancy.
    ///   On such platforms, if caching
    ///   is enabled, but per-block SM resource usage would result in zero occupancy, the occupancy calculator will calculate the occupancy
    ///   as if caching is disabled.
    ///   Setting [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`] makes the occupancy calculator return 0 in such cases.
    ///   More information can be found about this feature in the "Unified
    ///   L1/Texture Cache" section of the Maxwell tuning guide.
    ///
    /// For context-less kernels queried via [`Library::kernel`](crate::library::Library::kernel).
    /// Here, this wrapper uses the current context for calculations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_active_blocks_per_multiprocessor_with_flags(
        &self,
        block_size: i32,
        dynamic_shared_memory_bytes: usize,
        flags: OccupancyFlags,
    ) -> Result<i32> {
        self.module.ctx.bind()?;
        let mut blocks = 0;
        unsafe {
            try_ffi!(
                driver::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &raw mut blocks,
                    self.handle.as_raw(),
                    block_size,
                    dynamic_shared_memory_bytes as _,
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
    /// Returns an error if the module context cannot be bound, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_available_dynamic_shared_memory_per_block(
        &self,
        num_blocks: i32,
        block_size: i32,
    ) -> Result<usize> {
        self.module.ctx.bind()?;
        let mut bytes = 0;
        unsafe {
            try_ffi!(driver::cuOccupancyAvailableDynamicSMemPerBlock(
                &raw mut bytes,
                self.handle.as_raw(),
                num_blocks,
                block_size,
            ))?;
        }
        Ok(bytes as usize)
    }

    pub fn occupancy_max_potential_block_size(
        &self,
        dynamic_shared_memory_bytes: usize,
        block_size_limit: i32,
    ) -> Result<OccupancyMaxPotentialBlockSize> {
        self.occupancy_max_potential_block_size_with_flags(
            dynamic_shared_memory_bytes,
            block_size_limit,
            OccupancyFlags::DEFAULT,
        )
    }

    /// An extended version of [`sys::cuOccupancyMaxPotentialBlockSize`](singe_cuda_sys::driver::cuOccupancyMaxPotentialBlockSize).
    /// In addition to arguments passed to [`sys::cuOccupancyMaxPotentialBlockSize`](singe_cuda_sys::driver::cuOccupancyMaxPotentialBlockSize), [`KernelFunction::occupancy_max_potential_block_size_with_flags`] also takes `flags`.
    ///
    /// `flags` controls how special cases are handled.
    /// The valid flags are:
    ///
    /// * [`OccupancyFlags::DEFAULT`], which maintains the default behavior as [`sys::cuOccupancyMaxPotentialBlockSize`](singe_cuda_sys::driver::cuOccupancyMaxPotentialBlockSize);
    ///
    /// * [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`], which suppresses the default behavior on platforms where global caching affects occupancy.
    ///   On such platforms, the launch
    ///   configurations that produce maximal occupancy might not support global caching.
    ///   Setting [`OccupancyFlags::DISABLE_CACHING_OVERRIDE`] guarantees that the produced launch configuration is global caching compatible at a potential cost of occupancy.
    ///   More
    ///   information can be found about this feature in the "Unified L1/Texture Cache" section of the Maxwell tuning guide.
    ///
    /// For context-less kernels queried via [`Library::kernel`](crate::library::Library::kernel).
    /// Here, this wrapper uses the current context for calculations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_potential_block_size_with_flags(
        &self,
        dynamic_shared_memory_bytes: usize,
        block_size_limit: i32,
        flags: OccupancyFlags,
    ) -> Result<OccupancyMaxPotentialBlockSize> {
        self.module.ctx.bind()?;
        let mut min_grid_size = 0;
        let mut block_size = 0;
        unsafe {
            try_ffi!(driver::cuOccupancyMaxPotentialBlockSizeWithFlags(
                &raw mut min_grid_size,
                &raw mut block_size,
                self.handle.as_raw(),
                None,
                dynamic_shared_memory_bytes as _,
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
    /// Returns an error if the module context cannot be bound, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_potential_cluster_size(&self, config: ClusterLaunchConfig) -> Result<i32> {
        self.module.ctx.bind()?;
        let mut cluster_size = 0;
        let config = driver::CUlaunchConfig {
            gridDimX: config.grid_dim.x,
            gridDimY: config.grid_dim.y,
            gridDimZ: config.grid_dim.z,
            blockDimX: config.block_dim.x,
            blockDimY: config.block_dim.y,
            blockDimZ: config.block_dim.z,
            sharedMemBytes: config.shared_memory_bytes as _,
            hStream: ptr::null_mut(),
            attrs: ptr::null_mut(),
            numAttrs: 0,
        };
        unsafe {
            try_ffi!(driver::cuOccupancyMaxPotentialClusterSize(
                &raw mut cluster_size,
                self.handle.as_raw(),
                &raw const config,
            ))?;
        }
        Ok(cluster_size)
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
    /// Returns an error if the module context cannot be bound, `config` does
    /// not specify a valid cluster size for this kernel, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_active_clusters(&self, config: ClusterLaunchConfig) -> Result<i32> {
        self.module.ctx.bind()?;
        let mut clusters = 0;
        let config = driver::CUlaunchConfig {
            gridDimX: config.grid_dim.x,
            gridDimY: config.grid_dim.y,
            gridDimZ: config.grid_dim.z,
            blockDimX: config.block_dim.x,
            blockDimY: config.block_dim.y,
            blockDimZ: config.block_dim.z,
            sharedMemBytes: config.shared_memory_bytes as _,
            hStream: ptr::null_mut(),
            attrs: ptr::null_mut(),
            numAttrs: 0,
        };
        unsafe {
            try_ffi!(driver::cuOccupancyMaxActiveClusters(
                &raw mut clusters,
                self.handle.as_raw(),
                &raw const config,
            ))?;
        }
        Ok(clusters)
    }

    pub const fn as_raw(&self) -> DeviceFunction {
        self.handle
    }
}

impl LaunchConfig {
    pub const fn new(grid_dim: Dim3, block_dim: Dim3, shared_memory_bytes: usize) -> Self {
        Self {
            grid_dim,
            block_dim,
            shared_memory_bytes,
        }
    }

    pub fn try_for_1d_grid(element_count: usize, block_size: usize) -> Result<Self> {
        validate_block_dimension(block_size, "block_size")?;
        let grid_size = element_count.div_ceil(block_size);

        Ok(Self::new(
            Dim3::new(to_u32(grid_size, "grid_size")?, 1, 1),
            Dim3::new(to_u32(block_size, "block_size")?, 1, 1),
            0,
        ))
    }

    pub fn for_1d_grid(element_count: usize, block_size: usize) -> Self {
        Self::try_for_1d_grid(element_count, block_size)
            .expect("invalid 1d cuda launch configuration")
    }

    pub fn try_for_num_elems(element_count: usize, block_size: usize) -> Result<Self> {
        Self::try_for_1d_grid(element_count, block_size)
    }

    pub fn for_num_elems(element_count: usize, block_size: usize) -> Self {
        Self::try_for_num_elems(element_count, block_size)
            .expect("invalid cuda launch configuration")
    }

    pub fn try_for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Result<Self> {
        validate_block_dimension(block_width, "block_width")?;
        validate_block_dimension(block_height, "block_height")?;
        let grid_x = width.div_ceil(block_width);
        let grid_y = height.div_ceil(block_height);

        Ok(Self::new(
            Dim3::new(to_u32(grid_x, "grid_x")?, to_u32(grid_y, "grid_y")?, 1),
            Dim3::new(
                to_u32(block_width, "block_width")?,
                to_u32(block_height, "block_height")?,
                1,
            ),
            0,
        ))
    }

    pub fn for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Self {
        Self::try_for_2d_grid(width, height, block_width, block_height)
            .expect("invalid 2d cuda launch configuration")
    }

    pub fn try_for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Result<Self> {
        validate_block_dimension(block_width, "block_width")?;
        validate_block_dimension(block_height, "block_height")?;
        validate_block_dimension(block_depth, "block_depth")?;
        let grid_x = width.div_ceil(block_width);
        let grid_y = height.div_ceil(block_height);
        let grid_z = depth.div_ceil(block_depth);

        Ok(Self::new(
            Dim3::new(
                to_u32(grid_x, "grid_x")?,
                to_u32(grid_y, "grid_y")?,
                to_u32(grid_z, "grid_z")?,
            ),
            Dim3::new(
                to_u32(block_width, "block_width")?,
                to_u32(block_height, "block_height")?,
                to_u32(block_depth, "block_depth")?,
            ),
            0,
        ))
    }

    pub fn for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Self {
        Self::try_for_3d_grid(width, height, depth, block_width, block_height, block_depth)
            .expect("invalid 3d cuda launch configuration")
    }
}

fn validate_block_dimension(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::ZeroValue {
            name: name.to_owned(),
        });
    }
    Ok(())
}

fn to_u32(value: usize, name: &str) -> Result<u32> {
    checked_int(value, name, |name| Error::OutOfRange { name })
}

impl<'a> KernelParameters<'a> {
    pub const fn new() -> Self {
        Self {
            arguments: Vec::new(),
        }
    }

    pub fn arg<T: 'a>(&mut self, value: &'a T) -> &mut Self {
        self.arguments.push(KernelParameter::Borrowed {
            ptr: ptr::from_ref(value).cast_mut().cast::<()>(),
            _marker: PhantomData,
        });
        self
    }

    pub fn arg_mut<T: 'a>(&mut self, value: &'a mut T) -> &mut Self {
        self.arguments.push(KernelParameter::Borrowed {
            ptr: ptr::from_mut(value).cast::<()>(),
            _marker: PhantomData,
        });
        self
    }

    /// Pushes a copied kernel argument whose storage is owned by this list.
    ///
    /// Small scalar and pointer values are stored inline. Larger values fall
    /// back to heap storage, while keeping the argument pointee stable until
    /// CUDA has copied it during launch or graph-node creation.
    pub fn owned_arg<T: Copy + 'static>(&mut self, value: T) -> &mut Self {
        let value = OwnedKernelArgument::from_value(value);
        self.arguments.push(KernelParameter::Owned(value));
        self
    }

    pub fn push<A: PushKernelArg>(&mut self, arg: A) -> &mut Self {
        arg.push_to(self);
        self
    }

    pub fn device_slice<T: DeviceRepr, S: DeviceSlice<T> + ?Sized>(
        &mut self,
        slice: &S,
    ) -> &mut Self {
        // Kernels take the device address for slice-like wrappers; length is a
        // separate scalar argument when the kernel needs it.
        self.owned_arg(slice.as_device_ptr())
    }

    pub fn device_slice_mut<T: DeviceRepr, S: DeviceSliceMut<T> + ?Sized>(
        &mut self,
        slice: &mut S,
    ) -> &mut Self {
        self.owned_arg(slice.as_device_mut_ptr())
    }

    fn raw_pointers(&mut self) -> RawKernelPointers {
        RawKernelPointers::from_parameters(self.arguments.as_mut_slice())
    }
}

impl<'a> KernelParameter<'a> {
    fn as_mut_ptr(&mut self) -> *mut () {
        match self {
            Self::Borrowed { ptr, .. } => *ptr,
            Self::Owned(value) => value.as_mut_ptr(),
        }
    }
}

impl OwnedKernelArgument {
    fn from_value<T: Copy + 'static>(value: T) -> Self {
        if size_of::<T>() <= INLINE_KERNEL_ARGUMENT_BYTES
            && align_of::<T>() <= align_of::<InlineKernelArgument>()
        {
            Self::Inline(InlineKernelArgument::from_value(value))
        } else {
            Self::Boxed(Box::new(value))
        }
    }

    fn as_mut_ptr(&mut self) -> *mut () {
        match self {
            Self::Inline(value) => value.as_mut_ptr(),
            Self::Boxed(value) => value.as_mut().as_mut_ptr(),
        }
    }
}

impl InlineKernelArgument {
    fn from_value<T: Copy>(value: T) -> Self {
        let mut storage = Self {
            bytes: [MaybeUninit::uninit(); INLINE_KERNEL_ARGUMENT_BYTES],
        };
        unsafe {
            ptr::write(storage.as_mut_ptr().cast::<T>(), value);
        }
        storage
    }

    fn as_mut_ptr(&mut self) -> *mut () {
        self.bytes.as_mut_ptr().cast()
    }
}

enum RawKernelPointers {
    Inline {
        pointers: [*mut (); INLINE_KERNEL_ARGUMENTS],
        len: usize,
    },
    Heap(Vec<*mut ()>),
}

impl RawKernelPointers {
    fn from_parameters(parameters: &mut [KernelParameter<'_>]) -> Self {
        if parameters.len() <= INLINE_KERNEL_ARGUMENTS {
            let mut pointers = [ptr::null_mut(); INLINE_KERNEL_ARGUMENTS];
            for (dst, parameter) in pointers.iter_mut().zip(&mut *parameters) {
                *dst = parameter.as_mut_ptr();
            }
            Self::Inline {
                pointers,
                len: parameters.len(),
            }
        } else {
            Self::Heap(
                parameters
                    .iter_mut()
                    .map(KernelParameter::as_mut_ptr)
                    .collect(),
            )
        }
    }

    fn as_mut_slice(&mut self) -> &mut [*mut ()] {
        match self {
            Self::Inline { pointers, len } => &mut pointers[..*len],
            Self::Heap(pointers) => pointers.as_mut_slice(),
        }
    }
}

impl<'a> KernelLaunchArgs<'a> for KernelParameters<'a> {
    fn with_raw_pointers<R>(mut self, f: impl FnOnce(&mut [*mut ()]) -> R) -> R {
        let mut arguments = self.raw_pointers();
        f(arguments.as_mut_slice())
    }
}

impl private::Sealed for KernelParameters<'_> {}

impl<'a> KernelLaunchArgs<'a> for &mut KernelParameters<'a> {
    fn with_raw_pointers<R>(self, f: impl FnOnce(&mut [*mut ()]) -> R) -> R {
        let mut arguments = self.raw_pointers();
        f(arguments.as_mut_slice())
    }
}

impl private::Sealed for &mut KernelParameters<'_> {}

impl<'a> KernelLaunchArgs<'a> for () {
    fn with_raw_pointers<R>(self, f: impl FnOnce(&mut [*mut ()]) -> R) -> R {
        let mut arguments: [*mut (); 0] = [];
        f(&mut arguments)
    }
}

impl private::Sealed for () {}

macro_rules! impl_kernel_arguments_for_tuple {
    ($($arg:ident),+ $(,)?) => {
        impl<'a, $($arg),+> private::Sealed for ($($arg,)+)
        where
            $($arg: KernelTupleArgument<'a>,)+
        {
        }

        impl<'a, $($arg),+> KernelLaunchArgs<'a> for ($($arg,)+)
        where
            $($arg: KernelTupleArgument<'a>,)+
        {
            fn with_raw_pointers<R>(self, f: impl FnOnce(&mut [*mut ()]) -> R) -> R {
                #[allow(non_snake_case)]
                let ($($arg,)+) = self;
                let mut arguments = [
                    $($arg.into_kernel_argument_ptr(),)+
                ];
                f(&mut arguments)
            }
        }
    };
}

impl<'a, T: 'a> KernelTupleArgument<'a> for &'a T {
    fn into_kernel_argument_ptr(self) -> *mut () {
        ptr::from_ref(self).cast_mut().cast()
    }
}

impl<'a, T: 'a> KernelTupleArgument<'a> for &'a mut T {
    fn into_kernel_argument_ptr(self) -> *mut () {
        ptr::from_mut(self).cast()
    }
}

impl_kernel_arguments_for_tuple!(A);
impl_kernel_arguments_for_tuple!(A, B);
impl_kernel_arguments_for_tuple!(A, B, C);
impl_kernel_arguments_for_tuple!(A, B, C, D);
impl_kernel_arguments_for_tuple!(A, B, C, D, E);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_kernel_arguments_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

macro_rules! impl_push_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl PushKernelArg for $ty {
                fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
                    params.owned_arg(self);
                }
            }
        )*
    };
}

impl_push_scalar!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64,
);

impl<T: DeviceRepr> PushKernelArg for &DeviceMemory<T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.device_slice(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut DeviceMemory<T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.device_slice_mut(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &ManagedMemory<T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.device_slice(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut ManagedMemory<T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.device_slice_mut(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for DeviceView<'_, T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.owned_arg(self.as_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceView<'_, T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.owned_arg(self.as_device_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceViewMut<'_, T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.owned_arg(self.as_device_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut DeviceViewMut<'_, T> {
    fn push_to<'a>(self, params: &mut KernelParameters<'a>) {
        params.owned_arg(self.as_device_mut_ptr());
    }
}

impl Default for KernelParameters<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LargeArgument {
        words: [u64; 3],
    }

    #[test]
    fn boxed_owned_kernel_argument_points_to_inner_value() {
        let mut argument = OwnedKernelArgument::from_value(LargeArgument { words: [1, 2, 3] });
        assert!(matches!(argument, OwnedKernelArgument::Boxed(_)));

        let expected = match &mut argument {
            OwnedKernelArgument::Boxed(value) => value.as_mut().as_mut_ptr(),
            OwnedKernelArgument::Inline(_) => unreachable!(),
        };

        assert_eq!(argument.as_mut_ptr(), expected);
    }
}
