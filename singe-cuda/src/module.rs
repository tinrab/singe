use crate::utility::to_usize;

pub mod artifact;

use std::{
    borrow::Cow,
    ffi,
    ffi::CString,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem::{MaybeUninit, align_of, size_of},
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use bitflags::parser;
use singe_cuda_sys::driver;

use crate::{
    context::Context,
    dim::Dim3,
    error::{Error, Result},
    graph::{ExecutableGraph, Graph, GraphNode},
    green::GreenStream,
    kernel::{self, ModuleKernelHandle},
    library::Library,
    memory::{DeviceMemory, ManagedMemory},
    module::artifact::ArtifactMetadata,
    stream::{GraphRecordable, Stream, StreamCaptureScope},
    try_ffi,
    types::{
        ClusterSchedulingPolicy, DeviceFunction, FunctionAttribute, SharedMemoryCarveout,
        SharedMemoryMode,
    },
    utility::{to_u32, to_u64},
    view::{DeviceInput, DeviceInputView, DeviceRepr, DeviceSliceMut, DeviceView, DeviceViewMut},
};

bitflags::bitflags! {
    /// Occupancy calculator flag
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct OccupancyFlags: u32 {
        /// Default behavior.
        const DEFAULT = driver::CUoccupancy_flags::CU_OCCUPANCY_DEFAULT as _;
        /// Assume global caching is enabled and cannot be automatically turned off.
        const DISABLE_CACHING_OVERRIDE = driver::CUoccupancy_flags::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE as _;
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
    /// PTX virtual architecture version, encoded as `major * 10 + minor`. May be zero for cubins compiled before CUDA 3.0.
    pub ptx_version: i32,
    /// Binary architecture version, encoded as `major * 10 + minor`. Legacy cubins without an encoded version report 10.
    pub binary_version: i32,
    /// Whether the function was compiled with the `-Xptxas --dlcm=ca` cache setting.
    pub cache_mode_ca: bool,
    /// Maximum dynamic shared memory permitted per block for this function, in bytes.
    pub max_dynamic_shared_size_bytes: i32,
    /// Preferred percentage of shared L1/shared-memory resources assigned to shared memory. This is a hint; CUDA may choose another ratio.
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

/// A shared CUDA module and its retained loading context.
///
/// Clones, functions, and retained graph owners keep the module alive. An owned native module is unloaded when its last owner is dropped. Library-backed modules instead retain their library; externally borrowed modules rely on the external owner.
#[derive(Debug, Clone)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

#[derive(Debug)]
struct ModuleInner {
    handle: driver::CUmodule,
    context: Arc<Context>,
    ownership: ModuleOwnership,
    artifact_metadata: Option<Arc<ArtifactMetadata>>,
}

#[derive(Debug)]
enum ModuleOwnership {
    Owned,
    External,
    Library { owner: Library },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CodeOwnerId {
    Module(usize),
    Library(usize),
}

/// The address and byte size of a global variable in a CUDA module.
///
/// This view borrows the module to keep the variable loaded. It does not establish host accessibility, initialization, or exclusive access to the storage.
#[derive(Debug, Clone, Copy)]
pub struct Global<'a> {
    ptr: *mut (),
    size: usize,
    _module: &'a Module,
}

/// A native texture reference borrowed from a module.
///
/// The module borrow keeps the reference loaded.
#[derive(Debug, Clone, Copy)]
pub struct TextureReference<'a> {
    handle: driver::CUtexref,
    _module: &'a Module,
}

/// A native surface reference borrowed from a module.
///
/// The module borrow keeps the reference loaded.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceReference<'a> {
    handle: driver::CUsurfref,
    _module: &'a Module,
}

/// Borrowed or owned bytes containing CUDA device code.
///
/// Constructors preserve the supplied bytes without validating their format or adding a NUL terminator. Module and library loaders accept self-describing cubin, fatbin, or Tile IR images, or NUL-terminated PTX text. The caller must supply the representation required by the loader.
#[derive(Debug, Clone)]
pub struct ModuleImage<'a> {
    data: Cow<'a, [u8]>,
}

/// A context-specific CUDA kernel function that retains its module and loading context.
#[derive(Debug, Clone)]
pub struct KernelFunction {
    handle: DeviceFunction,
    module: Module,
}

/// A kernel launch and its arguments, ready to record in a stream capture.
///
/// The operation borrows its function and configuration and owns its argument list. Construct it with [`KernelFunction::launch_operation`].
#[derive(Debug)]
pub struct KernelLaunchOperation<'kernel, 'config, P> {
    function: &'kernel KernelFunction,
    config: &'config LaunchConfig,
    params: P,
}

/// Grid, block and dynamic shared-memory dimensions, independent of launch attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchGeometry {
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_memory_bytes: usize,
}

/// Launch geometry and attributes used when submitting or constructing a kernel node.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    geometry: LaunchGeometry,
    attributes: Vec<LaunchAttribute>,
}

/// Dynamically built CUDA kernel argument list.
///
/// Use this builder when the argument list depends on runtime conditions:
///
/// ```ignore
/// let mut params = KernelParameters::new();
/// params.arg(&input_ptr).arg(&output_ptr).push(len);
/// // SAFETY: the buffers match the kernel's parameters and outlive the launch.
/// unsafe { function.launch(&config, params)? };
/// ```
///
/// For fixed argument lists, passing a tuple of references directly to
/// [`KernelFunction::launch`] avoids building a dynamic list:
///
/// ```ignore
/// // SAFETY: the buffers match the kernel's parameters and outlive the launch.
/// unsafe { function.launch(&config, (&input_ptr, &mut output_ptr, &len))? };
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
    /// Appends this value in the kernel ABI representation selected by its implementation.
    fn push_to(self, params: &mut KernelParameters<'_>);
}

/// Kernel launch arguments accepted by launch and graph-node APIs.
///
/// This sealed trait is implemented for [`KernelParameters`], `()`, and tuples
/// of shared or mutable references up to 16 elements.
pub trait KernelLaunchArgs<'a>: private::Sealed {
    #[doc(hidden)]
    fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R;
}

/// Encoded CUDA kernel arguments for one launch or graph-node update call.
///
/// This is an implementation detail of [`KernelLaunchArgs`]. CUDA receives a
/// temporary array of pointers to encoded argument values; callers should use
/// [`KernelParameters`] or generated launch methods instead of constructing raw
/// argument arrays directly.
#[derive(Debug)]
#[doc(hidden)]
pub struct EncodedKernelArgs<'a> {
    pointers: &'a mut [*mut ()],
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
    /// Takes ownership of a native module and its context association.
    ///
    /// # Safety
    /// `handle` must be a live module loaded in `context` and must not be unloaded elsewhere.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null. The caller must satisfy the remaining validity and ownership requirements.
    pub unsafe fn from_raw(handle: driver::CUmodule, context: Arc<Context>) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_raw_with_metadata(handle, context, None) }
    }

    pub(crate) unsafe fn from_raw_with_metadata(
        handle: driver::CUmodule,
        context: Arc<Context>,
        artifact_metadata: Option<Arc<ArtifactMetadata>>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(ModuleInner {
                handle,
                context,
                ownership: ModuleOwnership::Owned,
                artifact_metadata,
            }),
        })
    }

    /// Borrows an externally owned module.
    ///
    /// # Safety
    /// `handle` must belong to `context` and stay loaded until every clone, function,
    /// prepared launch and graph retaining this wrapper has been released.
    pub unsafe fn from_borrowed_raw(handle: driver::CUmodule, context: Arc<Context>) -> Self {
        Self {
            inner: Arc::new(ModuleInner {
                handle,
                context,
                ownership: ModuleOwnership::External,
                artifact_metadata: None,
            }),
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
    pub fn function(&self, name: &str) -> Result<KernelFunction> {
        self.inner.context.bind()?;
        let c_name = CString::new(name)?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cuModuleGetFunction` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(driver::cuModuleGetFunction(
                &raw mut handle,
                self.inner.handle,
                c_name.as_ptr()
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(KernelFunction {
            // SAFETY: CUDA returned this function handle for the module retained by `self`.
            handle: unsafe { DeviceFunction::from_raw(handle) },
            module: self.clone(),
        })
    }

    /// Returns the number of functions in this module.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Driver cannot report the function count.
    pub fn function_count(&self) -> Result<usize> {
        self.inner.context.bind()?;
        // SAFETY: `cuModuleGetFunctionCount` receives valid output locations.
        unsafe {
            let mut count = 0;
            try_ffi!(driver::cuModuleGetFunctionCount(
                &raw mut count,
                self.inner.handle
            ))?;
            Ok(count as usize)
        }
    }

    /// Returns the borrowed native module handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> driver::CUmodule {
        self.inner.handle
    }

    /// Consumes the module and returns the raw CUDA module handle without
    /// unloading it.
    ///
    /// The caller becomes responsible for eventually unloading the returned
    /// handle with CUDA while retaining the returned context.
    /// Returns the unchanged wrapper if functions, graphs or other clones retain this module,
    /// or if its native handle is externally owned or owned by a library.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the module is externally owned, owned by a library, or still shared with another clone, function, or graph owner. The returned wrapper remains usable and retains its ownership mode.
    pub fn into_raw(self) -> StdResult<(driver::CUmodule, Arc<Context>), Self> {
        if !matches!(self.inner.ownership, ModuleOwnership::Owned) {
            return Err(self);
        }
        match Arc::try_unwrap(self.inner) {
            Ok(mut inner) => {
                inner.ownership = ModuleOwnership::External;
                Ok((inner.handle, Arc::clone(&inner.context)))
            }
            Err(inner) => Err(Self { inner }),
        }
    }

    /// Returns the CUDA context in which this module was loaded.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.inner.context
    }

    pub(crate) fn code_owner_id(&self) -> CodeOwnerId {
        match &self.inner.ownership {
            ModuleOwnership::Library { owner } => CodeOwnerId::Library(owner.code_owner_id()),
            ModuleOwnership::Owned | ModuleOwnership::External => {
                CodeOwnerId::Module(Arc::as_ptr(&self.inner) as usize)
            }
        }
    }

    pub(crate) fn from_library(
        handle: driver::CUmodule,
        context: Arc<Context>,
        library: Library,
    ) -> Self {
        Self {
            inner: Arc::new(ModuleInner {
                handle,
                context,
                ownership: ModuleOwnership::Library { owner: library },
                artifact_metadata: None,
            }),
        }
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
        self.inner.context.bind()?;
        // SAFETY: `cuModuleGetGlobal_v2` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(driver::cuModuleGetGlobal_v2(
                &raw mut ptr,
                &raw mut size,
                self.inner.handle,
                c_name.as_ptr(),
            ))?;
        }
        Ok(Global {
            ptr: ptr as _,
            size: to_usize(size, "global byte count")?,
            _module: self,
        })
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if !matches!(self.ownership, ModuleOwnership::Owned) {
            return;
        }

        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind context before unloading module: {error}");
            return;
        }

        // SAFETY: `cuModuleUnload` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(driver::cuModuleUnload(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to unload cuda module: {error}");
            }
        }
    }
}

// SAFETY: CUDA modules are immutable after loading in this wrapper. Kernel/function
// lookups use shared references and CUDA owns internal synchronization.
unsafe impl Send for ModuleInner {}

unsafe impl Sync for ModuleInner {}

impl<'a> ModuleImage<'a> {
    /// Borrows `data` as a module image without validating or modifying its bytes.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data: Cow::Borrowed(data),
        }
    }

    /// Takes ownership of `data` as a module image without validating or modifying its bytes.
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self {
            data: Cow::Owned(data),
        }
    }

    /// Takes ownership of the string's UTF-8 bytes as a module image.
    ///
    /// No NUL terminator is appended. PTX text passed to CUDA's module or library loaders must already contain its terminating NUL byte.
    #[must_use]
    pub fn from_string(data: String) -> Self {
        Self::from_vec(data.into_bytes())
    }

    /// Returns a pointer to the image bytes, valid while the image storage remains alive.
    #[must_use]
    pub fn as_ptr(&self) -> *const () {
        self.data.as_ptr().cast()
    }

    /// Borrows the complete image bytes, including any supplied NUL terminator.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_ref()
    }
}

impl Global<'_> {
    /// Returns the variable's raw address.
    ///
    /// Obtaining the pointer does not make device-only memory accessible to the host.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut () {
        self.ptr
    }

    /// Returns the variable's size in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.size
    }
}

impl TextureReference<'_> {
    /// Returns the native texture reference borrowed from the module.
    #[must_use]
    pub const fn as_raw(&self) -> driver::CUtexref {
        self.handle
    }
}

impl SurfaceReference<'_> {
    /// Returns the native surface reference borrowed from the module.
    #[must_use]
    pub const fn as_raw(&self) -> driver::CUsurfref {
        self.handle
    }
}

// SAFETY: the retained module owns this immutable handle and CUDA synchronizes attribute operations.
unsafe impl Send for KernelFunction {}

unsafe impl Sync for KernelFunction {}

impl KernelFunction {
    /// Adopts a function handle while retaining its module.
    ///
    /// # Safety
    /// `handle` must identify a live function in `module` and its actual context.
    #[must_use]
    pub unsafe fn from_raw(handle: DeviceFunction, module: &Module) -> Self {
        Self {
            handle,
            module: module.clone(),
        }
    }

    /// Creates a stream operation that launches this kernel.
    ///
    /// # Safety
    ///
    /// If this operation is recorded during stream capture, CUDA copies kernel argument values into the captured graph.
    /// For pointer arguments, only the pointer address is copied.
    /// Arguments must match the kernel ABI and satisfy its access and value requirements.
    /// The caller must ensure every copied pointer value remains valid for every captured graph execution that can use this operation, and mutable pointer arguments must remain exclusive for the work ordered by those graph launches.
    /// Keep the captured graph, executable or another code owner alive until queued executions complete.
    pub const unsafe fn launch_operation<'kernel, 'config, P>(
        &'kernel self,
        config: &'config LaunchConfig,
        params: P,
    ) -> KernelLaunchOperation<'kernel, 'config, P> {
        KernelLaunchOperation {
            function: self,
            config,
            params,
        }
    }

    fn check_graph_context(&self, graph: &Graph) -> Result<()> {
        if matches!(graph.context(), Some(context) if context != &self.module.inner.context) {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    fn check_executable_graph_context(&self, executable: &ExecutableGraph) -> Result<()> {
        if matches!(executable.context(), Some(context) if context != &self.module.inner.context) {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
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
    /// # Safety
    ///
    /// Arguments must match the kernel ABI and satisfy its access and value requirements.
    /// Keep the function and all pointed-to storage alive until execution completes.
    /// Order conflicting accesses and preserve exclusive writable access during execution.
    /// Successful submission does not establish output initialization or completion.
    pub unsafe fn launch<'a, P>(&self, config: &LaunchConfig, params: P) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        self.module.inner.context.bind()?;
        // SAFETY: the caller upholds this function's safety contract.
        params.with_encoded_arguments(|mut arguments| unsafe {
            config.launch_raw(
                self.handle.as_raw(),
                ptr::null_mut(),
                arguments.as_mut_ptr().cast(),
            )?;
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
    /// # Safety
    ///
    /// Arguments must match the kernel ABI and satisfy its access and value requirements.
    /// Keep the function and all pointed-to storage alive until execution completes.
    /// Order conflicting accesses and preserve exclusive writable access during execution.
    /// Successful submission does not establish output initialization or completion.
    pub unsafe fn launch_on<'a, P>(
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
        // SAFETY: the caller upholds this function's safety contract.
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
    /// CUDA copies each kernel argument value during this call. Non-pointer
    /// argument values may be borrowed from stack or temporary storage that
    /// outlives this call. If an argument value is a pointer, CUDA stores only
    /// the pointer address. The caller must ensure every copied pointer value
    /// remains valid for every graph instantiation, update, and launch that can
    /// execute the created node. Mutable pointer arguments must remain exclusive
    /// for the work ordered by those launches.
    /// Arguments must match the function ABI and satisfy its value and access requirements.
    /// Keep the graph, executable or another code owner alive until queued executions complete.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphContextMismatch` if the graph is associated with another context. Also propagates context binding errors, dependency ownership errors, and CUDA failures when creating the kernel node or setting its launch attributes.
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
        // SAFETY: the caller upholds this function's safety contract.
        let node = unsafe { graph.add_kernel_node(dependencies, self.handle, config, params) }?;
        graph.retain_module(&self.module);
        Ok(node)
    }

    /// Updates this kernel's parameters in an executable graph node.
    ///
    /// # Safety
    ///
    /// CUDA copies each kernel argument value during this call. Non-pointer
    /// argument values may be borrowed from stack or temporary storage that
    /// outlives this call. If an argument value is a pointer, CUDA stores only
    /// the pointer address. The caller must ensure every copied pointer value
    /// remains valid for every future launch that can execute `node`. Mutable
    /// pointer arguments must remain exclusive for the work ordered by those
    /// launches.
    /// Arguments must match the function ABI and satisfy its value and access requirements.
    /// Keep the graph, executable or another code owner alive until queued executions complete.
    /// Launch attributes are preserved; they must remain compatible with this kernel and geometry.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphContextMismatch` if the executable is associated with another context. Also propagates context binding errors, node ownership errors, and CUDA failures when updating the node.
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
        // SAFETY: the caller upholds this function's safety contract.
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
    /// The requested size plus static shared memory must fit the device's opt-in per-block limit. This attribute does not enable oversized shared memory; use [`Self::set_shared_memory_mode`] for that.
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
    /// Returns an error if the module context cannot be bound or any constituent attribute query fails. No partial result is returned.
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

    /// Returns the maximum number of active blocks per streaming multiprocessor for `block_size` threads and the specified dynamic shared-memory bytes per block.
    ///
    /// Uses [`OccupancyFlags::DEFAULT`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::occupancy_max_active_blocks_per_multiprocessor_with_flags`].
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
        self.module.inner.context.bind()?;
        let dynamic_shared_memory_bytes =
            validate_dynamic_shared_memory_bytes(dynamic_shared_memory_bytes)?;
        let mut blocks = 0;
        // SAFETY: `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if the module context cannot be bound, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_available_dynamic_shared_memory_per_block(
        &self,
        num_blocks: i32,
        block_size: i32,
    ) -> Result<usize> {
        self.module.inner.context.bind()?;
        let mut bytes = 0;
        // SAFETY: `cuOccupancyAvailableDynamicSMemPerBlock` receives live handles owned by the wrappers and valid output locations.
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
    /// `block_size_limit` restricts the block size; zero adds no limit. Uses [`OccupancyFlags::DEFAULT`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::occupancy_max_potential_block_size_with_flags`].
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
        self.module.inner.context.bind()?;
        let dynamic_shared_memory_bytes =
            validate_dynamic_shared_memory_bytes(dynamic_shared_memory_bytes)?;
        let mut min_grid_size = 0;
        let mut block_size = 0;
        // SAFETY: `cuOccupancyMaxPotentialBlockSizeWithFlags` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if the module context cannot be bound, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_potential_cluster_size(&self, config: &LaunchConfig) -> Result<i32> {
        self.module.inner.context.bind()?;
        let mut cluster_size = 0;
        // SAFETY: `cuOccupancyMaxPotentialClusterSize` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if the module context cannot be bound, `config` does
    /// not specify a valid cluster size for this kernel, CUDA rejects the
    /// occupancy query, or a previous asynchronous launch reports an error.
    pub fn occupancy_max_active_clusters(&self, config: &LaunchConfig) -> Result<i32> {
        self.module.inner.context.bind()?;
        let mut clusters = 0;
        // SAFETY: `cuOccupancyMaxActiveClusters` receives live handles owned by the wrappers and valid output locations.
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

unsafe impl<'a, P> GraphRecordable for KernelLaunchOperation<'_, '_, P>
where
    P: KernelLaunchArgs<'a>,
{
    type Output = ();

    fn record(self, scope: &StreamCaptureScope<'_>) -> Result<Self::Output> {
        scope.retain_module(self.function.module());
        // SAFETY: constructing this operation requires valid ABI, access and replay lifetimes.
        unsafe {
            self.function
                .launch_on(self.config, self.params, scope.stream())
        }
    }
}

impl LaunchGeometry {
    /// Creates nonzero grid and block dimensions with a representable shared-memory byte count.
    ///
    /// Device and kernel limits are checked during preparation or by CUDA during submission.
    ///
    /// # Errors
    /// Returns an error for zero dimensions or a shared-memory byte count exceeding `u32::MAX`.
    pub fn create(grid_dim: Dim3, block_dim: Dim3, shared_memory_bytes: usize) -> Result<Self> {
        validate_dim3(grid_dim, "grid_dim")?;
        validate_dim3(block_dim, "block_dim")?;
        validate_shared_memory_bytes(shared_memory_bytes)?;
        Ok(Self::from_validated(
            grid_dim,
            block_dim,
            shared_memory_bytes,
        ))
    }

    const fn from_validated(grid_dim: Dim3, block_dim: Dim3, shared_memory_bytes: usize) -> Self {
        Self {
            grid_dim,
            block_dim,
            shared_memory_bytes,
        }
    }

    /// Returns the number of blocks along each grid axis.
    #[must_use]
    pub const fn grid_dim(&self) -> Dim3 {
        self.grid_dim
    }

    /// Returns the number of threads along each block axis.
    #[must_use]
    pub const fn block_dim(&self) -> Dim3 {
        self.block_dim
    }

    /// Returns the dynamic shared-memory bytes per block.
    #[must_use]
    pub const fn shared_memory_bytes(&self) -> usize {
        self.shared_memory_bytes
    }

    pub(crate) fn shared_memory_bytes_u32(&self) -> u32 {
        u32::try_from(self.shared_memory_bytes)
            .expect("shared memory byte count is validated on construction")
    }

    /// Replaces the dynamic shared-memory byte count.
    ///
    /// # Errors
    /// Returns an error if the byte count exceeds `u32::MAX`.
    pub fn with_shared_memory_bytes(mut self, shared_memory_bytes: usize) -> Result<Self> {
        validate_shared_memory_bytes(shared_memory_bytes)?;
        self.shared_memory_bytes = shared_memory_bytes;
        Ok(self)
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_1d_grid(element_count: usize, block_size: usize) -> Result<Self> {
        validate_block_dimension(block_size, "block_size")?;
        let grid_size = element_count.div_ceil(block_size);

        validate_grid_dimension(grid_size, "grid_size")?;

        Ok(Self::from_validated(
            Dim3::new(to_u32(grid_size, "grid_size")?, 1, 1),
            Dim3::new(to_u32(block_size, "block_size")?, 1, 1),
            0,
        ))
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_1d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_1d_grid`] to handle these errors.
    #[must_use]
    pub fn for_1d_grid(element_count: usize, block_size: usize) -> Self {
        Self::try_for_1d_grid(element_count, block_size)
            .expect("invalid 1d cuda launch configuration")
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block. Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_num_elems(element_count: usize, block_size: usize) -> Result<Self> {
        Self::try_for_1d_grid(element_count, block_size)
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block. Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_num_elems`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_num_elems`] to handle these errors.
    #[must_use]
    pub fn for_num_elems(element_count: usize, block_size: usize) -> Self {
        Self::try_for_num_elems(element_count, block_size)
            .expect("invalid cuda launch configuration")
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
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
        validate_grid_dimension(grid_x, "grid_x")?;
        validate_grid_dimension(grid_y, "grid_y")?;

        Ok(Self::from_validated(
            Dim3::new(to_u32(grid_x, "grid_x")?, to_u32(grid_y, "grid_y")?, 1),
            Dim3::new(
                to_u32(block_width, "block_width")?,
                to_u32(block_height, "block_height")?,
                1,
            ),
            0,
        ))
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_2d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_2d_grid`] to handle these errors.
    #[must_use]
    pub fn for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Self {
        Self::try_for_2d_grid(width, height, block_width, block_height)
            .expect("invalid 2d cuda launch configuration")
    }

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
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
        validate_grid_dimension(grid_x, "grid_x")?;
        validate_grid_dimension(grid_y, "grid_y")?;
        validate_grid_dimension(grid_z, "grid_z")?;

        Ok(Self::from_validated(
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

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_3d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_3d_grid`] to handle these errors.
    #[must_use]
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

impl LaunchConfig {
    /// Creates a launch configuration with the supplied grid dimensions, block dimensions, and dynamic shared-memory bytes per block.
    ///
    /// No launch attributes are set. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if any grid or block axis is zero, or `Error::OutOfRange` if `shared_memory_bytes` exceeds `u32::MAX`.
    pub fn create(grid_dim: Dim3, block_dim: Dim3, shared_memory_bytes: usize) -> Result<Self> {
        Ok(LaunchGeometry::create(grid_dim, block_dim, shared_memory_bytes)?.into())
    }

    /// Returns the geometry used by this launch, without its attributes.
    #[must_use]
    pub const fn geometry(&self) -> &LaunchGeometry {
        &self.geometry
    }

    /// Returns the number of blocks along each grid axis.
    #[must_use]
    pub const fn grid_dim(&self) -> Dim3 {
        self.geometry.grid_dim()
    }

    /// Returns the number of threads along each block axis.
    #[must_use]
    pub const fn block_dim(&self) -> Dim3 {
        self.geometry.block_dim()
    }

    /// Returns the dynamic shared-memory bytes per block.
    #[must_use]
    pub const fn shared_memory_bytes(&self) -> usize {
        self.geometry.shared_memory_bytes()
    }

    pub(crate) fn shared_memory_bytes_u32(&self) -> u32 {
        self.geometry.shared_memory_bytes_u32()
    }

    /// Replaces the dynamic shared-memory byte count per block.
    ///
    /// # Errors
    ///
    /// Returns `Error::OutOfRange` if `bytes` exceeds `u32::MAX`.
    pub fn with_shared_memory_bytes(mut self, bytes: usize) -> Result<Self> {
        self.geometry = self.geometry.with_shared_memory_bytes(bytes)?;
        Ok(self)
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. The configuration starts with no launch attributes. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_1d_grid(element_count: usize, block_size: usize) -> Result<Self> {
        Ok(LaunchGeometry::try_for_1d_grid(element_count, block_size)?.into())
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_1d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_1d_grid`] to handle these errors.
    #[must_use]
    pub fn for_1d_grid(element_count: usize, block_size: usize) -> Self {
        LaunchGeometry::for_1d_grid(element_count, block_size).into()
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block. Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. The configuration starts with no launch attributes. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_num_elems(element_count: usize, block_size: usize) -> Result<Self> {
        Ok(LaunchGeometry::try_for_num_elems(element_count, block_size)?.into())
    }

    /// Creates a one-dimensional launch covering `element_count` elements with `block_size` threads per block. Equivalent to [`Self::try_for_1d_grid`].
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_num_elems`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_num_elems`] to handle these errors.
    #[must_use]
    pub fn for_num_elems(element_count: usize, block_size: usize) -> Self {
        LaunchGeometry::for_num_elems(element_count, block_size).into()
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. The configuration starts with no launch attributes. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Result<Self> {
        Ok(LaunchGeometry::try_for_2d_grid(width, height, block_width, block_height)?.into())
    }

    /// Creates a two-dimensional launch covering `width` by `height` elements with the supplied block width and height.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_2d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_2d_grid`] to handle these errors.
    #[must_use]
    pub fn for_2d_grid(
        width: usize,
        height: usize,
        block_width: usize,
        block_height: usize,
    ) -> Self {
        LaunchGeometry::for_2d_grid(width, height, block_width, block_height).into()
    }

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Each grid axis rounds up to a whole block. Dynamic shared memory starts at zero. The configuration starts with no launch attributes. Device and kernel limits are checked during preparation or submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if an element extent or block dimension is zero, or `Error::OutOfRange` if a computed grid dimension or supplied block dimension exceeds `u32::MAX`.
    pub fn try_for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Result<Self> {
        Ok(LaunchGeometry::try_for_3d_grid(
            width,
            height,
            depth,
            block_width,
            block_height,
            block_depth,
        )?
        .into())
    }

    /// Creates a three-dimensional launch covering `width` by `height` by `depth` elements with the supplied block dimensions.
    ///
    /// Uses the same rounding, shared-memory defaults, and validation as [`Self::try_for_3d_grid`].
    ///
    /// # Panics
    ///
    /// Panics if an element extent or block dimension is zero, or if a computed grid dimension or supplied block dimension exceeds `u32::MAX`. Use [`Self::try_for_3d_grid`] to handle these errors.
    #[must_use]
    pub fn for_3d_grid(
        width: usize,
        height: usize,
        depth: usize,
        block_width: usize,
        block_height: usize,
        block_depth: usize,
    ) -> Self {
        LaunchGeometry::for_3d_grid(width, height, depth, block_width, block_height, block_depth)
            .into()
    }
}

impl From<LaunchGeometry> for LaunchConfig {
    fn from(geometry: LaunchGeometry) -> Self {
        Self {
            geometry,
            attributes: Vec::new(),
        }
    }
}

fn validate_dim3(value: Dim3, name: &str) -> Result<()> {
    validate_grid_dimension(value.x as usize, &format!("{name}.x"))?;
    validate_grid_dimension(value.y as usize, &format!("{name}.y"))?;
    validate_grid_dimension(value.z as usize, &format!("{name}.z"))?;
    Ok(())
}

fn validate_grid_dimension(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::ZeroValue {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}

fn validate_block_dimension(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(Error::ZeroValue {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}

fn validate_shared_memory_bytes(value: usize) -> Result<u32> {
    to_u32(value, "shared_memory_bytes")
}

fn validate_dynamic_shared_memory_bytes(value: usize) -> Result<u64> {
    to_u64(value, "dynamic_shared_memory_bytes")
}

impl<'a> KernelParameters<'a> {
    /// Creates an empty kernel argument list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arguments: Vec::new(),
        }
    }

    /// Appends a borrowed argument value without converting its Rust representation.
    ///
    /// The value stays borrowed for this list's lifetime. If it contains a device pointer, the borrow covers the pointer value rather than the pointed-to allocation.
    pub fn arg<T: 'a>(&mut self, value: &'a T) -> &mut Self {
        self.arguments.push(KernelParameter::Borrowed {
            ptr: ptr::from_ref(value).cast_mut().cast::<()>(),
            _marker: PhantomData,
        });
        self
    }

    /// Appends an exclusively borrowed argument value without converting its Rust representation.
    ///
    /// CUDA reads the value as a kernel argument; this does not request a write back to the host argument slot. A pointer argument still needs the pointee lifetime and access guarantees required by the launch.
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

    /// Appends an argument using its [`PushKernelArg`] implementation to encode the kernel ABI value.
    pub fn push<A: PushKernelArg>(&mut self, arg: A) -> &mut Self {
        arg.push_to(self);
        self
    }

    /// Encodes a device input address, including deferred graph inputs.
    ///
    /// This copies metadata only. The launch contract determines when the
    /// pointed-to values must be initialized and how access is ordered.
    pub fn device_input<T: DeviceRepr, S: DeviceInput<T> + ?Sized>(
        &mut self,
        slice: &S,
    ) -> &mut Self {
        // Kernels take the device address for slice-like wrappers; length is a
        // separate scalar argument when the kernel needs it.
        self.owned_arg(slice.as_device_ptr())
    }

    /// Appends a copied device address for a mutable slice.
    ///
    /// This does not append the slice length or retain the allocation. Keep the allocation alive and preserve exclusive writable access for every execution that uses the copied pointer.
    pub fn device_slice_mut<T: DeviceRepr, S: DeviceSliceMut<T> + ?Sized>(
        &mut self,
        slice: &mut S,
    ) -> &mut Self {
        self.owned_arg(slice.as_mut_ptr())
    }

    fn raw_pointers(&mut self) -> RawKernelPointers {
        RawKernelPointers::from_parameters(self.arguments.as_mut_slice())
    }
}

impl KernelParameter<'_> {
    fn as_mut_ptr(&mut self) -> *mut () {
        match self {
            Self::Borrowed { ptr, .. } => *ptr,
            Self::Owned(value) => value.as_mut_ptr(),
        }
    }
}

impl OwnedKernelArgument {
    fn from_value<T: Copy + 'static>(value: T) -> Self {
        InlineKernelArgument::from_value(value)
            .map_or_else(|| Self::Boxed(Box::new(value)), Self::Inline)
    }

    fn as_mut_ptr(&mut self) -> *mut () {
        match self {
            Self::Inline(value) => value.as_mut_ptr(),
            Self::Boxed(value) => value.as_mut().as_mut_ptr(),
        }
    }
}

impl InlineKernelArgument {
    /// Stores `value` inline, or returns `None` if it does not fit the inline storage.
    fn from_value<T: Copy>(value: T) -> Option<Self> {
        if size_of::<T>() > INLINE_KERNEL_ARGUMENT_BYTES || align_of::<T>() > align_of::<Self>() {
            return None;
        }
        let mut storage = Self {
            bytes: [MaybeUninit::uninit(); INLINE_KERNEL_ARGUMENT_BYTES],
        };
        // SAFETY: the checks above guarantee the storage is large enough and aligned for `T`.
        unsafe {
            ptr::write(storage.as_mut_ptr().cast::<T>(), value);
        }
        Some(storage)
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

impl EncodedKernelArgs<'_> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut *mut () {
        self.pointers.as_mut_ptr()
    }
}

impl<'a> KernelLaunchArgs<'a> for KernelParameters<'a> {
    fn with_encoded_arguments<R>(mut self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
        let mut pointers = self.raw_pointers();
        f(EncodedKernelArgs {
            pointers: pointers.as_mut_slice(),
        })
    }
}

impl private::Sealed for KernelParameters<'_> {}

impl<'a> KernelLaunchArgs<'a> for &mut KernelParameters<'a> {
    fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
        let mut pointers = self.raw_pointers();
        f(EncodedKernelArgs {
            pointers: pointers.as_mut_slice(),
        })
    }
}

impl private::Sealed for &mut KernelParameters<'_> {}

impl KernelLaunchArgs<'_> for () {
    fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
        let mut pointers: [*mut (); 0] = [];
        f(EncodedKernelArgs {
            pointers: &mut pointers,
        })
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
            fn with_encoded_arguments<R>(self, f: impl FnOnce(EncodedKernelArgs<'_>) -> R) -> R {
                #[allow(non_snake_case, reason = "tuple fields are bound to the macro's uppercase type parameter names")]
                let ($($arg,)+) = self;
                let mut pointers = [
                    $($arg.into_kernel_argument_ptr(),)+
                ];
                f(EncodedKernelArgs {
                    pointers: &mut pointers,
                })
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
                fn push_to(self, params: &mut KernelParameters<'_>) {
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
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut DeviceMemory<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_slice_mut(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &ManagedMemory<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut ManagedMemory<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_slice_mut(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for DeviceInputView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(&self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceInputView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for DeviceView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceView<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &DeviceViewMut<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_ptr());
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut DeviceViewMut<'_, T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.owned_arg(self.as_mut_ptr());
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
    fn test_boxed_owned_kernel_argument_points_to_inner_value() {
        let mut argument = OwnedKernelArgument::from_value(LargeArgument { words: [1, 2, 3] });
        assert!(matches!(argument, OwnedKernelArgument::Boxed(_)));

        let expected = match &mut argument {
            OwnedKernelArgument::Boxed(value) => value.as_mut().as_mut_ptr(),
            OwnedKernelArgument::Inline(_) => unreachable!(),
        };

        assert_eq!(argument.as_mut_ptr(), expected);
    }

    #[test]
    fn test_launch_config_rejects_zero_grid_dimensions() {
        let error = LaunchConfig::try_for_1d_grid(0, 128).unwrap_err();
        assert!(matches!(error, Error::ZeroValue { name } if name == "grid_size"));

        let error = LaunchConfig::create(Dim3::new(0, 1, 1), Dim3::new(128, 1, 1), 0).unwrap_err();
        assert!(matches!(error, Error::ZeroValue { name } if name == "grid_dim.x"));
    }

    #[test]
    fn test_launch_config_rejects_invalid_shared_memory_size() {
        let error = LaunchConfig::try_for_1d_grid(1, 128)
            .unwrap()
            .with_shared_memory_bytes(u32::MAX as usize + 1)
            .unwrap_err();
        assert!(matches!(error, Error::OutOfRange { name } if name == "shared_memory_bytes"));
    }

    #[test]
    fn test_launch_config_exposes_checked_shared_memory_u32() {
        let config = LaunchConfig::try_for_1d_grid(1, 128)
            .unwrap()
            .with_shared_memory_bytes(u32::MAX as usize)
            .unwrap();

        assert_eq!(config.shared_memory_bytes(), u32::MAX as usize);
        assert_eq!(config.shared_memory_bytes_u32(), u32::MAX);
    }

    #[test]
    fn test_occupancy_dynamic_shared_memory_uses_checked_driver_width() {
        assert_eq!(validate_dynamic_shared_memory_bytes(0).unwrap(), 0);
        assert_eq!(
            validate_dynamic_shared_memory_bytes(usize::MAX).unwrap(),
            usize::MAX as u64
        );
    }

    #[test]
    fn test_launch_config_uses_checked_construction() {
        let config = LaunchConfig::create(Dim3::new(1, 1, 1), Dim3::new(32, 1, 1), 0).unwrap();

        assert_eq!(config.grid_dim(), Dim3::new(1, 1, 1));
        assert_eq!(config.block_dim(), Dim3::new(32, 1, 1));
        assert_eq!(config.shared_memory_bytes(), 0);
    }
}

/// Per-launch execution configuration.
#[derive(Debug, Clone, Copy)]
pub enum LaunchAttribute {
    /// Whether the kernel is launched as a cooperative kernel.
    Cooperative(bool),
    /// Dimensions of the cluster in blocks.
    ClusterDimension(Dim3),
    /// Cluster scheduling policy preference.
    ClusterSchedulingPolicy(ClusterSchedulingPolicy),
    /// Shared memory configuration for the kernel launch.
    SharedMemoryMode(SharedMemoryMode),
    /// Priority of the kernel launch.
    Priority(i32),
    /// Preferred shared memory carveout percentage.
    PreferredSharedMemoryCarveout(u32),
}

impl LaunchAttribute {
    pub(crate) fn raw(self) -> driver::CUlaunchAttribute {
        use driver::CUlaunchAttributeID as Id;
        let mut raw = driver::CUlaunchAttribute::default();
        match self {
            Self::Cooperative(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
                raw.value.cooperative = i32::from(value);
            }
            Self::ClusterDimension(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
                raw.value.clusterDim = driver::CUlaunchAttributeValue_union__bindgen_ty_1 {
                    x: value.x,
                    y: value.y,
                    z: value.z,
                };
            }
            Self::ClusterSchedulingPolicy(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
                raw.value.clusterSchedulingPolicyPreference = value.into();
            }
            Self::SharedMemoryMode(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
                raw.value.sharedMemoryMode = value.into();
            }
            Self::Priority(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_PRIORITY;
                raw.value.priority = value;
            }
            Self::PreferredSharedMemoryCarveout(value) => {
                raw.id = Id::CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
                raw.value.sharedMemCarveout = value;
            }
        }
        raw
    }
}

impl LaunchConfig {
    /// Configures attributes for an extended kernel launch.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if a cluster dimension is zero, or `Error::InvalidValue` if the preferred shared-memory carveout exceeds 100 percent. Device and kernel compatibility are checked later during preparation or submission.
    pub fn with_attribute(mut self, attribute: LaunchAttribute) -> Result<Self> {
        match attribute {
            LaunchAttribute::ClusterDimension(value) => validate_dim3(value, "cluster_dim")?,
            LaunchAttribute::PreferredSharedMemoryCarveout(value) if value > 100 => {
                return Err(Error::InvalidValue);
            }
            _ => {}
        }
        let id = attribute.raw().id;
        self.attributes.retain(|existing| existing.raw().id != id);
        self.attributes.push(attribute);
        Ok(self)
    }

    /// Returns the configured launch attributes.
    #[must_use]
    pub fn attributes(&self) -> &[LaunchAttribute] {
        &self.attributes
    }

    pub(crate) fn with_raw_config<R>(
        &self,
        stream: driver::CUstream,
        f: impl FnOnce(&driver::CUlaunchConfig) -> Result<R>,
    ) -> Result<R> {
        let mut attributes = self.attributes.iter().map(|a| a.raw()).collect::<Vec<_>>();
        let config = driver::CUlaunchConfig {
            gridDimX: self.geometry.grid_dim.x,
            gridDimY: self.geometry.grid_dim.y,
            gridDimZ: self.geometry.grid_dim.z,
            blockDimX: self.geometry.block_dim.x,
            blockDimY: self.geometry.block_dim.y,
            blockDimZ: self.geometry.block_dim.z,
            sharedMemBytes: self.shared_memory_bytes_u32(),
            hStream: stream,
            attrs: attributes.as_mut_ptr(),
            numAttrs: to_u32(attributes.len(), "launch attributes")?,
        };
        f(&config)
    }

    pub(crate) unsafe fn launch_raw(
        &self,
        function: driver::CUfunction,
        stream: driver::CUstream,
        arguments: *mut *mut ffi::c_void,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        self.with_raw_config(stream, |config| unsafe {
            try_ffi!(driver::cuLaunchKernelEx(
                config,
                function,
                arguments,
                ptr::null_mut()
            ))
        })
    }
}

impl KernelFunction {
    /// Specifies whether the function may use oversized shared memory configurations.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the selected attribute or value.
    pub fn set_shared_memory_mode(&self, mode: SharedMemoryMode) -> Result<()> {
        self.set_attribute(FunctionAttribute::SharedMemoryMode, mode as i32)
    }

    /// Returns the shared memory mode configured for this function.
    ///
    /// # Errors
    ///
    /// Returns an error if the module context cannot be bound or CUDA rejects the requested function attribute. Returns `Error::InvalidValue` if CUDA returns an unrecognized shared-memory mode.
    pub fn shared_memory_mode(&self) -> Result<SharedMemoryMode> {
        SharedMemoryMode::try_from(self.attribute(FunctionAttribute::SharedMemoryMode)? as u32)
            .map_err(|_| Error::InvalidValue)
    }

    /// Invokes this kernel on a stream provisioned by a green context.
    /// # Safety
    ///
    /// Arguments must match the kernel ABI and satisfy its access and value requirements.
    /// Keep the function and all pointed-to storage alive until execution completes.
    /// Order conflicting accesses and preserve exclusive writable access during execution.
    /// Successful submission does not establish output initialization or completion.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if the stream's green context uses another device. Also returns an error if the module context cannot be bound, argument encoding fails, CUDA rejects the launch, or previous asynchronous work reports an error.
    pub unsafe fn launch_on_green<'a, P: KernelLaunchArgs<'a>>(
        &self,
        config: &LaunchConfig,
        params: P,
        stream: &GreenStream,
    ) -> Result<()> {
        if self.module.inner.context.device() != stream.context().device() {
            return Err(Error::StreamContextMismatch);
        }
        self.module.inner.context.bind()?;
        // SAFETY: the caller upholds this function's safety contract.
        params.with_encoded_arguments(|mut arguments| unsafe {
            config.launch_raw(
                self.handle.as_raw(),
                stream.as_raw(),
                arguments.as_mut_ptr().cast(),
            )
        })
    }
}
