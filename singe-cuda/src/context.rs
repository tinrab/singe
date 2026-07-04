use std::{
    ffi::CString,
    fmt::{self, Display, Formatter},
    mem, ptr,
    sync::Arc,
};

use singe_cuda_sys::driver;

use crate::{
    device::Device,
    error::{Error, Result},
    graph::Graph,
    jit::JitOptions,
    library::Library,
    module::{Module, ModuleImage},
    nvrtc::{self, CompilationArtifact, OutputKind},
    try_ffi,
    types::Limit,
};

bitflags::bitflags! {
    /// Context creation flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ContextFlags: u32 {
        const SCHEDULE_AUTO = driver::CUctx_flags::CU_CTX_SCHED_AUTO as _;
        const SCHEDULE_SPIN = driver::CUctx_flags::CU_CTX_SCHED_SPIN as _;
        const SCHEDULE_YIELD = driver::CUctx_flags::CU_CTX_SCHED_YIELD as _;
        const SCHEDULE_BLOCKING_SYNC = driver::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC as _;
        const MAP_HOST = driver::CUctx_flags::CU_CTX_MAP_HOST as _;
        const LOCAL_MEMORY_RESIZE_TO_MAX = driver::CUctx_flags::CU_CTX_LMEM_RESIZE_TO_MAX as _;
        const COREDUMP_ENABLE = driver::CUctx_flags::CU_CTX_COREDUMP_ENABLE as _;
        const USER_COREDUMP_ENABLE = driver::CUctx_flags::CU_CTX_USER_COREDUMP_ENABLE as _;
        const SYNC_MEMORY_OPERATIONS = driver::CUctx_flags::CU_CTX_SYNC_MEMOPS as _;
    }
}

impl Display for ContextFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

/// A shared CUDA driver context.
///
/// Unlike cuBLAS, cuDNN, cuFFT, and similar library handles, a CUDA context is
/// the underlying execution environment for a device. It is intended to be
/// shared by streams, modules, libraries, events, allocations, and higher-level
/// library wrappers.
///
/// This type is therefore reference-counted by returning [`Arc<Self>`] from the
/// constructors, and it remains `Send + Sync`. Shared references do not mutate
/// Rust-visible state on the [`Context`] object itself; methods such as `bind`
/// update the calling thread's current CUDA context in the driver.
///
/// Prefer one long-lived context per device and share it across dependent CUDA
/// objects instead of creating many short-lived contexts.
#[derive(Debug)]
pub struct Context {
    handle: driver::CUcontext,
    device: Device,
    ownership: ContextOwnership,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextOwnership {
    Created,
    Primary,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RawContextOwnership {
    Created,
    Primary,
}

impl From<RawContextOwnership> for ContextOwnership {
    fn from(value: RawContextOwnership) -> Self {
        match value {
            RawContextOwnership::Created => Self::Created,
            RawContextOwnership::Primary => Self::Primary,
        }
    }
}

impl From<ContextOwnership> for RawContextOwnership {
    fn from(value: ContextOwnership) -> Self {
        match value {
            ContextOwnership::Created => Self::Created,
            ContextOwnership::Primary => Self::Primary,
        }
    }
}

impl Context {
    pub fn create() -> Result<Arc<Self>> {
        Self::create_with_flags(ContextFlags::empty())
    }

    pub fn create_with_flags(flags: ContextFlags) -> Result<Arc<Self>> {
        let device = Device::current()?;
        Self::create_for_device_with_flags(device, flags)
    }

    pub fn create_for_device(device: Device) -> Result<Arc<Self>> {
        Self::create_for_device_with_flags(device, ContextFlags::empty())
    }

    pub fn create_for_device_with_flags(device: Device, flags: ContextFlags) -> Result<Arc<Self>> {
        unsafe {
            try_ffi!(driver::cuInit(0))?;

            let mut handle = ptr::null_mut();
            try_ffi!(driver::cuCtxCreate_v4(
                &raw mut handle,
                ptr::null_mut(), // CUctxCreateParams
                flags.bits(),
                device.id() as _,
            ))?;

            if handle.is_null() {
                return Err(Error::NullHandle);
            }

            Ok(Arc::new(Self {
                handle,
                device,
                ownership: ContextOwnership::Created,
            }))
        }
    }

    pub fn retain_primary_for_device(device: Device) -> Result<Arc<Self>> {
        unsafe {
            try_ffi!(driver::cuInit(0))?;

            let mut handle = ptr::null_mut();
            try_ffi!(driver::cuDevicePrimaryCtxRetain(
                &raw mut handle,
                device.id() as _,
            ))?;

            if handle.is_null() {
                return Err(Error::NullHandle);
            }

            try_ffi!(driver::cuCtxSetCurrent(handle))?;

            Ok(Arc::new(Self {
                handle,
                device,
                ownership: ContextOwnership::Primary,
            }))
        }
    }

    /// Binds this CUDA context to the calling CPU thread.
    ///
    /// The "current context" is thread-local driver state. Calling this method
    /// does not mutate the Rust [`Context`] value itself; it makes this context
    /// current for subsequent CUDA driver and interoperating runtime calls on
    /// the current host thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Driver cannot query or set the current context.
    pub fn bind(&self) -> Result<()> {
        unsafe {
            let mut current_ctx = ptr::null_mut();
            try_ffi!(driver::cuCtxGetCurrent(&raw mut current_ctx))?;
            if current_ctx == self.as_raw() {
                return Ok(());
            }
            try_ffi!(driver::cuCtxSetCurrent(self.as_raw()))?;
        }
        Ok(())
    }

    /// Loads the corresponding module from the given image into the current context.
    /// The image may be a cubin or fatbin as output by **nvcc**, or a NUL-terminated PTX string, either as output by **nvcc** or hand-written, or Tile IR data.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot load the module, or a
    /// previous asynchronous launch reported an error.
    pub fn load_module(self: &Arc<Self>, image: &ModuleImage<'_>) -> Result<Module> {
        self.bind()?;

        unsafe {
            let mut module_handle = ptr::null_mut();
            try_ffi!(driver::cuModuleLoadData(
                &raw mut module_handle,
                image.as_ptr() as _,
            ))?;
            if module_handle.is_null() {
                return Err(Error::NullHandle);
            }
            Module::from_raw(module_handle, Arc::clone(self))
        }
    }

    /// Creates an empty CUDA graph associated with this context.
    ///
    /// Prefer this over [`RawGraph::create`](crate::graph::RawGraph::create)
    /// for ordinary Singe code. The returned graph carries its context
    /// association into instantiated executable graphs, allowing launches and
    /// uploads to reject streams from another context before calling CUDA.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot create the graph.
    pub fn create_graph(self: &Arc<Self>) -> Result<Graph> {
        Graph::create_in_context(Arc::clone(self))
    }

    pub fn unload_module(self: &Arc<Self>, module: Module) -> Result<()> {
        drop(module);
        Ok(())
    }

    /// Loads the corresponding module from the given image into the current context.
    /// The image may be a cubin or fatbin as output by **nvcc**, or a NUL-terminated PTX string, either as output by **nvcc** or hand-written, or Tile IR data.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot load the module, JIT options
    /// are rejected, or a previous asynchronous launch reported an error.
    pub fn load_module_with_options(
        self: &Arc<Self>,
        image: &ModuleImage<'_>,
        mut jit_options: JitOptions<'_>,
    ) -> Result<Module> {
        self.bind()?;

        let mut jit_options = jit_options.build();
        unsafe {
            let mut module_handle = ptr::null_mut();
            try_ffi!(driver::cuModuleLoadDataEx(
                &raw mut module_handle,
                image.as_ptr() as _,
                jit_options.names.len() as _,
                jit_options.names.as_mut_ptr() as _,
                jit_options.values.as_mut_ptr() as _,
            ))?;
            if module_handle.is_null() {
                return Err(Error::NullHandle);
            }
            Module::from_raw(module_handle, Arc::clone(self))
        }
    }

    pub fn load_nvrtc_module(
        self: &Arc<Self>,
        program: &nvrtc::Program,
        output: OutputKind,
    ) -> Result<Module> {
        self.load_nvrtc_module_with_options(program, output, JitOptions::default())
    }

    pub fn load_nvrtc_module_with_options(
        self: &Arc<Self>,
        program: &nvrtc::Program,
        output: OutputKind,
        jit_options: JitOptions<'_>,
    ) -> Result<Module> {
        let image = module_loadable_image(program.artifact(output)?)?;
        self.load_module_with_options(&image, jit_options)
    }

    pub fn load_library(self: &Arc<Self>, image: &ModuleImage<'_>) -> Result<Library> {
        self.load_library_with_options(image, JitOptions::default())
    }

    /// Loads the corresponding library from the given image based on the application defined library loading mode:
    ///
    /// * If module loading is set to EAGER by the environment variables described in "Module loading", the library is loaded eagerly into all contexts at the time of the call and future contexts at the time of creation until the library
    ///   is unloaded with [`sys::cuLibraryUnload`](singe_cuda_sys::driver::cuLibraryUnload).
    /// * If the environment variables are set to LAZY, the library is not immediately loaded into existing contexts and is loaded only when a function is needed for that context,
    ///   such as a kernel launch.
    ///
    /// These environment variables are described in the CUDA programming guide under the "CUDA environment variables" section.
    ///
    /// The code may be a cubin or fatbin emitted by **nvcc**, a NUL-terminated PTX string emitted by **nvcc** or written by hand, or Tile IR data.
    /// A fatbin must also contain relocatable code when doing separate compilation.
    ///
    /// If the library contains managed variables and no device in the system supports them, this call returns [`crate::error::Status::NotSupported`].
    pub fn load_library_with_options(
        self: &Arc<Self>,
        image: &ModuleImage<'_>,
        mut jit_options: JitOptions<'_>,
    ) -> Result<Library> {
        self.bind()?;

        let mut jit_options = jit_options.build();
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(driver::cuLibraryLoadData(
                &raw mut handle,
                image.as_ptr() as _,
                jit_options.names.as_mut_ptr() as _,
                jit_options.values.as_mut_ptr() as _,
                jit_options.names.len() as _,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        unsafe { Library::from_raw(handle, Arc::clone(self)) }
    }

    pub fn load_nvrtc_library(
        self: &Arc<Self>,
        program: &nvrtc::Program,
        output: OutputKind,
    ) -> Result<Library> {
        self.load_nvrtc_library_with_options(program, output, JitOptions::default())
    }

    pub fn load_nvrtc_library_with_options(
        self: &Arc<Self>,
        program: &nvrtc::Program,
        output: OutputKind,
        jit_options: JitOptions<'_>,
    ) -> Result<Library> {
        let image = library_loadable_image(program.artifact(output)?)?;
        self.load_library_with_options(&image, jit_options)
    }

    /// Loads the corresponding library from the given file based on the application defined library loading mode:
    ///
    /// * If module loading is set to EAGER by the environment variables described in "Module loading", the library is loaded eagerly into all contexts at the time of the call and future contexts at the time of creation until the library
    ///   is unloaded with [`sys::cuLibraryUnload`](singe_cuda_sys::driver::cuLibraryUnload).
    /// * If the environment variables are set to LAZY, the library is not immediately loaded into existing contexts and is loaded only when a function is needed for that context,
    ///   such as a kernel launch.
    ///
    /// These environment variables are described in the CUDA programming guide under the "CUDA environment variables" section.
    ///
    /// The file must be a cubin emitted by **nvcc**, a PTX file emitted by **nvcc** or written by hand, a fatbin emitted by **nvcc** or written by hand, or a Tile IR file.
    /// A fatbin must also contain relocatable code when doing separate compilation.
    ///
    /// If the library contains managed variables and no device in the system supports them, this call returns [`crate::error::Status::NotSupported`].
    ///
    /// # Errors
    ///
    /// Returns an error if this context cannot be bound, if `path` contains an
    /// interior NUL byte, or if CUDA Driver cannot load the library.
    pub fn load_library_from_file(self: &Arc<Self>, path: &str) -> Result<Library> {
        self.bind()?;
        let path = CString::new(path)?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(driver::cuLibraryLoadFromFile(
                &raw mut handle,
                path.as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        unsafe { Library::from_raw(handle, Arc::clone(self)) }
    }

    /// Blocks until the current context has completed all preceding requested tasks.
    /// If the current context is the primary context, child contexts that have been created are also synchronized.
    /// [`Context::synchronize`] returns an error if one of the preceding tasks failed.
    /// If the context was created with [`ContextFlags::SCHEDULE_BLOCKING_SYNC`], the CPU thread blocks until the GPU context has finished its work.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, a preceding task failed, or a previous
    /// asynchronous launch reported an error.
    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(driver::cuCtxSynchronize())?;
        }
        Ok(())
    }

    /// Returns the flags of the current context.
    /// See [`ContextFlags`] for flag values.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the flags, or a
    /// previous asynchronous launch reported an error.
    pub fn flags(&self) -> Result<ContextFlags> {
        self.bind()?;
        unsafe {
            let mut flags = 0;
            try_ffi!(driver::cuCtxGetFlags(&raw mut flags))?;
            Ok(ContextFlags::from_bits_truncate(flags))
        }
    }

    /// Returns the current size of limit.
    /// The supported [`Limit`] values are:
    ///
    /// * [`Limit::StackSize`]: stack size in bytes of each GPU thread.
    /// * [`Limit::PrintfFifoSize`]: size in bytes of the FIFO used by the `printf()` device system call.
    /// * [`Limit::MallocHeapSize`]: size in bytes of the heap used by the `malloc()` and `free()` device system calls.
    /// * [`Limit::DevRuntimeSyncDepth`]: maximum grid depth at which a thread can issue the device runtime call [`Device::synchronize`] to wait on child grid launches to complete.
    /// * [`Limit::DevRuntimePendingLaunchCount`]: maximum number of outstanding device runtime launches that can be made from this context.
    /// * [`Limit::MaxL2FetchGranularity`]: L2 cache fetch granularity.
    /// * [`Limit::PersistingL2CacheSize`]: persisting L2 cache size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `limit` is unsupported, CUDA cannot query
    /// the limit, or a previous asynchronous launch reported an error.
    pub fn limit(&self, limit: Limit) -> Result<usize> {
        self.bind()?;
        unsafe {
            let mut value = 0;
            try_ffi!(driver::cuCtxGetLimit(&raw mut value, limit.into()))?;
            Ok(value as usize)
        }
    }

    /// Setting limit to value is a request by the application to update the current limit maintained by the context.
    /// The driver may modify the requested value to meet hardware requirements, such as clamping to minimum or maximum values or rounding up to the nearest element size.
    /// Use [`Context::limit`] to query the effective value.
    ///
    /// Setting each [`Limit`] has its own restrictions.
    ///
    /// * [`Limit::StackSize`] controls the stack size in bytes of each GPU thread.
    ///   The driver automatically increases the per-thread stack size for each
    ///   kernel launch as needed.
    ///   This size is not reset back to the original value after each launch.
    ///   Setting this value will take
    ///   effect immediately, and if necessary, the device will block until all preceding requested tasks are complete.
    ///
    /// * [`Limit::PrintfFifoSize`] controls the size in bytes of the FIFO used by the `printf()` device system call.
    ///   Configure [`Limit::PrintfFifoSize`] before launching any kernel that uses the `printf()` device system call; otherwise [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// * [`Limit::MallocHeapSize`] controls the size in bytes of the heap used by the `malloc()` and `free()` device system calls.
    ///   Configure [`Limit::MallocHeapSize`] before launching any kernel that uses the `malloc()` or `free()` device system calls; otherwise [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// * [`Limit::DevRuntimeSyncDepth`] controls the maximum nesting depth of a grid at which a thread can safely call [`Device::synchronize`].
    ///   Setting this limit must be performed before any launch of a kernel that uses the device runtime and calls [`Device::synchronize`] above the default sync depth, two levels of grids.
    ///   Calls to [`Device::synchronize`] fail if this limit is violated.
    ///   This limit can be set smaller than the default or up to the maximum launch depth of 24.
    ///   Additional sync-depth levels require the driver to reserve large amounts of device memory that can no longer be used for application allocations.
    ///   If these reservations of device memory fail, [`Context::set_limit`] returns [`crate::error::Status::OutOfMemory`], and the limit can be reset to a lower value.
    ///   This limit is only applicable to devices of compute capability &lt; 9.0.
    ///   Setting this limit on devices of other compute capability versions returns [`crate::error::Status::UnsupportedLimit`].
    ///
    /// * [`Limit::DevRuntimePendingLaunchCount`] controls the maximum number of outstanding device runtime launches that can be made from the current context.
    ///   A grid is outstanding from launch until it is known to have completed.
    ///   Device runtime launches that violate this limit fail.
    ///   If a module using the device runtime needs more pending launches than the default 2048 launches, this limit can be increased.
    ///   Sustaining additional pending launches requires the driver to reserve larger amounts of device memory up front, which can no longer be used for allocations.
    ///   If these reservations fail, [`Context::set_limit`] returns [`crate::error::Status::OutOfMemory`], and the limit can be reset to a lower value.
    ///   This limit is only applicable to devices of compute capability 3.5 and higher.
    ///   Attempting to set this limit on devices of compute capability less than 3.5 returns [`crate::error::Status::UnsupportedLimit`].
    ///
    /// * [`Limit::MaxL2FetchGranularity`] controls the L2 cache fetch granularity.
    ///   Values can range from 0B to 128B.
    ///   Performance hint that may be ignored or clamped depending on the platform.
    ///
    /// * [`Limit::PersistingL2CacheSize`] controls size in bytes available for persisting L2 cache.
    ///   Performance hint that may be ignored or clamped depending on the platform.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `limit` is unsupported, CUDA rejects the
    /// requested value, or a previous asynchronous launch reported an error.
    pub fn set_limit(&self, limit: Limit, value: usize) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(driver::cuCtxSetLimit(limit.into(), value as _))?;
        }
        Ok(())
    }

    pub const fn device(&self) -> Device {
        self.device
    }

    pub const fn as_raw(&self) -> driver::CUcontext {
        self.handle
    }

    /// Takes ownership of a raw CUDA context.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA context for `device`, and no other Rust
    /// wrapper may own the same release responsibility. `ownership` must match
    /// how the context should be released: created contexts are destroyed with
    /// `cuCtxDestroy`, while primary contexts are released with
    /// `cuDevicePrimaryCtxRelease`.
    pub unsafe fn from_raw(
        handle: driver::CUcontext,
        device: Device,
        ownership: RawContextOwnership,
    ) -> Result<Arc<Self>> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Arc::new(Self {
            handle,
            device,
            ownership: ownership.into(),
        }))
    }

    /// Transfers ownership of the raw CUDA context to the caller.
    ///
    /// The caller becomes responsible for releasing the returned context
    /// according to the returned ownership mode.
    pub fn into_raw_parts(self) -> (driver::CUcontext, Device, RawContextOwnership) {
        let raw = (self.handle, self.device, self.ownership.into());
        mem::forget(self);
        raw
    }
}

// CUDA driver contexts are shared execution environments, not per-thread
// library handles. The Rust wrapper only stores the raw context pointer and the
// owning device, while current-context selection is maintained by CUDA as
// thread-local driver state.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let result = match self.ownership {
                ContextOwnership::Created => try_ffi!(driver::cuCtxDestroy_v2(self.handle)),
                ContextOwnership::Primary => {
                    try_ffi!(driver::cuDevicePrimaryCtxRelease_v2(self.device.id() as _))
                }
            };

            if let Err(err) = result {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy CUDA context wrapper: {err}");
            }
        }
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.as_raw() == other.as_raw()
    }
}

impl Eq for Context {}

fn module_loadable_image(artifact: CompilationArtifact) -> Result<ModuleImage<'static>> {
    match artifact {
        CompilationArtifact::Ptx(image) | CompilationArtifact::Cubin(image) => Ok(image),
        CompilationArtifact::LtoIr(_) | CompilationArtifact::OptixIr(_) => Err(Error::InvalidValue),
    }
}

fn library_loadable_image(artifact: CompilationArtifact) -> Result<ModuleImage<'static>> {
    match artifact {
        CompilationArtifact::Ptx(image) | CompilationArtifact::Cubin(image) => Ok(image),
        CompilationArtifact::LtoIr(_) | CompilationArtifact::OptixIr(_) => Err(Error::InvalidValue),
    }
}
