use std::{
    any::Any,
    ffi::CString,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    mem, ptr,
    result::Result as StdResult,
    sync::{Arc, Mutex},
};

use bitflags::parser;
use singe_cuda_sys::driver;

use crate::{
    device::Device,
    error::{Error, Result},
    graph::Graph,
    jit::JitOptions,
    library::Library,
    module::{Module, ModuleImage, artifact::ArtifactMetadata},
    nvrtc::{self, CompilationArtifact, OutputKind},
    submission::{Access, AccessGuard, AccessMode, AccessTracker, lock_access},
    try_ffi,
    types::Limit,
    utility::{to_u32, to_usize},
    view::DeviceRegion,
};

bitflags::bitflags! {
    /// Context creation flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ContextFlags: u32 {
        /// Automatic scheduling.
        const SCHEDULE_AUTO = driver::CUctx_flags::CU_CTX_SCHED_AUTO as _;
        /// Set spin as default scheduling.
        const SCHEDULE_SPIN = driver::CUctx_flags::CU_CTX_SCHED_SPIN as _;
        /// Set yield as default scheduling.
        const SCHEDULE_YIELD = driver::CUctx_flags::CU_CTX_SCHED_YIELD as _;
        /// Set blocking synchronization as default scheduling.
        const SCHEDULE_BLOCKING_SYNC = driver::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC as _;
        /// Deprecated: This flag was deprecated as of CUDA 11.0 and it no longer has any effect. All contexts as of CUDA 3.2 behave as though the flag is enabled.
        const MAP_HOST = driver::CUctx_flags::CU_CTX_MAP_HOST as _;
        /// Keep local memory allocation after launch.
        const LOCAL_MEMORY_RESIZE_TO_MAX = driver::CUctx_flags::CU_CTX_LMEM_RESIZE_TO_MAX as _;
        /// Trigger coredumps from exceptions in this context.
        const COREDUMP_ENABLE = driver::CUctx_flags::CU_CTX_COREDUMP_ENABLE as _;
        /// Enable user pipe to trigger coredumps in this context.
        const USER_COREDUMP_ENABLE = driver::CUctx_flags::CU_CTX_USER_COREDUMP_ENABLE as _;
        /// Ensure synchronous memory operations on this context will synchronize.
        const SYNC_MEMORY_OPERATIONS = driver::CUctx_flags::CU_CTX_SYNC_MEMOPS as _;
    }
}

impl Display for ContextFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Flags for [`Context::enable_peer_access`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PeerAccessFlags: u32 {
        /// Default peer-access behavior.
        const DEFAULT = 0;
    }
}

impl Display for PeerAccessFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
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
    ownership: RawContextOwnership,
    pub(crate) access_tracker: Arc<Mutex<AccessTracker>>,
}

/// Native release responsibility, including a retained external ownership token.
#[derive(Clone)]
#[non_exhaustive]
pub enum RawContextOwnership {
    /// Destroys the context with `cuCtxDestroy` when the last wrapper is dropped.
    Created,
    /// Releases one retained primary-context reference with `cuDevicePrimaryCtxRelease`.
    Primary,
    /// Keeps an external owner alive without destroying or releasing the native context.
    Borrowed(
        /// External ownership token retained for the lifetime of this wrapper.
        Arc<dyn Any + Send + Sync>,
    ),
}

impl fmt::Debug for RawContextOwnership {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Created => "Created",
            Self::Primary => "Primary",
            Self::Borrowed(_) => "Borrowed",
        })
    }
}

impl Context {
    /// Holds submitted accesses out of the supplied ranges while host code runs.
    /// Nested tracked access returns an error instead of waiting on its own host access guard.
    ///
    /// # Errors
    ///
    /// Returns `Error::MemoryContextMismatch` for a range belonging to another context, `Error::ReentrantAccess` for nested tracked access, or `Error::AccessTrackingPoisoned` if tracking has failed or a tracking lock is poisoned. Also propagates CUDA synchronization or event-query errors and errors returned by `access`. The callback runs only after all required waits succeed.
    pub fn with_host_access<T>(
        &self,
        regions: &[DeviceRegion<'_>],
        access: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let _guard = self.acquire_host_access(regions)?;
        access()
    }

    /// Waits for conflicting submitted uses and declared readiness streams, then holds
    /// these access declarations against concurrent tracked submissions while `access` runs.
    /// The declarations retain context and ordering metadata, not allocation owners.
    ///
    /// # Safety
    /// Keep the declared storage live and obey each declaration's access mode.
    /// Any asynchronous work using these ranges must complete before the callback returns
    /// or unwinds. Nested tracked access is rejected rather than waiting on its own guard.
    ///
    /// # Errors
    ///
    /// Converts tracking errors into `E` if a nonportable access belongs to another context, access is reentrant, tracking has failed, or a tracking lock is poisoned. Also converts errors from synchronizing conflicting uses or readiness streams and querying completion events. Errors returned by `access` pass through unchanged; the callback runs only after all required waits succeed.
    pub unsafe fn with_synchronized_accesses<T, E>(
        &self,
        accesses: &[Access],
        access: impl FnOnce() -> StdResult<T, E>,
    ) -> StdResult<T, E>
    where
        E: From<Error>,
    {
        let _guard = self.acquire_accesses(accesses)?;
        access()
    }

    pub(crate) fn acquire_host_access(
        &self,
        regions: &[DeviceRegion<'_>],
    ) -> Result<AccessGuard<'_>> {
        let accesses: Vec<_> = regions
            .iter()
            .map(|&region| Access::new(region, AccessMode::ReadWrite))
            .collect();
        self.acquire_accesses(&accesses)
    }

    pub(crate) fn acquire_accesses(&self, accesses: &[Access]) -> Result<AccessGuard<'_>> {
        for access in accesses {
            access.validate_context(self)?;
        }
        let mut tracker = lock_access(&self.access_tracker, accesses)?;
        for access in accesses {
            tracker.synchronize_access(access)?;
            if let Some(stream) = &access.readiness {
                stream.synchronize()?;
            }
        }
        Ok(tracker)
    }

    /// Enables this actual context to directly access `peer` allocations.
    /// Already-enabled access succeeds. The permission is unidirectional and
    /// persists until explicitly disabled or either context is destroyed.
    /// Pool allocations additionally follow their pool's access descriptors.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` for nondefault flags. Also returns an error if this context cannot be bound or CUDA rejects peer access, for example because the device pair does not support it or the peer-access limit is reached. Already-enabled access succeeds.
    pub fn enable_peer_access(&self, peer: &Self, flags: PeerAccessFlags) -> Result<()> {
        if flags != PeerAccessFlags::DEFAULT {
            return Err(Error::InvalidValue);
        }
        self.bind()?;
        // SAFETY: `cuCtxEnablePeerAccess` receives live handles owned by the wrappers.
        let status = unsafe { driver::cuCtxEnablePeerAccess(peer.as_raw(), flags.bits()) };
        if status == driver::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            return Ok(());
        }
        try_ffi!(status)
    }

    /// Disables this actual context's direct access to `peer` allocations.
    ///
    /// # Safety
    /// Complete every access that relies on this permission, including graph
    /// replays, before disabling it. Do not use retained peer-copy objects or
    /// peer pointers again until access is re-enabled. Exclude concurrent uses.
    ///
    /// # Errors
    ///
    /// Returns an error if this context cannot be bound or CUDA cannot disable the permission, including when peer access is not enabled.
    pub unsafe fn disable_peer_access(&self, peer: &Self) -> Result<()> {
        self.bind()?;
        // SAFETY: `cuCtxDisablePeerAccess` receives live handles owned by the wrappers.
        unsafe { try_ffi!(driver::cuCtxDisablePeerAccess(peer.as_raw())) }
    }

    /// Creates a context for the calling thread's current runtime device using default context flags.
    ///
    /// The new context becomes current on this thread. The last owning wrapper destroys it.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the current runtime device or from [`Self::create_for_device`].
    pub fn create() -> Result<Arc<Self>> {
        Self::create_with_flags(ContextFlags::empty())
    }

    /// Creates a context for the calling thread's current runtime device with `flags`.
    ///
    /// The new context becomes current on this thread. The last owning wrapper destroys it.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the current runtime device or from [`Self::create_for_device_with_flags`].
    pub fn create_with_flags(flags: ContextFlags) -> Result<Arc<Self>> {
        let device = Device::current()?;
        Self::create_for_device_with_flags(device, flags)
    }

    /// Creates a context for `device` using default context flags.
    ///
    /// The new context becomes current on the calling thread. The last owning wrapper destroys it.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization or context creation fails, including invalid creation flags, an unavailable device, or insufficient resources. Returns `Error::NullHandle` if CUDA reports success with a null context handle.
    pub fn create_for_device(device: Device) -> Result<Arc<Self>> {
        Self::create_for_device_with_flags(device, ContextFlags::empty())
    }

    /// Creates a context for `device` with `flags` and makes it current on the calling thread.
    ///
    /// The last owning wrapper destroys this context. Creating another context for the same device creates a separate execution environment.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization or context creation fails, including invalid creation flags, an unavailable device, or insufficient resources. Returns `Error::NullHandle` if CUDA reports success with a null context handle.
    pub fn create_for_device_with_flags(device: Device, flags: ContextFlags) -> Result<Arc<Self>> {
        // SAFETY: `cuInit`, `cuCtxCreate_v4` receive valid output locations.
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
                ownership: RawContextOwnership::Created,
                access_tracker: AccessTracker::for_context(handle as usize),
            }))
        }
    }

    /// Retains the primary context for `device` and makes it current on the calling thread.
    ///
    /// The device's primary context is shared with the CUDA runtime. Dropping the last owning wrapper releases this retain; other native retainers may keep the context alive.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization, retaining the primary context, or setting it current fails. Returns `Error::NullHandle` if CUDA reports success with a null context handle.
    pub fn retain_primary_for_device(device: Device) -> Result<Arc<Self>> {
        // SAFETY: `cuInit`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent` receive valid output locations.
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
                ownership: RawContextOwnership::Primary,
                access_tracker: AccessTracker::for_context(handle as usize),
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
        // SAFETY: `cuCtxGetCurrent`, `cuCtxSetCurrent` receive live handles owned by the wrappers and valid output locations.
        unsafe {
            let mut current_context = ptr::null_mut();
            try_ffi!(driver::cuCtxGetCurrent(&raw mut current_context))?;
            if current_context == self.as_raw() {
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
        self.load_module_with_metadata(image, None)
    }

    pub(crate) fn load_module_with_metadata(
        self: &Arc<Self>,
        image: &ModuleImage<'_>,
        metadata: Option<Arc<ArtifactMetadata>>,
    ) -> Result<Module> {
        self.bind()?;

        // SAFETY: the image is a borrowed, NUL-terminated or self-describing module image, and the options arrays have matching lengths.
        unsafe {
            let mut module_handle = ptr::null_mut();
            try_ffi!(driver::cuModuleLoadData(
                &raw mut module_handle,
                image.as_ptr().cast(),
            ))?;
            Module::from_raw_with_metadata(module_handle, Arc::clone(self), metadata)
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

    /// Drops this module handle.
    ///
    /// The module remains loaded while other module handles or objects retaining it exist. Cleanup uses the module's own context; `self` is not used to select a context.
    ///
    /// # Errors
    ///
    /// This method always returns `Ok(())`. Native unloading failures during final cleanup are not returned.
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
        // SAFETY: the image is a borrowed, NUL-terminated or self-describing module image, and the options arrays have matching lengths.
        unsafe {
            let mut module_handle = ptr::null_mut();
            try_ffi!(driver::cuModuleLoadDataEx(
                &raw mut module_handle,
                image.as_ptr().cast(),
                to_u32(jit_options.names.len(), "JIT option count")?,
                jit_options.names.as_mut_ptr().cast(),
                jit_options.values.as_mut_ptr().cast(),
            ))?;
            Module::from_raw(module_handle, Arc::clone(self))
        }
    }

    /// Retrieves `output` from a compiled NVRTC program and loads it as a module using default JIT settings.
    ///
    /// Only PTX, cubin, and Tile IR can be loaded directly. LTO IR and `OptiX` IR are rejected; link LTO IR before loading it.
    ///
    /// # Errors
    ///
    /// Propagates errors from retrieving the selected NVRTC artifact or loading the module. Returns `Error::InvalidValue` for LTO IR or `OptiX` IR; artifact retrieval also rejects empty PTX and cubin output. Loading can fail if the context cannot be bound, CUDA rejects the image or JIT settings, the JIT option count exceeds `u32`, or CUDA returns a null module handle.
    pub fn load_nvrtc_module(
        self: &Arc<Self>,
        program: &nvrtc::Program,
        output: OutputKind,
    ) -> Result<Module> {
        self.load_nvrtc_module_with_options(program, output, JitOptions::default())
    }

    /// Retrieves `output` from a compiled NVRTC program and loads it as a module with `jit_options`.
    ///
    /// Only PTX, cubin, and Tile IR can be loaded directly. LTO IR and `OptiX` IR are rejected; link LTO IR before loading it.
    ///
    /// # Errors
    ///
    /// Propagates errors from retrieving the selected NVRTC artifact or loading the module. Returns `Error::InvalidValue` for LTO IR or `OptiX` IR; artifact retrieval also rejects empty PTX and cubin output. Loading can fail if the context cannot be bound, CUDA rejects the image or JIT settings, the JIT option count exceeds `u32`, or CUDA returns a null module handle.
    pub fn load_nvrtc_module_with_options(
        self: &Arc<Self>,
        program: &nvrtc::Program,
        output: OutputKind,
        jit_options: JitOptions<'_>,
    ) -> Result<Module> {
        let image = module_loadable_image(program.artifact(output)?)?;
        self.load_module_with_options(&image, jit_options)
    }

    /// Loads a CUDA library from `image` using default JIT settings.
    ///
    /// See [`Self::load_library_with_options`] for eager and lazy loading behavior. The returned library retains this context.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::load_library_with_options`].
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
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the JIT option count exceeds `u32`, or CUDA rejects the image, options, or required device features. Returns `Error::NullHandle` if CUDA reports success with a null library handle. CUDA may also report errors from previous asynchronous work.
    pub fn load_library_with_options(
        self: &Arc<Self>,
        image: &ModuleImage<'_>,
        mut jit_options: JitOptions<'_>,
    ) -> Result<Library> {
        self.bind()?;

        let mut jit_options = jit_options.build();
        let mut handle = ptr::null_mut();
        // SAFETY: `cuLibraryLoadData` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(driver::cuLibraryLoadData(
                &raw mut handle,
                image.as_ptr().cast(),
                jit_options.names.as_mut_ptr().cast(),
                jit_options.values.as_mut_ptr().cast(),
                to_u32(jit_options.names.len(), "JIT option count")?,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: CUDA just created this handle in this context, and nothing else owns it.
        unsafe { Library::from_raw(handle, Arc::clone(self)) }
    }

    /// Retrieves `output` from a compiled NVRTC program and loads it as a CUDA library using default JIT settings.
    ///
    /// Only PTX, cubin, and Tile IR can be loaded directly. LTO IR and `OptiX` IR are rejected; link LTO IR before loading it.
    ///
    /// See [`Self::load_library_with_options`] for eager and lazy loading behavior.
    ///
    /// # Errors
    ///
    /// Propagates errors from retrieving the selected NVRTC artifact or from [`Self::load_library_with_options`]. Returns `Error::InvalidValue` for LTO IR or `OptiX` IR; artifact retrieval also rejects empty PTX and cubin output.
    pub fn load_nvrtc_library(
        self: &Arc<Self>,
        program: &nvrtc::Program,
        output: OutputKind,
    ) -> Result<Library> {
        self.load_nvrtc_library_with_options(program, output, JitOptions::default())
    }

    /// Retrieves `output` from a compiled NVRTC program and loads it as a CUDA library with `jit_options`.
    ///
    /// Only PTX, cubin, and Tile IR can be loaded directly. LTO IR and `OptiX` IR are rejected; link LTO IR before loading it.
    ///
    /// See [`Self::load_library_with_options`] for eager and lazy loading behavior.
    ///
    /// # Errors
    ///
    /// Propagates errors from retrieving the selected NVRTC artifact or from [`Self::load_library_with_options`]. Returns `Error::InvalidValue` for LTO IR or `OptiX` IR; artifact retrieval also rejects empty PTX and cubin output.
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
        // SAFETY: `cuLibraryLoadFromFile` receives valid output locations and pointers into buffers borrowed for the call.
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
        // SAFETY: CUDA just created this handle in this context, and nothing else owns it.
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
        // SAFETY: `cuCtxSynchronize` takes no arguments.
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
        // SAFETY: `cuCtxGetFlags` receives valid output locations.
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
        // SAFETY: `cuCtxGetLimit` receives valid output locations.
        unsafe {
            let mut value = 0;
            try_ffi!(driver::cuCtxGetLimit(&raw mut value, limit.into()))?;
            to_usize(value, "context limit")
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
        // SAFETY: `cuCtxSetLimit` receives only plain values.
        unsafe {
            try_ffi!(driver::cuCtxSetLimit(limit.into(), value as _))?;
        }
        Ok(())
    }

    /// Returns the device associated with this context.
    #[must_use]
    pub const fn device(&self) -> Device {
        self.device
    }

    /// Returns the borrowed native context handle.
    ///
    /// The handle remains owned according to this wrapper's [`RawContextOwnership`]. Returning it does not retain the context or transfer release responsibility.
    #[must_use]
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
    /// Borrowed contexts retain an external owner instead of acquiring release responsibility.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null. Other handle validity and ownership requirements are the caller's responsibility.
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
            ownership,
            access_tracker: AccessTracker::for_context(handle as usize),
        }))
    }

    /// Transfers ownership of the raw CUDA context to the caller.
    ///
    /// The caller becomes responsible for releasing the returned context
    /// according to the returned ownership mode.
    /// For borrowed contexts, keep the returned external token live while using the raw handle.
    #[must_use]
    pub fn into_raw_parts(mut self) -> (driver::CUcontext, Device, RawContextOwnership) {
        let handle = mem::replace(&mut self.handle, ptr::null_mut());
        (handle, self.device, self.ownership.clone())
    }
}

// SAFETY: CUDA driver contexts are shared execution environments, not per-thread
// library handles. The Rust wrapper only stores the raw context pointer and the
// owning device, while current-context selection is maintained by CUDA as
// thread-local driver state.
unsafe impl Send for Context {}

unsafe impl Sync for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `cuCtxDestroy_v2`, `cuDevicePrimaryCtxRelease_v2` receive live handles owned by the wrappers.
        unsafe {
            let result = match &self.ownership {
                RawContextOwnership::Created => try_ffi!(driver::cuCtxDestroy_v2(self.handle)),
                RawContextOwnership::Primary => {
                    try_ffi!(driver::cuDevicePrimaryCtxRelease_v2(self.device.id() as _))
                }
                RawContextOwnership::Borrowed(_) => Ok(()),
            };

            if let Err(error) = result {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda context wrapper: {error}");
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

impl Hash for Context {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_raw().hash(state);
    }
}

fn module_loadable_image(artifact: CompilationArtifact) -> Result<ModuleImage<'static>> {
    match artifact {
        CompilationArtifact::Ptx(image)
        | CompilationArtifact::Cubin(image)
        | CompilationArtifact::TileIr(image) => Ok(image),
        CompilationArtifact::LtoIr(_) | CompilationArtifact::OptixIr(_) => Err(Error::InvalidValue),
    }
}

fn library_loadable_image(artifact: CompilationArtifact) -> Result<ModuleImage<'static>> {
    match artifact {
        CompilationArtifact::Ptx(image)
        | CompilationArtifact::Cubin(image)
        | CompilationArtifact::TileIr(image) => Ok(image),
        CompilationArtifact::LtoIr(_) | CompilationArtifact::OptixIr(_) => Err(Error::InvalidValue),
    }
}
