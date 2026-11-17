//! CUDA libraries, kernel lookup, and library-owned globals.

use std::{ffi::CString, ptr, result::Result as StdResult, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::driver;

use crate::{
    context::Context,
    error::{Error, Result},
    graph::{ExecutableGraph, Graph, GraphNode},
    kernel::{self, LibraryKernelHandle},
    module::{KernelFunction, KernelLaunchArgs, LaunchConfig, LaunchGeometry, Module},
    try_ffi,
    types::{DeviceFunction, FunctionAttribute, FunctionCache, SharedMemoryMode},
    utility::{to_i32, to_usize},
};

/// Raw ownership of a CUDA library and its lookup context.
///
/// Unload `handle` while `context` is alive, or re-adopt the value with [`Library::from_raw`].
#[derive(Debug)]
pub struct RawLibrary {
    /// Native library handle.
    pub handle: driver::CUlibrary,
    /// Context used for lookups in the library.
    pub context: Arc<Context>,
}

/// A shared CUDA library and its retained lookup context.
///
/// Clones, kernels, and modules share ownership.
/// The native library is unloaded when its last owner is dropped, unless ownership has been transferred with [`Self::into_raw`].
#[derive(Debug, Clone)]
pub struct Library {
    inner: Arc<LibraryInner>,
}

#[derive(Debug)]
struct LibraryInner {
    handle: driver::CUlibrary,
    context: Arc<Context>,
    owns_handle: bool,
}

/// The address and byte size of a global or managed variable in a CUDA library.
///
/// This view borrows the library to keep its code and variable storage loaded.
/// It does not establish host accessibility, initialization, or exclusive access to the variable.
#[derive(Debug, Clone, Copy)]
pub struct LibraryGlobal<'a> {
    ptr: *mut (),
    size: usize,
    _library: &'a Library,
}

/// A CUDA library kernel handle that keeps its library and lookup context alive.
#[derive(Debug, Clone)]
pub struct LibraryKernel {
    handle: driver::CUkernel,
    library: Library,
}

/// Byte layout of one kernel argument in the device-side parameter buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelParamInfo {
    /// Byte offset of this argument in the parameter buffer.
    pub offset: usize,
    /// Size of this argument in bytes.
    pub size: usize,
}

impl Library {
    /// Takes ownership of a CUDA library, retaining its lookup context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be live and must not be unloaded elsewhere.
    /// `raw.context` must support the library's device code.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if the handle is null.
    pub unsafe fn from_raw(raw: RawLibrary) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(LibraryInner {
                handle: raw.handle,
                context: raw.context,
                owns_handle: true,
            }),
        })
    }

    /// Returns the handle of the kernel with the given name located in this library.
    /// If kernel handle is not found, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte, if the CUDA context cannot be bound, if CUDA Driver cannot find the kernel, or if it returns a null handle.
    pub fn kernel(&self, name: &str) -> Result<LibraryKernel> {
        let c_name = CString::new(name)?;
        let mut handle = ptr::null_mut();
        self.inner.context.bind()?;
        // SAFETY: `handle` is a writable local, `self.inner.handle` is the live library this owner retains, and `c_name` is NUL-terminated for the call.
        unsafe {
            try_ffi!(driver::cuLibraryGetKernel(
                &raw mut handle,
                self.inner.handle,
                c_name.as_ptr(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(LibraryKernel {
            handle,
            library: self.clone(),
        })
    }

    /// Returns the number of kernels in this library.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if CUDA Driver cannot report the kernel count.
    pub fn kernel_count(&self) -> Result<usize> {
        let mut count = 0;
        self.inner.context.bind()?;
        // SAFETY: `count` is a writable local, and `self.inner.handle` is the live library this owner retains.
        unsafe {
            try_ffi!(driver::cuLibraryGetKernelCount(
                &raw mut count,
                self.inner.handle
            ))?;
        }
        to_usize(count, "library kernel count")
    }

    /// Returns the module handle associated with the current context located in this library.
    /// If module handle is not found, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if CUDA Driver cannot find the module, or if it returns a null handle.
    pub fn module(&self) -> Result<Module> {
        let mut handle = ptr::null_mut();
        self.inner.context.bind()?;
        // SAFETY: `handle` is a writable local, and `self.inner.handle` is the live library this owner retains.
        unsafe {
            try_ffi!(driver::cuLibraryGetModule(
                &raw mut handle,
                self.inner.handle
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Module::from_library(
            handle,
            Arc::clone(&self.inner.context),
            self.clone(),
        ))
    }

    /// Returns the base pointer and size of the global with the given name for the requested library and the current context.
    /// If no global for the requested name exists, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte, if the CUDA context cannot be bound, or if CUDA Driver cannot find the global.
    pub fn global(&self, name: &str) -> Result<LibraryGlobal<'_>> {
        let c_name = CString::new(name)?;
        let mut ptr = 0;
        let mut size = 0;
        self.inner.context.bind()?;
        // SAFETY: `ptr` and `size` are writable locals, `self.inner.handle` is the live library this owner retains, and `c_name` is NUL-terminated for the call.
        unsafe {
            try_ffi!(driver::cuLibraryGetGlobal(
                &raw mut ptr,
                &raw mut size,
                self.inner.handle,
                c_name.as_ptr(),
            ))?;
        }
        Ok(LibraryGlobal {
            ptr: ptr as *mut (),
            size: to_usize(size, "library byte count")?,
            _library: self,
        })
    }

    /// Returns the base pointer and size of the managed memory with the given name for the requested library.
    /// If no managed memory with the requested name exists, the call returns [`crate::error::Status::NotFound`].
    /// Managed memory for the library is shared across devices and is registered when the library is loaded into at least one context.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte, if the CUDA context cannot be bound, or if CUDA Driver cannot find the managed allocation.
    pub fn managed(&self, name: &str) -> Result<LibraryGlobal<'_>> {
        let c_name = CString::new(name)?;
        let mut ptr = 0;
        let mut size = 0;
        self.inner.context.bind()?;
        // SAFETY: `ptr` and `size` are writable locals, `self.inner.handle` is the live library this owner retains, and `c_name` is NUL-terminated for the call.
        unsafe {
            try_ffi!(driver::cuLibraryGetManaged(
                &raw mut ptr,
                &raw mut size,
                self.inner.handle,
                c_name.as_ptr(),
            ))?;
        }
        Ok(LibraryGlobal {
            ptr: ptr as *mut (),
            size: to_usize(size, "library byte count")?,
            _library: self,
        })
    }

    /// Returns the borrowed native library handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> driver::CUlibrary {
        self.inner.handle
    }

    /// Consumes the library and returns the raw CUDA library handle without unloading it.
    ///
    /// The caller becomes responsible for eventually unloading the returned handle with CUDA while retaining the returned context.
    /// Returns the unchanged wrapper while kernels, modules, graphs or other clones retain this library.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if another library clone, kernel, module, or retained graph owner still shares the library.
    /// The returned wrapper remains usable and retains ownership.
    pub fn into_raw(self) -> StdResult<RawLibrary, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(mut inner) => {
                inner.owns_handle = false;
                Ok(RawLibrary {
                    handle: inner.handle,
                    context: Arc::clone(&inner.context),
                })
            }
            Err(inner) => Err(Self { inner }),
        }
    }

    pub(crate) fn code_owner_id(&self) -> usize {
        Arc::as_ptr(&self.inner).addr()
    }

    /// Returns the context used for library lookups and device attributes.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.inner.context
    }
}

impl Drop for LibraryInner {
    fn drop(&mut self) {
        if !self.owns_handle {
            return;
        }

        if let Err(error) = self.context.bind() {
            report_cleanup_failure("bind the cuda context to unload", "cuda library", &error);
            return;
        }

        // SAFETY: the last owner of `self.handle` unloads it once, with its lookup context bound, and no kernel or module that uses it remains.
        if let Err(error) = unsafe { try_ffi!(driver::cuLibraryUnload(self.handle)) } {
            report_cleanup_failure("unload", "cuda library", &error);
        }
    }
}

// SAFETY: a CUDA library handle may be used and unloaded from any thread once its context is bound, and the retained context is `Send`.
unsafe impl Send for LibraryInner {}

// SAFETY: shared access only passes the immutable handle to thread-safe driver lookups; `owns_handle` changes only on an owned value.
unsafe impl Sync for LibraryInner {}

impl LibraryGlobal<'_> {
    /// Returns the variable's raw address without changing its access permissions.
    ///
    /// The address may refer to device-only memory; obtaining it does not make it safe to dereference on the host.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut () {
        self.ptr
    }

    /// Returns the size of the variable in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.size
    }
}

// SAFETY: a CUDA kernel handle is usable from any thread while its library is loaded, which the retained `library` guarantees.
unsafe impl Send for LibraryKernel {}

// SAFETY: `&self` methods only pass the immutable kernel handle to thread-safe driver calls; attribute setters race only as CUDA documents for concurrent configuration.
unsafe impl Sync for LibraryKernel {}

impl LibraryKernel {
    /// Returns the native kernel name, replacing invalid UTF-8 sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup context cannot be bound, CUDA cannot query the name, or CUDA returns a null name pointer.
    pub fn name(&self) -> Result<String> {
        // SAFETY: the kernel handle belongs to the module or library retained by `self`.
        unsafe {
            kernel::name::<LibraryKernelHandle>(self.library.inner.context.as_ref(), self.handle)
        }
    }

    /// Returns the device function handle for this kernel and the current context.
    /// If the handle is not found, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if CUDA Driver cannot find the function, or if it returns a null handle.
    pub fn function(&self) -> Result<KernelFunction> {
        self.library.inner.context.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local, and `self.handle` is a live kernel of the library `self` retains, with its context bound above.
        unsafe {
            try_ffi!(driver::cuKernelGetFunction(&raw mut handle, self.handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let module = self.library.module()?;
        // SAFETY: CUDA returned this function handle for the module retained by the kernel.
        Ok(unsafe { KernelFunction::from_raw(DeviceFunction::from_raw(handle), &module) })
    }

    /// Adds this kernel to `graph` as a kernel node.
    ///
    /// # Safety
    ///
    /// The caller must ensure every pointer value passed through `params` remains valid for every graph instantiation, update, and launch that can execute the created node.
    /// Mutable pointer arguments must remain exclusive for the work ordered by those launches.
    /// Arguments must match the function ABI and satisfy its value and access requirements.
    /// Keep the graph, executable or another code owner alive until queued executions complete.
    ///
    /// # Errors
    ///
    /// Propagates errors from resolving this kernel's function and module.
    /// Also returns an error if the graph belongs to another context, a dependency belongs to another graph, the context cannot be bound, CUDA rejects the node or its launch attributes.
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
        let function = self.function()?;
        // SAFETY: `function` is this kernel's function, and the caller guarantees the argument validity, exclusivity, and code-owner lifetime that `KernelFunction::add_to_graph` requires.
        unsafe { function.add_to_graph(graph, dependencies, config, params) }
    }

    /// Updates this kernel's parameters in an executable graph node.
    ///
    /// # Safety
    ///
    /// The caller must ensure every pointer value passed through `params` remains valid for every future launch that can execute `node`.
    /// Mutable pointer arguments must remain exclusive for the work ordered by those launches.
    /// Arguments must match the function ABI and satisfy its value and access requirements.
    /// Keep the graph, executable or another code owner alive until queued executions complete.
    /// Launch attributes are preserved; they must remain compatible with this kernel and geometry.
    ///
    /// # Errors
    ///
    /// Propagates errors from resolving this kernel's function and module.
    /// Also returns an error if the executable belongs to another context, the node does not belong to its source graph, the context cannot be bound, CUDA rejects the parameter update.
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
        let function = self.function()?;
        // SAFETY: `function` is this kernel's function, and the caller guarantees the argument validity, exclusivity, code-owner lifetime, and attribute compatibility that `KernelFunction::set_graph_node_params` requires.
        unsafe { function.set_graph_node_params(executable, node, geometry, params) }
    }

    /// Returns `attribute` for this kernel on the library's lookup device.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup context cannot be bound or CUDA rejects the attribute query for this kernel and device.
    pub fn attribute(&self, attribute: FunctionAttribute) -> Result<i32> {
        // SAFETY: the kernel handle belongs to the module or library retained by `self`.
        unsafe {
            kernel::attribute::<LibraryKernelHandle>(
                self.library.inner.context.as_ref(),
                self.handle,
                attribute,
            )
        }
    }

    /// Specifies whether the kernel may use oversized shared memory configurations.
    ///
    /// # Errors
    ///
    /// Propagates errors from setting [`FunctionAttribute::SharedMemoryMode`], including a context binding failure or a mode unsupported by the kernel or device.
    pub fn set_shared_memory_mode(&self, mode: SharedMemoryMode) -> Result<()> {
        self.set_attribute(
            FunctionAttribute::SharedMemoryMode,
            to_i32(u32::from(mode), "shared memory mode")?,
        )
    }

    /// Returns the shared memory mode configured for this kernel.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying [`FunctionAttribute::SharedMemoryMode`].
    /// Returns `Error::UnknownEnumValue` if the returned native value is not a recognized [`SharedMemoryMode`].
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

    /// Sets `attribute` to `value` for this kernel on the library's lookup device.
    ///
    /// Only attributes supported for mutation by CUDA can be set; see [`FunctionAttribute`] for attribute-specific constraints.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup context cannot be bound or CUDA rejects the selected attribute or value for this kernel and device.
    pub fn set_attribute(&self, attribute: FunctionAttribute, value: i32) -> Result<()> {
        // SAFETY: the kernel handle belongs to the module or library retained by `self`.
        unsafe {
            kernel::set_attribute::<LibraryKernelHandle>(
                self.library.inner.context.as_ref(),
                self.handle,
                attribute,
                value,
            )
        }
    }

    /// Sets the preferred cache configuration for this kernel on devices where L1 cache and shared memory use the same hardware resources.
    /// This setting is only a preference.
    /// The driver uses the requested configuration if possible, but it may choose a different configuration if required to execute the kernel.
    /// This per-kernel setting overrides any context-wide preference set via [`sys::cuCtxSetCacheConfig`](singe_cuda_sys::driver::cuCtxSetCacheConfig).
    ///
    /// Attributes set using [`sys::cuFuncSetCacheConfig`](singe_cuda_sys::driver::cuFuncSetCacheConfig) override this preference regardless of call order.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.
    ///
    /// Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point.
    ///
    /// The supported cache configurations are:
    ///
    /// * [`FunctionCache::PreferNone`]: no preference for shared memory or L1 (default)
    /// * [`FunctionCache::PreferShared`]: prefer larger shared memory and smaller L1 cache
    /// * [`FunctionCache::PreferL1`]: prefer larger L1 cache and smaller shared memory
    /// * [`FunctionCache::PreferEqual`]: prefer equal sized L1 cache and shared memory
    ///
    /// This has stricter locking requirements than its legacy counterpart [`sys::cuFuncSetCacheConfig`](singe_cuda_sys::driver::cuFuncSetCacheConfig) because the setting has device-wide semantics.
    /// If multiple threads try to set a configuration on the same device simultaneously, the final cache configuration depends on OS scheduler interleaving and memory consistency.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if CUDA Driver rejects the cache configuration.
    pub fn set_cache_config(&self, config: FunctionCache) -> Result<()> {
        self.library.inner.context.bind()?;
        // SAFETY: `self.handle` is a live kernel of the library `self` retains, with its context bound above.
        unsafe {
            try_ffi!(driver::cuKernelSetCacheConfig(
                self.handle,
                config.into(),
                self.library.inner.context.device().id(),
            ))?;
        }
        Ok(())
    }

    /// Queries the kernel parameter at the given index, returning the offset and size where the parameter resides in the device-side parameter layout.
    /// Use this information to update kernel node parameters from the device.
    /// The index must be less than the number of parameters that the kernel takes.
    ///
    /// # Errors
    ///
    /// Returns an error if the library context cannot be bound, `index` is not a valid kernel parameter index, CUDA cannot query the parameter layout, or a previous asynchronous launch reported an error.
    pub fn param_info(&self, index: usize) -> Result<KernelParamInfo> {
        self.library.inner.context.bind()?;
        let mut offset = 0;
        let mut size = 0;
        let index = index
            .try_into()
            .map_err(|_| Error::out_of_range("kernel parameter index"))?;
        // SAFETY: `offset` and `size` are writable locals, and `self.handle` is a live kernel of the library `self` retains.
        unsafe {
            try_ffi!(driver::cuKernelGetParamInfo(
                self.handle,
                index,
                &raw mut offset,
                &raw mut size,
            ))?;
        }
        Ok(KernelParamInfo {
            offset: to_usize(offset, "kernel parameter offset")?,
            size: to_usize(size, "library byte count")?,
        })
    }

    /// Returns the borrowed native kernel handle.
    ///
    /// The handle is kept live by this kernel's library owner; the returned pointer does not retain that owner.
    #[must_use]
    pub const fn as_raw(&self) -> driver::CUkernel {
        self.handle
    }
}
