use std::{ffi::CString, mem::ManuallyDrop, ptr, sync::Arc};

use singe_cuda_sys::driver;

use crate::{
    context::Context,
    error::{Error, Result},
    graph::{ExecutableGraph, Graph, GraphNode},
    kernel::{self, LibraryKernelHandle},
    module::{KernelFunction, KernelLaunchArgs, LaunchConfig, Module},
    try_ffi,
    types::{DeviceFunction, FunctionAttribute, FunctionCache},
};

#[derive(Debug)]
pub struct Library {
    handle: driver::CUlibrary,
    ctx: Arc<Context>,
}

#[derive(Debug, Clone, Copy)]
pub struct LibraryGlobal<'a> {
    ptr: *mut (),
    size: usize,
    _library: &'a Library,
}

#[derive(Debug, Clone, Copy)]
pub struct LibraryKernel<'a> {
    handle: driver::CUkernel,
    library: &'a Library,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelParamInfo {
    pub offset: usize,
    pub size: usize,
}

impl Library {
    pub unsafe fn from_raw(handle: driver::CUlibrary, ctx: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, ctx })
    }

    /// Returns the handle of the kernel with the given name located in this library.
    /// If kernel handle is not found, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte, if the CUDA
    /// context cannot be bound, if CUDA Driver cannot find the kernel, or if it
    /// returns a null handle.
    pub fn kernel(&self, name: &str) -> Result<LibraryKernel<'_>> {
        let c_name = CString::new(name)?;
        let mut handle = ptr::null_mut();
        self.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuLibraryGetKernel(
                &raw mut handle,
                self.handle,
                c_name.as_ptr(),
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(LibraryKernel {
            handle,
            library: self,
        })
    }

    /// Returns the number of kernels in this library.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if CUDA Driver
    /// cannot report the kernel count.
    pub fn kernel_count(&self) -> Result<usize> {
        let mut count = 0;
        self.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuLibraryGetKernelCount(&raw mut count, self.handle))?;
        }
        Ok(count as usize)
    }

    /// Returns the module handle associated with the current context located in this library.
    /// If module handle is not found, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if CUDA Driver
    /// cannot find the module, or if it returns a null handle.
    pub fn module(&self) -> Result<Module> {
        let mut handle = ptr::null_mut();
        self.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuLibraryGetModule(&raw mut handle, self.handle))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(unsafe { Module::from_borrowed_raw(handle, Arc::clone(&self.ctx)) })
    }

    /// Returns the base pointer and size of the global with the given name for the requested library and the current context.
    /// If no global for the requested name exists, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte, if the CUDA
    /// context cannot be bound, or if CUDA Driver cannot find the global.
    pub fn global(&self, name: &str) -> Result<LibraryGlobal<'_>> {
        let c_name = CString::new(name)?;
        let mut ptr = 0;
        let mut size = 0;
        self.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuLibraryGetGlobal(
                &raw mut ptr,
                &raw mut size,
                self.handle,
                c_name.as_ptr(),
            ))?;
        }
        Ok(LibraryGlobal {
            ptr: ptr as *mut (),
            size: size as usize,
            _library: self,
        })
    }

    /// Returns the base pointer and size of the managed memory with the given name for the requested library.
    /// If no managed memory with the requested name exists, the call returns [`crate::error::Status::NotFound`].
    /// Managed memory for the library is shared across devices and is registered when the library is loaded into at least one context.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` contains an interior NUL byte, if the CUDA
    /// context cannot be bound, or if CUDA Driver cannot find the managed
    /// allocation.
    pub fn managed(&self, name: &str) -> Result<LibraryGlobal<'_>> {
        let c_name = CString::new(name)?;
        let mut ptr = 0;
        let mut size = 0;
        self.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuLibraryGetManaged(
                &raw mut ptr,
                &raw mut size,
                self.handle,
                c_name.as_ptr(),
            ))?;
        }
        Ok(LibraryGlobal {
            ptr: ptr as *mut (),
            size: size as usize,
            _library: self,
        })
    }

    /// Returns the pointer to the unified function named by `symbol`.
    /// If no unified function with that name exists, the call returns [`crate::error::Status::NotFound`].
    /// If no device in the system supports unified function pointers, the call may return [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if `symbol` contains an interior NUL byte, if the CUDA
    /// context cannot be bound, or if CUDA Driver cannot find the unified
    /// function.
    pub fn unified_function(&self, symbol: &str) -> Result<*mut ()> {
        let c_symbol = CString::new(symbol)?;
        let mut ptr = ptr::null_mut();
        self.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuLibraryGetUnifiedFunction(
                &raw mut ptr,
                self.handle,
                c_symbol.as_ptr(),
            ))?;
        }
        if ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(ptr.cast())
    }

    pub const fn as_raw(&self) -> driver::CUlibrary {
        self.handle
    }

    /// Consumes the library and returns the raw CUDA library handle without
    /// unloading it.
    ///
    /// The caller becomes responsible for eventually unloading the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> driver::CUlibrary {
        let library = ManuallyDrop::new(self);
        library.handle
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if let Err(err) = self.ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind context before unloading library: {err}");
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(driver::cuLibraryUnload(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to unload cuda library: {err}");
            }
        }
    }
}

impl LibraryGlobal<'_> {
    pub const fn as_ptr(&self) -> *mut () {
        self.ptr
    }

    pub const fn byte_len(&self) -> usize {
        self.size
    }
}

impl LibraryKernel<'_> {
    pub fn name(&self) -> Result<String> {
        kernel::name::<LibraryKernelHandle>(self.library.ctx.as_ref(), self.handle)
    }

    /// Returns the device function handle for this kernel and the current context.
    /// If the handle is not found, the call returns [`crate::error::Status::NotFound`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if CUDA Driver
    /// cannot find the function, or if it returns a null handle.
    pub fn function(&self) -> Result<DeviceFunction> {
        self.library.ctx.bind()?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(driver::cuKernelGetFunction(&raw mut handle, self.handle))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(unsafe { DeviceFunction::from_raw(handle) })
    }

    /// Adds this kernel to `graph` as a kernel node.
    ///
    /// # Safety
    ///
    /// The caller must ensure every pointer value passed through `params`
    /// remains valid for every graph instantiation, update, and launch that can
    /// execute the created node. Mutable pointer arguments must remain
    /// exclusive for the work ordered by those launches.
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
        let module = self.library.module()?;
        let function = unsafe { KernelFunction::from_raw(function, &module) };
        unsafe { function.add_to_graph(graph, dependencies, config, params) }
    }

    /// Updates this kernel's parameters in an executable graph node.
    ///
    /// # Safety
    ///
    /// The caller must ensure every pointer value passed through `params`
    /// remains valid for every future launch that can execute `node`. Mutable
    /// pointer arguments must remain exclusive for the work ordered by those
    /// launches.
    pub unsafe fn set_graph_node_params<'a, P>(
        &self,
        executable: &mut ExecutableGraph,
        node: GraphNode,
        config: &LaunchConfig,
        params: P,
    ) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        let function = self.function()?;
        let module = self.library.module()?;
        let function = unsafe { KernelFunction::from_raw(function, &module) };
        unsafe { function.set_graph_node_params(executable, node, config, params) }
    }

    pub fn attribute(&self, attribute: FunctionAttribute) -> Result<i32> {
        kernel::attribute::<LibraryKernelHandle>(self.library.ctx.as_ref(), self.handle, attribute)
    }

    pub fn set_attribute(&self, attribute: FunctionAttribute, value: i32) -> Result<()> {
        kernel::set_attribute::<LibraryKernelHandle>(
            self.library.ctx.as_ref(),
            self.handle,
            attribute,
            value,
        )
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
    /// Returns an error if the CUDA context cannot be bound or if CUDA Driver
    /// rejects the cache configuration.
    pub fn set_cache_config(&self, config: FunctionCache) -> Result<()> {
        self.library.ctx.bind()?;
        unsafe {
            try_ffi!(driver::cuKernelSetCacheConfig(
                self.handle,
                config.into(),
                self.library.ctx.device().id() as _,
            ))?;
        }
        Ok(())
    }

    /// Queries the kernel parameter at the given index, returning the offset and size where the parameter resides in the device-side parameter layout.
    /// Use this information to update kernel node parameters from the device. The index must be less than the number of parameters that the kernel takes.
    ///
    /// # Errors
    ///
    /// Returns an error if the library context cannot be bound, `index` is not a valid kernel
    /// parameter index, CUDA cannot query the parameter layout, or a previous asynchronous launch
    /// reported an error.
    pub fn param_info(&self, index: usize) -> Result<KernelParamInfo> {
        self.library.ctx.bind()?;
        let mut offset = 0;
        let mut size = 0;
        unsafe {
            try_ffi!(driver::cuKernelGetParamInfo(
                self.handle,
                index as _,
                &raw mut offset,
                &raw mut size,
            ))?;
        }
        Ok(KernelParamInfo {
            offset: offset as usize,
            size: size as usize,
        })
    }

    pub const fn as_raw(&self) -> driver::CUkernel {
        self.handle
    }
}
