//! CUDA modules, device-code images, kernel arguments, and launch configuration.

mod config;
mod function;
mod image;
mod parameters;

pub mod artifact;

use std::{ffi::CString, ptr, result::Result as StdResult, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::driver;

use crate::{
    context::Context,
    error::{Error, Result},
    library::Library,
    module::artifact::ArtifactMetadata,
    try_ffi,
    types::DeviceFunction,
    utility::to_usize,
};

pub use self::{
    config::{LaunchAttribute, LaunchConfig, LaunchGeometry},
    function::{
        FunctionAttributes, KernelFunction, OccupancyFlags, OccupancyMaxPotentialBlockSize,
    },
    image::ModuleImage,
    parameters::{EncodedKernelArgs, KernelLaunchArgs, KernelParameters, PushKernelArg},
};

/// Raw ownership of a loaded CUDA module and its context.
///
/// Unload `handle` while `context` is alive, or re-adopt the value with [`Module::from_raw`].
#[derive(Debug)]
pub struct RawModule {
    /// Native module handle.
    pub handle: driver::CUmodule,
    /// Context the module was loaded in.
    pub context: Arc<Context>,
}

/// A shared CUDA module and its retained loading context.
///
/// Clones, functions, and retained graph owners keep the module alive.
/// An owned native module is unloaded when its last owner is dropped.
/// Library-backed modules instead retain their library; externally borrowed modules rely on the external owner.
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
/// This view borrows the module to keep the variable loaded.
/// It does not establish host accessibility, initialization, or exclusive access to the storage.
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

impl Module {
    /// Takes ownership of a native module and its context association.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a live module loaded in `raw.context` and must not be unloaded elsewhere.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if the handle is null.
    pub unsafe fn from_raw(raw: RawModule) -> Result<Self> {
        // SAFETY: the caller guarantees `raw.handle` is a live module of `raw.context` that nothing else unloads, which `from_raw_with_metadata` requires.
        unsafe { Self::from_raw_with_metadata(raw.handle, raw.context, None) }
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
    ///
    /// `handle` must belong to `context` and stay loaded until every clone, function, prepared launch and graph retaining this wrapper has been released.
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
    /// Returns [`crate::error::Status::NotFound`] if the module has no kernel function named `name`.
    /// Also returns an error if `name` contains an interior NUL byte, the module context cannot be bound, CUDA rejects the lookup, or a previous asynchronous launch reports an error.
    pub fn function(&self, name: &str) -> Result<KernelFunction> {
        self.inner.context.bind()?;
        let c_name = CString::new(name)?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local, `self.inner.handle` is the live module this owner retains, and `c_name` is NUL-terminated for the call.
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
        let mut count = 0;
        // SAFETY: `count` is a writable local, and `self.inner.handle` is the live module this owner retains.
        unsafe {
            try_ffi!(driver::cuModuleGetFunctionCount(
                &raw mut count,
                self.inner.handle
            ))?;
        }
        to_usize(count, "module function count")
    }

    /// Returns the borrowed native module handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> driver::CUmodule {
        self.inner.handle
    }

    /// Consumes the module and returns the raw CUDA module handle without unloading it.
    ///
    /// The caller becomes responsible for eventually unloading the returned handle with CUDA while retaining the returned context.
    /// Returns the unchanged wrapper if functions, graphs or other clones retain this module, or if its native handle is externally owned or owned by a library.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the module is externally owned, owned by a library, or still shared with another clone, function, or graph owner.
    /// The returned wrapper remains usable and retains its ownership mode.
    pub fn into_raw(self) -> StdResult<RawModule, Self> {
        if !matches!(self.inner.ownership, ModuleOwnership::Owned) {
            return Err(self);
        }
        match Arc::try_unwrap(self.inner) {
            Ok(mut inner) => {
                inner.ownership = ModuleOwnership::External;
                Ok(RawModule {
                    handle: inner.handle,
                    context: Arc::clone(&inner.context),
                })
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
                CodeOwnerId::Module(Arc::as_ptr(&self.inner).addr())
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
    /// The returned [`Global`] borrows this module, so the module remains loaded for at least as long as the global reference is usable.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Status::NotFound`] if the module has no global variable named `name`.
    /// Also returns an error if `name` contains an interior NUL byte, the module context cannot be bound, CUDA rejects the lookup, or a previous asynchronous launch reports an error.
    pub fn global(&self, name: &str) -> Result<Global<'_>> {
        let c_name = CString::new(name)?;
        let mut ptr = 0;
        let mut size = 0;
        self.inner.context.bind()?;
        // SAFETY: `ptr` and `size` are writable locals, `self.inner.handle` is the live module this owner retains, and `c_name` is NUL-terminated for the call.
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
            report_cleanup_failure("bind the cuda context to unload", "cuda module", &error);
            return;
        }

        // SAFETY: the last owner of an owned module unloads `self.handle` once, with its context bound, after every function and graph retaining it has dropped.
        unsafe {
            if let Err(error) = try_ffi!(driver::cuModuleUnload(self.handle)) {
                report_cleanup_failure("unload", "cuda module", &error);
            }
        }
    }
}

// SAFETY: a CUDA module handle may be used and unloaded from any thread once its context is bound, and the other fields are `Send`.
unsafe impl Send for ModuleInner {}

// SAFETY: shared access only passes the immutable handle to thread-safe driver lookups; `ownership` changes only on an owned value in `into_raw`.
unsafe impl Sync for ModuleInner {}

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
