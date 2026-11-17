use std::{ffi::CString, mem, ptr, result::Result as StdResult, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{DebugMessageCategory, DebugMessageSeverity, NativeStruct},
    utility::optional_cstring,
};

/// Settings for creating an [`Instance`].
///
/// The default loads the built-in modules, discovers extension modules in the default location, and creates no debug messenger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceOptions {
    /// Whether to load the built-in modules.
    pub load_builtin_modules: bool,
    /// Whether to discover and load extension modules.
    pub load_extension_modules: bool,
    /// Where to look for extension modules, or `None` for the default location.
    ///
    /// Separate several paths with `:` on Linux or `;` on Windows.
    pub extension_modules_path: Option<String>,
    /// Whether the instance creates nvImageCodec's default debug messenger.
    pub create_debug_messenger: bool,
    /// Severities the default debug messenger reports, if it is created.
    pub debug_message_severity: DebugMessageSeverity,
    /// Categories the default debug messenger reports, if it is created.
    pub debug_message_category: DebugMessageCategory,
}

impl Default for InstanceOptions {
    fn default() -> Self {
        Self {
            load_builtin_modules: true,
            load_extension_modules: true,
            extension_modules_path: None,
            create_debug_messenger: false,
            debug_message_severity: DebugMessageSeverity::empty(),
            debug_message_category: DebugMessageCategory::empty(),
        }
    }
}

/// Loaded nvImageCodec library, which code streams, images, and codecs are created from.
///
/// Everything created from an instance keeps it alive, so the library is shut down only after the last of them is dropped.
#[derive(Debug, Clone)]
pub struct Instance {
    handle: Arc<Handle>,
}

#[derive(Debug)]
pub(crate) struct Handle {
    raw: sys::nvimgcodecInstance_t,
}

#[derive(Debug, Clone)]
pub(crate) struct InstanceRef {
    handle: Arc<Handle>,
}

// SAFETY: nvImageCodec does not bind an instance to its creating thread, so the last `Arc` owner can destroy it exactly once from whichever thread drops it.
unsafe impl Send for Handle {}

// SAFETY: shared owners only pass the instance handle to object-creation calls, which nvImageCodec allows concurrently, and never mutate it.
unsafe impl Sync for Handle {}

impl InstanceRef {
    pub(crate) fn into_instance(self) -> Instance {
        Instance {
            handle: self.handle,
        }
    }

    pub(crate) fn require_same(&self, other: &Self, name: &'static str) -> Result<()> {
        if Arc::ptr_eq(&self.handle, &other.handle) {
            Ok(())
        } else {
            Err(Error::ContextMismatch { name: name.into() })
        }
    }

    pub(crate) fn as_raw(&self) -> sys::nvimgcodecInstance_t {
        self.handle.raw
    }
}

impl Instance {
    /// Creates an instance with `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the extension module path contains a NUL byte or nvImageCodec fails to initialize.
    pub fn create(options: &InstanceOptions) -> Result<Self> {
        let extension_modules_path: Option<CString> =
            optional_cstring(options.extension_modules_path.as_deref())?;
        let create_info = sys::nvimgcodecInstanceCreateInfo_t {
            load_builtin_modules: i32::from(options.load_builtin_modules),
            load_extension_modules: i32::from(options.load_extension_modules),
            extension_modules_path: extension_modules_path
                .as_ref()
                .map_or(ptr::null(), |path| path.as_ptr()),
            create_debug_messenger: i32::from(options.create_debug_messenger),
            message_severity: options.debug_message_severity.bits(),
            message_category: options.debug_message_category.bits(),
            ..sys::nvimgcodecInstanceCreateInfo_t::init()
        };
        let mut raw = ptr::null_mut();
        // SAFETY: the output location is valid for writes, and `create_info` and the path it points to outlive the call.
        unsafe {
            try_ffi!(sys::nvimgcodecInstanceCreate(
                &raw mut raw,
                &raw const create_info
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle { raw }),
        })
    }

    /// Returns the native handle, which this instance still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecInstance_t {
        self.handle.raw
    }

    /// Takes ownership of a native instance created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid instance that is owned by nobody else; it is destroyed when the last owner drops.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvimgcodecInstance_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle { raw }),
        })
    }

    /// Releases the native handle to the caller, who must destroy it.
    ///
    /// # Errors
    ///
    /// Returns an error holding the instance unchanged if a code stream, image, codec, or other value still shares it.
    pub fn into_raw(self) -> StdResult<sys::nvimgcodecInstance_t, Self> {
        match Arc::try_unwrap(self.handle) {
            Ok(mut handle) => Ok(mem::replace(&mut handle.raw, ptr::null_mut())),
            Err(handle) => Err(Self { handle }),
        }
    }

    pub(crate) fn as_instance_ref(&self) -> InstanceRef {
        InstanceRef {
            handle: Arc::clone(&self.handle),
        }
    }

    pub(crate) fn into_instance_ref(self) -> InstanceRef {
        InstanceRef {
            handle: self.handle,
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `self.raw` is the live instance this handle owns, and every object created from it holds an owner of this `Arc`, so none outlives it.
            if let Err(error) = unsafe { try_ffi!(sys::nvimgcodecInstanceDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvimagecodec handle", &error);
            }
        }
    }
}
