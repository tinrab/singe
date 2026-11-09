use std::{ffi::CString, ptr, sync::Arc};

use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{DebugMessageCategory, DebugMessageSeverity},
    utility::{NativeStruct, optional_cstring},
};

/// Settings for creating an [`Instance`].
#[derive(Debug)]
pub struct InstanceCreateInfo {
    raw: sys::nvimgcodecInstanceCreateInfo_t,
    extension_modules_path: Option<CString>,
}

impl Default for InstanceCreateInfo {
    fn default() -> Self {
        Self {
            raw: sys::nvimgcodecInstanceCreateInfo_t::init(),
            extension_modules_path: None,
        }
    }
}

impl InstanceCreateInfo {
    /// Creates settings that load the built-in modules and discover extension modules, with the default extension path and no default debug messenger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to load the built-in modules; `true` by default.
    pub fn set_load_builtin_modules(&mut self, load: bool) -> &mut Self {
        self.raw.load_builtin_modules = i32::from(load);
        self
    }

    /// Sets whether to discover and load extension modules; `true` by default.
    pub fn set_load_extension_modules(&mut self, load: bool) -> &mut Self {
        self.raw.load_extension_modules = i32::from(load);
        self
    }

    /// Sets where to look for extension modules, or `None` for the default location.
    ///
    /// Separate several paths with `:` on Linux or `;` on Windows.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` contains a NUL byte.
    pub fn set_extension_modules_path(&mut self, path: Option<&str>) -> Result<&mut Self> {
        self.extension_modules_path = optional_cstring(path)?;
        self.raw.extension_modules_path = self
            .extension_modules_path
            .as_ref()
            .map_or(ptr::null(), |path| path.as_ptr());
        Ok(self)
    }

    /// Sets whether the instance creates nvImageCodec's default debug messenger.
    pub fn set_create_debug_messenger(&mut self, create: bool) -> &mut Self {
        self.raw.create_debug_messenger = i32::from(create);
        self
    }

    /// Sets which severities and categories of messages the default debug messenger reports.
    ///
    /// They take effect only if [`set_create_debug_messenger`](Self::set_create_debug_messenger) is enabled.
    pub fn set_debug_message_filters(
        &mut self,
        severity: DebugMessageSeverity,
        category: DebugMessageCategory,
    ) -> &mut Self {
        self.raw.message_severity = severity.bits();
        self.raw.message_category = category.bits();
        self
    }

    /// Returns a pointer to the native settings, valid while `self` is.
    #[must_use]
    pub fn as_raw(&self) -> *const sys::nvimgcodecInstanceCreateInfo_t {
        &raw const self.raw
    }
}

impl Clone for InstanceCreateInfo {
    fn clone(&self) -> Self {
        let extension_modules_path = self.extension_modules_path.clone();
        let mut raw = self.raw;
        raw.extension_modules_path = extension_modules_path
            .as_ref()
            .map_or(ptr::null(), |path| path.as_ptr());

        Self {
            raw,
            extension_modules_path,
        }
    }
}

/// Loaded nvImageCodec library, which code streams, images, and codecs are created from.
///
/// Everything created from an instance keeps it alive, so the library is shut down only after the last of them is dropped.
#[derive(Debug)]
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

// SAFETY: shared owners only pass the instance handle to object-creation calls and never
// mutate it; the last `Arc` owner destroys it exactly once.
unsafe impl Send for Handle {}

unsafe impl Sync for Handle {}

impl InstanceRef {
    pub(crate) fn require_same(&self, other: &Self) -> Result<()> {
        if Arc::ptr_eq(&self.handle, &other.handle) {
            Ok(())
        } else {
            Err(Error::InstanceMismatch)
        }
    }

    pub(crate) fn as_raw(&self) -> sys::nvimgcodecInstance_t {
        self.handle.raw
    }
}

impl Instance {
    /// Creates an instance with `create_info`.
    ///
    /// # Errors
    ///
    /// Returns an error if nvImageCodec fails to initialize.
    pub fn create(create_info: &InstanceCreateInfo) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecInstanceCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecInstanceCreate(
                &raw mut raw,
                create_info.as_raw()
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle { raw }),
        })
    }

    /// Creates an instance that loads the built-in modules and discovers extension modules.
    ///
    /// # Errors
    ///
    /// Returns an error if nvImageCodec fails to initialize.
    pub fn create_default() -> Result<Self> {
        Self::create(&InstanceCreateInfo::default())
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
    /// `raw` must be a valid instance that is owned by nobody else; it is destroyed with `nvimgcodecInstanceDestroy` when the last owner drops.
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
    /// Returns an error if a code stream, image, codec, or other value still shares the instance.
    pub fn into_raw(self) -> Result<sys::nvimgcodecInstance_t> {
        let mut handle = Arc::try_unwrap(self.handle).map_err(|_| Error::OwnershipShared)?;
        let raw = handle.raw;
        handle.raw = ptr::null_mut();
        Ok(raw)
    }

    pub(crate) fn as_instance_ref(&self) -> InstanceRef {
        InstanceRef {
            handle: Arc::clone(&self.handle),
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `nvimgcodecInstanceDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvimgcodecInstanceDestroy(self.raw)) {
                    eprintln!(
                        "failed to free nvimagecodec handle: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn test_instance_create_info_clone_rebuilds_extension_path_pointer() -> Result<()> {
        let mut original = InstanceCreateInfo::new();
        original.set_extension_modules_path(Some("/tmp/nvimgcodec"))?;

        let cloned = original.clone();

        assert!(!original.raw.extension_modules_path.is_null());
        assert!(!cloned.raw.extension_modules_path.is_null());
        assert_ne!(
            original.raw.extension_modules_path,
            cloned.raw.extension_modules_path
        );
        assert_eq!(
            cloned.raw.extension_modules_path,
            cloned
                .extension_modules_path
                .as_ref()
                .ok_or_else(|| Error::Empty {
                    name: "extension_modules_path".into(),
                })?
                .as_ptr()
        );
        Ok(())
    }
}
