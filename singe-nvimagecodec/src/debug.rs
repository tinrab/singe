use std::{
    ffi::c_void,
    fmt::{self, Debug, Formatter},
    ptr,
};

use singe_core::string_from_c_ptr;
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    instance::{Instance, InstanceRef},
    try_ffi,
    types::{DebugMessageCategory, DebugMessageSeverity},
    utility::{NativeStruct, debug_callback},
};

pub(crate) type DebugCallbackFn = dyn for<'a> Fn(DebugMessage<'a>) + Send + Sync + 'static;

/// Receives nvImageCodec's debug messages and passes them to a Rust callback.
///
/// It keeps its [`Instance`] alive and owns the callback.
#[derive(Debug)]
pub struct DebugMessenger {
    raw: sys::nvimgcodecDebugMessenger_t,
    _instance: InstanceRef,
    _callback: Option<Box<DebugCallback>>,
}

pub(crate) struct DebugCallback(pub(crate) Box<DebugCallbackFn>);

impl Debug for DebugCallback {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DebugCallback").finish_non_exhaustive()
    }
}

/// Debug message passed to a [`DebugMessenger`] callback, valid during the call.
#[derive(Debug, Clone, Copy)]
pub struct DebugMessage<'a> {
    severity: DebugMessageSeverity,
    category: DebugMessageCategory,
    raw: &'a sys::nvimgcodecDebugMessageData_t,
}

/// Native debug-messenger handle with the values that keep it valid, from [`DebugMessenger::into_raw`].
///
/// Destroy the handle before dropping this value, or pass it back to [`DebugMessenger::from_raw`].
pub struct RawDebugMessenger {
    /// Native debug-messenger handle.
    pub handle: sys::nvimgcodecDebugMessenger_t,
    owner: DebugMessenger,
}

impl DebugMessenger {
    /// Creates a messenger that calls `callback` for each message whose severity is in `severity` and whose category is in `category`.
    ///
    /// A panic in `callback` is caught and does not cross into nvImageCodec.
    ///
    /// # Errors
    ///
    /// Returns an error if nvImageCodec fails to create the messenger.
    pub fn create(
        instance: &Instance,
        severity: DebugMessageSeverity,
        category: DebugMessageCategory,
        callback: impl for<'a> Fn(DebugMessage<'a>) + Send + Sync + 'static,
    ) -> Result<Self> {
        let callback = Box::new(DebugCallback(Box::new(callback)));
        let mut desc = sys::nvimgcodecDebugMessengerDesc_t::init();
        desc.message_severity = severity.bits();
        desc.message_category = category.bits();
        desc.user_callback = Some(debug_callback);
        desc.user_data = ptr::from_ref::<DebugCallback>(callback.as_ref()) as *mut c_void;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvimgcodecDebugMessengerCreate` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvimgcodecDebugMessengerCreate(
                instance.as_raw(),
                &raw mut raw,
                &raw const desc,
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw,
            _instance: instance.as_instance_ref(),
            _callback: Some(callback),
        })
    }

    /// Returns the native handle, which this messenger still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecDebugMessenger_t {
        self.raw
    }

    /// Takes ownership of a native debug messenger created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `instance`, and owned by nobody else.
    /// If it has a callback, the caller must keep the callback's data alive separately.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw_parts(
        instance: &Instance,
        raw: sys::nvimgcodecDebugMessenger_t,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw,
            _instance: instance.as_instance_ref(),
            _callback: None,
        })
    }

    /// Takes back ownership of a handle from [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The handle must still be valid and owned by nobody else, with its original callback.
    #[must_use]
    pub unsafe fn from_raw(mut raw: RawDebugMessenger) -> Self {
        raw.owner.raw = raw.handle;
        raw.owner
    }

    /// Releases the native handle with the values that keep it valid, including the callback.
    ///
    /// Destroy the handle or pass it back to [`from_raw`](Self::from_raw).
    #[must_use]
    pub fn into_raw(mut self) -> RawDebugMessenger {
        let handle = self.raw;
        self.raw = ptr::null_mut();
        RawDebugMessenger {
            handle,
            owner: self,
        }
    }
}

impl DebugMessage<'_> {
    pub(crate) const fn from_raw(
        severity: DebugMessageSeverity,
        category: DebugMessageCategory,
        raw: &sys::nvimgcodecDebugMessageData_t,
    ) -> DebugMessage<'_> {
        DebugMessage {
            severity,
            category,
            raw,
        }
    }

    /// Returns the message's severity.
    #[must_use]
    pub const fn severity(&self) -> DebugMessageSeverity {
        self.severity
    }

    /// Returns the message's category.
    #[must_use]
    pub const fn category(&self) -> DebugMessageCategory {
        self.category
    }

    /// Returns the message text, with invalid UTF-8 replaced and an empty string for a missing message.
    #[must_use]
    pub fn message(&self) -> String {
        // SAFETY: `string_from_c_ptr` accepts null, and non-null values are NUL-terminated strings owned by the library.
        unsafe { string_from_c_ptr(self.raw.message) }
    }

    /// Returns nvImageCodec's internal status ID for the message.
    #[must_use]
    pub const fn internal_status_id(&self) -> u32 {
        self.raw.internal_status_id
    }

    /// Returns the name of the codec that sent the message, with invalid UTF-8 replaced and an empty string if there is none.
    #[must_use]
    pub fn codec(&self) -> String {
        // SAFETY: `string_from_c_ptr` accepts null, and non-null values are NUL-terminated strings owned by the library.
        unsafe { string_from_c_ptr(self.raw.codec) }
    }

    /// Returns the ID of the codec plugin that sent the message, with invalid UTF-8 replaced and an empty string if there is none.
    #[must_use]
    pub fn codec_id(&self) -> String {
        // SAFETY: `string_from_c_ptr` accepts null, and non-null values are NUL-terminated strings owned by the library.
        unsafe { string_from_c_ptr(self.raw.codec_id) }
    }

    /// Returns the version of the codec plugin that sent the message.
    #[must_use]
    pub const fn codec_version(&self) -> u32 {
        self.raw.codec_version
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `nvimgcodecDebugMessengerDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvimgcodecDebugMessengerDestroy(self.raw)) {
                    eprintln!(
                        "failed to free nvimagecodec debug messenger: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
