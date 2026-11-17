use std::{
    ffi::c_void,
    fmt::{self, Debug, Formatter},
    mem::ManuallyDrop,
    panic::{self, AssertUnwindSafe},
    ptr,
};

use singe_core::{report_cleanup_failure, string_from_c_ptr};
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    instance::{Instance, InstanceRef},
    try_ffi,
    types::{DebugMessageCategory, DebugMessageSeverity, NativeStruct},
};

pub(crate) type DebugCallbackFn = dyn for<'a> Fn(DebugMessage<'a>) + Send + Sync + 'static;

/// Receives nvImageCodec's debug messages and passes them to a Rust callback.
///
/// It keeps its [`Instance`] alive and owns the callback.
#[derive(Debug)]
pub struct DebugMessenger {
    raw: sys::nvimgcodecDebugMessenger_t,
    instance: InstanceRef,
    callback: Option<Box<DebugCallback>>,
}

/// Rust callback that a [`DebugMessenger`] passes nvImageCodec's messages to.
pub struct DebugCallback(pub(crate) Box<DebugCallbackFn>);

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
#[derive(Debug)]
pub struct RawDebugMessenger {
    /// Native debug-messenger handle, owned exclusively.
    pub handle: sys::nvimgcodecDebugMessenger_t,
    /// Instance the handle was created with.
    pub instance: Instance,
    /// Callback whose address the handle was given as user data, or `None` if the caller keeps its callback data alive.
    pub callback: Option<Box<DebugCallback>>,
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
        desc.user_data = ptr::from_ref::<DebugCallback>(callback.as_ref())
            .cast_mut()
            .cast::<c_void>();

        let mut raw = ptr::null_mut();
        // SAFETY: `instance` is live, `desc` is a local that outlives the call, its user data points to the boxed callback the messenger keeps, which `debug_callback` only reads, and `raw` is a writable local.
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
            instance: instance.as_instance_ref(),
            callback: Some(callback),
        })
    }

    /// Returns the native handle, which this messenger still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecDebugMessenger_t {
        self.raw
    }

    /// Takes ownership of a native debug messenger, from [`into_raw`](Self::into_raw) or created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.instance`, and owned by nobody else.
    /// `raw.callback` must be the callback the handle was created with, or `None`, in which case the caller must keep the callback's data alive separately.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawDebugMessenger) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw: raw.handle,
            instance: raw.instance.into_instance_ref(),
            callback: raw.callback,
        })
    }

    /// Releases the native handle with the values that keep it valid, including the callback.
    ///
    /// Destroy the handle or pass it back to [`from_raw`](Self::from_raw).
    #[must_use]
    pub fn into_raw(self) -> RawDebugMessenger {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so each owning field moves out exactly once.
        let (instance, callback) = unsafe {
            (
                ptr::read(&raw const this.instance),
                ptr::read(&raw const this.callback),
            )
        };
        RawDebugMessenger {
            handle: this.raw,
            instance: instance.into_instance(),
            callback,
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
            // SAFETY: `self.raw` is the live messenger this value owns, and its callback box is dropped only after this call, so no message arrives after it is freed.
            if let Err(error) = unsafe { try_ffi!(sys::nvimgcodecDebugMessengerDestroy(self.raw)) }
            {
                report_cleanup_failure("destroy", "nvimagecodec debug messenger", &error);
            }
        }
    }
}

unsafe extern "C" fn debug_callback(
    message_severity: sys::nvimgcodecDebugMessageSeverity_t,
    message_category: sys::nvimgcodecDebugMessageCategory_t,
    callback_data: *const sys::nvimgcodecDebugMessageData_t,
    user_data: *mut c_void,
) -> i32 {
    if user_data.is_null() || callback_data.is_null() {
        return 1;
    }

    // SAFETY: nvImageCodec passes back the `DebugCallback` registered with this messenger, which outlives it.
    let callback = unsafe { &*user_data.cast::<DebugCallback>().cast_const() };
    // SAFETY: nvImageCodec passes message data that stays valid for the duration of the callback.
    let data = unsafe { &*callback_data };
    let message = DebugMessage::from_raw(
        DebugMessageSeverity::from_bits_retain(message_severity.0),
        DebugMessageCategory::from_bits_retain(message_category.0),
        data,
    );

    // Unwinding must not reach nvImageCodec, so a panicking callback loses this message.
    if panic::catch_unwind(AssertUnwindSafe(|| (callback.0)(message))).is_err() {
        if cfg!(debug_assertions) {
            eprintln!("nvimagecodec debug messenger callback panicked");
        }

        return 1;
    }

    0
}
