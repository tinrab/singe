use std::{
    collections::BTreeMap,
    ffi,
    ffi::{CStr, CString},
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
    sync::{
        Arc, PoisonError, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use singe_core::{report_cleanup_failure, string_from_c_chars};
use singe_cupti_sys as sys;

use crate::{
    error::{Error, Result, Status},
    try_ffi,
    types::{
        ApiCallbackSite, CallbackDomain, CallbackId, CallbackIdResource, CallbackIdState,
        CallbackIdSync, ContextId, CorrelationId, SubscriberHandle,
    },
    utility::{to_u32, to_usize},
};

type CallbackFn =
    dyn for<'a> Fn(CallbackDomain, CallbackId, CallbackDataPointer<'a>) + Send + Sync + 'static;

// Callbacks by subscriber key, which CUPTI passes back as the callback's user data.
// The trampoline clones a callback out before calling it, so removing a subscriber's entry never frees a callback still running on another thread.
static SUBSCRIBER_CALLBACKS: RwLock<BTreeMap<usize, Arc<CallbackFn>>> =
    RwLock::new(BTreeMap::new());

static NEXT_SUBSCRIBER_KEY: AtomicUsize = AtomicUsize::new(1);

/// Opaque callback-data pointer borrowed from a CUPTI callback invocation, with the domain CUPTI delivered it for.
#[derive(Debug, Clone, Copy)]
pub struct CallbackDataPointer<'a> {
    ptr: *const (),
    domain: CallbackDomain,
    _a: PhantomData<&'a ()>,
}

impl CallbackDataPointer<'_> {
    /// Returns whether CUPTI supplied a null callback-data pointer.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.ptr.is_null()
    }

    /// Returns the callback domain CUPTI delivered this pointer for.
    #[must_use]
    pub const fn domain(self) -> CallbackDomain {
        self.domain
    }

    /// Decodes this pointer for its callback domain and passes the borrowed payload to `f`.
    ///
    /// Returns `None` without calling `f` when the pointer is null; domains without a decoder produce [`CallbackPayload::Unknown`].
    pub fn with_decoded<R>(self, f: impl FnOnce(CallbackPayload<'_>) -> R) -> Option<R> {
        if self.ptr.is_null() {
            return None;
        }

        // SAFETY: only the trampoline builds a pointer, from the callback data and domain of one CUPTI invocation, so `self.domain` names the type CUPTI passed, and CUPTI keeps that data valid for the callback that `'_` borrows.
        let payload = unsafe { CallbackPayload::from_raw(self.domain, self.ptr) };
        Some(f(payload))
    }
}

/// Callback data decoded according to its CUPTI callback domain.
///
/// Payload references remain borrowed for the callback invocation.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum CallbackPayload<'a> {
    /// CUDA driver or runtime API callback.
    Api(ApiCallbackInfo<'a>),
    /// NVTX callback.
    Nvtx(NvtxCallbackInfo<'a>),
    /// CUDA resource creation or destruction callback.
    Resource(ResourceCallbackInfo<'a>),
    /// CUDA context or stream synchronization callback.
    Synchronize(SynchronizeCallbackInfo<'a>),
    /// CUPTI state notification.
    State(StateCallbackInfo<'a>),
    /// Callback domain without a supported payload decoder.
    Unknown,
}

impl CallbackPayload<'_> {
    unsafe fn from_raw(domain: CallbackDomain, ptr: *const ()) -> Self {
        match domain {
            CallbackDomain::DriverApi | CallbackDomain::RuntimeApi => Self::Api(ApiCallbackInfo {
                // SAFETY: the caller guarantees that `ptr` is non-null and points to the live `CUpti_CallbackData` CUPTI passes for API domains.
                raw: unsafe { &*ptr.cast::<sys::CUpti_CallbackData>() },
            }),
            CallbackDomain::Resource => Self::Resource(ResourceCallbackInfo {
                // SAFETY: the caller guarantees that `ptr` is non-null and points to the live `CUpti_ResourceData` CUPTI passes for the resource domain.
                raw: unsafe { &*ptr.cast::<sys::CUpti_ResourceData>() },
            }),
            CallbackDomain::Synchronize => Self::Synchronize(SynchronizeCallbackInfo {
                // SAFETY: the caller guarantees that `ptr` is non-null and points to the live `CUpti_SynchronizeData` CUPTI passes for the synchronize domain.
                handle: unsafe { &*ptr.cast::<sys::CUpti_SynchronizeData>() },
            }),
            CallbackDomain::State => Self::State(StateCallbackInfo {
                // SAFETY: the caller guarantees that `ptr` is non-null and points to the live `CUpti_StateData` CUPTI passes for the state domain.
                raw: unsafe { &*ptr.cast::<sys::CUpti_StateData>() },
            }),
            CallbackDomain::Nvtx => Self::Nvtx(NvtxCallbackInfo {
                // SAFETY: the caller guarantees that `ptr` is non-null and points to the live `CUpti_NvtxData` CUPTI passes for the NVTX domain.
                raw: unsafe { &*ptr.cast::<sys::CUpti_NvtxData>() },
            }),
            _ => Self::Unknown,
        }
    }
}

/// Borrowed metadata for a CUDA driver or runtime API callback.
#[derive(Debug, Clone, Copy)]
pub struct ApiCallbackInfo<'a> {
    raw: &'a sys::CUpti_CallbackData,
}

impl<'a> ApiCallbackInfo<'a> {
    /// Returns whether the callback occurs at API entry or exit.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI reports a site this crate does not name.
    pub fn site(self) -> Result<ApiCallbackSite> {
        Ok(ApiCallbackSite::try_from(self.raw.callbackSite)?)
    }

    /// Returns the borrowed name of the API function issuing the callback, or `None` for a null native name pointer.
    #[must_use]
    pub fn function_name(self) -> Option<&'a CStr> {
        if self.raw.functionName.is_null() {
            return None;
        }
        // SAFETY: the pointer was checked for null, and CUPTI keeps the callback data's NUL-terminated strings alive for the callback that `'a` borrows.
        Some(unsafe { CStr::from_ptr(self.raw.functionName) })
    }

    /// Returns the borrowed symbol name for a driver or runtime launch callback, or `None` for a null pointer.
    ///
    /// For launch callbacks, the symbol is the kernel name; the native field is not meaningful for other API calls.
    #[must_use]
    pub fn symbol_name(self) -> Option<&'a CStr> {
        if self.raw.symbolName.is_null() {
            return None;
        }
        // SAFETY: the pointer was checked for null, and CUPTI keeps the callback data's NUL-terminated strings alive for the callback that `'a` borrows.
        Some(unsafe { CStr::from_ptr(self.raw.symbolName) })
    }

    /// Returns the process-unique identifier of the associated CUDA context.
    #[must_use]
    pub fn context_id(self) -> ContextId {
        ContextId::from(self.raw.contextUid)
    }

    /// Returns the identifier matching the corresponding driver or runtime API activity record.
    #[must_use]
    pub fn correlation_id(self) -> CorrelationId {
        CorrelationId::from(self.raw.correlationId)
    }
}

/// Borrowed metadata for an NVTX callback.
#[derive(Debug, Clone, Copy)]
pub struct NvtxCallbackInfo<'a> {
    raw: &'a sys::CUpti_NvtxData,
}

impl<'a> NvtxCallbackInfo<'a> {
    /// Returns the borrowed name of the API function issuing the callback, or `None` for a null native name pointer.
    #[must_use]
    pub fn function_name(self) -> Option<&'a CStr> {
        if self.raw.functionName.is_null() {
            return None;
        }
        // SAFETY: the pointer was checked for null, and CUPTI keeps the callback data's NUL-terminated strings alive for the callback that `'a` borrows.
        Some(unsafe { CStr::from_ptr(self.raw.functionName) })
    }
}

/// Borrowed resource creation or destruction callback metadata.
#[derive(Clone, Copy)]
pub struct ResourceCallbackInfo<'a> {
    raw: &'a sys::CUpti_ResourceData,
}

impl ResourceCallbackInfo<'_> {
    /// Returns whether the native callback includes a non-null CUDA context handle.
    #[must_use]
    pub fn has_context(self) -> bool {
        !self.raw.context.is_null()
    }

    /// Returns whether the native callback includes a non-null resource descriptor.
    #[must_use]
    pub fn has_resource_descriptor(self) -> bool {
        !self.raw.resourceDescriptor.is_null()
    }
}

impl Debug for ResourceCallbackInfo<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceCallbackInfo")
            .field("has_context", &self.has_context())
            .field("has_resource_descriptor", &self.has_resource_descriptor())
            .finish()
    }
}

/// Borrowed CUDA context or stream synchronization callback metadata.
#[derive(Debug, Clone, Copy)]
pub struct SynchronizeCallbackInfo<'a> {
    handle: &'a sys::CUpti_SynchronizeData,
}

impl SynchronizeCallbackInfo<'_> {
    /// Returns whether the native callback includes a non-null CUDA context handle.
    #[must_use]
    pub fn has_context(self) -> bool {
        !self.handle.context.is_null()
    }

    /// Returns whether the native stream handle is non-null.
    ///
    /// A null handle can represent the default stream; this query does not distinguish that case from an absent stream.
    #[must_use]
    pub fn has_stream(self) -> bool {
        !self.handle.stream.is_null()
    }
}

/// Borrowed CUPTI state-notification metadata.
#[derive(Clone, Copy)]
pub struct StateCallbackInfo<'a> {
    raw: &'a sys::CUpti_StateData,
}

impl<'a> StateCallbackInfo<'a> {
    /// Returns the CUPTI status associated with this state notification.
    #[must_use]
    pub fn notification_result(self) -> Status {
        // SAFETY: `notification` is the union's only field, and CUPTI initializes it for every state callback.
        let notification = unsafe { self.raw.__bindgen_anon_1.notification };
        notification.result.into()
    }

    /// Returns the borrowed notification message, or `None` for a null native message pointer.
    #[must_use]
    pub fn notification_message(self) -> Option<&'a CStr> {
        // SAFETY: `notification` is the union's only field, and CUPTI initializes it for every state callback.
        let notification = unsafe { self.raw.__bindgen_anon_1.notification };
        if notification.message.is_null() {
            return None;
        }
        // SAFETY: the pointer was checked for null, and CUPTI keeps the callback data's NUL-terminated strings alive for the callback that `'a` borrows.
        Some(unsafe { CStr::from_ptr(notification.message) })
    }
}

impl Debug for StateCallbackInfo<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateCallbackInfo")
            .field("notification_result", &self.notification_result())
            .field("notification_message", &self.notification_message())
            .finish()
    }
}

/// Settings for [`Subscriber::create`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SubscriberOptions {
    /// Name CUPTI copies and reports to later subscribers that conflict with this one.
    pub name: Option<CString>,
    /// Whether CUPTI may keep other subscribers active alongside this one.
    ///
    /// CUPTI rejects the subscription if it cannot honor the request.
    pub allow_multiple_subscribers: bool,
}

/// Owns a CUPTI callback subscription and its Rust callback.
///
/// CUPTI invokes the callback on the host thread issuing the traced CUDA call, so it may run concurrently on several threads and re-entrantly when it makes CUDA calls itself.
/// The callback is therefore `Fn + Send + Sync`; keep mutable state behind synchronization.
/// Callback panics are caught at the native trampoline.
/// Subscription creation leaves callbacks disabled; enable the desired domains or identifiers explicitly.
pub struct Subscriber {
    handle: Option<SubscriberHandle>,
    key: usize,
}

impl Debug for Subscriber {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Subscriber").finish_non_exhaustive()
    }
}

impl Subscriber {
    /// Subscribes `callback` to CUPTI and retains it until unsubscription or drop.
    ///
    /// Individual callbacks remain disabled until explicitly enabled.
    ///
    /// # Errors
    ///
    /// - Returns [`Error::SubscriberActive`] if another subscriber that reported its name prevents registration.
    /// - Returns [`crate::error::Status::MultipleSubscribersNotSupported`] if another subscriber or NVIDIA profiling or debugging tool already owns CUPTI callbacks.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`Error::NullHandle`] if CUPTI returns a null handle.
    pub fn create<F>(callback: F, options: &SubscriberOptions) -> Result<Self>
    where
        F: for<'a> Fn(CallbackDomain, CallbackId, CallbackDataPointer<'a>) + Send + Sync + 'static,
    {
        let key = NEXT_SUBSCRIBER_KEY.fetch_add(1, Ordering::Relaxed);
        SUBSCRIBER_CALLBACKS
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, Arc::new(callback));
        let mut handle = ptr::null_mut();
        let mut existing_name: Vec<ffi::c_char> =
            vec![0; sys::CUPTI_OLD_SUBSCRIBER_NAME_MIN_LEN as usize];
        let mut params = raw_subscriber_params(options, &mut existing_name);
        // SAFETY: `handle` and `params` are valid for the call, `params` points into `options` and `existing_name`, which outlive it, and the user data is a key the trampoline only looks up, never dereferences.
        let result = unsafe {
            try_ffi!(sys::cuptiSubscribe_v2(
                &raw mut handle,
                Some(callback_trampoline),
                ptr::without_provenance_mut(key),
                &raw mut params,
            ))
        };
        if result.is_err() || handle.is_null() {
            remove_subscriber_callback(key);
        }

        if let Err(error) = result {
            let existing = string_from_c_chars(&existing_name);
            return Err(if existing.is_empty() {
                error
            } else {
                Error::SubscriberActive { existing }
            });
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Some(handle),
            key,
        })
    }

    pub(crate) fn as_raw(&self) -> Result<SubscriberHandle> {
        self.handle.ok_or(Error::NullHandle)
    }

    /// Unregisters this CUPTI callback subscriber.
    ///
    /// After this returns, CUPTI will not issue more callbacks to the subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn unsubscribe(mut self) -> Result<()> {
        if let Some(handle) = self.handle.take() {
            // SAFETY: `handle` was taken out of `self.handle`, so it is this subscriber's live handle and is unsubscribed only once.
            unsafe { try_ffi!(sys::cuptiUnsubscribe(handle)) }?;
            // A failure returns above with the entry still registered, since CUPTI may still call the callback.
            remove_subscriber_callback(self.key);
        }

        Ok(())
    }

    /// Enables or disables every callback domain for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn set_all_domains_enabled(&self, enabled: bool) -> Result<()> {
        let handle = self.as_raw()?;
        // SAFETY: `handle` is this subscriber's live handle, and the other arguments are plain values.
        unsafe {
            try_ffi!(sys::cuptiEnableAllDomains(u32::from(enabled), handle))?;
        }
        Ok(())
    }

    /// Enables or disables every callback in `domain` for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle or callback domain is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn set_domain_enabled(&self, domain: CallbackDomain, enabled: bool) -> Result<()> {
        let handle = self.as_raw()?;
        // SAFETY: `handle` is this subscriber's live handle, and the other arguments are plain values.
        unsafe {
            try_ffi!(sys::cuptiEnableDomain(
                u32::from(enabled),
                handle,
                domain.into()
            ))?;
        }
        Ok(())
    }

    /// Enables or disables one callback in `domain` for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle, callback domain, or callback ID is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn set_callback_enabled(
        &self,
        domain: CallbackDomain,
        callback_id: CallbackId,
        enabled: bool,
    ) -> Result<()> {
        let handle = self.as_raw()?;
        // SAFETY: `handle` is this subscriber's live handle, and the other arguments are plain values.
        unsafe {
            try_ffi!(sys::cuptiEnableCallback(
                u32::from(enabled),
                handle,
                domain.into(),
                callback_id.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Returns whether one callback in `domain` is enabled for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle, callback domain, callback ID, or CUPTI rejects the request.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn is_callback_enabled(
        &self,
        domain: CallbackDomain,
        callback_id: CallbackId,
    ) -> Result<bool> {
        let mut enabled = 0u32;
        let handle = self.as_raw()?;
        // SAFETY: `handle` is this subscriber's live handle, and `enabled` is a valid output location.
        unsafe {
            try_ffi!(sys::cuptiGetCallbackState(
                &raw mut enabled,
                handle,
                domain.into(),
                callback_id.as_raw(),
            ))?;
        }
        Ok(enabled != 0)
    }

    /// Returns the enabled callback identifiers in `domain` for this subscriber.
    ///
    /// # Errors
    ///
    /// Returns an error if the subscriber handle is absent, a native callback-list query rejects the handle or domain, or a returned count cannot be represented as `usize`.
    pub fn enabled_callbacks(&self, domain: CallbackDomain) -> Result<Vec<CallbackId>> {
        let mut buffer_size = 0u32;
        let mut enabled_count = 0u32;
        let handle = self.as_raw()?;
        // SAFETY: `handle` is this subscriber's live handle, the null list asks only for the count, and both counts are local outputs.
        unsafe {
            try_ffi!(sys::cuptiGetEnabledCallbacks(
                handle,
                domain.into(),
                ptr::null_mut(),
                &raw mut buffer_size,
                &raw mut enabled_count,
            ))?;
        }

        if enabled_count == 0 {
            return Ok(Vec::new());
        }

        let len = to_usize(enabled_count, "enabled_count")?;
        let mut callbacks = vec![0u32; len];
        // CUPTI takes the buffer size in bytes, as NVIDIA's multiple-subscriber sample passes it.
        let mut buffer_size = to_u32(
            len.checked_mul(size_of::<sys::CUpti_CallbackId>())
                .ok_or_else(|| Error::out_of_range("enabled_callbacks_size"))?,
            "enabled_callbacks_size",
        )?;
        let mut enabled_count = 0u32;
        // SAFETY: `handle` is this subscriber's live handle, and `callbacks` holds `buffer_size` writable bytes of IDs, matching the size passed.
        unsafe {
            try_ffi!(sys::cuptiGetEnabledCallbacks(
                handle,
                domain.into(),
                callbacks.as_mut_ptr(),
                &raw mut buffer_size,
                &raw mut enabled_count,
            ))?;
        }

        // CUPTI reports the true count, which exceeds the filled entries when more callbacks were enabled since the first call.
        callbacks.truncate(to_usize(enabled_count, "enabled_callbacks_count")?.min(len));
        Ok(callbacks.into_iter().map(CallbackId::from).collect())
    }

    /// Returns CUPTI's name for a callback ID in a callback domain.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the callback domain or callback ID is invalid.
    pub fn callback_name(domain: CallbackDomain, callback_id: CallbackId) -> Result<String> {
        let mut name = ptr::null();
        // SAFETY: the domain and callback ID are passed by value and checked by CUPTI, and `name` is a local output.
        unsafe {
            try_ffi!(sys::cuptiGetCallbackName(
                domain.into(),
                callback_id.as_raw(),
                &raw mut name
            ))?;
        }

        if name.is_null() {
            return Err(Error::NullHandle);
        }

        // SAFETY: `name` was checked for null, and CUPTI returns a static NUL-terminated name, copied here before returning.
        Ok(unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned())
    }

    /// Returns the callback domains supported by the loaded CUPTI runtime.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn supported_domains() -> Result<Vec<CallbackDomain>> {
        let mut domain_count = 0;
        let mut domain_table = ptr::null_mut();
        // SAFETY: `domain_count` and `domain_table` are local outputs for the call.
        unsafe {
            try_ffi!(sys::cuptiSupportedDomains(
                &raw mut domain_count,
                &raw mut domain_table,
            ))?;
        }

        if domain_count == 0 || domain_table.is_null() {
            return Ok(Vec::new());
        }

        let count = to_usize(domain_count, "domain_count")?;
        // SAFETY: CUPTI returned a table of `count` domains, released only after this copy.
        let domains = unsafe { slice::from_raw_parts(domain_table.cast_const(), count) };
        domains
            .iter()
            .map(|&domain| Ok(CallbackDomain::try_from(domain)?))
            .collect()
    }

    /// Returns CUPTI's name for a resource callback identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot name the callback or rejects its domain or identifier, or returns a null name pointer.
    pub fn resource_callback_name(callback_id: CallbackIdResource) -> Result<String> {
        Self::callback_name(CallbackDomain::Resource, callback_id.into())
    }

    /// Returns CUPTI's name for a synchronization callback identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot name the callback or rejects its domain or identifier, or returns a null name pointer.
    pub fn sync_callback_name(callback_id: CallbackIdSync) -> Result<String> {
        Self::callback_name(CallbackDomain::Synchronize, callback_id.into())
    }

    /// Returns CUPTI's name for a state callback identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if CUPTI cannot name the callback or rejects its domain or identifier, or returns a null name pointer.
    pub fn state_callback_name(callback_id: CallbackIdState) -> Result<String> {
        Self::callback_name(CallbackDomain::State, callback_id.into())
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // SAFETY: `handle` was taken out of `self.handle`, so it is this subscriber's live handle and is unsubscribed only once.
            if let Err(error) = unsafe { try_ffi!(sys::cuptiUnsubscribe(handle)) } {
                report_cleanup_failure("unsubscribe", "cupti subscriber", &error);
                // CUPTI may still call the callback, so its entry stays registered.
                return;
            }

            remove_subscriber_callback(self.key);
        }
    }
}

// Removes a subscriber's callback; a trampoline already running keeps its own clone alive until it returns.
fn remove_subscriber_callback(key: usize) {
    SUBSCRIBER_CALLBACKS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&key);
}

fn raw_subscriber_params(
    options: &SubscriberOptions,
    existing_name: &mut [ffi::c_char],
) -> sys::CUpti_SubscriberParams {
    sys::CUpti_SubscriberParams {
        structSize: size_of::<sys::CUpti_SubscriberParams>() as u64,
        subscriberName: options
            .name
            .as_ref()
            .map_or(ptr::null(), |name| name.as_ptr()),
        oldSubscriberName: existing_name.as_mut_ptr(),
        oldSubscriberSize: existing_name.len() as u64,
        allowMultipleSubscribers: u8::from(options.allow_multiple_subscribers),
        padding: [0; 7],
    }
}

unsafe extern "C" fn callback_trampoline(
    userdata: *mut ffi::c_void,
    domain: sys::CUpti_CallbackDomain,
    callback_id: sys::CUpti_CallbackId,
    callback_data: *const ffi::c_void,
) {
    // A domain this crate does not name has no typed callback data to report.
    let Ok(domain) = CallbackDomain::try_from(domain) else {
        return;
    };

    // The read guard drops before the call, so a re-entrant subscribe or unsubscribe cannot deadlock.
    let Some(callback) = SUBSCRIBER_CALLBACKS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&userdata.addr())
        .cloned()
    else {
        return;
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        // CUPTI may call concurrently and re-entrantly, which the `Fn + Sync` callback permits.
        callback(
            domain,
            CallbackId::from(callback_id),
            CallbackDataPointer {
                ptr: callback_data.cast(),
                domain,
                _a: PhantomData,
            },
        );
    }));
    if result.is_err() {
        #[cfg(debug_assertions)]
        eprintln!("cupti callback panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_data_pointer_decodes_api_callback_data() {
        let function_name = CString::new("cuLaunchKernel").unwrap();
        let symbol_name = CString::new("kernel").unwrap();
        let raw = sys::CUpti_CallbackData {
            callbackSite: sys::CUpti_ApiCallbackSite::CUPTI_API_ENTER,
            functionName: function_name.as_ptr(),
            functionParams: ptr::null(),
            functionReturnValue: ptr::null_mut(),
            symbolName: symbol_name.as_ptr(),
            context: ptr::null_mut(),
            contextUid: 7,
            correlationData: ptr::null_mut(),
            correlationId: 11,
        };
        let pointer = CallbackDataPointer {
            ptr: (&raw const raw).cast(),
            domain: CallbackDomain::DriverApi,
            _a: PhantomData,
        };

        let decoded = pointer.with_decoded(|data| match data {
            CallbackPayload::Api(data) => {
                assert_eq!(data.site().unwrap(), ApiCallbackSite::Enter);
                assert_eq!(
                    data.function_name().unwrap().to_str().unwrap(),
                    "cuLaunchKernel"
                );
                assert_eq!(data.symbol_name().unwrap().to_str().unwrap(), "kernel");
                assert_eq!(data.context_id(), ContextId::from(7));
                assert_eq!(data.correlation_id(), CorrelationId::from(11));
                true
            }
            _ => false,
        });

        assert_eq!(decoded, Some(true));
    }

    #[test]
    fn test_callback_data_pointer_decodes_state_notification() {
        let message = CString::new("state changed").unwrap();
        let raw = sys::CUpti_StateData {
            __bindgen_anon_1: sys::CUpti_StateData__bindgen_ty_1 {
                notification: sys::CUpti_StateData__bindgen_ty_1__bindgen_ty_1 {
                    result: sys::CUptiResult::CUPTI_SUCCESS,
                    message: message.as_ptr(),
                },
            },
        };
        let pointer = CallbackDataPointer {
            ptr: (&raw const raw).cast(),
            domain: CallbackDomain::State,
            _a: PhantomData,
        };

        let decoded = pointer.with_decoded(|data| match data {
            CallbackPayload::State(data) => {
                assert_eq!(data.notification_result(), Status::Success);
                assert_eq!(
                    data.notification_message().unwrap().to_str().unwrap(),
                    "state changed"
                );
                true
            }
            _ => false,
        });

        assert_eq!(decoded, Some(true));
    }

    #[test]
    fn test_callback_data_pointer_decodes_nvtx_callback_data() {
        let function_name = CString::new("nvtxRangePushEx").unwrap();
        let raw = sys::CUpti_NvtxData {
            functionName: function_name.as_ptr(),
            functionParams: ptr::null(),
            functionReturnValue: ptr::null(),
        };
        let pointer = CallbackDataPointer {
            ptr: (&raw const raw).cast(),
            domain: CallbackDomain::Nvtx,
            _a: PhantomData,
        };

        let decoded = pointer.with_decoded(|data| match data {
            CallbackPayload::Nvtx(data) => {
                assert_eq!(
                    data.function_name().unwrap().to_str().unwrap(),
                    "nvtxRangePushEx"
                );
                true
            }
            _ => false,
        });

        assert_eq!(decoded, Some(true));
    }

    #[test]
    fn test_null_callback_data_pointer_does_not_decode() {
        let pointer = CallbackDataPointer {
            ptr: ptr::null(),
            domain: CallbackDomain::DriverApi,
            _a: PhantomData,
        };

        assert!(pointer.is_null());
        assert_eq!(pointer.with_decoded(|_| true), None);
    }

    #[test]
    fn test_subscriber_options_default_raw_state_is_valid() {
        let mut existing_name: Vec<ffi::c_char> =
            vec![0; sys::CUPTI_OLD_SUBSCRIBER_NAME_MIN_LEN as usize];
        let raw = raw_subscriber_params(&SubscriberOptions::default(), &mut existing_name);

        assert_eq!(
            raw.structSize,
            size_of::<sys::CUpti_SubscriberParams>() as u64
        );
        assert!(raw.subscriberName.is_null());
        assert!(!raw.oldSubscriberName.is_null());
        assert_eq!(
            raw.oldSubscriberSize,
            u64::from(sys::CUPTI_OLD_SUBSCRIBER_NAME_MIN_LEN)
        );
        assert_eq!(raw.allowMultipleSubscribers, 0);
        assert!(string_from_c_chars(&existing_name).is_empty());
    }

    #[test]
    fn test_subscriber_options_set_name_and_multiple_subscribers() {
        let options = SubscriberOptions {
            name: Some(CString::new("singe-cupti-test").unwrap()),
            allow_multiple_subscribers: true,
        };
        let mut existing_name: Vec<ffi::c_char> =
            vec![0; sys::CUPTI_OLD_SUBSCRIBER_NAME_MIN_LEN as usize];
        let raw = raw_subscriber_params(&options, &mut existing_name);

        assert!(!raw.subscriberName.is_null());
        assert_eq!(
            // SAFETY: the subscriber name was checked for null and points into `options`, which is still alive.
            unsafe { CStr::from_ptr(raw.subscriberName) }
                .to_str()
                .unwrap(),
            "singe-cupti-test"
        );
        assert_eq!(raw.allowMultipleSubscribers, 1);
    }
}
