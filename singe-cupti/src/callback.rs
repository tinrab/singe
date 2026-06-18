use std::{
    ffi::{CStr, CString},
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

use singe_cupti_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{
        ApiCallbackSite, CallbackDomain, CallbackId, CallbackIdResource, CallbackIdState,
        CallbackIdSync, ContextId, CorrelationId, SubscriberHandle,
    },
    utility::to_usize,
};

type CallbackFn =
    dyn for<'a> FnMut(CallbackDomain, CallbackId, CallbackDataPointer<'a>) + Send + 'static;
type CallbackBox = Box<CallbackFn>;

#[derive(Debug, Clone, Copy)]
pub struct CallbackDataPointer<'a> {
    ptr: *const (),
    _a: PhantomData<&'a ()>,
}

impl CallbackDataPointer<'_> {
    pub const fn is_null(self) -> bool {
        self.ptr.is_null()
    }

    pub fn with_decoded<R>(
        self,
        domain: CallbackDomain,
        f: impl FnOnce(CallbackData<'_>) -> R,
    ) -> Option<R> {
        if self.ptr.is_null() {
            return None;
        }
        Some(f(unsafe { CallbackData::from_raw(domain, self.ptr) }))
    }
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum CallbackData<'a> {
    Api(ApiCallbackData<'a>),
    Nvtx(NvtxData<'a>),
    Resource(ResourceData<'a>),
    Synchronize(SynchronizeData<'a>),
    State(StateData<'a>),
    Unknown,
}

impl<'a> CallbackData<'a> {
    unsafe fn from_raw(domain: CallbackDomain, ptr: *const ()) -> Self {
        match domain {
            CallbackDomain::DriverApi | CallbackDomain::RuntimeApi => Self::Api(ApiCallbackData {
                raw: unsafe { &*ptr.cast::<sys::CUpti_CallbackData>() },
            }),
            CallbackDomain::Resource => Self::Resource(ResourceData {
                raw: unsafe { &*ptr.cast::<sys::CUpti_ResourceData>() },
            }),
            CallbackDomain::Synchronize => Self::Synchronize(SynchronizeData {
                handle: unsafe { &*ptr.cast::<sys::CUpti_SynchronizeData>() },
            }),
            CallbackDomain::State => Self::State(StateData {
                raw: unsafe { &*ptr.cast::<sys::CUpti_StateData>() },
            }),
            CallbackDomain::Nvtx => Self::Nvtx(NvtxData {
                raw: unsafe { &*ptr.cast::<sys::CUpti_NvtxData>() },
            }),
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ApiCallbackData<'a> {
    raw: &'a sys::CUpti_CallbackData,
}

impl<'a> ApiCallbackData<'a> {
    pub fn site(self) -> ApiCallbackSite {
        ApiCallbackSite::from(self.raw.callbackSite)
    }

    pub fn function_name(self) -> Option<&'a CStr> {
        if self.raw.functionName.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(self.raw.functionName) })
    }

    pub fn symbol_name(self) -> Option<&'a CStr> {
        if self.raw.symbolName.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(self.raw.symbolName) })
    }

    pub fn context_id(self) -> ContextId {
        ContextId::from(self.raw.contextUid)
    }

    pub fn correlation_id(self) -> CorrelationId {
        CorrelationId::from(self.raw.correlationId)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NvtxData<'a> {
    raw: &'a sys::CUpti_NvtxData,
}

impl<'a> NvtxData<'a> {
    pub fn function_name(self) -> Option<&'a CStr> {
        if self.raw.functionName.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(self.raw.functionName) })
    }
}

#[derive(Clone, Copy)]
pub struct ResourceData<'a> {
    raw: &'a sys::CUpti_ResourceData,
}

impl ResourceData<'_> {
    pub fn has_context(self) -> bool {
        !self.raw.context.is_null()
    }

    pub fn has_resource_descriptor(self) -> bool {
        !self.raw.resourceDescriptor.is_null()
    }
}

impl Debug for ResourceData<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceData")
            .field("has_context", &self.has_context())
            .field("has_resource_descriptor", &self.has_resource_descriptor())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SynchronizeData<'a> {
    handle: &'a sys::CUpti_SynchronizeData,
}

impl SynchronizeData<'_> {
    pub fn has_context(self) -> bool {
        !self.handle.context.is_null()
    }

    pub fn has_stream(self) -> bool {
        !self.handle.stream.is_null()
    }
}

#[derive(Clone, Copy)]
pub struct StateData<'a> {
    raw: &'a sys::CUpti_StateData,
}

impl<'a> StateData<'a> {
    pub fn notification_result(self) -> crate::error::Status {
        let notification = unsafe { self.raw.__bindgen_anon_1.notification };
        notification.result.into()
    }

    pub fn notification_message(self) -> Option<&'a CStr> {
        let notification = unsafe { self.raw.__bindgen_anon_1.notification };
        if notification.message.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(notification.message) })
    }
}

impl Debug for StateData<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateData")
            .field("notification_result", &self.notification_result())
            .field("notification_message", &self.notification_message())
            .finish()
    }
}

#[derive(Debug)]
pub struct SubscriberParams {
    subscriber_name: Option<CString>,
    old_subscriber_name: Vec<i8>,
    allow_multiple_subscribers: bool,
}

impl SubscriberParams {
    pub fn new() -> Self {
        Self {
            subscriber_name: None,
            old_subscriber_name: vec![0; sys::CUPTI_OLD_SUBSCRIBER_NAME_MIN_LEN as usize],
            allow_multiple_subscribers: false,
        }
    }

    pub fn with_subscriber_name(mut self, name: &str) -> Result<Self> {
        self.subscriber_name = Some(CString::new(name)?);
        Ok(self)
    }

    pub fn with_multiple_subscribers(mut self, allow: bool) -> Self {
        self.allow_multiple_subscribers = allow;
        self
    }

    pub fn old_subscriber_name(&self) -> Option<String> {
        let ptr = self.old_subscriber_name.as_ptr();
        if ptr.is_null() || self.old_subscriber_name.first().copied() == Some(0) {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn as_raw_mut(&mut self) -> sys::CUpti_SubscriberParams {
        sys::CUpti_SubscriberParams {
            structSize: size_of::<sys::CUpti_SubscriberParams>() as u64,
            subscriberName: self
                .subscriber_name
                .as_ref()
                .map_or(ptr::null(), |name| name.as_ptr()),
            oldSubscriberName: self.old_subscriber_name.as_mut_ptr(),
            oldSubscriberSize: self.old_subscriber_name.len() as u64,
            allowMultipleSubscribers: self.allow_multiple_subscribers as u8,
            padding: [0; 7],
        }
    }
}

impl Default for SubscriberParams {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Subscriber {
    handle: Option<SubscriberHandle>,
    callback: Option<Box<CallbackBox>>,
}

impl Subscriber {
    pub fn create<F>(callback: F) -> Result<Self>
    where
        F: for<'a> FnMut(CallbackDomain, CallbackId, CallbackDataPointer<'a>) + Send + 'static,
    {
        let mut params = SubscriberParams::default();
        Self::create_with_params(callback, &mut params)
    }

    /// Creates a CUPTI callback subscriber using explicit subscription parameters.
    ///
    /// The returned [`Subscriber`] owns the CUPTI subscription and keeps the Rust callback alive until unsubscribe or drop.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber parameters are invalid.
    /// - Returns [`crate::error::Status::MultipleSubscribersNotSupported`] if another subscriber or NVIDIA profiling/debugging tool already owns CUPTI callbacks.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn create_with_params<F>(callback: F, params: &mut SubscriberParams) -> Result<Self>
    where
        F: for<'a> FnMut(CallbackDomain, CallbackId, CallbackDataPointer<'a>) + Send + 'static,
    {
        let mut callback: Box<CallbackBox> = Box::new(Box::new(callback));
        let userdata = callback.as_mut() as *mut CallbackBox;
        let mut handle = ptr::null_mut();
        let mut params = params.as_raw_mut();
        unsafe {
            try_ffi!(sys::cuptiSubscribe_v2(
                &mut handle,
                Some(callback_trampoline),
                userdata.cast(),
                &mut params,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Some(handle),
            callback: Some(callback),
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
            unsafe {
                try_ffi!(sys::cuptiUnsubscribe(handle))?;
            }
        }
        drop(self.callback.take());
        Ok(())
    }

    /// Enables or disables every callback domain for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn enable_all_domains(&self, enable: bool) -> Result<()> {
        let handle = self.as_raw()?;
        unsafe {
            try_ffi!(sys::cuptiEnableAllDomains(enable as u32, handle))?;
        }
        Ok(())
    }

    /// Enables or disables every callback in one callback domain for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle or callback domain is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn enable_domain(&self, enable: bool, domain: CallbackDomain) -> Result<()> {
        let handle = self.as_raw()?;
        unsafe {
            try_ffi!(sys::cuptiEnableDomain(enable as u32, handle, domain.into()))?;
        }
        Ok(())
    }

    /// Enables or disables one callback ID in one callback domain for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle, callback domain, or callback ID is invalid.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn enable_callback(
        &self,
        enable: bool,
        domain: CallbackDomain,
        callback_id: CallbackId,
    ) -> Result<()> {
        let handle = self.as_raw()?;
        unsafe {
            try_ffi!(sys::cuptiEnableCallback(
                enable as u32,
                handle,
                domain.into(),
                callback_id.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Returns whether one callback ID is enabled for this subscriber.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the subscriber handle, callback domain, callback ID, or CUPTI rejects the request.
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    pub fn callback_state(&self, domain: CallbackDomain, callback_id: CallbackId) -> Result<bool> {
        let mut enable = 0u32;
        let handle = self.as_raw()?;
        unsafe {
            try_ffi!(sys::cuptiGetCallbackState(
                &mut enable,
                handle,
                domain.into(),
                callback_id.as_raw(),
            ))?;
        }
        Ok(enable != 0)
    }

    pub fn enabled_callbacks(&self, domain: CallbackDomain) -> Result<Vec<CallbackId>> {
        let mut buffer_size = 0u32;
        let mut enabled_count = 0u32;
        let handle = self.as_raw()?;
        unsafe {
            try_ffi!(sys::cuptiGetEnabledCallbacks(
                handle,
                domain.into(),
                ptr::null_mut(),
                &mut buffer_size,
                &mut enabled_count,
            ))?;
        }

        if enabled_count == 0 {
            return Ok(Vec::new());
        }

        let len = to_usize(enabled_count, "enabled_count")?;
        let mut callbacks = vec![0u32; len];
        let mut buffer_size = enabled_count;
        let mut enabled_count = 0u32;
        unsafe {
            try_ffi!(sys::cuptiGetEnabledCallbacks(
                handle,
                domain.into(),
                callbacks.as_mut_ptr(),
                &mut buffer_size,
                &mut enabled_count,
            ))?;
        }
        callbacks.truncate(to_usize(enabled_count, "enabled_callbacks_count")?);
        Ok(callbacks.into_iter().map(CallbackId::from).collect())
    }

    /// Returns CUPTI's name for a callback ID in a callback domain.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if the callback domain or callback ID is invalid.
    pub fn callback_name(domain: CallbackDomain, callback_id: CallbackId) -> Result<String> {
        let mut name = ptr::null();
        unsafe {
            try_ffi!(sys::cuptiGetCallbackName(
                domain.into(),
                callback_id.as_raw(),
                &mut name
            ))?;
        }

        if name.is_null() {
            return Err(Error::NullHandle);
        }

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
        unsafe {
            try_ffi!(sys::cuptiSupportedDomains(
                &mut domain_count,
                &mut domain_table,
            ))?;
        }

        if domain_count == 0 || domain_table.is_null() {
            return Ok(Vec::new());
        }

        let count = to_usize(domain_count, "domain_count")?;
        let domains = unsafe { slice::from_raw_parts(domain_table.cast_const(), count) };
        Ok(domains.iter().copied().map(CallbackDomain::from).collect())
    }

    pub fn resource_callback_name(callback_id: CallbackIdResource) -> Result<String> {
        Self::callback_name(CallbackDomain::Resource, callback_id.into())
    }

    pub fn sync_callback_name(callback_id: CallbackIdSync) -> Result<String> {
        Self::callback_name(CallbackDomain::Synchronize, callback_id.into())
    }

    pub fn state_callback_name(callback_id: CallbackIdState) -> Result<String> {
        Self::callback_name(CallbackDomain::State, callback_id.into())
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            unsafe {
                let _ = sys::cuptiUnsubscribe(handle);
            }
        }
        drop(self.callback.take());
    }
}

unsafe extern "C" fn callback_trampoline(
    userdata: *mut std::ffi::c_void,
    domain: sys::CUpti_CallbackDomain,
    callback_id: sys::CUpti_CallbackId,
    callback_data: *const std::ffi::c_void,
) {
    if userdata.is_null() {
        return;
    }

    let _ = catch_unwind(AssertUnwindSafe(|| {
        let callback = unsafe { &mut *userdata.cast::<CallbackBox>() };
        callback(
            CallbackDomain::from(domain),
            CallbackId::from(callback_id),
            CallbackDataPointer {
                ptr: callback_data as _,
                _a: PhantomData,
            },
        );
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_data_pointer_decodes_api_callback_data() {
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
            ptr: (&raw as *const sys::CUpti_CallbackData).cast(),
            _a: PhantomData,
        };

        let decoded = pointer.with_decoded(CallbackDomain::DriverApi, |data| match data {
            CallbackData::Api(data) => {
                assert_eq!(data.site(), ApiCallbackSite::Enter);
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
    fn callback_data_pointer_decodes_state_notification() {
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
            ptr: (&raw as *const sys::CUpti_StateData).cast(),
            _a: PhantomData,
        };

        let decoded = pointer.with_decoded(CallbackDomain::State, |data| match data {
            CallbackData::State(data) => {
                assert_eq!(data.notification_result(), crate::error::Status::Success);
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
    fn callback_data_pointer_decodes_nvtx_callback_data() {
        let function_name = CString::new("nvtxRangePushEx").unwrap();
        let raw = sys::CUpti_NvtxData {
            functionName: function_name.as_ptr(),
            functionParams: ptr::null(),
            functionReturnValue: ptr::null(),
        };
        let pointer = CallbackDataPointer {
            ptr: (&raw as *const sys::CUpti_NvtxData).cast(),
            _a: PhantomData,
        };

        let decoded = pointer.with_decoded(CallbackDomain::Nvtx, |data| match data {
            CallbackData::Nvtx(data) => {
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
    fn null_callback_data_pointer_does_not_decode() {
        let pointer = CallbackDataPointer {
            ptr: ptr::null(),
            _a: PhantomData,
        };

        assert!(pointer.is_null());
        assert_eq!(
            pointer.with_decoded(CallbackDomain::DriverApi, |_| true),
            None
        );
    }

    #[test]
    fn subscriber_params_default_raw_state_is_valid() {
        let mut params = SubscriberParams::new();
        let raw = params.as_raw_mut();

        assert_eq!(
            raw.structSize,
            size_of::<sys::CUpti_SubscriberParams>() as u64
        );
        assert!(raw.subscriberName.is_null());
        assert!(!raw.oldSubscriberName.is_null());
        assert_eq!(
            raw.oldSubscriberSize,
            sys::CUPTI_OLD_SUBSCRIBER_NAME_MIN_LEN as u64
        );
        assert_eq!(raw.allowMultipleSubscribers, 0);
        assert_eq!(params.old_subscriber_name(), None);
    }

    #[test]
    fn subscriber_params_builder_sets_name_and_multiple_subscribers() -> Result<()> {
        let mut params = SubscriberParams::new()
            .with_subscriber_name("singe-cupti-test")?
            .with_multiple_subscribers(true);
        let raw = params.as_raw_mut();

        assert!(!raw.subscriberName.is_null());
        assert_eq!(
            unsafe { CStr::from_ptr(raw.subscriberName) }
                .to_str()
                .unwrap(),
            "singe-cupti-test"
        );
        assert_eq!(raw.allowMultipleSubscribers, 1);
        Ok(())
    }

    #[test]
    fn subscriber_params_rejects_interior_nul_names() {
        let error = SubscriberParams::new()
            .with_subscriber_name("bad\0name")
            .unwrap_err();

        assert!(matches!(error, Error::InteriorNul));
    }
}
