use std::{ffi, ffi::CStr, ptr, sync::Arc};

use singe_cuda::{
    context::Context,
    event::{Event, EventFlags},
};

use crate::{
    communicator::Communicator,
    error::{Error, Result},
    group::validate_collective,
    sys,
    types::CtaPolicy,
};

/// Caller-owned, rank-local CUDA event recorded at collective kernel launch completion.
/// Every rank must provide an event or every rank must omit it.
/// The communicator retains the event through queued waits and captured-graph executions.
#[derive(Debug, Clone)]
pub struct LaunchCompletionEvent(Arc<Event>);

impl LaunchCompletionEvent {
    /// Create it with `cudaEventDisableTiming`; interprocess and interop events are unsupported.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot create a timing-disabled event in `context`.
    pub fn create(context: &Arc<Context>) -> Result<Self> {
        Ok(Self(Arc::new(
            context.create_event_with_flags(EventFlags::DISABLE_TIMING)?,
        )))
    }

    /// Returns the retained CUDA launch-completion event.
    #[must_use]
    pub fn event(&self) -> &Event {
        &self.0
    }
}

/// A single vendor-specific option expressed as a key-value pair.
/// The official NCCL library ignores every extension; a vendor library
/// should assign a unique value to `vendor_id` to avoid collision among vendor libraries.
#[derive(Debug, Clone, Copy)]
pub struct Extension<'a> {
    vendor_id: i32,
    option_id: i32,
    value: ExtensionValue<'a>,
}

#[derive(Debug, Clone, Copy)]
enum ExtensionValue<'a> {
    Integer(i32),
    String(&'a CStr),
    Raw(*mut ffi::c_void),
}

impl<'a> Extension<'a> {
    /// Integer value.
    #[must_use]
    pub fn integer(vendor_id: i32, option_id: i32, value: i32) -> Self {
        Self {
            vendor_id,
            option_id,
            value: ExtensionValue::Integer(value),
        }
    }

    /// C-style string value.
    #[must_use]
    pub fn string(vendor_id: i32, option_id: i32, value: &'a CStr) -> Self {
        Self {
            vendor_id,
            option_id,
            value: ExtensionValue::String(value),
        }
    }

    /// Opaque pointer-sized value.
    ///
    /// # Safety
    /// The pointer must satisfy the vendor option's validity and lifetime requirements
    /// for every collective call that uses this extension.
    pub unsafe fn raw(vendor_id: i32, option_id: i32, value: *mut ffi::c_void) -> Self {
        Self {
            vendor_id,
            option_id,
            value: ExtensionValue::Raw(value),
        }
    }
}

/// Per-collective configuration. Users assign values to attributes to customize an
/// individual collective call issued through one of the configured entry points.
/// The same configuration must be set on every rank; NCCL validates it only locally.
#[derive(Debug, Clone)]
pub struct CollectiveConfig<'a> {
    /// Lower bound on channels/CTAs.
    pub min_ctas: Option<i32>,
    /// Upper bound on channels/CTAs, must be less than or equal to the communicator's maximum.
    pub max_ctas: Option<i32>,
    /// NVLS-pool-specific channel cap.
    pub nvls_ctas: Option<i32>,
    /// CUDA thread-block-cluster size (Hopper+), should be the same for all
    /// collectives in the same group. Inconsistent sizes in a group are undefined behavior.
    pub cluster_size: Option<i32>,
    /// Selection string; `None` or an empty string means automatic.
    pub algorithm: Option<&'a CStr>,
    /// Default true: unsatisfiable selection is an error.
    pub force_algorithm: bool,
    /// Inherit the communicator-level CTA policy when unset.
    pub cta_policy: Option<CtaPolicy>,
    /// An opaque value delivered verbatim to profiler plugins with this call's profiler events.
    /// It does not affect execution. Applications should use values with the MSB clear.
    pub profiler_tag: u64,
    /// At most one non-null event per communicator is supported in a group.
    pub launch_completion_event: Option<&'a LaunchCompletionEvent>,
    /// Vendor-specific extension options. The list is unordered and must not contain duplicate keys.
    pub extensions: &'a [Extension<'a>],
}

impl Default for CollectiveConfig<'_> {
    fn default() -> Self {
        Self {
            min_ctas: None,
            max_ctas: None,
            nvls_ctas: None,
            cluster_size: None,
            algorithm: None,
            force_algorithm: true,
            cta_policy: None,
            profiler_tag: 0,
            launch_completion_event: None,
            extensions: &[],
        }
    }
}

pub(crate) struct RawConfig {
    pub(crate) value: sys::ncclCollConfig_t,
    _extensions: Vec<sys::ncclConfigExt_t>,
}

impl CollectiveConfig<'_> {
    pub(crate) fn prepare(&self, communicator: &Communicator) -> Result<RawConfig> {
        if self.profiler_tag >> 63 != 0
            || [
                self.min_ctas,
                self.max_ctas,
                self.nvls_ctas,
                self.cluster_size,
            ]
            .into_iter()
            .flatten()
            .any(|v| v < 0)
            || matches!((self.min_ctas,self.max_ctas),(Some(min),Some(max)) if min > max)
        {
            return Err(Error::out_of_range("collective configuration"));
        }
        let mut extensions = Vec::with_capacity(self.extensions.len());
        for (index, extension) in self.extensions.iter().enumerate() {
            if !(1..1 << 24).contains(&extension.vendor_id)
                || self.extensions[..index].iter().any(|other| {
                    other.vendor_id == extension.vendor_id && other.option_id == extension.option_id
                })
            {
                return Err(Error::out_of_range("vendor extension key"));
            }
            extensions.push(sys::ncclConfigExt_t {
                next: ptr::null_mut(),
                key: sys::ncclConfigExt__bindgen_ty_1 {
                    vendorId: extension.vendor_id,
                    optionId: extension.option_id,
                },
                val: match extension.value {
                    ExtensionValue::Integer(i) => sys::ncclConfigExt__bindgen_ty_2 { i },
                    ExtensionValue::String(s) => sys::ncclConfigExt__bindgen_ty_2 { s: s.as_ptr() },
                    ExtensionValue::Raw(raw) => sys::ncclConfigExt__bindgen_ty_2 { raw },
                },
            });
        }
        for index in 1..extensions.len() {
            extensions[index - 1].next = &raw mut extensions[index];
        }
        let event = if let Some(event) = self.launch_completion_event {
            if event.event().context() != communicator.cuda_context() {
                return Err(Error::StreamContextMismatch);
            }
            let mut retained = communicator.handle.launch_events.borrow_mut();
            if !retained.iter().any(|old| Arc::ptr_eq(old, &event.0)) {
                retained.push(Arc::clone(&event.0));
            }
            event.event().as_raw()
        } else {
            ptr::null_mut()
        };
        validate_collective(communicator.as_raw(), self.cluster_size, !event.is_null())?;
        Ok(RawConfig {
            value: sys::ncclCollConfig_t {
                size: size_of::<sys::ncclCollConfig_t>() as _,
                magic: sys::NCCL_API_MAGIC,
                version: sys::NCCL_VERSION_CODE,
                ext: extensions.first_mut().map_or(ptr::null_mut(), |v| v),
                minCTAs: self.min_ctas.unwrap_or(i32::MIN),
                maxCTAs: self.max_ctas.unwrap_or(i32::MIN),
                nvlsCTAs: self.nvls_ctas.unwrap_or(i32::MIN),
                cgaClusterSize: self.cluster_size.unwrap_or(i32::MIN),
                algSelection: self.algorithm.map_or(ptr::null(), CStr::as_ptr),
                forceAlgSelection: self.force_algorithm.into(),
                CTAPolicy: self.cta_policy.map_or(i32::MIN, |policy| policy as i32),
                userProfilerTag: self.profiler_tag,
                launchCompletionEvent: event,
            },
            _extensions: extensions,
        })
    }
}
