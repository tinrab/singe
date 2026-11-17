//! Green contexts provision subsets of GPU resources for work submission.

use std::{
    fmt::{self, Debug, Formatter},
    mem, ptr,
    result::Result as StdResult,
    sync::Arc,
};

use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    device::Device,
    error::{Error, Result},
    event::Event,
    try_ffi,
    utility::to_usize,
};

/// Data for SM-related resources.
/// Properties are supplied by the driver.
#[derive(Clone)]
pub struct SmResource {
    raw: sys::CUdevResource,
    device: Device,
}

impl Debug for SmResource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmResource").finish_non_exhaustive()
    }
}

/// Input data for splitting SMs into structured groups.
#[derive(Debug, Clone, Copy)]
pub struct SmGroup {
    /// The amount of SMs available in this resource.
    pub sm_count: u32,
    /// The amount of co-scheduled SMs grouped together for locality purposes.
    pub coscheduled_sm_count: u32,
    /// When possible, combine co-scheduled groups into larger groups of this size.
    pub preferred_coscheduled_sm_count: u32,
    /// Locality domain to allocate the SM group from.
    pub locality_domain_id: Option<u32>,
    /// Allow unused SMs to be backfilled into this group.
    pub backfill: bool,
}

impl SmResource {
    /// Gets the SM resources available to the device.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query SM resources for `device`, including an invalid device ordinal, an uninitialized driver, or unsupported resource provisioning.
    pub fn query(device: Device) -> Result<Self> {
        let mut raw = sys::CUdevResource::default();
        // SAFETY: `raw` is a local the driver writes through an exclusive pointer, and the driver validates the device ordinal.
        unsafe {
            try_ffi!(sys::cuDeviceGetDevResource(
                device.id(),
                &raw mut raw,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM
            ))?;
        }
        Ok(Self { raw, device })
    }

    /// Gets the SM resources available to an ordinary CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the context's SM resource descriptor, including an unsupported resource type or invalid native context.
    pub fn for_context(context: &Context) -> Result<Self> {
        let mut raw = sys::CUdevResource::default();
        // SAFETY: `context` keeps its native context alive for the call, and `raw` is a local output.
        unsafe {
            try_ffi!(sys::cuCtxGetDevResource(
                context.as_raw(),
                &raw mut raw,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM
            ))?;
        }
        Ok(Self {
            raw,
            device: context.device(),
        })
    }

    /// Returns the number of streaming multiprocessors available in this resource.
    #[must_use]
    pub fn sm_count(&self) -> u32 {
        // SAFETY: the resource type is SM, so the `sm` field is the active union member.
        unsafe { self.raw.__bindgen_anon_1.sm.smCount }
    }

    /// Returns the locality domain the SMs must be located on, if the resource names one.
    /// With backfill enabled, SMs may be assigned from other locality domains.
    #[must_use]
    pub fn locality_domain_id(&self) -> Option<u32> {
        // SAFETY: the resource type is SM, so the `sm` field is the active union member.
        let sm = unsafe { self.raw.__bindgen_anon_1.sm };
        (sm.flags & sys::CUdevSmResourceGroup_flags::CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID.0
            != 0)
            .then_some(sm.localityDomainId)
    }

    /// Returns the minimum number of streaming multiprocessors required to partition this resource.
    #[must_use]
    pub fn minimum_partition_size(&self) -> u32 {
        // SAFETY: the resource type is SM, so the `sm` field is the active union member.
        unsafe { self.raw.__bindgen_anon_1.sm.minSmPartitionSize }
    }

    /// Returns the number of streaming multiprocessors guaranteed to be co-scheduled on the same GPU processing cluster.
    ///
    /// The resource's SM count is a multiple of this value unless backfill is enabled.
    #[must_use]
    pub fn coscheduled_alignment(&self) -> u32 {
        // SAFETY: the resource type is SM, so the `sm` field is the active union member.
        unsafe { self.raw.__bindgen_anon_1.sm.smCoscheduledAlignment }
    }

    /// Splits an SM resource into groups with a minimum SM count.
    ///
    /// # Errors
    ///
    /// Returns an error if `minimum_count` or `maximum_groups` is zero, or CUDA cannot split the resource, such as for an unsupported partition size.
    /// Successful results include the groups actually produced and the remainder.
    pub fn split_by_count(
        &self,
        minimum_count: u32,
        maximum_groups: u32,
    ) -> Result<(Vec<Self>, Self)> {
        if minimum_count == 0 {
            return Err(Error::not_positive("minimum count"));
        }

        if maximum_groups == 0 {
            return Err(Error::not_positive("maximum groups"));
        }

        let mut groups = maximum_groups;
        let mut results = vec![sys::CUdevResource::default(); to_usize(groups, "maximum groups")?];
        let mut remainder = sys::CUdevResource::default();
        // SAFETY: `results` holds `groups` descriptors for the driver to write, `self.raw` is an SM descriptor the driver produced, and the other outputs are locals.
        unsafe {
            try_ffi!(sys::cuDevSmResourceSplitByCount(
                results.as_mut_ptr(),
                &raw mut groups,
                &raw const self.raw,
                &raw mut remainder,
                0,
                minimum_count
            ))?;
        }

        results.truncate(to_usize(groups, "split group count")?);
        Ok((
            results
                .into_iter()
                .map(|raw| Self {
                    raw,
                    device: self.device,
                })
                .collect(),
            Self {
                raw: remainder,
                device: self.device,
            },
        ))
    }

    /// Splits an SM resource into structured groups and an unstructured remainder.
    ///
    /// # Errors
    ///
    /// Returns an error if the group count does not fit a `u32`, or CUDA cannot split the resource, such as for invalid group constraints or insufficient resources.
    pub fn split(&self, groups: &[SmGroup]) -> Result<(Vec<Self>, Self)> {
        let count = groups
            .len()
            .try_into()
            .map_err(|_| Error::out_of_range("group count"))?;
        let mut parameters = groups
            .iter()
            .map(|g| sys::CU_DEV_SM_RESOURCE_GROUP_PARAMS {
                smCount: g.sm_count,
                coscheduledSmCount: g.coscheduled_sm_count,
                preferredCoscheduledSmCount: g.preferred_coscheduled_sm_count,
                flags: if g.backfill {
                    sys::CUdevSmResourceGroup_flags::CU_DEV_SM_RESOURCE_GROUP_BACKFILL.0
                } else {
                    0
                } | if g.locality_domain_id.is_some() {
                    sys::CUdevSmResourceGroup_flags::CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID.0
                } else {
                    0
                },
                localityDomainId: g.locality_domain_id.unwrap_or(0),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let mut results = vec![sys::CUdevResource::default(); groups.len()];
        let mut remainder = sys::CUdevResource::default();
        // SAFETY: `results` and `parameters` each hold `count` elements, `self.raw` is an SM descriptor the driver produced, and `remainder` is a local output.
        unsafe {
            try_ffi!(sys::cuDevSmResourceSplit(
                results.as_mut_ptr(),
                count,
                &raw const self.raw,
                &raw mut remainder,
                0,
                parameters.as_mut_ptr()
            ))?;
        }
        Ok((
            results
                .into_iter()
                .map(|raw| Self {
                    raw,
                    device: self.device,
                })
                .collect(),
            Self {
                raw: remainder,
                device: self.device,
            },
        ))
    }
}

/// Sharing scope for workqueues.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
#[non_exhaustive]
pub enum WorkqueueSharingScope {
    /// Use all shared workqueue resources across all contexts.
    /// Default driver behaviour.
    DeviceContext = sys::CUdevWorkqueueConfigScope::CU_WORKQUEUE_SCOPE_DEVICE_CTX.0,
    /// When possible, use non-overlapping workqueue resources with other balanced green contexts.
    BalancedGreenContext = sys::CUdevWorkqueueConfigScope::CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED.0,
}

/// Data for workqueue configuration related resources.
#[derive(Clone)]
pub struct WorkqueueConfig {
    raw: sys::CUdevResource,
    device: Device,
}

impl Debug for WorkqueueConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkqueueConfig").finish_non_exhaustive()
    }
}

impl WorkqueueConfig {
    /// Gets the workqueue configuration resources available to the device.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query workqueue configuration resources for `device`, including an invalid device ordinal, an uninitialized driver, or unsupported resource provisioning.
    pub fn query(device: Device) -> Result<Self> {
        let mut raw = sys::CUdevResource::default();
        // SAFETY: `raw` is a local the driver writes through an exclusive pointer, and the driver validates the device ordinal.
        unsafe {
            try_ffi!(sys::cuDeviceGetDevResource(
                device.id(),
                &raw mut raw,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
            ))?;
        }
        Ok(Self { raw, device })
    }

    /// Sets the expected maximum number of concurrent stream-ordered workloads.
    #[must_use]
    pub fn with_concurrency_limit(mut self, limit: u32) -> Self {
        self.raw.__bindgen_anon_1.wqConfig.wqConcurrencyLimit = limit;
        self
    }

    /// Sets the sharing scope for the workqueue resources.
    #[must_use]
    pub fn with_sharing_scope(mut self, scope: WorkqueueSharingScope) -> Self {
        self.raw.__bindgen_anon_1.wqConfig.sharingScope = match scope {
            WorkqueueSharingScope::DeviceContext => {
                sys::CUdevWorkqueueConfigScope::CU_WORKQUEUE_SCOPE_DEVICE_CTX
            }
            WorkqueueSharingScope::BalancedGreenContext => {
                sys::CUdevWorkqueueConfigScope::CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
            }
        };
        self
    }
}

/// A green context with a specified set of resources.
///
/// A context handle obtained from [`Self::context_handle`] can be current to only one thread at a time.
#[derive(Debug)]
pub struct GreenContext {
    handle: sys::CUgreenCtx,
    device: Device,
}

/// Raw ownership of a green context and its device.
///
/// Destroy `handle` with CUDA, or re-adopt the value with [`GreenContext::from_raw`].
#[derive(Debug)]
pub struct RawGreenContext {
    /// Native green-context handle.
    pub handle: sys::CUgreenCtx,
    /// Device whose resources the green context provisions.
    pub device: Device,
}

// SAFETY: the wrapper never makes the green context current, and the driver calls it makes on the handle are thread-safe from any thread.
unsafe impl Send for GreenContext {}

// SAFETY: shared methods only pass the handle to thread-safe driver calls, and destruction needs the last owner.
unsafe impl Sync for GreenContext {}

impl GreenContext {
    /// Creates a green context from SM resources and an optional workqueue configuration, retaining the device primary context.
    ///
    /// # Errors
    ///
    /// Returns an error if `resources` is empty, the resources or the workqueue refer to different devices, the descriptor count does not fit a `u32`, or CUDA cannot generate the descriptor or create the green context.
    pub fn create(
        resources: &[SmResource],
        workqueue: Option<WorkqueueConfig>,
        default_stream: bool,
    ) -> Result<Arc<Self>> {
        let device = resources
            .first()
            .ok_or_else(|| Error::empty("sm resources"))?
            .device;
        if resources.iter().any(|r| r.device != device) {
            return Err(Error::out_of_range("sm resource device"));
        }

        let mut raw = resources.iter().map(|r| r.raw).collect::<Vec<_>>();
        if let Some(workqueue) = workqueue {
            if workqueue.device != device {
                return Err(Error::out_of_range("workqueue device"));
            }

            raw.push(workqueue.raw);
        }

        let mut descriptor = ptr::null_mut();
        // SAFETY: `raw` holds the counted descriptors the driver produced for one device, and `descriptor` is a local output.
        unsafe {
            try_ffi!(sys::cuDevResourceGenerateDesc(
                &raw mut descriptor,
                raw.as_mut_ptr(),
                raw.len()
                    .try_into()
                    .map_err(|_| Error::out_of_range("resource descriptor count"))?
            ))?;
        }

        let mut handle = ptr::null_mut();
        let flags = if default_stream {
            sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM.0
        } else {
            0
        };
        // SAFETY: `descriptor` was just generated for `device`'s resources, and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cuGreenCtxCreate(
                &raw mut handle,
                descriptor,
                device.id(),
                flags
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Arc::new(Self { handle, device }))
    }

    /// Returns the borrowed native green-context handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUgreenCtx {
        self.handle
    }

    /// Returns the device whose resources this green context provisions.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Takes ownership of a green context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be live, uniquely owned, and associated with `raw.device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the raw owner by value transfers ownership of its handle"
    )]
    pub unsafe fn from_raw(raw: RawGreenContext) -> Result<Arc<Self>> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Arc::new(Self {
            handle: raw.handle,
            device: raw.device,
        }))
    }

    /// Transfers ownership of the green context and its device to the caller.
    ///
    /// # Errors
    ///
    /// Returns an error holding the green context unchanged if another owner, such as a [`GreenStream`], still retains it.
    pub fn into_raw(this: Arc<Self>) -> StdResult<RawGreenContext, Arc<Self>> {
        let mut owner = Arc::try_unwrap(this)?;
        Ok(RawGreenContext {
            handle: mem::replace(&mut owner.handle, ptr::null_mut()),
            device: owner.device,
        })
    }

    /// Returns the unique ID associated with this green context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the green context's ID, including an invalid native handle or unavailable driver support.
    pub fn id(&self) -> Result<u64> {
        let mut id = 0;
        // SAFETY: `self` owns the live green context, and `id` is a local output.
        unsafe {
            try_ffi!(sys::cuGreenCtxGetId(self.handle, &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns a context handle representing this green context.
    ///
    /// The returned handle is borrowed and must not be destroyed or used after this owner is dropped.
    /// It can be current to only one thread at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot convert this green-context handle to its ordinary context representation.
    /// The returned handle does not acquire an independent ownership reference.
    pub fn context_handle(&self) -> Result<sys::CUcontext> {
        let mut context = ptr::null_mut();
        // SAFETY: `self` owns the live green context, and `context` is a local output.
        unsafe {
            try_ffi!(sys::cuCtxFromGreenCtx(&raw mut context, self.handle))?;
        }
        Ok(context)
    }

    /// Gets the SM resources provisioned for this green context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query this green context's provisioned SM resources.
    pub fn resources(&self) -> Result<SmResource> {
        let mut raw = sys::CUdevResource::default();
        // SAFETY: `self` owns the live green context, and `raw` is a local output.
        unsafe {
            try_ffi!(sys::cuGreenCtxGetDevResource(
                self.handle,
                &raw mut raw,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM
            ))?;
        }
        Ok(SmResource {
            raw,
            device: self.device,
        })
    }

    /// Records an event in this green context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the event recording, including invalid or incompatible event and green-context handles or prior asynchronous failures.
    pub fn record_event(&self, event: &Event) -> Result<()> {
        // SAFETY: `self` owns the live green context, and `event` keeps its native event alive for the call.
        unsafe { try_ffi!(sys::cuGreenCtxRecordEvent(self.handle, event.as_raw())) }
    }

    /// Makes all future work submitted to this green context wait for the event.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot order future green-context work after the event, including invalid handles, unsupported ordering, or prior asynchronous failures.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        // SAFETY: `self` owns the live green context, and `event` keeps its native event alive for the call.
        unsafe { try_ffi!(sys::cuGreenCtxWaitEvent(self.handle, event.as_raw())) }
    }

    /// Creates a nonblocking stream for use in this green context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot create a nonblocking stream for this green context with the requested priority.
    /// The successful stream retains the green context.
    pub fn create_stream(self: &Arc<Self>, priority: i32) -> Result<GreenStream> {
        let mut handle = ptr::null_mut();
        // SAFETY: `self` owns the live green context, which the returned stream retains, and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cuGreenCtxStreamCreate(
                &raw mut handle,
                self.handle,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING.0,
                priority
            ))?;
        }
        Ok(GreenStream {
            handle,
            context: Arc::clone(self),
        })
    }
}

impl Drop for GreenContext {
    fn drop(&mut self) {
        // SAFETY: a non-null handle is owned by this wrapper alone, and every stream that retained it has dropped.
        unsafe {
            if !self.handle.is_null()
                && let Err(error) = try_ffi!(sys::cuGreenCtxDestroy(self.handle))
            {
                report_cleanup_failure("destroy", "cuda green context", &error);
            }
        }
    }
}

/// A stream retaining the green context that provisioned its resources.
#[derive(Debug)]
pub struct GreenStream {
    handle: sys::CUstream,
    context: Arc<GreenContext>,
}

// SAFETY: a `CUstream` may be used and destroyed from any thread, and the retained green context is `Send + Sync`.
unsafe impl Send for GreenStream {}

// SAFETY: shared methods only pass the handle to thread-safe driver calls, and the wrapper has no interior mutability.
unsafe impl Sync for GreenStream {}

impl GreenStream {
    /// Returns the borrowed native green-context stream handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUstream {
        self.handle
    }

    /// Returns the green context retained by this stream.
    #[must_use]
    pub fn context(&self) -> &Arc<GreenContext> {
        &self.context
    }

    /// Waits until all work in this stream has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot synchronize the stream or reports an error from preceding asynchronous work.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `self` owns the live stream, whose green context it retains.
        unsafe { try_ffi!(sys::cuStreamSynchronize(self.handle)) }
    }
}

impl Drop for GreenStream {
    fn drop(&mut self) {
        // SAFETY: this wrapper owns the stream, and `self.context` keeps its green context alive until after the destroy.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuStreamDestroy(self.handle)) {
                report_cleanup_failure("destroy", "cuda green stream", &error);
            }
        }
    }
}

impl_enum_display!(WorkqueueSharingScope, {
    WorkqueueSharingScope::DeviceContext => "CU_WORKQUEUE_SCOPE_DEVICE_CTX",
    WorkqueueSharingScope::BalancedGreenContext => "CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED",
});
