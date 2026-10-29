//! Green contexts provision subsets of GPU resources for work submission.

use std::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem, ptr,
    rc::Rc,
};

use singe_core::impl_enum_display;
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    device::Device,
    error::{Error, Result},
    event::Event,
    try_ffi,
};

/// Data for SM-related resources. Properties are supplied by the driver.
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
        // SAFETY: `cuDeviceGetDevResource` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuDeviceGetDevResource(
                device.id() as _,
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
        // SAFETY: `cuCtxGetDevResource` receives live handles owned by the wrappers and valid output locations.
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

    /// The amount of streaming multiprocessors available in this resource.
    #[must_use]
    pub fn sm_count(&self) -> u32 {
        // SAFETY: the resource type is SM, so the `sm` field is the active union member.
        unsafe { self.raw.__bindgen_anon_1.sm.smCount }
    }

    /// Locality domain that the SM must be located on, if specified in the flags.
    /// With backfill enabled, SMs may be assigned from other locality domains.
    #[must_use]
    pub fn locality_domain_id(&self) -> Option<u32> {
        // SAFETY: the resource type is SM, so the `sm` field is the active union member.
        let sm = unsafe { self.raw.__bindgen_anon_1.sm };
        (sm.flags
            & sys::CUdevSmResourceGroup_flags::CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID as u32
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
    /// Returns `Error::InvalidValue` if `minimum_count` or `maximum_groups` is zero. Also propagates native split errors for unavailable resources, unsupported partition sizes, or invalid resource descriptors. Successful results include the groups actually produced and the remainder.
    pub fn split_by_count(
        &self,
        minimum_count: u32,
        maximum_groups: u32,
    ) -> Result<(Vec<Self>, Self)> {
        if minimum_count == 0 || maximum_groups == 0 {
            return Err(Error::InvalidValue);
        }
        let mut groups = maximum_groups;
        let mut results = vec![sys::CUdevResource::default(); groups as usize];
        let mut remainder = sys::CUdevResource::default();
        // SAFETY: `cuDevSmResourceSplitByCount` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
        results.truncate(groups as usize);
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
    /// Returns `Error::InvalidValue` if the group count does not fit the native integer type. Also propagates native split errors for invalid group constraints, unsupported locality or co-scheduling settings, or insufficient resources.
    pub fn split(&self, groups: &[SmGroup]) -> Result<(Vec<Self>, Self)> {
        let count = groups.len().try_into().map_err(|_| Error::InvalidValue)?;
        let mut parameters = groups
            .iter()
            .map(|g| sys::CU_DEV_SM_RESOURCE_GROUP_PARAMS {
                smCount: g.sm_count,
                coscheduledSmCount: g.coscheduled_sm_count,
                preferredCoscheduledSmCount: g.preferred_coscheduled_sm_count,
                flags: if g.backfill {
                    sys::CUdevSmResourceGroup_flags::CU_DEV_SM_RESOURCE_GROUP_BACKFILL as u32
                } else {
                    0
                } | if g.locality_domain_id.is_some() {
                    sys::CUdevSmResourceGroup_flags::CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
                        as u32
                } else {
                    0
                },
                localityDomainId: g.locality_domain_id.unwrap_or(0),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let mut results = vec![sys::CUdevResource::default(); groups.len()];
        let mut remainder = sys::CUdevResource::default();
        // SAFETY: `cuDevSmResourceSplit` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    /// Use all shared workqueue resources across all contexts. Default driver behaviour.
    DeviceContext = sys::CUdevWorkqueueConfigScope::CU_WORKQUEUE_SCOPE_DEVICE_CTX as _,
    /// When possible, use non-overlapping workqueue resources with other balanced green contexts.
    BalancedGreenContext =
        sys::CUdevWorkqueueConfigScope::CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED as _,
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
        // SAFETY: `cuDeviceGetDevResource` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuDeviceGetDevResource(
                device.id() as _,
                &raw mut raw,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
            ))?;
        }
        Ok(Self { raw, device })
    }

    /// The expected maximum number of concurrent stream-ordered workloads.
    #[must_use]
    pub fn with_concurrency_limit(mut self, limit: u32) -> Self {
        self.raw.__bindgen_anon_1.wqConfig.wqConcurrencyLimit = limit;
        self
    }

    /// The sharing scope for the workqueue resources.
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
/// A green context can be current to only one thread at a time. This owner and
/// its streams are deliberately neither `Send` nor `Sync`.
#[derive(Debug)]
pub struct GreenContext {
    handle: sys::CUgreenCtx,
    device: Device,
    _thread: PhantomData<Rc<()>>,
}

impl GreenContext {
    /// Creates a green context from the specified resources, retaining the device primary context.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::create_with_workqueue`] with no explicit workqueue configuration.
    pub fn create(resources: &[SmResource], default_stream: bool) -> Result<Rc<Self>> {
        Self::create_with_workqueue(resources, None, default_stream)
    }

    /// Creates a green context with SM and optional workqueue configuration resources.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if the SM resource list is empty, resources refer to different devices, the workqueue refers to another device, or the descriptor count does not fit the native integer type. Also propagates native resource-descriptor generation and green-context creation errors, or `Error::NullHandle` for a null created handle.
    pub fn create_with_workqueue(
        resources: &[SmResource],
        workqueue: Option<WorkqueueConfig>,
        default_stream: bool,
    ) -> Result<Rc<Self>> {
        let device = resources.first().ok_or(Error::InvalidValue)?.device;
        if resources.iter().any(|r| r.device != device) {
            return Err(Error::InvalidValue);
        }
        let mut raw = resources.iter().map(|r| r.raw).collect::<Vec<_>>();
        if let Some(workqueue) = workqueue {
            if workqueue.device != device {
                return Err(Error::InvalidValue);
            }
            raw.push(workqueue.raw);
        }
        let mut descriptor = ptr::null_mut();
        // SAFETY: `cuDevResourceGenerateDesc` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuDevResourceGenerateDesc(
                &raw mut descriptor,
                raw.as_mut_ptr(),
                raw.len().try_into().map_err(|_| Error::InvalidValue)?
            ))?;
        }
        let mut handle = ptr::null_mut();
        let flags = if default_stream {
            sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32
        } else {
            0
        };
        // SAFETY: `cuGreenCtxCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuGreenCtxCreate(
                &raw mut handle,
                descriptor,
                device.id() as _,
                flags
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Rc::new(Self {
            handle,
            device,
            _thread: PhantomData,
        }))
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

    /// Takes ownership of a green context on this thread.
    ///
    /// # Safety
    /// The handle must be live, uniquely owned and associated with `device`.
    /// It must not be current to another thread.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null. The caller must satisfy the remaining validity, ownership, and thread-affinity requirements.
    pub unsafe fn from_raw(handle: sys::CUgreenCtx, device: Device) -> Result<Rc<Self>> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Rc::new(Self {
            handle,
            device,
            _thread: PhantomData,
        }))
    }

    /// Transfers ownership of a uniquely owned green context to the caller.
    #[must_use]
    pub fn into_raw(mut self) -> sys::CUgreenCtx {
        mem::replace(&mut self.handle, ptr::null_mut())
    }

    /// Returns the unique ID associated with this green context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the green context's ID, including an invalid native handle or unavailable driver support.
    pub fn id(&self) -> Result<u64> {
        let mut id = 0;
        // SAFETY: `cuGreenCtxGetId` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuGreenCtxGetId(self.handle, &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns a context handle representing this green context.
    ///
    /// The returned handle is borrowed and must not be destroyed or used after
    /// this owner is dropped. All use must remain on this thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot convert this green-context handle to its ordinary context representation. The returned handle does not acquire an independent ownership reference.
    pub fn context_handle(&self) -> Result<sys::CUcontext> {
        let mut context = ptr::null_mut();
        // SAFETY: `cuCtxFromGreenCtx` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `cuGreenCtxGetDevResource` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `cuGreenCtxRecordEvent` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::cuGreenCtxRecordEvent(self.handle, event.as_raw())) }
    }

    /// Makes all future work submitted to this green context wait for the event.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot order future green-context work after the event, including invalid handles, unsupported ordering, or prior asynchronous failures.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        // SAFETY: `cuGreenCtxWaitEvent` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::cuGreenCtxWaitEvent(self.handle, event.as_raw())) }
    }

    /// Creates a nonblocking stream for use in this green context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot create a nonblocking stream for this green context with the requested priority. The successful stream retains the green context.
    pub fn create_stream(self: &Rc<Self>, priority: i32) -> Result<GreenStream> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cuGreenCtxStreamCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuGreenCtxStreamCreate(
                &raw mut handle,
                self.handle,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as _,
                priority
            ))?;
        }
        Ok(GreenStream {
            handle,
            context: Rc::clone(self),
        })
    }
}

impl Drop for GreenContext {
    fn drop(&mut self) {
        // SAFETY: `cuGreenCtxDestroy` receives live handles owned by the wrappers.
        unsafe {
            if !self.handle.is_null()
                && let Err(error) = try_ffi!(sys::cuGreenCtxDestroy(self.handle))
            {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda green context: {error}");
            }
        }
    }
}

/// A stream retaining the green context that provisioned its resources.
#[derive(Debug)]
pub struct GreenStream {
    handle: sys::CUstream,
    context: Rc<GreenContext>,
}

impl GreenStream {
    /// Returns the borrowed native green-context stream handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUstream {
        self.handle
    }

    /// Returns the green context retained by this stream.
    #[must_use]
    pub fn context(&self) -> &Rc<GreenContext> {
        &self.context
    }

    /// Waits until all work in this stream has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot synchronize the stream or reports an error from preceding asynchronous work.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `cuStreamSynchronize` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::cuStreamSynchronize(self.handle)) }
    }
}

impl Drop for GreenStream {
    fn drop(&mut self) {
        // SAFETY: `cuStreamDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuStreamDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda green stream: {error}");
            }
        }
    }
}

impl_enum_display!(WorkqueueSharingScope, {
    WorkqueueSharingScope::DeviceContext => "CU_WORKQUEUE_SCOPE_DEVICE_CTX",
    WorkqueueSharingScope::BalancedGreenContext => "CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED",
});
