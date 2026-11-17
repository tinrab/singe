use std::{
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    ops::Deref,
    ptr,
};

use singe_core::report_cleanup_failure;
use singe_nvml_sys as sys;

use crate::{
    compute_instance::{ComputeInstance, OwnedComputeInstance, RawComputeInstance},
    device::Device,
    error::{Error, Result, Status},
    library::Library,
    try_ffi,
    types::{
        ComputeInstanceEngineProfile, ComputeInstancePlacement, ComputeInstanceProfileInfo,
        EnableState, GpuInstanceInfo, VgpuInstanceId, VgpuPlacementId, VgpuSchedulerEngine,
        VgpuSchedulerLog, VgpuSchedulerState, VgpuTypeId,
    },
    utility::{struct_version, to_u32, to_usize},
    vgpu_instance::VgpuInstance,
    vgpu_type::VgpuType,
};

/// NVML GPU instance handle that does not own the instance.
///
/// It keeps NVML initialized.
/// Queries such as [`Device::gpu_instances`] return handles to instances the caller may not own, so the instance can be destroyed while a handle exists, by an [`OwnedGpuInstance`], another process, or a device reset.
/// NVML treats such a handle as invalid and rejects calls through it.
/// Equality and hashing compare the native handle.
#[derive(Debug)]
pub struct GpuInstance {
    raw: sys::nvmlGpuInstance_t,
    library: Library,
}

// SAFETY: NVML documents its API as thread-safe, the handle is an opaque identifier the wrapper never dereferences, and the retained `Library` is `Send`.
unsafe impl Send for GpuInstance {}

// SAFETY: NVML documents its API as thread-safe, and every `&self` method only passes the opaque handle to NVML.
unsafe impl Sync for GpuInstance {}

impl PartialEq for GpuInstance {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for GpuInstance {}

impl Hash for GpuInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

/// Owns an NVML GPU instance and requests its destruction on drop.
///
/// It lends out its instance's methods as a borrowed [`GpuInstance`].
/// [`Self::into_inner`] and [`Self::into_raw`] transfer destruction responsibility to the caller.
#[derive(Debug)]
pub struct OwnedGpuInstance {
    instance: GpuInstance,
}

/// A transferred NVML GPU instance handle with the initialization it was created under.
#[derive(Debug)]
pub struct RawGpuInstance {
    /// The owned native GPU instance handle.
    pub handle: sys::nvmlGpuInstance_t,
    /// The NVML initialization kept alive until `handle` is destroyed.
    pub library: Library,
}

// SAFETY: the record owns its handle exclusively, NVML documents its API as thread-safe, and the retained `Library` is `Send`.
unsafe impl Send for RawGpuInstance {}

impl GpuInstance {
    /// Wraps an NVML GPU instance handle and keeps `library`'s initialization alive.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlGpuInstance_t` obtained while `library`'s initialization was alive, and it must stay valid for the duration of any calls made through the returned wrapper.
    pub unsafe fn from_raw(handle: sys::nvmlGpuInstance_t, library: &Library) -> Self {
        Self {
            raw: handle,
            library: library.clone(),
        }
    }

    /// Returns the borrowed native instance handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlGpuInstance_t {
        self.raw
    }

    /// Returns whether the stored native handle is null.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.raw.is_null()
    }

    /// Returns the NVML initialization that this instance keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Returns GPU instance information.
    ///
    /// Requires an Ampere or newer GPU on Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle or output arguments are rejected by NVML, if the current process does not have permission to query the instance, or if NVML has not been initialized.
    pub fn info(&self) -> Result<GpuInstanceInfo> {
        let mut info = sys::nvmlGpuInstanceInfo_t::default();
        // SAFETY: `self.raw` is a GPU instance handle valid per `from_raw`'s contract, and `info` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetInfo(self.raw, &raw mut info))?;
        }
        // SAFETY: NVML filled `info` above, and its device handle stays valid while `self.library` keeps NVML initialized.
        Ok(unsafe { GpuInstanceInfo::from_raw(info, &self.library) })
    }

    /// Returns the physical GPU containing this GPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn parent_device(&self) -> Result<Device> {
        Ok(self.info()?.device)
    }

    /// Returns the GPU instance identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn id(&self) -> Result<u32> {
        Ok(self.info()?.id)
    }

    /// Returns the GPU instance profile identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::info`] fails.
    pub fn profile_id(&self) -> Result<u32> {
        Ok(self.info()?.profile_id)
    }

    /// Returns information about compute instance profile `profile` with engine profile `engine_profile`.
    ///
    /// Requires an Ampere or newer GPU on Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance, profile, engine profile, or structure version is invalid, if the profile is not supported, if the current process does not have permission to perform the operation, or if NVML has not been initialized.
    pub fn compute_instance_profile_info(
        &self,
        profile: u32,
        engine_profile: ComputeInstanceEngineProfile,
    ) -> Result<ComputeInstanceProfileInfo> {
        let mut info = sys::nvmlComputeInstanceProfileInfo_v3_t {
            version: struct_version::<sys::nvmlComputeInstanceProfileInfo_v3_t>(3),
            ..Default::default()
        };
        // SAFETY: `self.raw` is a live GPU instance handle, and `info` is a writable local struct of the version it declares.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstanceProfileInfoV(
                self.raw,
                profile,
                engine_profile.into(),
                (&raw mut info).cast(),
            ))?;
        }
        Ok(info.into())
    }

    /// Returns how many more compute instances of profile `profile_id` this GPU instance can create.
    ///
    /// Requires an Ampere or newer GPU on Linux, and privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or `profile_id` is invalid, if the profile is not supported, if the current process does not have permission to perform the operation, or if NVML has not been initialized.
    pub fn compute_instance_remaining_capacity(&self, profile_id: u32) -> Result<u32> {
        let mut count = 0;
        // SAFETY: `self.raw` is a live GPU instance handle, and `count` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstanceRemainingCapacity(
                self.raw,
                profile_id,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns every placement a compute instance of profile `profile_id` can take within this GPU instance.
    ///
    /// Requires an Ampere or newer GPU on Linux, and privileged access.
    ///
    /// Creating a compute instance fails if its placement overlaps occupied compute slices.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or `profile_id` is invalid, if MIG mode is not enabled or the profile is not supported, if the current process does not have permission to perform the operation, or if NVML has not been initialized.
    pub fn compute_instance_possible_placements(
        &self,
        profile_id: u32,
    ) -> Result<Vec<ComputeInstancePlacement>> {
        let mut count = 0;
        // SAFETY: `self.raw` is a live GPU instance handle, and a null array asks NVML only for the count, written to the local `count`.
        let status = unsafe {
            sys::nvmlGpuInstanceGetComputeInstancePossiblePlacements(
                self.raw,
                profile_id,
                ptr::null_mut(),
                &raw mut count,
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut placements = vec![
            sys::nvmlComputeInstancePlacement_t::default();
            to_usize(count, "placement count")?
        ];
        // SAFETY: `self.raw` is a live GPU instance handle, and `placements` holds the `count` writable entries passed as its length.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstancePossiblePlacements(
                self.raw,
                profile_id,
                placements.as_mut_ptr(),
                &raw mut count,
            ))?;
        }

        placements.truncate(to_usize(count, "placement count")?);
        Ok(placements.into_iter().map(Into::into).collect())
    }

    /// Returns the compute instance with ID `id`.
    ///
    /// Requires an Ampere or newer GPU on Linux, and privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or ID is invalid, if the compute instance is not found, if MIG mode is not enabled, if the current process lacks permission, or if NVML has not been initialized.
    pub fn compute_instance_by_id(&self, id: u32) -> Result<ComputeInstance> {
        let mut instance = ptr::null_mut();
        // SAFETY: `self.raw` is a live GPU instance handle, and NVML writes a valid compute instance handle on success, before it is wrapped with `self.library`.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstanceById(
                self.raw,
                id,
                &raw mut instance,
            ))?;
            Ok(ComputeInstance::from_raw(instance, &self.library))
        }
    }

    /// Creates a compute instance of profile `profile_id`, which is destroyed when the returned owner drops.
    /// The owner borrows this GPU instance, so the GPU instance cannot be destroyed first.
    ///
    /// Requires an Ampere or newer GPU on Linux, and privileged access.
    ///
    /// The handle becomes invalid if the parent GPU is unbound or reset or the parent GPU instance is destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested compute instance cannot be created, if the GPU instance or `profile_id` is invalid, if the profile is not supported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn create_compute_instance(&self, profile_id: u32) -> Result<OwnedComputeInstance<'_>> {
        let mut instance = ptr::null_mut();
        // SAFETY: `self.raw` is a live GPU instance handle, and NVML writes a new owned compute instance handle on success, before it is wrapped.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceCreateComputeInstance(
                self.raw,
                profile_id,
                &raw mut instance,
            ))?;
            OwnedComputeInstance::from_raw(RawComputeInstance {
                handle: instance,
                library: self.library.clone(),
            })
        }
    }

    /// Creates a compute instance of profile `profile_id` at `placement`, which is destroyed when the returned owner drops.
    /// The owner borrows this GPU instance, so the GPU instance cannot be destroyed first.
    ///
    /// Requires an Ampere or newer GPU on Linux, and privileged access.
    ///
    /// The handle becomes invalid if the parent GPU is unbound or reset or the parent GPU instance is destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested compute instance cannot be created, if the GPU instance, `profile_id`, or placement is invalid, if the profile is not supported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn create_compute_instance_with_placement(
        &self,
        profile_id: u32,
        placement: ComputeInstancePlacement,
    ) -> Result<OwnedComputeInstance<'_>> {
        let placement = sys::nvmlComputeInstancePlacement_t::from(placement);
        let mut instance = ptr::null_mut();
        // SAFETY: `self.raw` is a live GPU instance handle, `placement` is a local struct, and NVML writes a new owned compute instance handle on success, before it is wrapped.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceCreateComputeInstanceWithPlacement(
                self.raw,
                profile_id,
                &raw const placement,
                &raw mut instance,
            ))?;
            OwnedComputeInstance::from_raw(RawComputeInstance {
                handle: instance,
                library: self.library.clone(),
            })
        }
    }

    /// Returns the compute instances of profile `profile_id`.
    ///
    /// Requires an Ampere or newer GPU on Linux, and privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or `profile_id` is invalid, if the profile is not supported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn compute_instances(&self, profile_id: u32) -> Result<Vec<ComputeInstance>> {
        // NVML rejects a null array, and requires room for every instance the profile allows.
        let capacity = self.compute_instance_capacity(profile_id)?;
        let mut instances = vec![ptr::null_mut(); to_usize(capacity, "compute instance count")?];
        if instances.is_empty() {
            return Ok(Vec::new());
        }

        let mut count = 0;
        // SAFETY: `self.raw` is a live GPU instance handle, and `instances` holds the profile's `instanceCount` writable handle slots.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstances(
                self.raw,
                profile_id,
                instances.as_mut_ptr(),
                &raw mut count,
            ))?;
        }

        instances.truncate(to_usize(count, "compute instance count")?);
        Ok(instances
            .into_iter()
            // SAFETY: NVML filled these handles, truncated to the reported count, and each wrapper retains `self.library`.
            .map(|instance| unsafe { ComputeInstance::from_raw(instance, &self.library) })
            .collect())
    }

    // NVML looks compute instance profiles up only by index, so this finds the one with ID `profile_id`.
    fn compute_instance_capacity(&self, profile_id: u32) -> Result<u32> {
        for profile in 0..sys::NVML_COMPUTE_INSTANCE_PROFILE_COUNT {
            match self.compute_instance_profile_info(profile, ComputeInstanceEngineProfile::Shared)
            {
                Ok(info) if info.id == profile_id => return Ok(info.instance_count),
                Ok(_)
                | Err(Error::Nvml {
                    code: Status::NotSupported,
                }) => {}
                Err(error) => return Err(error),
            }
        }

        Err(Error::out_of_range("profile_id"))
    }

    /// Returns the vGPU types that can currently be created on this GPU instance.
    ///
    /// The set changes over time, since some vGPU types cannot run concurrently.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the intermediate size query reports a larger buffer requirement than expected, if this GPU instance or query arguments are invalid, if the host or GPU does not support vGPU creation, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn creatable_vgpus(&self) -> Result<Vec<VgpuType>> {
        let mut info = sys::nvmlVgpuTypeIdInfo_t {
            vgpuCount: 0,
            ..Default::default()
        };
        // SAFETY: `self.raw` is a live GPU instance handle, and `info` has a zero count and null array, so NVML only writes the count.
        let status = unsafe { sys::nvmlGpuInstanceGetCreatableVgpus(self.raw, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.vgpuCount == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut types = vec![0u32; to_usize(info.vgpuCount, "vGPU type count")?];
        info.vgpuTypeIds = types.as_mut_ptr();
        // SAFETY: `self.raw` is a live GPU instance handle, and `info.vgpuTypeIds` points to the `info.vgpuCount` writable entries of `types`.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetCreatableVgpus(
                self.raw,
                &raw mut info
            ))?;
        }

        types.truncate(to_usize(info.vgpuCount, "vGPU type count")?);
        Ok(types
            .into_iter()
            .map(|id| VgpuType::from_id(VgpuTypeId(id), &self.library))
            .collect())
    }

    /// Returns the active vGPU instances on this GPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the intermediate size query reports a larger buffer requirement than expected, if this GPU instance or query arguments are invalid, if the host or GPU does not support vGPU queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn active_vgpus(&self) -> Result<Vec<VgpuInstance>> {
        let mut info = sys::nvmlActiveVgpuInstanceInfo_t {
            version: struct_version::<sys::nvmlActiveVgpuInstanceInfo_t>(1),
            ..Default::default()
        };

        // SAFETY: `self.raw` is a live GPU instance handle, and `info` has a zero count and null array, so NVML only writes the count.
        let status = unsafe { sys::nvmlGpuInstanceGetActiveVgpus(self.raw, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.vgpuCount == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut instances = vec![0u32; to_usize(info.vgpuCount, "vGPU instance count")?];
        info.vgpuInstances = instances.as_mut_ptr();
        // SAFETY: `self.raw` is a live GPU instance handle, and `info.vgpuInstances` points to the `info.vgpuCount` writable entries of `instances`.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetActiveVgpus(self.raw, &raw mut info))?;
        }

        instances.truncate(to_usize(info.vgpuCount, "vGPU instance count")?);
        Ok(instances
            .into_iter()
            .map(|instance| VgpuInstance::from_id(VgpuInstanceId(instance), &self.library))
            .collect())
    }

    /// Returns whether vGPU heterogeneous mode is enabled on this GPU instance.
    ///
    /// In heterogeneous mode, time-sliced vGPUs with different framebuffer sizes run concurrently.
    /// Requires a Blackwell `GB20x` or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if this GPU instance or query arguments are invalid, if the host, GPU, or MIG mode does not support the query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_heterogeneous_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlVgpuHeterogeneousMode_t {
            version: struct_version::<sys::nvmlVgpuHeterogeneousMode_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is a live GPU instance handle, and `mode` is a writable local struct with its version set.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuHeterogeneousMode(
                self.raw,
                &raw mut mode,
            ))?;
        }
        Ok(EnableState::from(sys::nvmlEnableState_t(mode.mode)))
    }

    /// Returns the placement IDs where a vGPU of type `vgpu_type_id` can currently be created on this GPU instance.
    ///
    /// The set changes over time with the vGPUs already running.
    /// Requires a Blackwell `GB20x` or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the intermediate size query reports a larger buffer requirement than expected, if this GPU instance or query arguments are invalid, if the host or GPU does not support the query or vGPU heterogeneous mode is disabled, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_type_creatable_placements(
        &self,
        vgpu_type_id: VgpuTypeId,
    ) -> Result<Vec<VgpuPlacementId>> {
        let mut info = sys::nvmlVgpuCreatablePlacementInfo_t {
            version: struct_version::<sys::nvmlVgpuCreatablePlacementInfo_t>(1),
            vgpuTypeId: vgpu_type_id.0,
            placementSize: to_u32(size_of::<u32>(), "NVML buffer length")?,
            ..Default::default()
        };

        let status =
            // SAFETY: `self.raw` is a live GPU instance handle, and `info` has a zero count and null array, so NVML only writes the count.
            unsafe { sys::nvmlGpuInstanceGetVgpuTypeCreatablePlacements(self.raw, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut placements = vec![0u32; to_usize(info.count, "placement count")?];
        info.placementIds = placements.as_mut_ptr();
        // SAFETY: `self.raw` is a live GPU instance handle, and `info.placementIds` points to the `info.count` writable entries of `placements`.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuTypeCreatablePlacements(
                self.raw,
                &raw mut info,
            ))?;
        }

        placements.truncate(to_usize(info.count, "placement count")?);
        Ok(placements.into_iter().map(VgpuPlacementId).collect())
    }

    /// Returns the vGPU scheduler state of `engine` on this GPU instance.
    ///
    /// The details do not apply under the best-effort scheduler policy.
    /// Requires a Blackwell `GB20x` or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if this GPU instance or query arguments are invalid, if the host or GPU does not support vGPU scheduler queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_scheduler_state(&self, engine: VgpuSchedulerEngine) -> Result<VgpuSchedulerState> {
        let mut info = sys::nvmlVgpuSchedulerStateInfo_v2_t {
            engineId: engine.into(),
            ..Default::default()
        };
        // SAFETY: `self.raw` is a live GPU instance handle, and `info` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuSchedulerState_v2(
                self.raw,
                &raw mut info,
            ))?;
        }
        VgpuSchedulerState::from_raw(info)
    }

    /// Returns the recent vGPU scheduler log entries of `engine` on this GPU instance.
    ///
    /// NVML keeps a bounded number of entries, so reading the whole log takes at least five calls a second.
    /// Requires a Blackwell `GB20x` or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if this GPU instance or query arguments are invalid, if the host or GPU does not support vGPU scheduler queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_scheduler_log(&self, engine: VgpuSchedulerEngine) -> Result<VgpuSchedulerLog> {
        let mut info = sys::nvmlVgpuSchedulerLogInfo_v2_t {
            engineId: engine.into(),
            ..Default::default()
        };
        // SAFETY: `self.raw` is a live GPU instance handle, and `info` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuSchedulerLog_v2(
                self.raw,
                &raw mut info,
            ))?;
        }
        VgpuSchedulerLog::from_raw(&info)
    }
}

impl OwnedGpuInstance {
    /// Wraps an owned NVML GPU instance handle and keeps `raw.library`'s initialization alive.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid `nvmlGpuInstance_t` created while `raw.library`'s initialization was alive and owned by the caller, and ownership must not be used elsewhere after constructing this wrapper.
    /// The wrapper destroys the GPU instance with `nvmlGpuInstanceDestroy` unless ownership is transferred with [`OwnedGpuInstance::into_raw`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawGpuInstance) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            instance: GpuInstance {
                raw: raw.handle,
                library: raw.library,
            },
        })
    }

    /// Borrows the instance without transferring destruction responsibility.
    #[must_use]
    pub const fn as_gpu_instance(&self) -> &GpuInstance {
        &self.instance
    }

    /// Transfers destruction responsibility and returns the GPU instance wrapper.
    ///
    /// The caller must eventually destroy the native instance or place it in another owner.
    #[must_use]
    pub fn into_inner(self) -> GpuInstance {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so its only field moves out exactly once.
        unsafe { ptr::read(&raw const this.instance) }
    }

    /// Transfers this owned GPU instance handle to the caller.
    ///
    /// After this call, the wrapper will not destroy the GPU instance.
    /// The caller becomes responsible for eventually passing the handle to `nvmlGpuInstanceDestroy` or otherwise transferring ownership.
    /// The returned [`Library`] keeps NVML initialized; drop it only after the handle is destroyed.
    #[must_use]
    pub fn into_raw(self) -> RawGpuInstance {
        let GpuInstance { raw, library } = self.into_inner();
        RawGpuInstance {
            handle: raw,
            library,
        }
    }
}

impl Deref for OwnedGpuInstance {
    type Target = GpuInstance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for OwnedGpuInstance {
    fn drop(&mut self) {
        // SAFETY: `self` owns the non-null instance and destroys it once, while its retained library keeps NVML initialized.
        unsafe {
            if let Err(error) = try_ffi!(sys::nvmlGpuInstanceDestroy(self.instance.raw)) {
                report_cleanup_failure("destroy", "nvml gpu instance", &error);
            }
        }
    }
}
