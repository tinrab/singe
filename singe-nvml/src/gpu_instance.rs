use std::{hash::Hash, mem::ManuallyDrop, ops::Deref, ptr};

use singe_nvml_sys as sys;

use crate::{
    compute_instance::{ComputeInstance, OwnedComputeInstance},
    device::Device,
    error::Result,
    try_ffi,
    types::{
        ComputeInstanceEngineProfile, ComputeInstancePlacement, ComputeInstanceProfileInfo,
        EnableState, GpuInstanceInfo, VgpuInstanceId, VgpuPlacementId, VgpuSchedulerEngine,
        VgpuSchedulerLog, VgpuSchedulerState, VgpuTypeId, try_from_nvml_enum,
    },
    utility::struct_version,
    vgpu_instance::VgpuInstance,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GpuInstance(sys::nvmlGpuInstance_t);

#[derive(Debug)]
pub struct OwnedGpuInstance(GpuInstance);

impl GpuInstance {
    /// Wraps a borrowed NVML GPU instance handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlGpuInstance_t` for the duration of any
    /// calls made through the returned wrapper.
    pub const unsafe fn from_raw(handle: sys::nvmlGpuInstance_t) -> Self {
        Self(handle)
    }

    pub const fn as_raw(&self) -> sys::nvmlGpuInstance_t {
        self.0
    }

    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns GPU instance information.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle or output arguments are rejected by NVML, if
    /// the current process does not have permission to query the instance, or if
    /// NVML has not been initialized.
    pub fn info(&self) -> Result<GpuInstanceInfo> {
        let mut info = sys::nvmlGpuInstanceInfo_t::default();
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetInfo(self.0, &raw mut info))?;
        }
        Ok(info.into())
    }

    pub fn parent_device(&self) -> Result<Device> {
        Ok(self.info()?.device)
    }

    pub fn id(&self) -> Result<u32> {
        Ok(self.info()?.id)
    }

    pub fn profile_id(&self) -> Result<u32> {
        Ok(self.info()?.profile_id)
    }

    /// Versioned wrapper that requests compute-instance profile information using the latest supported NVML output layout.
    ///
    /// This wrapper sets the version field on the output structure before calling NVML.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance, profile, engine profile, or structure
    /// version is invalid, if the profile is not supported, if the current process
    /// does not have permission to perform the operation, or if NVML has not been
    /// initialized.
    pub fn compute_instance_profile_info(
        &self,
        profile: u32,
        engine_profile: ComputeInstanceEngineProfile,
    ) -> Result<ComputeInstanceProfileInfo> {
        let mut info = sys::nvmlComputeInstanceProfileInfo_v3_t {
            version: struct_version::<sys::nvmlComputeInstanceProfileInfo_v3_t>(3),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstanceProfileInfoV(
                self.0,
                profile,
                engine_profile.into(),
                (&raw mut info).cast(),
            ))?;
        }
        Ok(info.into())
    }

    /// Returns compute instance profile capacity.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or `profile_id` is invalid, if the
    /// profile is not supported, if the current process does not have permission
    /// to perform the operation, or if NVML has not been initialized.
    pub fn compute_instance_remaining_capacity(&self, profile_id: u32) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstanceRemainingCapacity(
                self.0,
                profile_id,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns compute instance placements.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// A placement represents the location of a compute instance within a GPU instance.
    /// Returns all possible placements for the given profile.
    /// A created compute instance occupies compute slices described by its placement.
    /// Creating a compute instance fails if its placement overlaps already
    /// occupied compute slices.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or `profile_id` is invalid, if MIG mode
    /// is not enabled or the profile is not supported, if the current process does
    /// not have permission to perform the operation, or if NVML has not been
    /// initialized.
    pub fn compute_instance_possible_placements(
        &self,
        profile_id: u32,
    ) -> Result<Vec<ComputeInstancePlacement>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlGpuInstanceGetComputeInstancePossiblePlacements(
                self.0,
                profile_id,
                ptr::null_mut(),
                &raw mut count,
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut placements = vec![sys::nvmlComputeInstancePlacement_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstancePossiblePlacements(
                self.0,
                profile_id,
                placements.as_mut_ptr(),
                &raw mut count,
            ))?;
        }
        placements.truncate(count as usize);
        Ok(placements.into_iter().map(Into::into).collect())
    }

    /// Returns compute instance for the given instance ID.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or ID is invalid, if the compute
    /// instance is not found, if MIG mode is not enabled, if the current process
    /// lacks permission, or if NVML has not been initialized.
    pub fn compute_instance_by_id(&self, id: u32) -> Result<ComputeInstance> {
        let mut instance = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstanceById(
                self.0,
                id,
                &raw mut instance,
            ))?;
            Ok(ComputeInstance::from_raw(instance))
        }
    }

    /// Creates a compute instance.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// If the parent device is unbound or reset, or if the parent GPU instance or compute instance is destroyed, the compute instance handle becomes invalid.
    /// The compute instance must be recreated to acquire a valid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested compute instance cannot be created, if the
    /// GPU instance or `profile_id` is invalid, if the profile is not supported, if
    /// the current process lacks permission, or if NVML has not been initialized.
    pub fn create_compute_instance(&self, profile_id: u32) -> Result<OwnedComputeInstance> {
        let mut instance = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceCreateComputeInstance(
                self.0,
                profile_id,
                &raw mut instance,
            ))?;
            Ok(OwnedComputeInstance::from_raw(instance))
        }
    }

    /// Creates a compute instance with the specified placement.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// If the parent device is unbound or reset, or if the parent GPU instance or compute instance is destroyed, the compute instance handle becomes invalid.
    /// The compute instance must be recreated to acquire a valid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested compute instance cannot be created, if the
    /// GPU instance, `profile_id`, or placement is invalid, if the profile is not
    /// supported, if the current process lacks permission, or if NVML has not been
    /// initialized.
    pub fn create_compute_instance_with_placement(
        &self,
        profile_id: u32,
        placement: ComputeInstancePlacement,
    ) -> Result<OwnedComputeInstance> {
        let placement = sys::nvmlComputeInstancePlacement_t::from(placement);
        let mut instance = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceCreateComputeInstanceWithPlacement(
                self.0,
                profile_id,
                &raw const placement,
                &raw mut instance,
            ))?;
            Ok(OwnedComputeInstance::from_raw(instance))
        }
    }

    /// Returns compute instances for the given profile ID.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU instance or `profile_id` is invalid, if the
    /// profile is not supported, if the current process lacks permission, or if
    /// NVML has not been initialized.
    pub fn compute_instances(&self, profile_id: u32) -> Result<Vec<ComputeInstance>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlGpuInstanceGetComputeInstances(
                self.0,
                profile_id,
                ptr::null_mut(),
                &raw mut count,
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut instances = vec![ptr::null_mut(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetComputeInstances(
                self.0,
                profile_id,
                instances.as_mut_ptr(),
                &raw mut count,
            ))?;
        }
        instances.truncate(count as usize);
        Ok(instances
            .into_iter()
            .map(|instance| unsafe { ComputeInstance::from_raw(instance) })
            .collect())
    }

    /// Query the currently creatable vGPU types on a specific GPU instance.
    ///
    /// Returns the vGPU types that can currently be created for this GPU instance.
    /// This wrapper performs the NVML size query internally and returns the results as a [`Vec`].
    ///
    /// The creatable vGPU types may differ over time, as there may be restrictions on what type of vGPUs can concurrently run on the device.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the intermediate
    /// size query reports a larger buffer requirement than expected, if this GPU
    /// instance or query arguments are invalid, if the host or GPU does not support
    /// vGPU creation, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn creatable_vgpus(&self) -> Result<Vec<VgpuTypeId>> {
        let mut info = sys::nvmlVgpuTypeIdInfo_t {
            vgpuCount: 0,
            ..Default::default()
        };
        let status = unsafe { sys::nvmlGpuInstanceGetCreatableVgpus(self.0, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.vgpuCount == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut types = vec![0u32; info.vgpuCount as usize];
        info.vgpuTypeIds = types.as_mut_ptr();
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetCreatableVgpus(self.0, &raw mut info))?;
        }
        types.truncate(info.vgpuCount as usize);
        Ok(types.into_iter().map(VgpuTypeId).collect())
    }

    /// Returns the active vGPU instances within a GPU instance.
    ///
    /// This wrapper performs the NVML size query internally and returns the active vGPU instances as a [`Vec`].
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the intermediate
    /// size query reports a larger buffer requirement than expected, if this GPU
    /// instance or query arguments are invalid, if the host or GPU does not support
    /// vGPU queries, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn active_vgpus(&self) -> Result<Vec<VgpuInstance>> {
        let mut info = sys::nvmlActiveVgpuInstanceInfo_t {
            version: struct_version::<sys::nvmlActiveVgpuInstanceInfo_t>(1),
            ..Default::default()
        };

        let status = unsafe { sys::nvmlGpuInstanceGetActiveVgpus(self.0, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.vgpuCount == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut instances = vec![0u32; info.vgpuCount as usize];
        info.vgpuInstances = instances.as_mut_ptr();
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetActiveVgpus(self.0, &raw mut info))?;
        }
        instances.truncate(info.vgpuCount as usize);
        Ok(instances
            .into_iter()
            .map(|instance| VgpuInstance::from_id(VgpuInstanceId(instance)))
            .collect())
    }

    /// Returns the vGPU heterogeneous mode for the GPU instance.
    ///
    /// When in heterogeneous mode, a vGPU can concurrently host timesliced vGPUs with differing framebuffer sizes.
    ///
    /// On success, returns the current vGPU heterogeneous mode as [`EnableState::Enabled`] or [`EnableState::Disabled`].
    ///
    /// For Blackwell &tm GB20x; or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if this GPU instance
    /// or query arguments are invalid, if the host, GPU, or MIG mode does not
    /// support the query, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn vgpu_heterogeneous_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlVgpuHeterogeneousMode_t {
            version: struct_version::<sys::nvmlVgpuHeterogeneousMode_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuHeterogeneousMode(
                self.0,
                &raw mut mode,
            ))?;
        }
        try_from_nvml_enum("enable state", mode.mode)
    }

    /// Query the creatable vGPU placement ID of the vGPU type within a GPU instance.
    ///
    /// For Blackwell &tm GB20x; or newer fully supported devices.
    ///
    /// The returned placement IDs correspond to the given `vgpu_type_id`.
    /// This wrapper performs the NVML size query internally and returns the placement IDs as a [`Vec`].
    /// The creatable vGPU placement IDs may differ over time, as there may be restrictions on what type of vGPU the vGPU instance is running.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the intermediate
    /// size query reports a larger buffer requirement than expected, if this GPU
    /// instance or query arguments are invalid, if the host or GPU does not support
    /// the query or vGPU heterogeneous mode is disabled, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_type_creatable_placements(
        &self,
        vgpu_type_id: VgpuTypeId,
    ) -> Result<Vec<VgpuPlacementId>> {
        let mut info = sys::nvmlVgpuCreatablePlacementInfo_t {
            version: struct_version::<sys::nvmlVgpuCreatablePlacementInfo_t>(1),
            vgpuTypeId: vgpu_type_id.0,
            placementSize: size_of::<u32>() as u32,
            ..Default::default()
        };

        let status =
            unsafe { sys::nvmlGpuInstanceGetVgpuTypeCreatablePlacements(self.0, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut placements = vec![0u32; info.count as usize];
        info.placementIds = placements.as_mut_ptr();
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuTypeCreatablePlacements(
                self.0,
                &raw mut info,
            ))?;
        }
        placements.truncate(info.count as usize);
        Ok(placements.into_iter().map(VgpuPlacementId).collect())
    }

    /// Returns the vGPU scheduler state for the given GPU instance.
    /// The returned scheduler-state details are not relevant when the scheduler policy is best effort.
    ///
    /// For Blackwell &tm GB20x; or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if this GPU instance or query arguments are invalid, if the
    /// host or GPU does not support vGPU scheduler queries, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_scheduler_state(&self, engine: VgpuSchedulerEngine) -> Result<VgpuSchedulerState> {
        let mut info = sys::nvmlVgpuSchedulerStateInfo_v2_t {
            engineId: engine as u32,
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuSchedulerState_v2(
                self.0,
                &raw mut info,
            ))?;
        }
        VgpuSchedulerState::from_raw(info)
    }

    /// Returns the vGPU scheduler logs for this GPU instance.
    /// The number of returned elements never exceeds
    /// `NVML_SCHEDULER_SW_MAX_LOG_ENTRIES`.
    ///
    /// To get the entire logs, call this method at least 5 times a second.
    ///
    /// For Blackwell &tm GB20x; or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if this GPU instance or query arguments are invalid, if the
    /// host or GPU does not support vGPU scheduler queries, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_scheduler_log(&self, engine: VgpuSchedulerEngine) -> Result<VgpuSchedulerLog> {
        let mut info = sys::nvmlVgpuSchedulerLogInfo_v2_t {
            engineId: engine as u32,
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlGpuInstanceGetVgpuSchedulerLog_v2(
                self.0,
                &raw mut info,
            ))?;
        }
        VgpuSchedulerLog::from_raw(info)
    }
}

impl OwnedGpuInstance {
    /// Wraps an owned NVML GPU instance handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlGpuInstance_t` owned by the caller, and
    /// ownership must not be used elsewhere after constructing this wrapper.
    /// The wrapper destroys the GPU instance with `nvmlGpuInstanceDestroy`
    /// unless ownership is transferred with [`OwnedGpuInstance::into_raw`].
    pub const unsafe fn from_raw(handle: sys::nvmlGpuInstance_t) -> Self {
        Self(GpuInstance(handle))
    }

    pub const fn as_gpu_instance(&self) -> &GpuInstance {
        &self.0
    }

    pub fn into_inner(self) -> GpuInstance {
        let this = ManuallyDrop::new(self);
        unsafe { ptr::read(&this.0) }
    }

    /// Transfers this owned GPU instance handle to the caller.
    ///
    /// After this call, the wrapper will not destroy the GPU instance. The
    /// caller becomes responsible for eventually passing the handle to
    /// `nvmlGpuInstanceDestroy` or otherwise transferring ownership.
    pub fn into_raw(self) -> sys::nvmlGpuInstance_t {
        self.into_inner().0
    }
}

impl Deref for OwnedGpuInstance {
    type Target = GpuInstance;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for OwnedGpuInstance {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::nvmlGpuInstanceDestroy(self.0.0);
        }
    }
}
