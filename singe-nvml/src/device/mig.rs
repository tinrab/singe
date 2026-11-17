use std::ptr;

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result, Status},
    gpu_instance::{GpuInstance, OwnedGpuInstance, RawGpuInstance},
    try_ffi,
    types::{
        CurrentPending, GpuInstancePlacement, GpuInstanceProfileInfo, MigMode, MigModeActivation,
        try_from_nvml_enum,
    },
    utility::{struct_version, to_usize},
};

impl Device {
    /// Returns MIG mode for the device.
    ///
    /// For Ampere or newer fully supported devices.
    ///
    /// Changing MIG modes may require device unbind or reset.
    /// The "pending" MIG mode refers to the target mode following the next activation trigger.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or outputs, if the device does not support MIG mode, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn mig_mode(&self) -> Result<CurrentPending<MigMode>> {
        let mut current = 0;
        let mut pending = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `current` and `pending` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMigMode(
                self.raw,
                &raw mut current,
                &raw mut pending,
            ))?;
        }
        Ok(CurrentPending {
            current: try_from_nvml_enum("mig mode", current)?,
            pending: try_from_nvml_enum("mig mode", pending)?,
        })
    }

    /// Sets MIG mode for the device.
    ///
    /// For Ampere or newer fully supported devices.
    /// Requires root privileges.
    ///
    /// This mode determines whether a GPU instance can be created.
    ///
    /// This may unbind or reset the device to activate the requested mode.
    /// Thus, the attributes associated with the device, such as minor number, might change.
    /// Query such attributes again after changing the mode.
    ///
    /// On certain platforms, such as pass-through virtualization, reset may not be exposed directly and a VM reboot is required.
    /// In those cases, the returned activation status is [`Status::ResetRequired`].
    ///
    /// The returned activation status contains the appropriate error code when activation is unsuccessful.
    /// For example, if device unbind fails because the device is not idle, the status is [`Status::InUse`].
    /// Idle the device and retry setting the mode in that case.
    ///
    /// On Windows, only disabling MIG mode is supported.
    /// `activation_status` returns [`Status::NotSupported`] because GPU reset is not supported on Windows through this operation.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, requested mode, or activation-status output, if the device does not support MIG mode, if the current process lacks permission, or if NVML has not been initialized.
    pub fn set_mig_mode(&self, mode: MigMode) -> Result<MigModeActivation> {
        let mut activation_status = sys::nvmlReturn_t::NVML_SUCCESS;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `activation_status` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetMigMode(
                self.raw,
                mode.into(),
                &raw mut activation_status,
            ))?;
        }
        Ok(MigModeActivation::from_raw(activation_status))
    }

    /// Returns the maximum number of MIG devices that can exist under a parent NVML device.
    ///
    /// Returns zero if MIG is not supported or enabled.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_mig_device_count(&self) -> Result<u32> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `count` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxMigDeviceCount(
                self.raw,
                &raw mut count
            ))?;
        }
        Ok(count)
    }

    /// Returns whether this handle refers to a MIG device.
    ///
    /// A MIG device handle is an NVML abstraction which maps to a MIG compute instance.
    /// These overloaded references can be used (with some restrictions) interchangeably with a GPU device handle to execute queries at a per-compute instance granularity.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support this check, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn is_mig_device_handle(&self) -> Result<bool> {
        let mut is_mig_device = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `is_mig_device` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceIsMigDeviceHandle(
                self.raw,
                &raw mut is_mig_device,
            ))?;
        }
        Ok(is_mig_device != 0)
    }

    /// Returns the parent device of this MIG device handle.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the MIG device handle or parent output, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn parent_device_from_mig_handle(&self) -> Result<Self> {
        let mut device = ptr::null_mut();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `device` is a writable local, and `try_ffi!` returns before wrapping it unless NVML wrote a valid handle under the same initialization.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
                self.raw,
                &raw mut device,
            ))?;
            Ok(Self::from_raw(device, &self.library))
        }
    }

    /// Returns the MIG device at `index` under this parent device.
    ///
    /// If the compute instance is destroyed, either explicitly or by destroying, resetting, or unbinding the parent GPU instance or GPU device, the MIG device handle remains invalid and must be requested again.
    /// Handles may be reused and their properties can change in the process.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, index, or output, if no MIG device exists at `index`, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn mig_device(&self, index: u32) -> Result<Self> {
        let mut device = ptr::null_mut();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `device` is a writable local, and `try_ffi!` returns before wrapping it unless NVML wrote a valid handle under the same initialization.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMigDeviceHandleByIndex(
                self.raw,
                index,
                &raw mut device,
            ))?;
            Ok(Self::from_raw(device, &self.library))
        }
    }

    /// Returns the MIG device handles currently present on this GPU.
    ///
    /// Unpopulated MIG slots are skipped.
    ///
    /// # Errors
    ///
    /// Propagates maximum-count and device-handle query errors.
    /// [`Status::NotFound`] for an unpopulated slot is skipped; other NVML failures terminate enumeration.
    pub fn mig_devices(&self) -> Result<Vec<Self>> {
        let max_count = self.max_mig_device_count()?;
        let mut devices = Vec::with_capacity(to_usize(max_count, "MIG device count")?);

        for index in 0..max_count {
            match self.mig_device(index) {
                Ok(device) => devices.push(device),
                Err(Error::Nvml {
                    code: Status::NotFound,
                }) => {}
                Err(error) => return Err(error),
            }
        }

        Ok(devices)
    }

    /// Returns the GPU instance profile information for the profile index `profile`.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, profile, or request layout, if MIG mode is disabled or the profile is unsupported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instance_profile_info(&self, profile: u32) -> Result<GpuInstanceProfileInfo> {
        let mut info = sys::nvmlGpuInstanceProfileInfo_v3_t {
            version: struct_version::<sys::nvmlGpuInstanceProfileInfo_v3_t>(3),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable version 3 local whose version field tells NVML which layout to write.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceProfileInfoV(
                self.raw,
                profile,
                (&raw mut info).cast(),
            ))?;
        }
        Ok(info.into())
    }

    /// Returns the GPU instance profile information for the profile ID `profile_id`.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, profile ID, or request layout, if MIG mode is disabled or the profile is unsupported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instance_profile_info_by_id(
        &self,
        profile_id: u32,
    ) -> Result<GpuInstanceProfileInfo> {
        let mut info = sys::nvmlGpuInstanceProfileInfo_v3_t {
            version: struct_version::<sys::nvmlGpuInstanceProfileInfo_v3_t>(3),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable version 3 local whose version field tells NVML which layout to write.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceProfileInfoByIdV(
                self.raw,
                profile_id,
                (&raw mut info).cast(),
            ))?;
        }
        Ok(info.into())
    }

    /// Returns how many more GPU instances of `profile_id` can be created.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if MIG mode is disabled or `profile_id` is unsupported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instance_remaining_capacity(&self, profile_id: u32) -> Result<u32> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `count` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceRemainingCapacity(
                self.raw,
                profile_id,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns the possible GPU instance placements for `profile_id`.
    ///
    /// A placement represents the location of a GPU instance within a device.
    /// Returns all possible placements for the given profile, regardless of whether MIG is enabled.
    /// A created GPU instance occupies memory slices described by its placement.
    /// Creating a GPU instance fails if its placement overlaps already occupied memory slices.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not support MIG or `profile_id` is unsupported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instance_possible_placements(
        &self,
        profile_id: u32,
    ) -> Result<Vec<GpuInstancePlacement>> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and a null array with a writable `count` asks NVML only for the placement count.
        let status = unsafe {
            sys::nvmlDeviceGetGpuInstancePossiblePlacements_v2(
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
            sys::nvmlGpuInstancePlacement_t::default();
            to_usize(count, "GPU instance placement count")?
        ];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `placements` holds `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstancePossiblePlacements_v2(
                self.raw,
                profile_id,
                placements.as_mut_ptr(),
                &raw mut count,
            ))?;
        }

        placements.truncate(to_usize(count, "GPU instance placement count")?);
        Ok(placements.into_iter().map(Into::into).collect())
    }

    /// Returns the GPU instance with the ID `id`.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if no GPU instance has the requested ID, if MIG mode is disabled, if the current process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instance_by_id(&self, id: u32) -> Result<GpuInstance> {
        let mut instance = ptr::null_mut();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `instance` is a writable local, and `try_ffi!` returns before wrapping it unless NVML wrote a valid handle under the same initialization.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceById(
                self.raw,
                id,
                &raw mut instance,
            ))?;
            Ok(GpuInstance::from_raw(instance, &self.library))
        }
    }

    /// Creates a GPU instance.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// If the parent device is unbound or reset, or if the GPU instance is destroyed, the GPU instance handle becomes invalid.
    /// The GPU instance must be recreated to acquire a valid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML cannot allocate the requested GPU instance, if NVML rejects the query, if MIG mode is disabled or the device is a vGPU guest, if the current process lacks permission, or if NVML has not been initialized.
    pub fn create_gpu_instance(&self, profile_id: u32) -> Result<OwnedGpuInstance> {
        let mut instance = ptr::null_mut();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `instance` is a writable local, and `try_ffi!` returns before taking ownership unless NVML created a GPU instance under the same initialization.
        unsafe {
            try_ffi!(sys::nvmlDeviceCreateGpuInstance(
                self.raw,
                profile_id,
                &raw mut instance,
            ))?;
            OwnedGpuInstance::from_raw(RawGpuInstance {
                handle: instance,
                library: self.library.clone(),
            })
        }
    }

    /// Creates a GPU instance with the specified placement.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// If the parent device is unbound or reset, or if the GPU instance is destroyed, the GPU instance handle becomes invalid.
    /// The GPU instance must be recreated to acquire a valid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML cannot allocate the requested GPU instance, if NVML rejects the query or placement, if MIG mode is disabled or the device is a vGPU guest, if the current process lacks permission, or if NVML has not been initialized.
    pub fn create_gpu_instance_with_placement(
        &self,
        profile_id: u32,
        placement: GpuInstancePlacement,
    ) -> Result<OwnedGpuInstance> {
        let placement = sys::nvmlGpuInstancePlacement_t::from(placement);
        let mut instance = ptr::null_mut();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `placement` is a local NVML only reads, and `try_ffi!` returns before taking ownership unless NVML created a GPU instance.
        unsafe {
            try_ffi!(sys::nvmlDeviceCreateGpuInstanceWithPlacement(
                self.raw,
                profile_id,
                &raw const placement,
                &raw mut instance,
            ))?;
            OwnedGpuInstance::from_raw(RawGpuInstance {
                handle: instance,
                library: self.library.clone(),
            })
        }
    }

    /// Returns the GPU instances of `profile_id`.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if MIG mode is disabled or `profile_id` is unsupported, if the current process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instances(&self, profile_id: u32) -> Result<Vec<GpuInstance>> {
        // NVML rejects a null array, and requires room for every instance the profile allows.
        let capacity = self
            .gpu_instance_profile_info_by_id(profile_id)?
            .instance_count;
        let mut instances = vec![ptr::null_mut(); to_usize(capacity, "GPU instance count")?];
        if instances.is_empty() {
            return Ok(Vec::new());
        }

        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `instances` holds the profile's `instanceCount` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstances(
                self.raw,
                profile_id,
                instances.as_mut_ptr(),
                &raw mut count,
            ))?;
        }

        instances.truncate(to_usize(count, "GPU instance count")?);
        Ok(instances
            .into_iter()
            // SAFETY: NVML wrote these handles under `self.library`'s initialization, and the vector was truncated to the count it reported.
            .map(|instance| unsafe { GpuInstance::from_raw(instance, &self.library) })
            .collect())
    }

    /// Returns the GPU instance ID for this MIG device handle.
    ///
    /// GPU instance IDs are unique per device and remain valid until the GPU instance is destroyed.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gpu_instance_id(&self) -> Result<u32> {
        let mut id = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `id` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceId(self.raw, &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns the compute instance ID for this MIG device handle.
    ///
    /// Compute instance IDs are unique per GPU instance and remain valid until the compute instance is destroyed.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn compute_instance_id(&self) -> Result<u32> {
        let mut id = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `id` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetComputeInstanceId(self.raw, &raw mut id))?;
        }
        Ok(id)
    }
}
