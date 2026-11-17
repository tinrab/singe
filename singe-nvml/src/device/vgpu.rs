use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result},
    try_ffi,
    types::{
        DeviceVgpuCapability, EnableState, GridLicenseFeatureCode, HostVgpuMode, PgpuMetadata,
        VgpuInstanceId, VgpuLicenseExpiry, VgpuPlacementId, VgpuPlacementMode, VgpuTypeId,
        VirtualizationMode,
    },
    utility::{query_sized_raw, query_u32_list, struct_version, to_u32, to_usize},
    vgpu_instance::VgpuInstance,
    vgpu_type::VgpuType,
};

/// vGPU software licensable feature and its license state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridLicensableFeature {
    /// Licensed feature code.
    pub code: GridLicenseFeatureCode,
    /// Whether the feature is currently licensed.
    pub licensed: bool,
    /// Product name of the feature.
    pub product_name: String,
    /// Whether the feature is enabled.
    pub enabled: bool,
    /// License expiry structure containing date and time.
    pub expiry: VgpuLicenseExpiry,
}

/// vGPU software licensing support and the licensable features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridLicensableFeatures {
    /// Whether vGPU Software Licensing is supported on the system.
    pub supported: bool,
    /// Array of vGPU software licensable features.
    pub features: Vec<GridLicensableFeature>,
}

impl Device {
    /// Returns the virtualization mode corresponding to the GPU.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn virtualization_mode(&self) -> Result<VirtualizationMode> {
        let mut mode = sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_NONE.0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local with the layout of the transparent `nvmlGpuVirtualizationMode_t`.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetVirtualizationMode(
                self.raw,
                (&raw mut mode).cast::<sys::nvmlGpuVirtualizationMode_t>(),
            ))?;
        }
        Ok(VirtualizationMode::from_raw(mode))
    }

    /// Returns whether this vGPU-capable device runs in SR-IOV host mode.
    ///
    /// The device is in SR-IOV mode when both the device and the driver support SR-IOV host operation.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not support host vGPU mode reporting, or if NVML reports an unexpected failure.
    pub fn host_vgpu_mode(&self) -> Result<HostVgpuMode> {
        let mut mode = sys::nvmlHostVgpuMode_t::NVML_HOST_VGPU_MODE_NON_SRIOV;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHostVgpuMode(self.raw, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Returns the supported vGPU types on a physical GPU (device).
    ///
    /// This wrapper performs the NVML size query internally and returns the supported vGPU types as a [`Vec`].
    ///
    /// # Errors
    ///
    /// Returns an error if the vGPU type list changes while the wrapper is fetching it, if NVML rejects the query, if vGPU is unsupported by the device, or if NVML reports an unexpected failure.
    pub fn supported_vgpus(&self) -> Result<Vec<VgpuType>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `query_u32_list` passes a writable count with either null or an array of that many entries.
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetSupportedVgpus(self.raw, count, values)
        })?
        .into_iter()
        .map(|id| VgpuType::from_id(VgpuTypeId(id), &self.library))
        .collect())
    }

    /// Returns the currently creatable vGPU types on a physical GPU.
    ///
    /// The creatable vGPU types for a device may differ over time because restrictions can limit which vGPU types may run concurrently.
    /// For example, if only one vGPU type is allowed at a time on a device, the creatable list is restricted to the currently running vGPU type.
    /// This wrapper performs the NVML size query internally and returns the creatable vGPU types as a [`Vec`].
    ///
    /// # Errors
    ///
    /// Returns an error if the vGPU type list changes while the wrapper is fetching it, if NVML rejects the query, if vGPU is unsupported by the device, or if NVML reports an unexpected failure.
    pub fn creatable_vgpus(&self) -> Result<Vec<VgpuType>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `query_u32_list` passes a writable count with either null or an array of that many entries.
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetCreatableVgpus(self.raw, count, values)
        })?
        .into_iter()
        .map(|id| VgpuType::from_id(VgpuTypeId(id), &self.library))
        .collect())
    }

    /// Returns placements supported for a vGPU type in the selected placement mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the device, vGPU type, or placement mode is invalid or unsupported, the native buffer size cannot be represented, or the placement list outgrows the allocated buffer.
    /// Also propagates NVML enumeration failures.
    pub fn vgpu_type_supported_placements(
        &self,
        vgpu_type_id: VgpuTypeId,
        mode: VgpuPlacementMode,
    ) -> Result<Vec<VgpuPlacementId>> {
        self.query_vgpu_placements(
            vgpu_type_id,
            mode,
            sys::nvmlDeviceGetVgpuTypeSupportedPlacements,
        )
    }

    /// Returns placements currently available for creating a vGPU of this type.
    ///
    /// # Errors
    ///
    /// Returns an error if the device, vGPU type, or placement mode is invalid or unsupported, the native buffer size cannot be represented, or the placement list outgrows the allocated buffer.
    /// Also propagates NVML enumeration failures.
    pub fn vgpu_type_creatable_placements(
        &self,
        vgpu_type_id: VgpuTypeId,
        mode: VgpuPlacementMode,
    ) -> Result<Vec<VgpuPlacementId>> {
        self.query_vgpu_placements(
            vgpu_type_id,
            mode,
            sys::nvmlDeviceGetVgpuTypeCreatablePlacements,
        )
    }

    /// Returns the active vGPU instances on a device.
    ///
    /// This wrapper performs the NVML size query internally and returns the active vGPU instances as a [`Vec`].
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the active vGPU list changes while the wrapper is fetching it, if NVML rejects the query, if vGPU is unsupported by the device, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn active_vgpus(&self) -> Result<Vec<VgpuInstance>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `query_u32_list` passes a writable count with either null or an array of that many entries.
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetActiveVgpus(self.raw, count, values)
        })?
        .into_iter()
        .map(|instance| VgpuInstance::from_id(VgpuInstanceId(instance), &self.library))
        .collect())
    }

    /// Returns the vGPU heterogeneous mode for the device.
    ///
    /// When in heterogeneous mode, a vGPU can concurrently host timesliced vGPUs with differing framebuffer sizes.
    ///
    /// On success, returns the current vGPU heterogeneous mode as [`EnableState::Enabled`] or [`EnableState::Disabled`].
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the query, if MIG is enabled or the device does not support vGPU heterogeneous mode, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_heterogeneous_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlVgpuHeterogeneousMode_t {
            version: struct_version::<sys::nvmlVgpuHeterogeneousMode_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetVgpuHeterogeneousMode(
                self.raw,
                &raw mut mode
            ))?;
        }
        Ok(EnableState::from(sys::nvmlEnableState_t(mode.mode)))
    }

    /// Returns a vGPU metadata structure for this physical GPU.
    /// The structure contains information about the GPU and the currently installed NVIDIA host driver version that's controlling it, together with an opaque data section containing internal state.
    ///
    /// This wrapper queries the metadata size first, then fetches it into a buffer of that size.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata buffer changes while the wrapper is fetching it, if NVML rejects the metadata request, if vGPU is unsupported by the system, or if NVML reports an unexpected failure.
    pub fn vgpu_metadata(&self) -> Result<PgpuMetadata> {
        let raw = query_sized_raw(|size, buffer| {
            // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `query_sized_raw` passes a writable size with either null or a buffer of that many bytes.
            unsafe { sys::nvmlDeviceGetVgpuMetadata(self.raw, buffer.cast(), size) }
        })?;
        PgpuMetadata::decode_raw(&raw)
    }

    /// Returns this physical GPU's properties as an ASCII-encoded string.
    ///
    /// This wrapper queries the string size first, then fetches it into a buffer of that size.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata buffer changes while the wrapper is fetching it, if NVML rejects the metadata request, if vGPU is unsupported by the system, or if NVML reports an unexpected failure.
    pub fn pgpu_metadata_string(&self) -> Result<String> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `query_sized_raw` passes a writable size with either null or a buffer of that many bytes.
        let raw = query_sized_raw(|size, buffer| unsafe {
            sys::nvmlDeviceGetPgpuMetadataString(self.raw, buffer.cast(), size)
        })?;
        let buffer = raw.into_iter().map(u8::cast_signed).collect::<Vec<_>>();
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the requested vGPU capability of this device.
    ///
    /// See [`DeviceVgpuCapability`] for the supported capabilities.
    /// Returns whether the capability is supported and any associated capability data.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device, capability, or output, if the current state does not support this vGPU capability query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_capability(&self, capability: DeviceVgpuCapability) -> Result<u32> {
        let mut result = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `result` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetVgpuCapabilities(
                self.raw,
                capability.into(),
                &raw mut result,
            ))?;
        }
        Ok(result)
    }

    fn query_vgpu_placements(
        &self,
        vgpu_type_id: VgpuTypeId,
        mode: VgpuPlacementMode,
        query: unsafe extern "C" fn(
            sys::nvmlDevice_t,
            sys::nvmlVgpuTypeId_t,
            *mut sys::nvmlVgpuPlacementList_t,
        ) -> sys::nvmlReturn_t,
    ) -> Result<Vec<VgpuPlacementId>> {
        let mut placement_list = sys::nvmlVgpuPlacementList_t {
            version: struct_version::<sys::nvmlVgpuPlacementList_t>(2),
            placementSize: to_u32(size_of::<u32>(), "NVML buffer length")?,
            mode: mode as u32,
            ..Default::default()
        };

        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `placement_list` is a writable local whose version matches its layout and whose null `placementIds` asks NVML only for the count.
        let status = unsafe { query(self.raw, vgpu_type_id.0, &raw mut placement_list) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && placement_list.count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut placements = vec![0u32; to_usize(placement_list.count, "vGPU placement count")?];
        placement_list.placementIds = placements.as_mut_ptr();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `placement_list.placementIds` points to `placements`, which holds `count` writable entries.
        unsafe {
            try_ffi!(query(self.raw, vgpu_type_id.0, &raw mut placement_list))?;
        }

        placements.truncate(to_usize(placement_list.count, "vGPU placement count")?);
        Ok(placements.into_iter().map(VgpuPlacementId).collect())
    }

    /// Returns the vGPU software licensable features.
    ///
    /// Identifies whether the system supports vGPU Software Licensing.
    /// If it does, returns the licensable features and their current license status.
    ///
    /// # Errors
    ///
    /// Propagates NVML license-feature query errors.
    /// Returns an error if the reported feature count exceeds the native array capacity.
    pub fn grid_licensable_features(&self) -> Result<GridLicensableFeatures> {
        let mut raw = sys::nvmlGridLicensableFeatures_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `raw` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGridLicensableFeatures_v4(
                self.as_raw(),
                &raw mut raw
            ))?;
        }

        let features = raw
            .gridLicensableFeatures
            .get(..to_usize(raw.licensableFeaturesCount, "licensable feature count")?)
            .ok_or_else(|| Error::out_of_range("licensable feature count"))?
            .iter()
            .map(|feature| GridLicensableFeature {
                code: GridLicenseFeatureCode(feature.featureCode.0),
                licensed: feature.featureState != 0,
                product_name: string_from_c_chars(&feature.productName),
                enabled: feature.featureEnabled != 0,
                expiry: VgpuLicenseExpiry {
                    year: feature.licenseExpiry.year,
                    month: feature.licenseExpiry.month,
                    day: feature.licenseExpiry.day,
                    hour: feature.licenseExpiry.hour,
                    minute: feature.licenseExpiry.min,
                    second: feature.licenseExpiry.sec,
                    status: feature.licenseExpiry.status,
                },
            })
            .collect();
        Ok(GridLicensableFeatures {
            supported: raw.isGridLicenseSupported != 0,
            features,
        })
    }
}
