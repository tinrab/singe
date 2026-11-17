mod clocks;
mod ecc;
mod memory;
mod mig;
mod nvlink;
mod pci;
mod power;
mod processes;
mod samples;
mod thermal;
mod topology;
mod utilization;
mod vgpu;

use std::{
    ffi::c_char,
    hash::{Hash, Hasher},
};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    error::{Error, Result},
    library::Library,
    try_ffi,
    types::{
        Architecture, BbxFlushTime, Brand, C2cModeInfo, ComputeCapability, ComputeMode,
        CurrentPending, DeviceAddressingMode, DeviceAttributes, DeviceCapabilities, DriverModel,
        DriverModelFlags, EnableState, EventTypes, FieldId, FieldQuery, FieldSample, FieldValue,
        GpuOperationMode, GspFirmwareMode, InforomObject, Pdi, PerfPolicyType, PlatformInfo,
        RepairStatus, RestrictedApi,
    },
    utility::{struct_version, to_u32},
};

pub use self::{
    ecc::BankRemapperStatus,
    memory::{MemoryLimit, MemoryLimits},
    nvlink::{
        NvLinkBandwidthMode, NvLinkTelemetryRequest, NvLinkTelemetrySamples, NvLinkTelemetryType,
    },
    power::{
        AdaptiveTgpModeInfo, CoreRailMetrics, ObservedMetrics, PerformanceControllerSample,
        PerformanceMetricsDlppc2xSample, PerformanceMetricsPfpp1xSample, PerformanceSample,
        PowerModelMetricsDlppm1x, PowerModelMetricsDlppm1xDramclkEstimates,
        PowerModelMetricsDlppm1xPerf, PowerModelMetricsPfpp1x, PowerModelMetricsSamplePfpp1x,
        PowerModelOperatingPointPfpp1x, PowerTuple, RailMetrics,
    },
    samples::{PrmCounterId, PrmCounterRequest},
    vgpu::{GridLicensableFeature, GridLicensableFeatures},
};

/// NVML GPU device handle.
///
/// It keeps NVML initialized, and clones refer to the same native device.
/// Equality and hashing compare the native handle.
#[derive(Debug, Clone)]
pub struct Device {
    raw: sys::nvmlDevice_t,
    library: Library,
}

// SAFETY: NVML is thread-safe, and a device handle is a process-wide identifier the wrapper never dereferences, so it may move to another thread together with its `Send` `Library`.
unsafe impl Send for Device {}

// SAFETY: NVML is thread-safe, every method only passes the handle to NVML by value, and the retained `Library` is `Sync`.
unsafe impl Sync for Device {}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for Device {}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl Device {
    /// Wraps an NVML device handle and keeps `library`'s initialization alive.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid NVML device handle obtained while `library`'s initialization was alive.
    /// NVML owns the underlying device handle; this wrapper does not destroy it.
    pub unsafe fn from_raw(handle: sys::nvmlDevice_t, library: &Library) -> Self {
        Self {
            raw: handle,
            library: library.clone(),
        }
    }

    /// Returns the borrowed native handle.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvmlDevice_t {
        self.raw
    }

    /// Returns whether the stored native handle is null.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.raw.is_null()
    }

    /// Returns the NVML initialization that this device keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Returns the NVML index of this device.
    ///
    /// For all products.
    ///
    /// Valid indices are derived from the accessible device count returned by [`Library::device_count`](crate::library::Library::device_count).
    /// For example, if the count is 2 the valid indices are 0 and 1, corresponding to GPU 0 and GPU 1.
    ///
    /// The order in which NVML enumerates devices has no guarantees of consistency between reboots.
    /// Prefer PCI bus IDs or GPU UUIDs for stable device lookup.
    /// See [`Library::device_by_pci_bus_id`](crate::library::Library::device_by_pci_bus_id) and [`Library::device_by_uuid`](crate::library::Library::device_by_uuid).
    ///
    /// With MIG device handles, this returns indices that can be passed to [`Device::mig_device`] to retrieve an identical handle.
    /// MIG device indices are unique within a device.
    ///
    /// The NVML index may not correlate with other APIs, such as the CUDA device index.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or index output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn index(&self) -> Result<u32> {
        let mut index = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `index` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetIndex(self.raw, &raw mut index))?;
        }
        Ok(index)
    }

    /// Returns the name of this device.
    ///
    /// For all products.
    ///
    /// The name is an alphanumeric product identifier such as `Tesla C2070`.
    /// It does not exceed 96 bytes including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// With MIG device handles, this returns MIG device names that can identify devices based on their attributes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the internal name buffer is too small, if NVML rejects the handle or output buffer, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn name(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_DEVICE_NAME_BUFFER_SIZE as usize];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the passed length is the length of the writable `buffer`.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetName(
                self.raw,
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the hostname for the device.
    ///
    /// For Blackwell or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or hostname output, if the device does not support hostnames, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn hostname(&self) -> Result<String> {
        let mut hostname = sys::nvmlHostname_v1_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `hostname` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHostname_v1(self.raw, &raw mut hostname))?;
        }
        Ok(string_from_c_chars(&hostname.value))
    }

    /// Returns the brand of this device.
    ///
    /// For all products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or brand output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn brand(&self) -> Result<Brand> {
        let mut brand = sys::nvmlBrandType_t::NVML_BRAND_UNKNOWN.0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `brand` is a writable local with the layout of the transparent `nvmlBrandType_t`.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBrand(
                self.raw,
                (&raw mut brand).cast::<sys::nvmlBrandType_t>(),
            ))?;
        }
        Ok(Brand::from_raw(brand))
    }

    /// Returns architecture for the device.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or architecture output, or if NVML has not been initialized.
    pub fn architecture(&self) -> Result<Architecture> {
        let mut architecture = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `architecture` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetArchitecture(
                self.raw,
                &raw mut architecture
            ))?;
        }
        Ok(Architecture::from_raw(architecture))
    }

    /// Returns the device serial number.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the query buffer is insufficient, or serial-number reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn serial(&self) -> Result<String> {
        device_string_query(
            self,
            sys::NVML_DEVICE_SERIAL_BUFFER_SIZE as usize,
            sys::nvmlDeviceGetSerial,
        )
    }

    /// Returns the device UUID.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the query buffer is insufficient, or UUID reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn uuid(&self) -> Result<String> {
        device_string_query(
            self,
            sys::NVML_DEVICE_UUID_V2_BUFFER_SIZE as usize,
            sys::nvmlDeviceGetUUID,
        )
    }

    /// Returns the video BIOS version string.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the query buffer is insufficient, or video BIOS version reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn vbios_version(&self) -> Result<String> {
        // From docs: "It will not exceed 32 characters in length (including the terminating NUL byte)"
        // https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g4a02015969489ad8a28d8c56b34c825e
        device_string_query(self, 32, sys::nvmlDeviceGetVbiosVersion)
    }

    /// Returns the board part number.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the query buffer is insufficient, or board part-number reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn board_part_number(&self) -> Result<String> {
        device_string_query(
            self,
            sys::NVML_DEVICE_PART_NUMBER_BUFFER_SIZE as usize,
            sys::nvmlDeviceGetBoardPartNumber,
        )
    }

    /// Returns a unique identifier for the device module on the baseboard.
    ///
    /// Returns a unique identifier for each GPU module on a given baseboard.
    /// For non-baseboard products, this ID would always be 0.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or module ID output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn module_id(&self) -> Result<u32> {
        let mut module_id = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `module_id` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetModuleId(self.raw, &raw mut module_id))?;
        }
        Ok(module_id)
    }

    /// Returns whether the device is on a multi-GPU board.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn is_multi_gpu_board(&self) -> Result<bool> {
        let mut multi_gpu = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `multi_gpu` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMultiGpuBoard(
                self.raw,
                &raw mut multi_gpu
            ))?;
        }
        Ok(multi_gpu != 0)
    }

    /// Returns attributes (engine counts etc.) for the given NVML device handle.
    ///
    /// This currently supports only MIG device handles.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device handle, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn attributes(&self) -> Result<DeviceAttributes> {
        let mut attributes = sys::nvmlDeviceAttributes_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `attributes` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAttributes_v2(
                self.raw,
                &raw mut attributes
            ))?;
        }
        Ok(attributes.into())
    }

    /// Returns the root/admin permissions for the target NVML operation.
    /// See [`RestrictedApi`] for the list of supported operations.
    /// If an operation is restricted, only callers with root privileges can call it.
    ///
    /// For all fully supported products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, API restriction, or output, if the device or queried feature does not support API restriction reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn is_api_restricted(&self, api: RestrictedApi) -> Result<EnableState> {
        let mut state = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `state` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAPIRestriction(
                self.raw,
                api.into(),
                &raw mut state,
            ))?;
        }
        Ok(state.into())
    }

    /// Returns platform information of this device.
    ///
    /// For Blackwell or newer fully supported devices.
    ///
    /// Returns the platform information reported by NVML for this device.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the request, if system memory is insufficient, if the device does not support this query, or if NVML reports an unexpected failure.
    pub fn platform_info(&self) -> Result<PlatformInfo> {
        let mut info = sys::nvmlPlatformInfo_t {
            version: struct_version::<sys::nvmlPlatformInfo_t>(2),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPlatformInfo(self.raw, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns the Per Device Identifier (PDI) associated with this device.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// Returns the per-device identifier reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML rejects the handle or output, if the device does not support PDI reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn pdi(&self) -> Result<Pdi> {
        let mut pdi = sys::nvmlPdi_t {
            version: struct_version::<sys::nvmlPdi_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `pdi` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPdi(self.raw, &raw mut pdi))?;
        }
        Ok(pdi.into())
    }

    /// Returns the Device's C2C Mode information.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support C2C mode reporting, or if NVML reports an unexpected failure.
    pub fn c2c_mode_info(&self) -> Result<C2cModeInfo> {
        let mut info = sys::nvmlC2cModeInfo_v1_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetC2cModeInfoV(self.raw, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns the CUDA compute capability of the device.
    ///
    /// For all products.
    ///
    /// Returns the major and minor compute capability version numbers of the device.
    /// The versions match the compute capability that the CUDA driver reports for the device.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or outputs, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn compute_capability(&self) -> Result<ComputeCapability> {
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `major` and `minor` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCudaComputeCapability(
                self.raw,
                &raw mut major,
                &raw mut minor,
            ))?;
        }
        Ok(ComputeCapability { major, minor })
    }

    /// Returns the event types supported by this device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// Events are not supported on Windows.
    /// Therefore, this call returns an empty event mask on Windows.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the event mask output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn supported_event_types(&self) -> Result<EventTypes> {
        let mut event_types = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `event_types` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSupportedEventTypes(
                self.raw,
                &raw mut event_types,
            ))?;
        }
        Ok(EventTypes::from_bits_retain(event_types))
    }

    /// Returns the device board ID in the range `0..N`.
    /// Devices with the same board ID indicate GPUs connected to the same PLX.
    /// Use in conjunction with [`Device::is_multi_gpu_board`] to decide if they are on the same board as well.
    /// The returned board ID is unique for the current configuration.
    /// Uniqueness and ordering across reboots and system configurations are not guaranteed, but IDs remain distinct within one configuration.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support board IDs, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn board_id(&self) -> Result<u32> {
        let mut board_id = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `board_id` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBoardId(self.raw, &raw mut board_id))?;
        }
        Ok(board_id)
    }

    /// Returns the display mode for the device.
    ///
    /// For all products.
    ///
    /// Indicates whether a physical display, such as a monitor, is currently connected to any of the device's connectors.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support display-mode reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn display_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDisplayMode(self.raw, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Returns the display active state for the device.
    ///
    /// For all products.
    ///
    /// Indicates whether a display is initialized on the device.
    /// For example whether X Server is attached to this device and has allocated memory for the screen.
    ///
    /// Display can be active even when no monitor is physically attached.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if the device does not support display-active reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn display_active(&self) -> Result<EnableState> {
        let mut active = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `active` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDisplayActive(self.raw, &raw mut active))?;
        }
        Ok(active.into())
    }

    /// Returns the current GPU operation mode and the pending mode that will take effect after reboot.
    ///
    /// For GK110 M-class and X-class Tesla products from the Kepler family.
    /// Modes [`GpuOperationMode::LowDp`] and [`GpuOperationMode::AllOn`] are supported on fully supported `GeForce` products.
    /// Not supported on Quadro and Tesla C-class products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or outputs, if the device does not support GPU operation modes, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gpu_operation_mode(&self) -> Result<CurrentPending<GpuOperationMode>> {
        let mut current = sys::nvmlGpuOperationMode_t::NVML_GOM_ALL_ON;
        let mut pending = sys::nvmlGpuOperationMode_t::NVML_GOM_ALL_ON;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `current` and `pending` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuOperationMode(
                self.raw,
                &raw mut current,
                &raw mut pending,
            ))?;
        }
        Ok(CurrentPending {
            current: current.into(),
            pending: pending.into(),
        })
    }

    /// Returns the current and pending driver model for the device.
    ///
    /// For Kepler or newer fully supported devices.
    /// For windows only.
    ///
    /// On Windows platforms the device driver can run in either WDDM, MCDM or WDM (TCC) modes.
    /// If a display is attached to the device it must run in WDDM mode.
    /// MCDM mode is preferred if a display is not attached.
    /// TCC mode is deprecated.
    ///
    /// See [`DriverModel`] for details on available driver models.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or outputs, if the platform is not Windows, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn driver_model(&self) -> Result<CurrentPending<DriverModel>> {
        let mut current = sys::nvmlDriverModel_t::NVML_DRIVER_WDDM;
        let mut pending = sys::nvmlDriverModel_t::NVML_DRIVER_WDDM;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `current` and `pending` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDriverModel_v2(
                self.raw,
                &raw mut current,
                &raw mut pending,
            ))?;
        }
        Ok(CurrentPending {
            current: current.into(),
            pending: pending.into(),
        })
    }

    /// Sets the driver model for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// For windows only.
    /// Requires root/admin permissions.
    ///
    /// On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode.
    /// If a display is attached to the device it must run in WDDM mode.
    ///
    /// It is possible to force the change to WDM (TCC) while the display is still attached with [`DriverModelFlags::FORCE`].
    /// Use the force flag only if the host is powered down afterward and the display is detached from the device before the next reboot.
    ///
    /// This operation takes effect after the next reboot.
    ///
    /// Windows driver model may only be set to WDDM when running in DEFAULT compute mode.
    ///
    /// Changing the driver model to WDDM is not supported when the GPU does not support graphics acceleration, or would not support it after reboot.
    /// See [`Device::set_gpu_operation_mode`].
    ///
    /// See [`DriverModel`] for details on available driver models.
    /// See [`DriverModelFlags`] for the available flag values.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, driver model, or flags, if the platform is not Windows or the device does not support driver-model changes, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_driver_model(&self, model: DriverModel, flags: DriverModelFlags) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the model and flags are passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDriverModel(
                self.raw,
                model.into(),
                flags.bits(),
            ))?;
        }
        Ok(())
    }

    /// Sets the GPU operation mode.
    /// See [`GpuOperationMode`] for details.
    ///
    /// For GK110 M-class and X-class Tesla products from the Kepler family.
    /// Modes [`GpuOperationMode::LowDp`] and [`GpuOperationMode::AllOn`] are supported on fully supported `GeForce` products.
    /// Not supported on Quadro and Tesla C-class products.
    /// Requires root/admin permissions.
    ///
    /// Changing GOMs requires a reboot.
    /// The reboot requirement might be removed in the future.
    ///
    /// Compute-only GOMs do not support graphics acceleration.
    /// On Windows, switching to these GOMs is not supported when the pending driver model is WDDM.
    /// See [`Device::set_driver_model`].
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or mode, if the device does not support GPU operation mode changes or the requested mode, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_gpu_operation_mode(&self, mode: GpuOperationMode) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the mode is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetGpuOperationMode(self.raw, mode.into()))?;
        }
        Ok(())
    }

    /// Returns the device information-ROM image version.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the query buffer is insufficient, or information-ROM version reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn inforom_image_version(&self) -> Result<String> {
        device_string_query(self, 16, sys::nvmlDeviceGetInforomImageVersion)
    }

    /// Returns the version information for the device's infoROM object.
    ///
    /// For all products with an inforom.
    ///
    /// Fermi and higher parts have non-volatile on-board memory for persisting device info, such as aggregate ECC counts.
    /// The version of the data structures in this memory may change from time to time.
    /// It does not exceed 16 bytes including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// See [`InforomObject`] for details on the available infoROM objects.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the internal infoROM version buffer is too small, if NVML rejects the output, if the device does not have an infoROM, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn inforom_version(&self, object: InforomObject) -> Result<String> {
        let mut buffer = vec![0; sys::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE as usize];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the passed length is the length of the writable `buffer`.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetInforomVersion(
                self.as_raw(),
                object.into(),
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the checksum of the configuration stored in the device's infoROM.
    ///
    /// For all products with an inforom.
    ///
    /// Can be used to make sure that two GPUs have the exact same configuration.
    /// Current checksum takes into account configuration stored in PWR and ECC infoROM objects.
    /// The checksum can change between driver releases or when configuration changes, such as disabling or enabling ECC.
    ///
    /// # Errors
    ///
    /// Returns an error if the infoROM checksum cannot be read because the infoROM is corrupted, if the device is inaccessible, if NVML rejects the output, if the device does not support checksums, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn inforom_configuration_checksum(&self) -> Result<u32> {
        let mut checksum = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `checksum` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetInforomConfigurationChecksum(
                self.raw,
                &raw mut checksum,
            ))?;
        }
        Ok(checksum)
    }

    /// Reads the infoROM from the flash and verifies the checksums.
    ///
    /// For all products with an inforom.
    ///
    /// # Errors
    ///
    /// Returns an error if the infoROM is corrupted, if the device is inaccessible, if the device does not support infoROM validation, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn validate_inforom(&self) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the handle is the only argument.
        unsafe { try_ffi!(sys::nvmlDeviceValidateInforom(self.raw)) }
    }

    /// Returns the timestamp and the duration of the last flush of the BBX (blackbox) infoROM object during the current run.
    ///
    /// For all products with an inforom.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the BBX object has not been flushed yet, if the device does not have an infoROM, or if NVML reports an unexpected failure.
    pub fn last_bbx_flush_time(&self) -> Result<BbxFlushTime> {
        let mut timestamp = 0;
        let mut duration_us = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `timestamp` and `duration_us` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetLastBBXFlushTime(
                self.raw,
                &raw mut timestamp,
                &raw mut duration_us,
            ))?;
        }
        Ok(BbxFlushTime {
            timestamp,
            duration_us,
        })
    }

    /// Returns the device's interrupt number.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support IRQ-number reporting, or if NVML has not been initialized.
    pub fn irq_number(&self) -> Result<u32> {
        let mut irq = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `irq` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetIrqNum(self.raw, &raw mut irq))?;
        }
        Ok(irq)
    }

    /// Returns the device's core count.
    ///
    /// On MIG-enabled GPUs, querying the device's core count is currently not supported by this operation.
    /// Use [`Device::gpu_instance_profile_info`] to fetch the MIG device's core count.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device or MIG device does not support core-count reporting, or if NVML has not been initialized.
    pub fn num_gpu_cores(&self) -> Result<u32> {
        let mut cores = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `cores` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNumGpuCores(self.raw, &raw mut cores))?;
        }
        Ok(cores)
    }

    fn single_field_value(&self, field: FieldId, scope_id: u32) -> Result<FieldSample> {
        self.field_values(&[FieldQuery { field, scope_id }])?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Empty {
                name: "nvmlDeviceGetFieldValues".into(),
            })
    }

    /// Returns GSP firmware version.
    ///
    /// This wrapper allocates the firmware-version buffer internally and returns it as a [`String`].
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device or output buffer, if GSP firmware is not enabled for the GPU, or if NVML reports an unexpected failure.
    pub fn gsp_firmware_version(&self) -> Result<String> {
        let mut buffer = vec![0; sys::NVML_GSP_FIRMWARE_VERSION_BUF_SIZE as usize];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `buffer` holds the `NVML_GSP_FIRMWARE_VERSION_BUF_SIZE` bytes NVML writes at most.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGspFirmwareVersion(
                self.raw,
                buffer.as_mut_ptr()
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns GSP firmware mode.
    ///
    /// Returns GSP firmware enablement and default mode information.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if GSP firmware is not enabled for the GPU, or if NVML reports an unexpected failure.
    pub fn gsp_firmware_mode(&self) -> Result<GspFirmwareMode> {
        let mut enabled = 0;
        let mut default_mode = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `enabled` and `default_mode` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGspFirmwareMode(
                self.raw,
                &raw mut enabled,
                &raw mut default_mode,
            ))?;
        }
        Ok(GspFirmwareMode {
            enabled: enabled != 0,
            default_mode: default_mode != 0,
        })
    }

    /// Returns minor number for the device.
    /// The minor number is the suffix in the Linux device node path `/dev/nvidia[minor_number]`.
    ///
    /// For all products.
    /// Supported only for Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support Linux minor-number reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn minor_number(&self) -> Result<u32> {
        let mut minor_number = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `minor_number` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMinorNumber(
                self.raw,
                &raw mut minor_number
            ))?;
        }
        Ok(minor_number)
    }

    /// Returns the current compute mode for the device.
    ///
    /// For all products.
    ///
    /// See [`ComputeMode`] for details on allowed compute modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support compute-mode reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn compute_mode(&self) -> Result<ComputeMode> {
        let mut compute_mode = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_DEFAULT;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `compute_mode` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetComputeMode(
                self.raw,
                &raw mut compute_mode
            ))?;
        }
        Ok(compute_mode.into())
    }

    /// Returns the persistence mode associated with this device.
    ///
    /// For all products.
    /// For Linux only.
    ///
    /// When driver persistence mode is enabled the driver software state is not torn down when the last client disconnects.
    /// By default this feature is disabled.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support persistence mode, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn persistence_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPersistenceMode(self.raw, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Sets the compute mode for the device.
    ///
    /// For all products.
    /// Requires root/admin permissions.
    ///
    /// The compute mode determines whether a GPU can be used for compute operations and whether it can be shared across contexts.
    ///
    /// This operation takes effect immediately.
    /// Under Linux it is not persistent across reboots and always resets to "Default".
    /// Under Windows it is persistent.
    ///
    /// Under Windows, compute mode may only be set to default when running in WDDM.
    ///
    /// On MIG-enabled GPUs, compute mode is set to default and changing it is not supported.
    ///
    /// See [`ComputeMode`] for details on available compute modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or mode, if the device does not support compute-mode changes, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_compute_mode(&self, mode: ComputeMode) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the mode is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetComputeMode(self.raw, mode.into()))?;
        }
        Ok(())
    }

    /// Sets the persistence mode for the device.
    ///
    /// For all products.
    /// For Linux only.
    /// Requires root/admin permissions.
    ///
    /// The persistence mode determines whether the GPU driver software is torn down after the last client exits.
    ///
    /// This operation takes effect immediately.
    /// It is not persistent across reboots.
    /// After each reboot the persistence mode is reset to "Disabled".
    ///
    /// See [`EnableState`] for available modes.
    ///
    /// After disabling persistence mode on a device that has its own NUMA memory, the current device handle is no longer valid.
    /// Acquire a fresh handle from the library before continuing to interact with the device.
    /// This limitation is currently only applicable to devices that have a coherent `NVLink` connection to system memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or mode, if the device does not support persistence mode, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_persistence_mode(&self, mode: EnableState) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the mode is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetPersistenceMode(self.raw, mode.into()))?;
        }
        Ok(())
    }

    /// Returns the addressing mode for the given GPU.
    ///
    /// [`DeviceAddressingMode::Hmm`] makes system-allocated memory (`malloc`, `mmap`) addressable from the GPU through software-based mirroring of the CPU page tables.
    /// [`DeviceAddressingMode::Ats`] makes system-allocated memory addressable from the GPU through Address Translation Services, which means there is effectively a single set of page tables used by both CPU and GPU.
    /// [`DeviceAddressingMode::None`] means neither HMM nor ATS is active.
    ///
    /// For Turing or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the handle, or if the current platform does not support addressing-mode reporting.
    pub fn addressing_mode(&self) -> Result<DeviceAddressingMode> {
        let mut mode = sys::nvmlDeviceAddressingMode_t {
            version: struct_version::<sys::nvmlDeviceAddressingMode_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAddressingMode(self.raw, &raw mut mode))?;
        }
        Ok(DeviceAddressingMode::from(
            sys::nvmlDeviceAddressingModeType_t(mode.value),
        ))
    }

    /// Returns the repair status for TPC/channel repair.
    ///
    /// For Ampere or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the handle or output, if the device does not support repair-status reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn repair_status(&self) -> Result<RepairStatus> {
        let mut status = sys::nvmlRepairStatus_t {
            version: struct_version::<sys::nvmlRepairStatus_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `status` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRepairStatus(self.raw, &raw mut status))?;
        }
        Ok(status.into())
    }

    /// Returns the device capability mask while preserving unknown native bits.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the native record version is unsupported, or capability reporting is unavailable.
    /// Also propagates NVML query failures.
    pub fn capabilities(&self) -> Result<DeviceCapabilities> {
        let mut caps = sys::nvmlDeviceCapabilities_t {
            version: struct_version::<sys::nvmlDeviceCapabilities_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `caps` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCapabilities(self.raw, &raw mut caps))?;
        }
        Ok(DeviceCapabilities::from_bits_retain(caps.capMask))
    }
}

fn field_id_for_perf_policy(perf_policy: PerfPolicyType) -> FieldId {
    match perf_policy {
        PerfPolicyType::Power => FieldId::PERF_POLICY_POWER,
        PerfPolicyType::Thermal => FieldId::PERF_POLICY_THERMAL,
        PerfPolicyType::SyncBoost => FieldId::PERF_POLICY_SYNC_BOOST,
        PerfPolicyType::BoardLimit => FieldId::PERF_POLICY_BOARD_LIMIT,
        PerfPolicyType::LowUtilization => FieldId::PERF_POLICY_LOW_UTILIZATION,
        PerfPolicyType::Reliability => FieldId::PERF_POLICY_RELIABILITY,
        PerfPolicyType::TotalAppClocks => FieldId::PERF_POLICY_TOTAL_APP_CLOCKS,
        PerfPolicyType::TotalBaseClocks => FieldId::PERF_POLICY_TOTAL_BASE_CLOCKS,
    }
}

fn field_sample_as_u64(sample: FieldSample) -> Result<u64> {
    Ok(match sample.result? {
        FieldValue::UnsignedInt(value) => u64::from(value),
        FieldValue::UnsignedLong(value) | FieldValue::UnsignedLongLong(value) => value,
        FieldValue::SignedInt(value) => {
            u64::try_from(value).map_err(|_| Error::out_of_range("field sample"))?
        }
        FieldValue::SignedLongLong(value) => {
            u64::try_from(value).map_err(|_| Error::out_of_range("field sample"))?
        }
        FieldValue::UnsignedShort(value) => u64::from(value),
        FieldValue::Double(_) => {
            return Err(Error::UnexpectedFieldValueType {
                name: "numeric field sample".into(),
                value: "double".into(),
            });
        }
    })
}

fn device_string_query(
    device: &Device,
    length: usize,
    function: unsafe extern "C" fn(sys::nvmlDevice_t, *mut c_char, u32) -> sys::nvmlReturn_t,
) -> Result<String> {
    let mut buffer = vec![0; length];
    // SAFETY: `device` holds a live NVML device handle, and `buffer` holds the `buffer.len()` writable bytes passed as its length.
    unsafe {
        try_ffi!(function(
            device.as_raw(),
            buffer.as_mut_ptr(),
            to_u32(buffer.len(), "NVML string buffer length")?
        ))?;
    }
    Ok(string_from_c_chars(&buffer))
}
