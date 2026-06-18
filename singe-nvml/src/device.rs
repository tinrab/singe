#![allow(deprecated)]

use std::{
    mem::{self, MaybeUninit},
    ptr,
};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    error::{Error, Result, Status},
    gpu_instance::{GpuInstance, OwnedGpuInstance},
    library::GpmSample,
    try_ffi,
    types::{
        AccountingStats, AdaptiveClockInfoStatus, AffinityScope, Architecture, AutoBoostClocks,
        Bar1MemoryInfo, BbxFlushTime, Brand, BridgeChipHierarchy, BusType, C2cModeInfo,
        ClkMonStatus, ClockId, ClockOffset, ClockRange, ClockRangeI32, ClockType,
        ComputeCapability, ComputeMode, ConfComputeGpuAttestationReport, ConfComputeGpuCertificate,
        ConfComputeMemSizeInfo, CoolerInfo, CurrentClockFreqs, CurrentPending,
        DeviceAddressingMode, DeviceAttributes, DeviceCapabilities, DeviceVgpuCapability,
        DramEncryptionInfo, DriverModel, DriverModelFlags, DynamicPstatesInfo, EccCounterType,
        EccErrorCounts, EccSramErrorStatus, EccSramUniqueUncorrectedErrorCounts, EnableState,
        EncoderSessionInfo, EncoderStats, EncoderType, EventTypes, FanPolicy, FanSpeedInfo,
        FbcSessionInfo, FbcStats, FieldId, FieldQuery, FieldSample, FieldValue, GpmSupport,
        GpuFabricInfo, GpuInstancePlacement, GpuInstanceProfileInfo, GpuOperationMode,
        GspFirmwareMode, HostVgpuMode, InforomObject, MarginTemperature, MemoryErrorType,
        MemoryInfo, MemoryLocation, MigMode, MigModeActivation, MinMaxFanSpeed, NvLinkBwMode,
        NvLinkCapability, NvLinkErrorCounter, NvLinkInfo, NvLinkRemoteDeviceType,
        NvLinkSupportedBwModes, NvLinkVersion, P2pCapabilityIndex, P2pStatus, PageRetirementCause,
        PciInfo, PciInfoExt, PcieUtilCounter, Pdi, PerfPolicyType, PerformanceModes,
        PerformanceState, PgpuMetadata, PlatformInfo, PowerLimits, PowerMizerMode, PowerMizerModes,
        PowerSource, ProcessDetail, ProcessInfo, ProcessMode, ProcessUtilizationInfo,
        ProcessUtilizationSample, RemappedRows, RepairStatus, RestrictedApi, RetiredPage,
        RowRemapperHistogram, Sample, Samples, SamplingType, TemperatureInfo, TemperatureSensor,
        TemperatureThreshold, ThermalSettings, TopologyLevel, Utilization, UtilizationCounter,
        VgpuInstanceId, VgpuPlacementId, VgpuPlacementMode, VgpuTypeId, ViolationTime,
        VirtualizationMode, WorkloadPowerCurrentProfiles, WorkloadPowerProfilesInfo,
        try_from_nvml_enum,
    },
    utility::{
        device_clock_offset_range, device_string_query, device_ulong_bitmask_list,
        device_utilization_counter, query_process_info_list, query_sized_raw, query_u32_list,
        struct_version,
    },
    vgpu_instance::VgpuInstance,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Device(sys::nvmlDevice_t);

impl Device {
    /// Wraps a borrowed NVML device handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `nvmlDevice_t` for the duration of any calls
    /// made through the returned wrapper. NVML owns the underlying device
    /// handle; this wrapper does not destroy it.
    pub const unsafe fn from_raw(handle: sys::nvmlDevice_t) -> Self {
        Self(handle)
    }

    pub const fn as_raw(self) -> sys::nvmlDevice_t {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0.is_null()
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
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or index output, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn index(self) -> Result<u32> {
        let mut index = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetIndex(self.0, &raw mut index))?;
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
    /// Returns an error if the device is inaccessible, if the internal name
    /// buffer is too small, if NVML rejects the handle or output buffer, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn name(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_DEVICE_NAME_BUFFER_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetName(
                self.0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the hostname for the device.
    ///
    /// For Blackwell or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// Returns the hostname string for the GPU device that was set using [`sys::nvmlDeviceSetHostname_v1`].
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or hostname output, if the device does not support hostnames, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn hostname(self) -> Result<String> {
        let mut hostname = sys::nvmlHostname_v1_t::default();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHostname_v1(self.0, &raw mut hostname))?;
        }
        Ok(string_from_c_chars(&hostname.value))
    }

    /// Returns the brand of this device.
    ///
    /// For all products.
    ///
    /// The type is a member of [`Brand`] defined above.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or brand output, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn brand(self) -> Result<Brand> {
        let mut brand = sys::nvmlBrandType_t::NVML_BRAND_UNKNOWN as u32;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBrand(
                self.0,
                (&raw mut brand).cast::<sys::nvmlBrandType_t>(),
            ))?;
        }
        Ok(Brand::from_raw(brand))
    }

    /// Returns architecture for the device.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or architecture output, or if
    /// NVML has not been initialized.
    pub fn architecture(self) -> Result<Architecture> {
        let mut architecture = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetArchitecture(
                self.0,
                &raw mut architecture
            ))?;
        }
        Ok(Architecture::from_raw(architecture))
    }

    pub fn serial(self) -> Result<String> {
        device_string_query(
            self,
            sys::NVML_DEVICE_SERIAL_BUFFER_SIZE as usize,
            sys::nvmlDeviceGetSerial,
        )
    }

    pub fn uuid(self) -> Result<String> {
        device_string_query(
            self,
            sys::NVML_DEVICE_UUID_V2_BUFFER_SIZE as usize,
            sys::nvmlDeviceGetUUID,
        )
    }

    pub fn vbios_version(self) -> Result<String> {
        // From docs: "It will not exceed 32 characters in length (including the terminating NUL byte)"
        // https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g4a02015969489ad8a28d8c56b34c825e
        device_string_query(self, 32, sys::nvmlDeviceGetVbiosVersion)
    }

    pub fn board_part_number(self) -> Result<String> {
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
    /// Returns an error if NVML rejects the handle or module ID output, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn module_id(self) -> Result<u32> {
        let mut module_id = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetModuleId(self.0, &raw mut module_id))?;
        }
        Ok(module_id)
    }

    /// Returns whether the device is on a multi-GPU board.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support this query, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn is_multi_gpu_board(self) -> Result<bool> {
        let mut multi_gpu = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMultiGpuBoard(self.0, &raw mut multi_gpu))?;
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
    /// Returns an error if NVML rejects the device handle, if the device does
    /// not support this query, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn attributes(self) -> Result<DeviceAttributes> {
        unsafe {
            let mut attributes = MaybeUninit::<sys::nvmlDeviceAttributes_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetAttributes_v2(
                self.0,
                attributes.as_mut_ptr()
            ))?;
            Ok(attributes.assume_init().into())
        }
    }

    /// Returns the root/admin permissions for the target NVML operation.
    /// See [`RestrictedApi`] for the list of supported operations.
    /// If an operation is restricted, only callers with root privileges can call it.
    /// See [`sys::nvmlDeviceSetAPIRestriction`] to change current permissions.
    ///
    /// For all fully supported products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, API restriction, or output, if the device or queried feature does
    /// not support API restriction reporting, if NVML has not been initialized,
    /// or if NVML reports an unexpected failure.
    pub fn is_api_restricted(self, api: RestrictedApi) -> Result<EnableState> {
        let mut state = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAPIRestriction(
                self.0,
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
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the request, if system memory is
    /// insufficient, if the device does not support this query, or if NVML
    /// reports an unexpected failure.
    pub fn platform_info(self) -> Result<PlatformInfo> {
        let mut info = sys::nvmlPlatformInfo_t {
            version: struct_version::<sys::nvmlPlatformInfo_t>(2),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPlatformInfo(self.0, &raw mut info))?;
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
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML rejects the handle
    /// or output, if the device does not support PDI reporting, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn pdi(self) -> Result<Pdi> {
        let mut pdi = sys::nvmlPdi_t {
            version: struct_version::<sys::nvmlPdi_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPdi(self.0, &raw mut pdi))?;
        }
        Ok(pdi.into())
    }

    /// Returns the Device's C2C Mode information.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support C2C mode reporting, or
    /// if NVML reports an unexpected failure.
    pub fn c2c_mode_info(self) -> Result<C2cModeInfo> {
        unsafe {
            let mut info = MaybeUninit::<sys::nvmlC2cModeInfo_v1_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetC2cModeInfoV(self.0, info.as_mut_ptr()))?;
            Ok(info.assume_init().into())
        }
    }

    /// Returns the current Auto Boosted clocks state for this device.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates to maximize performance as thermal limits allow.
    ///
    /// On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
    /// Use [`sys::nvmlDeviceSetApplicationsClocks`] and [`sys::nvmlDeviceResetApplicationsClocks`] to control Auto Boost behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support Auto Boosted clocks, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn auto_boosted_clocks(self) -> Result<AutoBoostClocks> {
        let mut enabled = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        let mut default_enabled = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAutoBoostedClocksEnabled(
                self.0,
                &raw mut enabled,
                &raw mut default_enabled,
            ))?;
        }
        Ok(AutoBoostClocks {
            enabled: enabled.into(),
            default_enabled: default_enabled.into(),
        })
    }

    /// Tries to set the default state of Auto Boosted clocks on a device.
    /// Auto Boosted clocks return to this default state when no compute
    /// processes, such as CUDA applications with active contexts, are running.
    ///
    /// For Kepler or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
    /// Requires root/admin permissions.
    ///
    /// Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates to maximize performance as thermal limits allow.
    /// Disable Auto Boosted clocks when fixed clock rates are required.
    ///
    /// On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
    /// Use [`sys::nvmlDeviceSetApplicationsClocks`] and [`sys::nvmlDeviceResetApplicationsClocks`] to control Auto Boost behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, if the device does not support Auto Boosted clocks, if the
    /// current process lacks permission to change the default state, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn set_default_auto_boosted_clocks_enabled(
        self,
        enabled: EnableState,
        flags: DriverModelFlags,
    ) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDefaultAutoBoostedClocksEnabled(
                self.0,
                enabled.into(),
                flags.bits(),
            ))?;
        }
        Ok(())
    }

    /// Returns the PCI attributes of this device.
    ///
    /// For all products.
    ///
    /// Returns the PCI information reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or PCI info output, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn pci_info(self) -> Result<PciInfo> {
        unsafe {
            let mut pci = MaybeUninit::<sys::nvmlPciInfo_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetPciInfo_v3(self.0, pci.as_mut_ptr()))?;
            Ok(pci.assume_init().into())
        }
    }

    /// Returns PCI attributes of this device.
    ///
    /// For all products.
    ///
    /// Returns the extended PCI information reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or PCI info output, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn pci_info_ext(self) -> Result<PciInfoExt> {
        let mut pci = sys::nvmlPciInfoExt_t {
            version: struct_version::<sys::nvmlPciInfoExt_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPciInfoExt(self.0, &raw mut pci))?;
        }
        Ok(pci.into())
    }

    /// Returns bridge chip information for all the bridge chips on the board.
    ///
    /// For all fully supported products.
    /// Only applicable to multi-GPU products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if bridge-chip reporting is not supported for the
    /// device, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn bridge_chip_hierarchy(self) -> Result<BridgeChipHierarchy> {
        unsafe {
            let mut hierarchy = MaybeUninit::<sys::nvmlBridgeChipHierarchy_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetBridgeChipInfo(
                self.0,
                hierarchy.as_mut_ptr()
            ))?;
            Ok(hierarchy.assume_init().into())
        }
    }

    /// Returns the current PCIe link generation.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if PCIe link information is unavailable, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn current_pcie_link_generation(self) -> Result<u32> {
        let mut generation = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrPcieLinkGeneration(
                self.0,
                &raw mut generation,
            ))?;
        }
        Ok(generation)
    }

    /// Returns the current PCIe link width.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if PCIe link information is unavailable, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn current_pcie_link_width(self) -> Result<u32> {
        let mut width = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrPcieLinkWidth(self.0, &raw mut width))?;
        }
        Ok(width)
    }

    /// Returns the maximum PCIe link generation possible with this device and system.
    ///
    /// For example, a generation 2 PCIe device attached to a generation 1 PCIe bus reports generation 1.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if PCIe link information is unavailable, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn max_pcie_link_generation(self) -> Result<u32> {
        let mut generation = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxPcieLinkGeneration(
                self.0,
                &raw mut generation,
            ))?;
        }
        Ok(generation)
    }

    /// Returns the maximum PCIe link generation supported by this device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if PCIe link information is unavailable, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn gpu_max_pcie_link_generation(self) -> Result<u32> {
        let mut generation = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuMaxPcieLinkGeneration(
                self.0,
                &raw mut generation,
            ))?;
        }
        Ok(generation)
    }

    /// Returns the maximum PCIe link width possible with this device and system.
    ///
    /// For example, a device with a 16x PCIe bus width attached to an 8x PCIe
    /// system bus reports a maximum link width of 8.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if PCIe link information is unavailable, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn max_pcie_link_width(self) -> Result<u32> {
        let mut width = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxPcieLinkWidth(self.0, &raw mut width))?;
        }
        Ok(width)
    }

    /// Returns PCIe utilization information.
    /// Queries a byte counter over a 20 ms interval to report PCIe throughput.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Not supported in virtual machines running virtual GPU (vGPU).
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, counter, or output, if the device does not support PCIe
    /// utilization queries, if NVML has not been initialized, or if NVML reports
    /// an unexpected failure.
    pub fn pcie_throughput(self, counter: PcieUtilCounter) -> Result<u32> {
        let mut throughput = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieThroughput(
                self.0,
                counter.into(),
                &raw mut throughput,
            ))?;
        }
        Ok(throughput)
    }

    /// Returns the PCIe replay counter.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support replay-counter queries,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn pcie_replay_counter(self) -> Result<u32> {
        let mut counter = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieReplayCounter(
                self.0,
                &raw mut counter
            ))?;
        }
        Ok(counter)
    }

    /// Returns the device's PCIe Max Link speed in MB/s.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support this query, or if NVML
    /// has not been initialized.
    pub fn pcie_link_max_speed(self) -> Result<u32> {
        let mut speed = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieLinkMaxSpeed(self.0, &raw mut speed))?;
        }
        Ok(speed)
    }

    /// Returns the device's PCIe Link speed in Mbps.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support PCIe speed queries, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn pcie_speed(self) -> Result<u32> {
        let mut speed = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieSpeed(self.0, &raw mut speed))?;
        }
        Ok(speed)
    }

    /// Indicates whether the supplied device supports GPM.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// This supports device handles and MIG device handles.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or cannot query GPM support.
    pub fn gpm_support(self) -> Result<GpmSupport> {
        let mut support = sys::nvmlGpmSupport_t {
            version: sys::NVML_GPM_SUPPORT_VERSION,
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlGpmQueryDeviceSupport(self.0, &raw mut support))?;
        }
        Ok(support.into())
    }

    /// Returns GPM stream state.
    ///
    /// For Hopper or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support GPM streaming-state queries, or if NVML has not been
    /// initialized.
    pub fn gpm_streaming_enabled(self) -> Result<EnableState> {
        let mut state = 0;
        unsafe {
            try_ffi!(sys::nvmlGpmQueryIfStreamingEnabled(self.0, &raw mut state))?;
        }
        try_from_nvml_enum("enable state", state)
    }

    /// Read a sample of GPM metrics into the provided `sample` buffer.
    /// After two samples are gathered, you can call [`Library::gpm_metrics`](crate::library::Library::gpm_metrics) on those samples to retrieve metrics.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// * The interval between two [`Device::gpm_sample`] calls must be greater than 100 ms due to the internal sample refresh rate.
    /// * Supports device handles and MIG device handles.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or sample buffer, if the
    /// device does not support GPM sampling, or if samples are requested too
    /// quickly.
    pub fn gpm_sample(self, sample: &GpmSample) -> Result<()> {
        unsafe { try_ffi!(sys::nvmlGpmSampleGet(self.0, sample.as_raw())) }
    }

    /// Read a sample of GPM metrics into the provided `sample` buffer for a MIG GPU instance.
    ///
    /// After two samples are gathered, you can call [`Library::gpm_metrics`](crate::library::Library::gpm_metrics) on those samples to retrieve metrics.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// The interval between two [`Device::gpm_mig_sample`] calls must be greater than 100 ms due to the internal sample refresh rate.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, GPU instance id, or sample
    /// buffer, if the device does not support GPM MIG sampling, or if samples
    /// are requested too quickly.
    pub fn gpm_mig_sample(self, gpu_instance_id: u32, sample: &GpmSample) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlGpmMigSampleGet(
                self.0,
                gpu_instance_id,
                sample.as_raw(),
            ))
        }
    }

    pub fn memory_info(self) -> Result<MemoryInfo> {
        let mut memory = sys::nvmlMemory_v2_t {
            version: struct_version::<sys::nvmlMemory_v2_t>(2),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemoryInfoV(self.0, &raw mut memory))?;
        }
        Ok(memory.into())
    }

    /// Returns the current utilization rates for the device's major subsystems.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// * During driver initialization when ECC is enabled, GPU and memory utilization readings can be high.
    ///   ECC memory scrubbing during driver initialization causes this.
    /// * On MIG-enabled GPUs, querying device utilization rates is not currently supported.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support utilization queries, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn utilization(self) -> Result<Utilization> {
        unsafe {
            let mut utilization = MaybeUninit::<sys::nvmlUtilization_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetUtilizationRates(
                self.0,
                utilization.as_mut_ptr(),
            ))?;
            Ok(utilization.assume_init().into())
        }
    }

    pub fn encoder_utilization(self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetEncoderUtilization)
    }

    pub fn decoder_utilization(self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetDecoderUtilization)
    }

    pub fn jpg_utilization(self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetJpgUtilization)
    }

    pub fn ofa_utilization(self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetOfaUtilization)
    }

    /// Returns the current capacity of the device's encoder, as a percentage of maximum encoder capacity with valid values in the range 0-100.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if the device does not support the requested encoder, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_capacity(self, encoder: EncoderType) -> Result<u32> {
        let mut capacity = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEncoderCapacity(
                self.0,
                encoder.into(),
                &raw mut capacity,
            ))?;
        }
        Ok(capacity)
    }

    /// Returns the current encoder statistics for the given device.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn encoder_stats(self) -> Result<EncoderStats> {
        let mut session_count = 0;
        let mut average_fps = 0;
        let mut average_latency_us = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEncoderStats(
                self.0,
                &raw mut session_count,
                &raw mut average_fps,
                &raw mut average_latency_us,
            ))?;
        }
        Ok(EncoderStats {
            session_count,
            average_fps,
            average_latency_us,
        })
    }

    /// Returns information about active encoder sessions on a target device.
    ///
    /// This wrapper queries the required session count first, then returns the active encoder sessions as a [`Vec`].
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the active-session
    /// count changes while the wrapper is fetching sessions, if NVML reports an
    /// invalid session count, if the device does not support this query, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_sessions(self) -> Result<Vec<EncoderSessionInfo>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetEncoderSessions(self.as_raw(), &raw mut count, ptr::null_mut())
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut sessions = vec![sys::nvmlEncoderSessionInfo_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEncoderSessions(
                self.as_raw(),
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }
        sessions.truncate(count as usize);
        Ok(sessions.into_iter().map(Into::into).collect())
    }

    /// Returns total, available, and used size of BAR1 memory.
    ///
    /// BAR1 maps framebuffer memory so the CPU or third-party PCIe peer devices can access it directly.
    ///
    /// In MIG mode, a device handle returns aggregate information only if the caller has appropriate privileges.
    /// Per-instance information can be queried by using specific MIG device handles.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if the device does not support BAR1 memory reporting, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn bar1_memory_info(self) -> Result<Bar1MemoryInfo> {
        unsafe {
            let mut memory = MaybeUninit::<sys::nvmlBAR1Memory_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetBAR1MemoryInfo(
                self.0,
                memory.as_mut_ptr()
            ))?;
            Ok(memory.assume_init().into())
        }
    }

    /// Returns the current clock speeds for the device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// See [`ClockType`] for details on available clock information.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, clock type, or output, if the device cannot report the requested
    /// clock, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn clock(self, kind: ClockType) -> Result<u32> {
        let mut clock = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetClockInfo(
                self.0,
                kind.into(),
                &raw mut clock
            ))?;
        }
        Ok(clock)
    }

    /// Returns the clock speed for the clock specified by the clock type and clock ID.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, clock type, clock ID, or output, if the device does not support
    /// this clock query, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn clock_with_id(self, kind: ClockType, clock_id: ClockId) -> Result<u32> {
        let mut clock = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetClock(
                self.0,
                kind.into(),
                clock_id.into(),
                &raw mut clock,
            ))?;
        }
        Ok(clock)
    }

    /// Returns the maximum clock speeds for the device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// See [`ClockType`] for details on available clock information.
    ///
    /// Current P0 clocks (reported by [`Device::clock`]) can differ from max clocks by a few MHz.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, clock type, or output, if the device cannot report the requested
    /// maximum clock, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn max_clock(self, kind: ClockType) -> Result<u32> {
        let mut clock = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxClockInfo(
                self.0,
                kind.into(),
                &raw mut clock,
            ))?;
        }
        Ok(clock)
    }

    /// Returns a string with the associated current GPU Clock and Memory Clock values.
    ///
    /// Not all tokens are reported on all GPUs, and additional tokens may be added in the future.
    ///
    /// These clock values include the offset set by clients through [`Device::set_clock_offsets`].
    ///
    /// Clock values are returned as a comma-separated list of "token=value" pairs.
    /// Valid tokens:
    ///
    /// - `perf`: performance level.
    /// - `nvclock`: GPU clock in MHz for the performance level.
    /// - `nvclockmin`: minimum GPU clock in MHz for the performance level.
    /// - `nvclockmax`: maximum GPU clock in MHz for the performance level.
    /// - `nvclockeditable`: whether the GPU clock domain is editable for the performance level.
    /// - `memclock`: memory clock in MHz for the performance level.
    /// - `memclockmin`: minimum memory clock in MHz for the performance level.
    /// - `memclockmax`: maximum memory clock in MHz for the performance level.
    /// - `memclockeditable`: whether the memory clock domain is editable for the performance level.
    /// - `memtransferrate`: memory transfer rate in MHz for the performance level.
    /// - `memtransferratemin`: minimum memory transfer rate in MHz for the performance level.
    /// - `memtransferratemax`: maximum memory transfer rate in MHz for the performance level.
    /// - `memtransferrateeditable`: whether the memory transfer rate is editable for the performance level.
    ///
    /// Example:
    ///
    /// `nvclock=324, nvclockmin=324, nvclockmax=324, nvclockeditable=0, memclock=324, memclockmin=324, memclockmax=324, memclockeditable=0, memtransferrate=648, memtransferratemin=648, memtransferratemax=648, memtransferrateeditable=0;`
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the internal
    /// clock-frequency buffer is too small, if NVML rejects the handle or
    /// output, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn current_clock_freqs(self) -> Result<CurrentClockFreqs> {
        let mut current = sys::nvmlDeviceCurrentClockFreqs_t {
            version: struct_version::<sys::nvmlDeviceCurrentClockFreqs_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrentClockFreqs(
                self.0,
                &raw mut current
            ))?;
        }
        Ok(current.into())
    }

    /// Returns a performance mode string with all the performance modes defined for this device along with their associated GPU Clock and Memory Clock values.
    /// Not all tokens are reported on all GPUs, and additional tokens may be added in the future.
    /// For backward compatibility, NVML still provides `nvclock` and `memclock`; those are the same as `nvclockmin` and `memclockmin`.
    ///
    /// These clock values include the offset set by clients through [`Device::set_clock_offsets`].
    ///
    /// Maximum available Pstate (P15) shows the minimum performance level (0) and vice versa.
    ///
    /// Each performance mode is returned as a comma-separated list of "token=value" pairs.
    /// Performance-mode token sets are separated by semicolons.
    /// Valid tokens:
    ///
    /// - `perf`: performance level.
    /// - `nvclock`: GPU clock in MHz for the performance level.
    /// - `nvclockmin`: minimum GPU clock in MHz for the performance level.
    /// - `nvclockmax`: maximum GPU clock in MHz for the performance level.
    /// - `nvclockeditable`: whether the GPU clock domain is editable for the performance level.
    /// - `memclock`: memory clock in MHz for the performance level.
    /// - `memclockmin`: minimum memory clock in MHz for the performance level.
    /// - `memclockmax`: maximum memory clock in MHz for the performance level.
    /// - `memclockeditable`: whether the memory clock domain is editable for the performance level.
    /// - `memtransferrate`: memory transfer rate in MHz for the performance level.
    /// - `memtransferratemin`: minimum memory transfer rate in MHz for the performance level.
    /// - `memtransferratemax`: maximum memory transfer rate in MHz for the performance level.
    /// - `memtransferrateeditable`: whether the memory transfer rate is editable for the performance level.
    ///
    /// Example:
    ///
    /// `perf=0, nvclock=324, nvclockmin=324, nvclockmax=324, nvclockeditable=0, memclock=324, memclockmin=324, memclockmax=324, memclockeditable=0, memtransferrate=648, memtransferratemin=648, memtransferratemax=648, memtransferrateeditable=0; perf=1, nvclock=324, nvclockmin=324, nvclockmax=640, nvclockeditable=0, memclock=810, memclockmin=810, memclockmax=810, memclockeditable=0, memtransferrate=1620, memtransferrate=1620, memtransferrate=1620, memtransferrateeditable=0;`
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the internal
    /// performance-mode buffer is too small, if NVML rejects the handle or
    /// output, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn performance_modes(self) -> Result<PerformanceModes> {
        let mut modes = sys::nvmlDevicePerfModes_t {
            version: struct_version::<sys::nvmlDevicePerfModes_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPerformanceModes(self.0, &raw mut modes))?;
        }
        Ok(modes.into())
    }

    pub fn applications_clock(self, kind: ClockType) -> Result<u32> {
        self.clock_with_id(kind, ClockId::AppClockTarget)
    }

    pub fn default_applications_clock(self, kind: ClockType) -> Result<u32> {
        self.clock_with_id(kind, ClockId::AppClockDefault)
    }

    /// Returns the customer-defined maximum boost clock speed specified by `kind`.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, clock type, or output, if the device or requested clock type does
    /// not support customer boost clocks, if NVML has not been initialized, or
    /// if NVML reports an unexpected failure.
    pub fn max_customer_boost_clock(self, kind: ClockType) -> Result<u32> {
        let mut clock = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxCustomerBoostClock(
                self.0,
                kind.into(),
                &raw mut clock,
            ))?;
        }
        Ok(clock)
    }

    /// Returns the minimum and maximum clocks of a clock domain for a P-state.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, clock type, P-state, or
    /// outputs, if the device does not support this query, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn min_max_clock_of_pstate(
        self,
        kind: ClockType,
        pstate: PerformanceState,
    ) -> Result<ClockRange> {
        let mut min = 0;
        let mut max = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMinMaxClockOfPState(
                self.0,
                kind.into(),
                pstate.into(),
                &raw mut min,
                &raw mut max,
            ))?;
        }
        Ok(ClockRange { min, max })
    }

    /// Returns the minimum, maximum, and current clock offset for a clock domain and P-state.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// [`Device::gpc_clock_vf_offset`], [`Device::memory_clock_vf_offset`],
    /// [`sys::nvmlDeviceGetGpcClkMinMaxVfOffset`], and
    /// [`sys::nvmlDeviceGetMemClkMinMaxVfOffset`] are deprecated and are planned
    /// for removal in a future release.
    /// Use [`Device::clock_offsets`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the device, clock type, P-state, or
    /// outputs, if the device does not support clock-offset queries, or if NVML
    /// has not been initialized.
    pub fn clock_offsets(self, kind: ClockType, pstate: PerformanceState) -> Result<ClockOffset> {
        let mut info = sys::nvmlClockOffset_t {
            version: struct_version::<sys::nvmlClockOffset_t>(1),
            type_: kind.into(),
            pstate: pstate.into(),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetClockOffsets(self.0, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Controls current clock offset of some clock domain for the given PState
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the handle, clock type, P-state, or
    /// offset, if the device does not support clock-offset control, if the
    /// current process lacks permission, or if NVML has not been initialized.
    pub fn set_clock_offsets(self, offset: ClockOffset) -> Result<()> {
        let mut info = sys::nvmlClockOffset_t {
            version: struct_version::<sys::nvmlClockOffset_t>(1),
            type_: offset.kind.into(),
            pstate: offset.pstate.into(),
            clockOffsetMHz: offset.clock_offset_mhz,
            minClockOffsetMHz: offset.min_clock_offset_mhz,
            maxClockOffsetMHz: offset.max_clock_offset_mhz,
        };
        unsafe { try_ffi!(sys::nvmlDeviceSetClockOffsets(self.0, &raw mut info)) }
    }

    /// Returns the GPCCLK VF offset value.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support this deprecated offset query, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn gpc_clock_vf_offset(self) -> Result<i32> {
        let mut offset = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpcClkVfOffset(self.0, &raw mut offset))?;
        }
        Ok(offset)
    }

    /// Returns the MemClk (Memory Clock) VF offset value.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support this deprecated offset query, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn memory_clock_vf_offset(self) -> Result<i32> {
        let mut offset = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemClkVfOffset(self.0, &raw mut offset))?;
        }
        Ok(offset)
    }

    pub fn gpc_clock_vf_offset_range(self) -> Result<ClockRangeI32> {
        device_clock_offset_range(self, sys::nvmlDeviceGetGpcClkMinMaxVfOffset)
    }

    pub fn memory_clock_vf_offset_range(self) -> Result<ClockRangeI32> {
        device_clock_offset_range(self, sys::nvmlDeviceGetMemClkMinMaxVfOffset)
    }

    /// Returns the list of possible memory clocks that can be used as an argument for [`sys::nvmlDeviceSetMemoryLockedClocks`].
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the clock list changes
    /// while the wrapper is fetching it, if NVML rejects the handle or count
    /// output, if the device does not support this query, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn supported_memory_clocks(self) -> Result<Vec<u32>> {
        query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetSupportedMemoryClocks(self.0, count, values)
        })
    }

    /// Returns the list of possible graphics clocks that can be used as an argument for [`sys::nvmlDeviceSetGpuLockedClocks`].
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the clock list changes
    /// while the wrapper is fetching it, if NVML rejects the handle, memory
    /// clock, or output, if `memory_clock_mhz` is unsupported, if the device does
    /// not support this query, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn supported_graphics_clocks(self, memory_clock_mhz: u32) -> Result<Vec<u32>> {
        query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetSupportedGraphicsClocks(self.0, memory_clock_mhz, count, values)
        })
    }

    pub fn temperature_reading(self, sensor: TemperatureSensor) -> Result<u32> {
        let value = self.temperature(sensor)?.temperature;
        u32::try_from(value).map_err(|_| Error::NegativeValue {
            name: "temperature".into(),
            value: i64::from(value),
        })
    }

    /// Returns the temperature threshold for the GPU with the specified threshold type in degrees C.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// See [`TemperatureThreshold`] for details on available temperature thresholds.
    ///
    /// This is no longer the preferred interface for retrieving the following temperature thresholds on Ada and later architectures: [`TemperatureThreshold::Shutdown`], [`TemperatureThreshold::Slowdown`], [`TemperatureThreshold::MemoryMax`] and [`TemperatureThreshold::GpuMax`].
    ///
    /// Support for reading these temperature thresholds for Ada and later architectures may be removed in future releases.
    /// Use [`Device::field_values`] with `NVML_FI_DEV_TEMPERATURE_*` fields to retrieve temperature thresholds on these architectures.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, threshold type, or output, if the device does not support the
    /// requested temperature threshold, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn temperature_threshold(self, threshold: TemperatureThreshold) -> Result<u32> {
        let mut temperature = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTemperatureThreshold(
                self.0,
                threshold.into(),
                &raw mut temperature,
            ))?;
        }
        Ok(temperature)
    }

    /// Sets the temperature threshold for the GPU with the specified threshold type in degrees C.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// See [`TemperatureThreshold`] for details on available temperature thresholds.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, threshold type, or temperature value, if the device does not
    /// support setting this threshold, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn set_temperature_threshold(
        self,
        threshold: TemperatureThreshold,
        temperature: i32,
    ) -> Result<i32> {
        let mut temperature = temperature;
        unsafe {
            try_ffi!(sys::nvmlDeviceSetTemperatureThreshold(
                self.0,
                threshold.into(),
                &raw mut temperature,
            ))?;
        }
        Ok(temperature)
    }

    /// Returns the thermal margin temperature (distance to nearest slowdown threshold).
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML rejects the handle
    /// or output, if the platform does not support this query, or if NVML
    /// reports an unexpected failure.
    pub fn margin_temperature(self) -> Result<MarginTemperature> {
        let mut margin = sys::nvmlMarginTemperature_t {
            version: struct_version::<sys::nvmlMarginTemperature_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMarginTemperature(self.0, &raw mut margin))?;
        }
        Ok(margin.into())
    }

    /// Used to execute a list of thermal system instructions.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, sensor index, or output, if the device does not support thermal
    /// settings, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn thermal_settings(self, sensor_index: u32) -> Result<ThermalSettings> {
        unsafe {
            let mut settings = MaybeUninit::<sys::nvmlGpuThermalSettings_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetThermalSettings(
                self.0,
                sensor_index,
                settings.as_mut_ptr(),
            ))?;
            Ok(settings.assume_init().into())
        }
    }

    /// Returns power usage for this GPU and its associated circuitry, such as memory, in milliwatts.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
    /// On Ampere (except GA100) or newer GPUs, this returns power averaged over a one-second interval.
    /// On GA100 and older architectures, instantaneous power is returned.
    ///
    /// See `NVML_FI_DEV_POWER_AVERAGE` on newer architectures and `NVML_FI_DEV_POWER_INSTANT` to query specific power values.
    ///
    /// It is only available if power management mode is supported.
    /// See [`sys::nvmlDeviceGetPowerManagementMode`].
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support power readings, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_usage(self) -> Result<u32> {
        let mut power = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerUsage(self.0, &raw mut power))?;
        }
        Ok(power)
    }

    pub fn power_management_mode(self) -> Result<EnableState> {
        let sample = self.single_field_value(FieldId::POWER_CURRENT_LIMIT, 0)?;
        match sample.result {
            Ok(_) => Ok(EnableState::Enabled),
            Err(Error::Nvml {
                code: Status::NotSupported,
                ..
            }) => Ok(EnableState::Disabled),
            Err(err) => Err(err),
        }
    }

    /// Returns the power management limit associated with this device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// The power limit defines the upper boundary for the card's power draw.
    /// If the card's total power draw reaches this limit the power management algorithm kicks in.
    ///
    /// This reading is only available if power management mode is supported.
    /// See [`sys::nvmlDeviceGetPowerManagementMode`].
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support power limits, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_management_limit(self) -> Result<u32> {
        let mut limit = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerManagementLimit(
                self.0,
                &raw mut limit
            ))?;
        }
        Ok(limit)
    }

    /// Sets a new power limit for this device.
    ///
    /// For Kepler or newer fully supported devices.
    /// Requires root/admin permissions.
    ///
    /// See [`Device::power_management_limit_constraints`] to check the allowed ranges of values.
    ///
    /// Limit is not persistent across reboots or driver unloads.
    /// Enable persistent mode to prevent driver from unloading when no application is using the device.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or power limit, if the device does not support power-limit
    /// control, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn set_power_management_limit(self, limit: u32) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetPowerManagementLimit(self.0, limit))?;
        }
        Ok(())
    }

    /// Returns default power management limit on this device, in milliwatts.
    /// Default power management limit is a power management limit that the device boots with.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support power limits, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_management_default_limit(self) -> Result<u32> {
        let mut limit = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerManagementDefaultLimit(
                self.0,
                &raw mut limit,
            ))?;
        }
        Ok(limit)
    }

    /// Returns the effective power limit that the driver enforces after taking into account all limiters.
    ///
    /// This can differ from [`Device::power_management_limit`] if other limits are set elsewhere.
    /// This includes the out-of-band power-limit interface.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support power limits, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn enforced_power_limit(self) -> Result<u32> {
        let mut limit = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEnforcedPowerLimit(self.0, &raw mut limit))?;
        }
        Ok(limit)
    }

    /// Returns information about possible values of power management limits on this device.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or outputs, if the device does not support power-limit ranges, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_management_limit_constraints(self) -> Result<PowerLimits> {
        let mut min = 0;
        let mut max = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerManagementLimitConstraints(
                self.0,
                &raw mut min,
                &raw mut max,
            ))?;
        }
        Ok(PowerLimits { min, max })
    }

    /// Returns current power mizer mode on this device.
    ///
    /// PowerMizerMode provides a hint to the driver as to how to manage the performance of the GPU.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support PowerMizer mode
    /// readings, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn power_mizer_mode(self) -> Result<PowerMizerModes> {
        unsafe {
            let mut modes = MaybeUninit::<sys::nvmlDevicePowerMizerModes_v1_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetPowerMizerMode_v1(
                self.0,
                modes.as_mut_ptr()
            ))?;
            Ok(modes.assume_init().into())
        }
    }

    /// Sets the new power mizer mode.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or mode, if the device does not support PowerMizer mode changes,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn set_power_mizer_mode(self, mode: PowerMizerMode) -> Result<()> {
        let mut power_mizer = sys::nvmlDevicePowerMizerModes_v1_t {
            mode: mode.into(),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceSetPowerMizerMode_v1(
                self.0,
                &raw mut power_mizer
            ))
        }
    }

    /// Returns total energy consumption for this GPU in millijoules (mJ) since the driver was last reloaded.
    ///
    /// For Volta or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support energy readings, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn total_energy_consumption(self) -> Result<u64> {
        let mut energy = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTotalEnergyConsumption(
                self.0,
                &raw mut energy
            ))?;
        }
        Ok(energy)
    }

    /// Returns the CUDA compute capability of the device.
    ///
    /// For all products.
    ///
    /// Returns the major and minor compute capability version numbers of the device.
    /// The major and minor versions are equivalent to the `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` and `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` attributes returned by `cuDeviceGetAttribute`.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or outputs, if NVML has not been initialized, or if NVML reports
    /// an unexpected failure.
    pub fn compute_capability(self) -> Result<ComputeCapability> {
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCudaComputeCapability(
                self.0,
                &raw mut major,
                &raw mut minor,
            ))?;
        }
        Ok(ComputeCapability { major, minor })
    }

    /// Returns the current performance state for the device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// See [`PerformanceState`] for details on allowed performance states.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support performance-state
    /// queries, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn performance_state(self) -> Result<PerformanceState> {
        let mut state = sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPerformanceState(self.0, &raw mut state))?;
        }
        Ok(state.into())
    }

    pub fn power_state(self) -> Result<PerformanceState> {
        self.performance_state()
    }

    /// Returns all supported performance states (P-states) for the device.
    ///
    /// The returned array contains a contiguous list of valid P-states supported
    /// by the device.
    /// If the number of supported P-states is smaller than the supplied array,
    /// missing elements contain [`PerformanceState::Unknown`].
    ///
    /// The number of returned elements never exceeds `NVML_MAX_GPU_PERF_PSTATES`.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed internal P-state buffer is too small, if
    /// NVML rejects the query, if the device does not support performance-state
    /// readings, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn supported_performance_states(self) -> Result<Vec<PerformanceState>> {
        let mut states =
            [sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN; sys::NVML_MAX_GPU_PERF_PSTATES as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSupportedPerformanceStates(
                self.0,
                states.as_mut_ptr(),
                states.len() as u32,
            ))?;
        }
        Ok(states
            .into_iter()
            .filter(|state| *state != sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN)
            .map(Into::into)
            .collect())
    }

    /// Returns current clocks event reasons.
    ///
    /// For all fully supported products.
    ///
    /// More than one bit can be enabled at the same time.
    /// Multiple reasons can be affecting clocks at once.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support clocks-event reasons, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn current_clocks_event_reasons(self) -> Result<u64> {
        let mut reasons = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrentClocksEventReasons(
                self.0,
                &raw mut reasons,
            ))?;
        }
        Ok(reasons)
    }

    pub fn current_clocks_throttle_reasons(self) -> Result<u64> {
        self.current_clocks_event_reasons()
    }

    /// Returns bitmask of supported clocks event reasons that can be returned by [`Device::current_clocks_event_reasons`].
    ///
    /// For all fully supported products.
    ///
    /// Not supported in virtual machines running virtual GPU (vGPU).
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn supported_clocks_event_reasons(self) -> Result<u64> {
        let mut reasons = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSupportedClocksEventReasons(
                self.0,
                &raw mut reasons,
            ))?;
        }
        Ok(reasons)
    }

    pub fn supported_clocks_throttle_reasons(self) -> Result<u64> {
        self.supported_clocks_event_reasons()
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
    /// Returns an error if the device is inaccessible, if NVML rejects the event
    /// mask output, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn supported_event_types(self) -> Result<EventTypes> {
        let mut event_types = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSupportedEventTypes(
                self.0,
                &raw mut event_types,
            ))?;
        }
        Ok(EventTypes::from_bits_retain(event_types))
    }

    /// Returns the device board ID in the range `0..N`.
    /// Devices with the same board ID indicate GPUs connected to the same PLX.
    /// Use in conjunction with [`Device::is_multi_gpu_board`] to decide if they are on the same board as well.
    /// The returned board ID is unique for the current configuration.
    /// Uniqueness and ordering across reboots and system configurations are not
    /// guaranteed, but IDs remain distinct within one configuration.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support board IDs, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn board_id(self) -> Result<u32> {
        let mut board_id = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBoardId(self.0, &raw mut board_id))?;
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
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support display-mode reporting,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn display_mode(self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDisplayMode(self.0, &raw mut mode))?;
        }
        try_from_nvml_enum("enable state", mode as u32)
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
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if the device does not support display-active reporting, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn display_active(self) -> Result<EnableState> {
        let mut active = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDisplayActive(self.0, &raw mut active))?;
        }
        Ok(active.into())
    }

    /// Returns the current and pending ECC modes for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`] version 1.0 or higher.
    ///
    /// Changing ECC modes requires a reboot.
    /// The "pending" ECC mode refers to the target mode following the next reboot.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or outputs, if the device does not support ECC mode reporting, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn ecc_mode(self) -> Result<CurrentPending<EnableState>> {
        let mut current = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        let mut pending = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEccMode(
                self.0,
                &raw mut current,
                &raw mut pending,
            ))?;
        }
        Ok(CurrentPending {
            current: current.into(),
            pending: pending.into(),
        })
    }

    /// Returns the default ECC modes for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`] version 1.0 or higher.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support ECC mode reporting, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn default_ecc_mode(self) -> Result<EnableState> {
        let mut default_mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDefaultEccMode(
                self.0,
                &raw mut default_mode
            ))?;
        }
        Ok(default_mode.into())
    }

    /// Returns the current GPU operation mode and the pending mode that will
    /// take effect after reboot.
    ///
    /// For GK110 M-class and X-class Tesla products from the Kepler family.
    /// Modes [`GpuOperationMode::LowDp`] and [`GpuOperationMode::AllOn`] are supported on fully supported GeForce products.
    /// Not supported on Quadro and Tesla C-class products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or outputs, if the device does not support GPU operation modes, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gpu_operation_mode(self) -> Result<CurrentPending<GpuOperationMode>> {
        let mut current = sys::nvmlGpuOperationMode_t::NVML_GOM_ALL_ON;
        let mut pending = sys::nvmlGpuOperationMode_t::NVML_GOM_ALL_ON;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuOperationMode(
                self.0,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or outputs, if the platform is not Windows, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn driver_model(self) -> Result<CurrentPending<DriverModel>> {
        let mut current = sys::nvmlDriverModel_t::NVML_DRIVER_WDDM;
        let mut pending = sys::nvmlDriverModel_t::NVML_DRIVER_WDDM;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDriverModel_v2(
                self.0,
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
    /// It is possible to force the change to WDM (TCC) while the display is still attached with a force flag (nvmlFlagForce).
    /// Use the force flag only if the host is powered down afterward and the display is detached from the device before the next reboot.
    ///
    /// This operation takes effect after the next reboot.
    ///
    /// Windows driver model may only be set to WDDM when running in DEFAULT compute mode.
    ///
    /// Changing the driver model to WDDM is not supported when the GPU does not
    /// support graphics acceleration, or would not support it after reboot.
    /// See [`Device::set_gpu_operation_mode`].
    ///
    /// See [`DriverModel`] for details on available driver models.
    /// See [`DriverModelFlags`] for the available flag values.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, driver model, or flags, if the platform is not Windows or the
    /// device does not support driver-model changes, if the current process
    /// lacks permission, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn set_driver_model(self, model: DriverModel, flags: DriverModelFlags) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDriverModel(
                self.0,
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
    /// Modes [`GpuOperationMode::LowDp`] and [`GpuOperationMode::AllOn`] are supported on fully supported GeForce products.
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
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or mode, if the device does not support GPU operation mode changes
    /// or the requested mode, if the current process lacks permission, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_gpu_operation_mode(self, mode: GpuOperationMode) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetGpuOperationMode(self.0, mode.into()))?;
        }
        Ok(())
    }

    pub fn inforom_image_version(self) -> Result<String> {
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
    /// Returns an error if the device is inaccessible, if the internal infoROM
    /// version buffer is too small, if NVML rejects the output, if the device
    /// does not have an infoROM, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn inforom_version(self, object: InforomObject) -> Result<String> {
        let mut buffer = vec![0i8; sys::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetInforomVersion(
                self.as_raw(),
                object.into(),
                buffer.as_mut_ptr(),
                buffer.len() as u32,
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
    /// Returns an error if the infoROM checksum cannot be read because the
    /// infoROM is corrupted, if the device is inaccessible, if NVML rejects the
    /// output, if the device does not support checksums, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn inforom_configuration_checksum(self) -> Result<u32> {
        let mut checksum = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetInforomConfigurationChecksum(
                self.0,
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
    /// Returns an error if the infoROM is corrupted, if the device is
    /// inaccessible, if the device does not support infoROM validation, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn validate_inforom(self) -> Result<()> {
        unsafe { try_ffi!(sys::nvmlDeviceValidateInforom(self.0)) }
    }

    /// Returns the timestamp and the duration of the last flush of the BBX (blackbox) infoROM object during the current run.
    ///
    /// For all products with an inforom.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the BBX object has not
    /// been flushed yet, if the device does not have an infoROM, or if NVML
    /// reports an unexpected failure.
    pub fn last_bbx_flush_time(self) -> Result<BbxFlushTime> {
        let mut timestamp = 0;
        let mut duration_us = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetLastBBXFlushTime(
                self.0,
                &raw mut timestamp,
                &raw mut duration_us,
            ))?;
        }
        Ok(BbxFlushTime {
            timestamp,
            duration_us,
        })
    }

    /// Returns the device's Adaptive Clock status.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support adaptive-clock status,
    /// or if NVML has not been initialized.
    pub fn adaptive_clock_info_status(self) -> Result<AdaptiveClockInfoStatus> {
        let mut status = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAdaptiveClockInfoStatus(
                self.0,
                &raw mut status
            ))?;
        }
        try_from_nvml_enum("adaptive clock info status", status)
    }

    /// Returns the frequency monitor fault status for the device.
    ///
    /// For Ampere or newer fully supported devices.
    /// Requires root privileges.
    ///
    /// Returns the decoded frequency-monitor fault status reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support frequency-monitor
    /// status, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn clk_mon_status(self) -> Result<ClkMonStatus> {
        unsafe {
            let mut status = MaybeUninit::<sys::nvmlClkMonStatus_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetClkMonStatus(self.0, status.as_mut_ptr()))?;
            Ok(status.assume_init().into())
        }
    }

    /// Returns SRAM ECC error status of this device.
    ///
    /// For Ampere or newer fully supported devices.
    /// Requires root/admin permissions.
    ///
    /// Returns the SRAM ECC error status reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML rejects the handle
    /// or output, if the device does not support SRAM ECC status, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn sram_ecc_error_status(self) -> Result<EccSramErrorStatus> {
        let mut status = sys::nvmlEccSramErrorStatus_t {
            version: struct_version::<sys::nvmlEccSramErrorStatus_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSramEccErrorStatus(
                self.0,
                &raw mut status,
            ))?;
        }
        Ok(status.into())
    }

    pub fn sram_unique_uncorrected_ecc_error_counts(
        self,
    ) -> Result<EccSramUniqueUncorrectedErrorCounts> {
        let mut counts = sys::nvmlEccSramUniqueUncorrectedErrorCounts_t {
            version: struct_version::<sys::nvmlEccSramUniqueUncorrectedErrorCounts_t>(1),
            ..Default::default()
        };

        unsafe {
            try_ffi!(sys::nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(
                self.0,
                &raw mut counts,
            ))?;

            let entries = if counts.entryCount == 0 || counts.entries.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(counts.entries, counts.entryCount as usize)
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect()
            };

            Ok(EccSramUniqueUncorrectedErrorCounts { entries })
        }
    }

    /// Returns the device's interrupt number.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support IRQ-number reporting, or
    /// if NVML has not been initialized.
    pub fn irq_number(self) -> Result<u32> {
        let mut irq = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetIrqNum(self.0, &raw mut irq))?;
        }
        Ok(irq)
    }

    /// Returns the device's core count.
    ///
    /// On MIG-enabled GPUs, querying the device's core count is currently not supported by this operation.
    /// Use [`sys::nvmlDeviceGetGpuInstanceProfileInfo`] to fetch the MIG device's core count.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device or MIG device does not support core-count
    /// reporting, or if NVML has not been initialized.
    pub fn num_gpu_cores(self) -> Result<u32> {
        let mut cores = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNumGpuCores(self.0, &raw mut cores))?;
        }
        Ok(cores)
    }

    /// Returns the device's memory bus width.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support memory-bus-width
    /// reporting, or if NVML has not been initialized.
    pub fn memory_bus_width(self) -> Result<u32> {
        let mut width = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemoryBusWidth(self.0, &raw mut width))?;
        }
        Ok(width)
    }

    /// Returns the GPU bus type, such as PCIe or PCI.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML has not been initialized, if NVML rejects the
    /// handle or output, or if NVML reports an unexpected failure.
    pub fn bus_type(self) -> Result<BusType> {
        let mut bus_type = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBusType(self.0, &raw mut bus_type))?;
        }
        Ok(BusType::from_raw(bus_type))
    }

    /// Returns the device power source.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support power-source reporting,
    /// or if NVML has not been initialized.
    pub fn power_source(self) -> Result<PowerSource> {
        let mut power_source = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerSource(self.0, &raw mut power_source))?;
        }
        Ok(PowerSource::from_raw(power_source))
    }

    /// Returns the total ECC error counts for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`] version 1.0 or higher.
    /// Requires ECC Mode to be enabled.
    ///
    /// The total error count is the sum of errors across each separate memory system, that is, the total set of errors across the entire device.
    ///
    /// See [`MemoryErrorType`] for a description of available error types.
    /// See [`EccCounterType`] for a description of available counter types.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, error type, counter type, or output, if the device does not
    /// support ECC error reporting, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn total_ecc_errors(
        self,
        error_type: MemoryErrorType,
        counter_type: EccCounterType,
    ) -> Result<u64> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTotalEccErrors(
                self.0,
                error_type.into(),
                counter_type.into(),
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    pub fn detailed_ecc_errors(
        self,
        error_type: MemoryErrorType,
        counter_type: EccCounterType,
    ) -> Result<EccErrorCounts> {
        Ok(EccErrorCounts {
            l1_cache: self.memory_error_counter(
                error_type,
                counter_type,
                MemoryLocation::L1Cache,
            )?,
            l2_cache: self.memory_error_counter(
                error_type,
                counter_type,
                MemoryLocation::L2Cache,
            )?,
            device_memory: self.memory_error_counter(
                error_type,
                counter_type,
                MemoryLocation::Dram,
            )?,
            register_file: self.memory_error_counter(
                error_type,
                counter_type,
                MemoryLocation::RegisterFile,
            )?,
        })
    }

    /// Returns the requested memory error counter for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// Requires [`InforomObject::Ecc`] version 2.0 or higher to report aggregate location-based memory error counts.
    /// Requires [`InforomObject::Ecc`] version 1.0 or higher to report all other memory error counts.
    ///
    /// Only applicable to devices with ECC.
    ///
    /// Requires ECC Mode to be enabled.
    ///
    /// On MIG-enabled GPUs, per instance information can be queried using specific MIG device handles.
    /// Per instance information is currently only supported for non-DRAM uncorrectable volatile errors.
    /// Querying volatile errors using device handles is currently not supported.
    ///
    /// See [`MemoryErrorType`] for a description of available memory error types.
    /// See [`EccCounterType`] for a description of available counter types.
    /// See [`MemoryLocation`] for a description of available counter locations.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, error type, counter type, memory location, or output, if the
    /// device does not support ECC error reporting for the requested memory
    /// location, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn memory_error_counter(
        self,
        error_type: MemoryErrorType,
        counter_type: EccCounterType,
        location: MemoryLocation,
    ) -> Result<u64> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemoryErrorCounter(
                self.0,
                error_type.into(),
                counter_type.into(),
                location.into(),
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    pub fn violation_status(self, perf_policy: PerfPolicyType) -> Result<ViolationTime> {
        let sample = self.single_field_value(field_id_for_perf_policy(perf_policy), 0)?;
        Ok(ViolationTime {
            reference_time: u64::try_from(sample.timestamp).unwrap_or_default(),
            violation_time: field_sample_as_u64(sample)?,
        })
    }

    /// Returns the list of retired pages by source, including pages that are pending retirement.
    /// The returned address information is the hardware address of the retired page.
    /// This does not match the virtual address used in CUDA, but it matches the
    /// address information in Xid 63.
    ///
    /// [`Device::retired_pages`] adds an additional timestamps parameter to return the time of each page's retirement.
    /// This is supported for Pascal and newer architecture.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the retired-page list
    /// changes while the wrapper is fetching it, if NVML rejects the handle,
    /// cause, count, or outputs, if the device does not support page retirement,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn retired_pages(self, cause: PageRetirementCause) -> Result<Vec<RetiredPage>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetRetiredPages_v2(
                self.0,
                cause.into(),
                &raw mut count,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut addresses = vec![0u64; count as usize];
        let mut timestamps = vec![0u64; count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRetiredPages_v2(
                self.0,
                cause.into(),
                &raw mut count,
                addresses.as_mut_ptr(),
                timestamps.as_mut_ptr(),
            ))?;
        }
        addresses.truncate(count as usize);
        timestamps.truncate(count as usize);
        Ok(addresses
            .into_iter()
            .zip(timestamps)
            .map(|(address, timestamp)| RetiredPage {
                address,
                timestamp: Some(timestamp),
            })
            .collect())
    }

    /// Check if any pages are pending retirement and need a reboot to fully retire.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support page-retirement status,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn retired_pages_pending_status(self) -> Result<EnableState> {
        let mut pending = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRetiredPagesPendingStatus(
                self.0,
                &raw mut pending,
            ))?;
        }
        Ok(pending.into())
    }

    /// Returns the number of remapped rows.
    /// The number of rows reported is based on the remapping cause.
    /// `is_pending` indicates whether there are pending remappings.
    /// A reset is required to actually remap the row.
    /// `failure_occurred` is set if a row remapping ever failed in the past.
    /// A pending remapping does not affect future GPU work because error
    /// containment and dynamic page blacklisting handle it.
    ///
    /// On MIG-enabled GPUs with active instances, querying the number of remapped rows is not supported.
    ///
    /// For Ampere or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or outputs, if MIG is enabled
    /// or the device does not support remapped-row reporting, or if NVML reports
    /// an unexpected failure.
    pub fn remapped_rows(self) -> Result<RemappedRows> {
        let mut corrected = 0;
        let mut uncorrected = 0;
        let mut pending = 0;
        let mut failure_occurred = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRemappedRows(
                self.0,
                &raw mut corrected,
                &raw mut uncorrected,
                &raw mut pending,
                &raw mut failure_occurred,
            ))?;
        }
        Ok(RemappedRows {
            corrected,
            uncorrected,
            pending: pending != 0,
            failure_occurred: failure_occurred != 0,
        })
    }

    /// Returns the row remapper histogram.
    /// Returns the remap availability for each bank on the GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML reports an unexpected row-remapper query
    /// failure.
    pub fn row_remapper_histogram(self) -> Result<RowRemapperHistogram> {
        unsafe {
            let mut histogram = MaybeUninit::<sys::nvmlRowRemapperHistogramValues_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetRowRemapperHistogram(
                self.0,
                histogram.as_mut_ptr(),
            ))?;
            Ok(histogram.assume_init().into())
        }
    }

    /// Requests values for a list of device fields.
    /// Allows multiple fields to be queried at once.
    /// If multiple field IDs are populated by the same driver call, NVML
    /// populates those results from one call rather than one call per field ID.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device handle or field-value buffer.
    pub fn field_values(self, queries: &[FieldQuery]) -> Result<Vec<FieldSample>> {
        let mut values = queries
            .iter()
            .map(|query| sys::nvmlFieldValue_t {
                fieldId: query.field.0,
                scopeId: query.scope_id,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFieldValues(
                self.0,
                values.len() as i32,
                values.as_mut_ptr(),
            ))?;
        }
        Ok(values.into_iter().map(FieldSample::from_raw).collect())
    }

    fn single_field_value(self, field: FieldId, scope_id: u32) -> Result<FieldSample> {
        self.field_values(&[FieldQuery { field, scope_id }])?
            .into_iter()
            .next()
            .ok_or(Error::EmptyOutput {
                name: "nvmlDeviceGetFieldValues".into(),
            })
    }

    /// Clear values for a list of fields for a device.
    /// Allows multiple fields to be cleared at once.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device handle or field-value buffer.
    pub fn clear_field_values(self, queries: &[FieldQuery]) -> Result<Vec<FieldSample>> {
        let mut values = queries
            .iter()
            .map(|query| sys::nvmlFieldValue_t {
                fieldId: query.field.0,
                scopeId: query.scope_id,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        unsafe {
            try_ffi!(sys::nvmlDeviceClearFieldValues(
                self.0,
                values.len() as i32,
                values.as_mut_ptr(),
            ))?;
        }
        Ok(values.into_iter().map(FieldSample::from_raw).collect())
    }

    /// Returns recent samples for the GPU.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// Fetches power, utilization, or clock samples maintained in the driver's buffer, depending on the requested sample type.
    ///
    /// Power, utilization, and clock samples are returned as unsigned integer values.
    ///
    /// This wrapper performs the size query internally and returns the samples as an owned collection.
    ///
    /// `last_seen_timestamp` represents a CPU timestamp in microseconds.
    /// Use `0` to fetch all samples maintained by the buffer.
    /// Use a timestamp returned by a previous query to get more recent samples.
    ///
    /// This wrapper performs the NVML size query internally and returns the samples that were actually retrieved.
    /// Compared with polling the current-value methods, samples provide higher-frequency data at lower polling cost.
    ///
    /// On MIG-enabled GPUs, querying the following sample types, [`SamplingType::GpuUtilization`], [`SamplingType::MemoryUtilization`], [`SamplingType::EncoderUtilization`], and [`SamplingType::DecoderUtilization`], is not currently supported.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if the device does not support the requested sample type, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    /// Missing sample entries are returned as an empty sample list.
    pub fn samples(self, kind: SamplingType, last_seen_timestamp: u64) -> Result<Samples> {
        let mut value_type = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_INT;
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetSamples(
                self.0,
                kind.into(),
                last_seen_timestamp,
                &raw mut value_type,
                &raw mut count,
                ptr::null_mut(),
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Samples {
                value_type: try_from_nvml_enum("field value type", u32::from(value_type))?,
                samples: Vec::new(),
            });
        }
        if status == sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND {
            return Ok(Samples {
                value_type: try_from_nvml_enum("field value type", u32::from(value_type))?,
                samples: Vec::new(),
            });
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut samples = vec![sys::nvmlSample_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSamples(
                self.0,
                kind.into(),
                last_seen_timestamp,
                &raw mut value_type,
                &raw mut count,
                samples.as_mut_ptr(),
            ))?;
        }
        samples.truncate(count as usize);
        Ok(Samples {
            value_type: try_from_nvml_enum("field value type", u32::from(value_type))?,
            samples: samples
                .into_iter()
                .map(|sample| Sample::from_raw(value_type, sample))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    /// Returns GSP firmware version.
    ///
    /// This wrapper allocates the firmware-version buffer internally and returns it as a [`String`].
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device or output buffer, if GSP
    /// firmware is not enabled for the GPU, or if NVML reports an unexpected
    /// failure.
    pub fn gsp_firmware_version(self) -> Result<String> {
        let mut buffer = vec![0i8; sys::NVML_GSP_FIRMWARE_VERSION_BUF_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGspFirmwareVersion(
                self.0,
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
    /// Returns an error if NVML rejects the query, if GSP firmware is not
    /// enabled for the GPU, or if NVML reports an unexpected failure.
    pub fn gsp_firmware_mode(self) -> Result<GspFirmwareMode> {
        let mut enabled = 0;
        let mut default_mode = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGspFirmwareMode(
                self.0,
                &raw mut enabled,
                &raw mut default_mode,
            ))?;
        }
        Ok(GspFirmwareMode {
            enabled: enabled != 0,
            default_mode: default_mode != 0,
        })
    }

    /// Returns the number of fans on the device.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not have a fan, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn num_fans(self) -> Result<u32> {
        let mut fans = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNumFans(self.0, &raw mut fans))?;
        }
        Ok(fans)
    }

    /// Returns the intended operating speed of the device's fan.
    ///
    /// The reported speed is the intended fan speed.
    /// If the fan is physically blocked and unable to spin, the output does not
    /// match the actual fan speed.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
    /// This value may exceed 100% in certain cases.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not have a fan, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn default_fan_speed(self) -> Result<u32> {
        let mut speed = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFanSpeed(self.0, &raw mut speed))?;
        }
        Ok(speed)
    }

    /// Returns the intended operating speed of the device's specified fan.
    ///
    /// The reported speed is the intended fan speed.
    /// If the fan is physically blocked and unable to spin, the output does not
    /// match the actual fan speed.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
    /// This value may exceed 100% in certain cases.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, fan index, or output, if the device does not have a supported fan
    /// interface, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn fan_speed(self, fan: u32) -> Result<u32> {
        let mut speed = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFanSpeed_v2(self.0, fan, &raw mut speed))?;
        }
        Ok(speed)
    }

    /// Returns the intended operating speed in rotations per minute (RPM) of the device's specified fan.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// The reported speed is the intended fan speed.
    /// If the fan is physically blocked and unable to spin, the output does not
    /// match the actual fan speed.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the handle, fan index, or output, if the
    /// device does not support fan RPM reporting, or if NVML has not been
    /// initialized.
    pub fn fan_speed_rpm(self, fan: u32) -> Result<FanSpeedInfo> {
        let mut info = sys::nvmlFanSpeedInfo_t {
            version: struct_version::<sys::nvmlFanSpeedInfo_t>(1),
            fan,
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFanSpeedRPM(self.0, &raw mut info))?;
        }
        Ok(FanSpeedInfo {
            fan: info.fan,
            speed: info.speed,
        })
    }

    /// Returns current fan control policy.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, fan index, or output, if
    /// the device does not support fan policies, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn fan_control_policy(self, fan: u32) -> Result<FanPolicy> {
        let mut policy = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFanControlPolicy_v2(
                self.0,
                fan,
                &raw mut policy
            ))?;
        }
        try_from_nvml_enum("fan policy", policy)
    }

    /// Sets the speed of the fan control policy to default.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or fan index, if the device
    /// does not support fan-speed control, if NVML has not been initialized, or
    /// if NVML reports an unexpected failure.
    pub fn set_default_fan_speed(self, fan: u32) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDefaultFanSpeed_v2(self.0, fan))?;
        }
        Ok(())
    }

    /// Sets current fan control policy.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Requires privileged access.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, fan index, or policy, if the
    /// device does not support fan policies, if NVML has not been initialized,
    /// or if NVML reports an unexpected failure.
    pub fn set_fan_control_policy(self, fan: u32, policy: FanPolicy) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetFanControlPolicy(
                self.0,
                fan,
                policy.into()
            ))?;
        }
        Ok(())
    }

    /// Sets the speed of a specified fan.
    ///
    /// Warning: this changes the fan control policy to manual.
    /// You must monitor the temperature and adjust the fan speed accordingly.
    /// Setting the fan speed too low can damage the GPU.
    /// Use [`Device::set_default_fan_speed`] to restore default control policy.
    ///
    /// For all cuda-capable discrete products with fans that are Maxwell or Newer.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML has not been initialized, if NVML rejects the
    /// handle, fan index, or speed, if the device does not support manual fan
    /// speed control, or if NVML reports an unexpected failure.
    pub fn set_fan_speed(self, fan: u32, speed: u32) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetFanSpeed_v2(self.0, fan, speed))?;
        }
        Ok(())
    }

    /// Returns the intended target speed of the device's specified fan.
    ///
    /// Normally, the driver dynamically adjusts the fan based on the needs of the GPU.
    /// When the caller sets fan speed with [`Device::set_fan_speed`], the driver attempts to make the fan achieve that setting.
    /// The actual current speed of the fan is reported in [`Device::fan_speed`].
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
    /// This value may exceed 100% in certain cases.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, fan index, or output, if the device does not have a supported fan
    /// interface, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn target_fan_speed(self, fan: u32) -> Result<u32> {
        let mut speed = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTargetFanSpeed(
                self.0,
                fan,
                &raw mut speed
            ))?;
        }
        Ok(speed)
    }

    /// Returns the min and max fan speed that can be set for the GPU fan.
    ///
    /// For all cuda-capable discrete products with fans.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support fan-speed range reporting, if NVML has not been initialized,
    /// or if NVML reports an unexpected failure.
    pub fn min_max_fan_speed(self) -> Result<MinMaxFanSpeed> {
        let mut min = 0;
        let mut max = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMinMaxFanSpeed(
                self.0,
                &raw mut min,
                &raw mut max,
            ))?;
        }
        Ok(MinMaxFanSpeed { min, max })
    }

    /// Returns the cooler's information.
    /// Returns a cooler's control signal characteristics.
    /// The possible types are restricted, Variable and Toggle.
    /// See [`CoolerControl`](crate::types::CoolerControl) for details on available signal types.
    /// Returns objects that cooler cools.
    /// Targets may be GPU, Memory, Power Supply or All of these.
    /// See [`CoolerTarget`](crate::types::CoolerTarget) for details on available targets.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// For all discrete products with dedicated fans.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the handle or cooler index, if the device
    /// does not support cooler reporting, or if NVML has not been initialized.
    pub fn cooler_info(self, index: u32) -> Result<CoolerInfo> {
        let mut info = sys::nvmlCoolerInfo_t {
            version: struct_version::<sys::nvmlCoolerInfo_t>(1),
            index,
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCoolerInfo(self.0, &raw mut info))?;
        }
        Ok(CoolerInfo {
            index: info.index,
            signal_type: info.signalType.into(),
            target: info.target.into(),
        })
    }

    /// Returns the current temperature readings (in degrees C) for the given device.
    ///
    /// For all products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, sensor type, or output, if the device does not have the requested
    /// sensor, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn temperature(self, sensor: TemperatureSensor) -> Result<TemperatureInfo> {
        let mut temperature = sys::nvmlTemperature_t {
            version: struct_version::<sys::nvmlTemperature_t>(1),
            sensorType: sensor.into(),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTemperatureV(self.0, &raw mut temperature))?;
        }
        Ok(TemperatureInfo {
            sensor: temperature.sensorType.into(),
            temperature: temperature.temperature,
        })
    }

    /// Returns minor number for the device.
    /// The minor number is the suffix in the Linux device node path
    /// `/dev/nvidia[minor_number]`.
    ///
    /// For all products.
    /// Supported only for Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support Linux minor-number
    /// reporting, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn minor_number(self) -> Result<u32> {
        let mut minor_number = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMinorNumber(self.0, &raw mut minor_number))?;
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
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support compute-mode reporting,
    /// if NVML has not been initialized, or if NVML reports an unexpected
    /// failure.
    pub fn compute_mode(self) -> Result<ComputeMode> {
        let mut compute_mode = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_DEFAULT;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetComputeMode(self.0, &raw mut compute_mode))?;
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
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output, if the device does not support persistence mode, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn persistence_mode(self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPersistenceMode(self.0, &raw mut mode))?;
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
    /// Under Windows, compute mode may only be set to default when running in
    /// WDDM.
    ///
    /// On MIG-enabled GPUs, compute mode is set to default and changing it is
    /// not supported.
    ///
    /// See [`ComputeMode`] for details on available compute modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or mode, if the device does not support compute-mode changes, if
    /// the current process lacks permission, if NVML has not been initialized,
    /// or if NVML reports an unexpected failure.
    pub fn set_compute_mode(self, mode: ComputeMode) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetComputeMode(self.0, mode.into()))?;
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
    /// After disabling persistence mode on a device that has its own NUMA memory,
    /// the current device handle is no longer valid. Acquire a fresh handle from
    /// the library before continuing to interact with the device.
    /// This limitation is currently only applicable to devices that have a coherent NVLink connection to system memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or mode, if the device does not support persistence mode, if the
    /// current process lacks permission, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn set_persistence_mode(self, mode: EnableState) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetPersistenceMode(self.0, mode.into()))?;
        }
        Ok(())
    }

    /// Sets the ECC mode for the device.
    ///
    /// For Kepler or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`] version 1.0 or higher.
    /// Requires root/admin permissions.
    ///
    /// The ECC mode determines whether the GPU enables its ECC support.
    ///
    /// This operation takes effect after the next reboot.
    ///
    /// See [`EnableState`] for details on available modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or mode, if the device does not support ECC mode changes, if the
    /// current process lacks permission, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn set_ecc_mode(self, mode: EnableState) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceSetEccMode(self.0, mode.into()))?;
        }
        Ok(())
    }

    /// Clear the ECC error and other memory error counts for the device.
    ///
    /// For Kepler or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`] version 2.0 or higher to clear aggregate location-based ECC counts.
    /// Requires [`InforomObject::Ecc`] version 1.0 or higher to clear all other ECC counts.
    /// Requires root/admin permissions.
    /// Requires ECC Mode to be enabled.
    ///
    /// Sets all of the specified ECC counters to 0, including both detailed and total counts.
    ///
    /// This operation takes effect immediately.
    ///
    /// See [`MemoryErrorType`] for details on available counter types.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or counter type, if the device does not support clearing ECC
    /// counters, if the current process lacks permission, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn clear_ecc_error_counts(self, counter_type: EccCounterType) -> Result<()> {
        unsafe {
            try_ffi!(sys::nvmlDeviceClearEccErrorCounts(
                self.0,
                counter_type.into()
            ))?;
        }
        Ok(())
    }

    /// Returns MIG mode for the device.
    ///
    /// For Ampere or newer fully supported devices.
    ///
    /// Changing MIG modes may require device unbind or reset.
    /// The "pending" MIG mode refers to the target mode following the next activation trigger.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or outputs, if the device
    /// does not support MIG mode, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn mig_mode(self) -> Result<CurrentPending<MigMode>> {
        let mut current = 0;
        let mut pending = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMigMode(
                self.0,
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
    /// On Windows, only disabling MIG mode is supported. `activation_status` returns [`Status::NotSupported`] because GPU reset is not supported on Windows through this operation.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, requested mode, or
    /// activation-status output, if the device does not support MIG mode, if the
    /// current process lacks permission, or if NVML has not been initialized.
    pub fn set_mig_mode(self, mode: MigMode) -> Result<MigModeActivation> {
        let mut activation_status = sys::nvmlReturn_t::NVML_SUCCESS;
        unsafe {
            try_ffi!(sys::nvmlDeviceSetMigMode(
                self.0,
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
    /// Returns an error if NVML rejects the handle or output, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn max_mig_device_count(self) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxMigDeviceCount(self.0, &raw mut count))?;
        }
        Ok(count)
    }

    /// Tests if this handle refers to a MIG device.
    ///
    /// A MIG device handle is an NVML abstraction which maps to a MIG compute instance.
    /// These overloaded references can be used (with some restrictions) interchangeably with a GPU device handle to execute queries at a per-compute instance granularity.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support this check, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn is_mig_device_handle(self) -> Result<bool> {
        let mut is_mig_device = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceIsMigDeviceHandle(
                self.0,
                &raw mut is_mig_device,
            ))?;
        }
        Ok(is_mig_device != 0)
    }

    /// Returns parent device handle from a MIG device handle.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the MIG device handle or parent output,
    /// if the device does not support this query, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn parent_device_from_mig_handle(self) -> Result<Self> {
        let mut device = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
                self.0,
                &raw mut device,
            ))?;
            Ok(Self::from_raw(device))
        }
    }

    /// Returns MIG device handle for the given index under its parent NVML device.
    ///
    /// If the compute instance is destroyed, either explicitly or by destroying, resetting, or unbinding the parent GPU instance or GPU device, the MIG device handle remains invalid and must be requested again.
    /// Handles may be reused and their properties can change in the process.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, index, or output, if no MIG
    /// device exists at `index`, if the device does not support this query, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn mig_device(self, index: u32) -> Result<Self> {
        let mut device = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMigDeviceHandleByIndex(
                self.0,
                index,
                &raw mut device,
            ))?;
            Ok(Self::from_raw(device))
        }
    }

    pub fn mig_devices(self) -> Result<Vec<Self>> {
        let max_count = self.max_mig_device_count()?;
        let mut devices = Vec::with_capacity(max_count as usize);

        for index in 0..max_count {
            match self.mig_device(index) {
                Ok(device) => devices.push(device),
                Err(Error::Nvml {
                    code: Status::NotFound,
                    ..
                }) => {}
                Err(error) => return Err(error),
            }
        }

        Ok(devices)
    }

    /// Versioned wrapper that requests GPU-instance profile information using the latest supported NVML output layout.
    ///
    /// This wrapper sets the version field on the output structure before calling NVML.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, profile, or request layout,
    /// if MIG mode is disabled or the profile is unsupported, if the current
    /// process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instance_profile_info(self, profile: u32) -> Result<GpuInstanceProfileInfo> {
        let mut info = sys::nvmlGpuInstanceProfileInfo_v3_t {
            version: struct_version::<sys::nvmlGpuInstanceProfileInfo_v3_t>(3),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceProfileInfoV(
                self.0,
                profile,
                (&raw mut info).cast(),
            ))?;
        }
        Ok(info.into())
    }

    /// GPU instance profile query function that accepts profile ID, instead of profile name.
    /// It requests the result using the latest supported NVML output layout.
    ///
    /// This wrapper sets the version field on the output structure before calling NVML.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, profile ID, or request
    /// layout, if MIG mode is disabled or the profile is unsupported, if the
    /// current process lacks permission, or if NVML has not been initialized.
    pub fn gpu_instance_profile_info_by_id(
        self,
        profile_id: u32,
    ) -> Result<GpuInstanceProfileInfo> {
        let mut info = sys::nvmlGpuInstanceProfileInfo_v3_t {
            version: struct_version::<sys::nvmlGpuInstanceProfileInfo_v3_t>(3),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceProfileInfoByIdV(
                self.0,
                profile_id,
                (&raw mut info).cast(),
            ))?;
        }
        Ok(info.into())
    }

    /// Returns GPU instance profile capacity.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if MIG mode is disabled or
    /// `profile_id` is unsupported, if the current process lacks permission, or
    /// if NVML has not been initialized.
    pub fn gpu_instance_remaining_capacity(self, profile_id: u32) -> Result<u32> {
        let mut count = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceRemainingCapacity(
                self.0,
                profile_id,
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns GPU instance placements.
    ///
    /// A placement represents the location of a GPU instance within a device.
    /// Returns all possible placements for the given profile, regardless of whether MIG is enabled.
    /// A created GPU instance occupies memory slices described by its placement.
    /// Creating a GPU instance fails if its placement overlaps already occupied
    /// memory slices.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not
    /// support MIG or `profile_id` is unsupported, if the current process lacks
    /// permission, or if NVML has not been initialized.
    pub fn gpu_instance_possible_placements(
        self,
        profile_id: u32,
    ) -> Result<Vec<GpuInstancePlacement>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetGpuInstancePossiblePlacements_v2(
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

        let mut placements = vec![sys::nvmlGpuInstancePlacement_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstancePossiblePlacements_v2(
                self.0,
                profile_id,
                placements.as_mut_ptr(),
                &raw mut count,
            ))?;
        }
        placements.truncate(count as usize);
        Ok(placements.into_iter().map(Into::into).collect())
    }

    /// Returns GPU instances for the given instance ID.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if no GPU instance has the
    /// requested ID, if MIG mode is disabled, if the current process lacks
    /// permission, or if NVML has not been initialized.
    pub fn gpu_instance_by_id(self, id: u32) -> Result<GpuInstance> {
        let mut instance = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceById(
                self.0,
                id,
                &raw mut instance,
            ))?;
            Ok(GpuInstance::from_raw(instance))
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
    /// Returns an error if NVML cannot allocate the requested GPU instance, if
    /// NVML rejects the query, if MIG mode is disabled or the device is a vGPU
    /// guest, if the current process lacks permission, or if NVML has not been
    /// initialized.
    pub fn create_gpu_instance(self, profile_id: u32) -> Result<OwnedGpuInstance> {
        let mut instance = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceCreateGpuInstance(
                self.0,
                profile_id,
                &raw mut instance,
            ))?;
            Ok(OwnedGpuInstance::from_raw(instance))
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
    /// Returns an error if NVML cannot allocate the requested GPU instance, if
    /// NVML rejects the query or placement, if MIG mode is disabled or the
    /// device is a vGPU guest, if the current process lacks permission, or if
    /// NVML has not been initialized.
    pub fn create_gpu_instance_with_placement(
        self,
        profile_id: u32,
        placement: GpuInstancePlacement,
    ) -> Result<OwnedGpuInstance> {
        let placement = sys::nvmlGpuInstancePlacement_t::from(placement);
        let mut instance = ptr::null_mut();
        unsafe {
            try_ffi!(sys::nvmlDeviceCreateGpuInstanceWithPlacement(
                self.0,
                profile_id,
                &raw const placement,
                &raw mut instance,
            ))?;
            Ok(OwnedGpuInstance::from_raw(instance))
        }
    }

    /// Returns GPU instances for the given profile ID.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux only.
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if MIG mode is disabled, if
    /// the current process lacks permission, or if NVML has not been
    /// initialized.
    pub fn gpu_instances(self, profile_id: u32) -> Result<Vec<GpuInstance>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetGpuInstances(self.0, profile_id, ptr::null_mut(), &raw mut count)
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut instances = vec![ptr::null_mut(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstances(
                self.0,
                profile_id,
                instances.as_mut_ptr(),
                &raw mut count,
            ))?;
        }
        instances.truncate(count as usize);
        Ok(instances
            .into_iter()
            .map(|instance| unsafe { GpuInstance::from_raw(instance) })
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
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support this query, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn gpu_instance_id(self) -> Result<u32> {
        let mut id = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuInstanceId(self.0, &raw mut id))?;
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
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support this query, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn compute_instance_id(self) -> Result<u32> {
        let mut id = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetComputeInstanceId(self.0, &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns the virtualization mode corresponding to the GPU.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn virtualization_mode(self) -> Result<VirtualizationMode> {
        let mut mode = sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_NONE as u32;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetVirtualizationMode(
                self.0,
                (&raw mut mode).cast::<sys::nvmlGpuVirtualizationMode_t>(),
            ))?;
        }
        Ok(VirtualizationMode::from_raw(mode))
    }

    /// Queries if SR-IOV host operation is supported on a vGPU supported device.
    ///
    /// Checks whether SR-IOV host capability is supported by the device and the driver, and indicates device is in SR-IOV mode if both of these conditions are true.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not
    /// support host vGPU mode reporting, or if NVML reports an unexpected
    /// failure.
    pub fn host_vgpu_mode(self) -> Result<HostVgpuMode> {
        let mut mode = sys::nvmlHostVgpuMode_t::NVML_HOST_VGPU_MODE_NON_SRIOV;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetHostVgpuMode(self.0, &raw mut mode))?;
        }
        try_from_nvml_enum("host vgpu mode", mode as u32)
    }

    /// Returns the supported vGPU types on a physical GPU (device).
    ///
    /// This wrapper performs the NVML size query internally and returns the supported vGPU types as a [`Vec`].
    ///
    /// # Errors
    ///
    /// Returns an error if the vGPU type list changes while the wrapper is
    /// fetching it, if NVML rejects the query, if vGPU is unsupported by the
    /// device, or if NVML reports an unexpected failure.
    pub fn supported_vgpus(self) -> Result<Vec<VgpuTypeId>> {
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetSupportedVgpus(self.0, count, values)
        })?
        .into_iter()
        .map(VgpuTypeId)
        .collect())
    }

    /// Returns the currently creatable vGPU types on a physical GPU.
    ///
    /// The creatable vGPU types for a device may differ over time because
    /// restrictions can limit which vGPU types may run concurrently.
    /// For example, if only one vGPU type is allowed at a time on a device, the
    /// creatable list is restricted to the currently running vGPU type.
    /// This wrapper performs the NVML size query internally and returns the creatable vGPU types as a [`Vec`].
    ///
    /// # Errors
    ///
    /// Returns an error if the vGPU type list changes while the wrapper is
    /// fetching it, if NVML rejects the query, if vGPU is unsupported by the
    /// device, or if NVML reports an unexpected failure.
    pub fn creatable_vgpus(self) -> Result<Vec<VgpuTypeId>> {
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetCreatableVgpus(self.0, count, values)
        })?
        .into_iter()
        .map(VgpuTypeId)
        .collect())
    }

    pub fn vgpu_type_supported_placements(
        self,
        vgpu_type_id: VgpuTypeId,
        mode: VgpuPlacementMode,
    ) -> Result<Vec<VgpuPlacementId>> {
        self.query_vgpu_placements(
            vgpu_type_id,
            mode,
            sys::nvmlDeviceGetVgpuTypeSupportedPlacements,
        )
    }

    pub fn vgpu_type_creatable_placements(
        self,
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
    /// Returns an error if the active vGPU list changes while the wrapper is
    /// fetching it, if NVML rejects the query, if vGPU is unsupported by the
    /// device, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn active_vgpus(self) -> Result<Vec<VgpuInstance>> {
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetActiveVgpus(self.0, count, values)
        })?
        .into_iter()
        .map(|instance| VgpuInstance::from_id(VgpuInstanceId(instance)))
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
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the query, if MIG is enabled or the
    /// device does not support vGPU heterogeneous mode, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_heterogeneous_mode(self) -> Result<EnableState> {
        let mut mode = sys::nvmlVgpuHeterogeneousMode_t {
            version: struct_version::<sys::nvmlVgpuHeterogeneousMode_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetVgpuHeterogeneousMode(
                self.0,
                &raw mut mode
            ))?;
        }
        try_from_nvml_enum("enable state", mode.mode)
    }

    /// Returns a vGPU metadata structure for this physical GPU.
    /// The structure contains information about the GPU and the currently installed NVIDIA host driver version that's controlling it, together with an opaque data section containing internal state.
    ///
    /// This wrapper allocates the metadata buffer internally and retries if NVML reports it is too small.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata buffer changes while the wrapper is
    /// fetching it, if NVML rejects the metadata request, if vGPU is unsupported
    /// by the system, or if NVML reports an unexpected failure.
    pub fn vgpu_metadata(self) -> Result<PgpuMetadata> {
        let raw = query_sized_raw(|size, buffer| unsafe {
            sys::nvmlDeviceGetVgpuMetadata(self.0, buffer.cast(), size)
        })?;

        let metadata = unsafe { &*raw.as_ptr().cast::<sys::nvmlVgpuPgpuMetadata_t>() };
        let opaque_offset = mem::offset_of!(sys::nvmlVgpuPgpuMetadata_t, opaqueData);
        let opaque_end = opaque_offset.saturating_add(metadata.opaqueDataSize as usize);
        let opaque_data = raw[opaque_offset..opaque_end].to_vec();

        Ok(PgpuMetadata::from_raw(metadata, opaque_data))
    }

    /// Returns this physical GPU's properties as an ASCII-encoded string.
    ///
    /// This wrapper allocates the metadata buffer internally and retries if NVML reports it is too small.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata buffer changes while the wrapper is
    /// fetching it, if NVML rejects the metadata request, if vGPU is unsupported
    /// by the system, or if NVML reports an unexpected failure.
    pub fn pgpu_metadata_string(self) -> Result<String> {
        let raw = query_sized_raw(|size, buffer| unsafe {
            sys::nvmlDeviceGetPgpuMetadataString(self.0, buffer.cast(), size)
        })?;
        let buffer = raw.into_iter().map(|byte| byte as i8).collect::<Vec<_>>();
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the requested vGPU capability for GPU.
    ///
    /// See [`DeviceVgpuCapability`] for the supported capabilities.
    /// Returns whether the capability is supported and any associated capability data.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device, capability, or output, if
    /// the current state does not support this vGPU capability query, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_capability(self, capability: DeviceVgpuCapability) -> Result<u32> {
        let mut result = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetVgpuCapabilities(
                self.0,
                capability.into(),
                &raw mut result,
            ))?;
        }
        Ok(result)
    }

    fn query_vgpu_placements(
        self,
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
            placementSize: size_of::<u32>() as u32,
            mode: mode as u32,
            ..Default::default()
        };

        let status = unsafe { query(self.0, vgpu_type_id.0, &raw mut placement_list) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && placement_list.count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut placements = vec![0u32; placement_list.count as usize];
        placement_list.placementIds = placements.as_mut_ptr();
        unsafe {
            try_ffi!(query(self.0, vgpu_type_id.0, &raw mut placement_list))?;
        }
        placements.truncate(placement_list.count as usize);
        Ok(placements.into_iter().map(VgpuPlacementId).collect())
    }

    /// Returns bitmasks with the ideal CPU affinity for the device.
    /// For example, if processors 0, 1, 32, and 33 are ideal for the device and two mask words are requested, then `result[0] = 0x3` and `result[1] = 0x3`.
    /// This is equivalent to calling [`Device::cpu_affinity_within_scope`] with `NVML_AFFINITY_SCOPE_NODE`.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle or output buffer, if the device does not support CPU affinity
    /// reporting, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn cpu_affinity(self) -> Result<Vec<u64>> {
        device_ulong_bitmask_list(|set_size, set| unsafe {
            sys::nvmlDeviceGetCpuAffinity(self.0, set_size, set)
        })
    }

    /// Returns bitmasks with the ideal CPU affinity within a node or socket for the device.
    /// For example, if processors 0, 1, 32, and 33 are ideal for the device and two mask words are requested, then `result[0] = 0x3` and `result[1] = 0x3`.
    ///
    /// If the requested scope is not applicable to the target topology, NVML
    /// falls back to reporting CPU affinity for the device's immediate non-I/O
    /// ancestor.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, scope, or output buffer, if the device does not support scoped CPU
    /// affinity reporting, if NVML has not been initialized, or if NVML reports
    /// an unexpected failure.
    pub fn cpu_affinity_within_scope(self, scope: AffinityScope) -> Result<Vec<u64>> {
        device_ulong_bitmask_list(|set_size, set| unsafe {
            sys::nvmlDeviceGetCpuAffinityWithinScope(self.0, set_size, set, scope.into())
        })
    }

    /// Sets the ideal affinity for the calling thread and device using the guidelines given in [`Device::cpu_affinity`].
    /// Note, this is a change as of version 8.0.
    /// Older versions set the affinity for a calling process and all children.
    /// Currently supports up to 1024 processors.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, if the device does not support CPU affinity binding, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn set_cpu_affinity(self) -> Result<()> {
        unsafe { try_ffi!(sys::nvmlDeviceSetCpuAffinity(self.0)) }
    }

    /// Clear all affinity bindings for the calling thread.
    /// Note, this is a change as of version 8.0 as older versions cleared the affinity for a calling process and all children.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn clear_cpu_affinity(self) -> Result<()> {
        unsafe { try_ffi!(sys::nvmlDeviceClearCpuAffinity(self.0)) }
    }

    /// Returns bitmasks with the ideal memory affinity within a node or socket for this device.
    /// For example, if NUMA nodes 0 and 1 are ideal within the socket and one mask word is requested, `result[0] = 0x3`.
    ///
    /// If the requested scope is not applicable to the target topology, NVML
    /// falls back to reporting memory affinity for the device's immediate
    /// non-I/O ancestor.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, scope, or output buffer, if the device does not support memory
    /// affinity reporting, if NVML has not been initialized, or if NVML reports
    /// an unexpected failure.
    pub fn memory_affinity(self, scope: AffinityScope) -> Result<Vec<u64>> {
        device_ulong_bitmask_list(|set_size, set| unsafe {
            sys::nvmlDeviceGetMemoryAffinity(self.0, set_size, set, scope.into())
        })
    }

    /// Returns the addressing mode for the given GPU.
    ///
    /// [`DeviceAddressingMode::Hmm`] makes system-allocated memory (`malloc`,
    /// `mmap`) addressable from the GPU through software-based mirroring of the
    /// CPU page tables. [`DeviceAddressingMode::Ats`] makes system-allocated
    /// memory addressable from the GPU through Address Translation Services,
    /// which means there is effectively a single set of page tables used by
    /// both CPU and GPU. [`DeviceAddressingMode::None`] means neither HMM nor
    /// ATS is active.
    ///
    /// For Turing or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the handle, or if the current platform
    /// does not support addressing-mode reporting.
    pub fn addressing_mode(self) -> Result<DeviceAddressingMode> {
        let mut mode = sys::nvmlDeviceAddressingMode_t {
            version: struct_version::<sys::nvmlDeviceAddressingMode_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAddressingMode(self.0, &raw mut mode))?;
        }
        try_from_nvml_enum("device addressing mode", mode.value)
    }

    /// Returns the repair status for TPC/channel repair.
    ///
    /// For Ampere or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the handle or output, if the device does
    /// not support repair-status reporting, if NVML has not been initialized, or
    /// if NVML reports an unexpected failure.
    pub fn repair_status(self) -> Result<RepairStatus> {
        let mut status = sys::nvmlRepairStatus_t {
            version: struct_version::<sys::nvmlRepairStatus_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRepairStatus(self.0, &raw mut status))?;
        }
        Ok(status.into())
    }

    /// Returns the NUMA node of the given GPU device.
    /// This only applies to platforms where the GPUs are NUMA nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, or if the current
    /// platform does not support GPU NUMA-node reporting.
    pub fn numa_node_id(self) -> Result<u32> {
        let mut node = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNumaNodeId(self.0, &raw mut node))?;
        }
        Ok(node)
    }

    pub fn capabilities(self) -> Result<DeviceCapabilities> {
        let mut caps = sys::nvmlDeviceCapabilities_t {
            version: struct_version::<sys::nvmlDeviceCapabilities_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCapabilities(self.0, &raw mut caps))?;
        }
        Ok(DeviceCapabilities::from_bits_retain(caps.capMask))
    }

    /// Returns performance monitor samples from the associated subdevice.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// query, if the device does not support dynamic P-state samples, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn dynamic_pstates_info(self) -> Result<DynamicPstatesInfo> {
        unsafe {
            let mut info = MaybeUninit::<sys::nvmlGpuDynamicPstatesInfo_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetDynamicPstatesInfo(
                self.0,
                info.as_mut_ptr()
            ))?;
            Ok(info.assume_init().into())
        }
    }

    /// Versioned wrapper that requests GPU fabric information using the latest supported NVML output layout.
    ///
    /// This wrapper sets the version field on the output structure before calling NVML.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device does not support GPU fabric reporting.
    pub fn gpu_fabric_info(self) -> Result<GpuFabricInfo> {
        let mut info = sys::nvmlGpuFabricInfoV_t {
            version: struct_version::<sys::nvmlGpuFabricInfoV_t>(3),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuFabricInfoV(self.0, &raw mut info))?;
        }
        GpuFabricInfo::from_raw(info)
    }

    /// Returns performance profiles information.
    ///
    /// For Blackwell or newer fully supported devices.
    /// Returns the workload power profile information reported by NVML.
    /// The `perf_profiles_mask` field is a bitmask of all supported mode indices where the mode is supported if the index is 1.
    /// Each supported mode has a corresponding entry in the profile array containing the profile ID, the priority of this mode, and a conflicting mask.
    /// Lower priority values indicate higher priority, and each bit set in the conflicting mask corresponds to a different profile that cannot be used with the given profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML reports that the
    /// output layout is too small, if NVML rejects the query, if the device does
    /// not support workload power profiles, if NVML has not been initialized, or
    /// if NVML reports an unexpected failure.
    pub fn workload_power_profiles_info(self) -> Result<WorkloadPowerProfilesInfo> {
        let mut info = sys::nvmlWorkloadPowerProfileProfilesInfo_t {
            version: struct_version::<sys::nvmlWorkloadPowerProfileProfilesInfo_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceWorkloadPowerProfileGetProfilesInfo(
                self.0,
                &raw mut info,
            ))?;
        }
        Ok(info.into())
    }

    /// Returns current performance profiles.
    ///
    /// For Blackwell or newer fully supported devices.
    /// Returns the current workload power profile state reported by NVML.
    /// Returns the current, requested, and enforced performance profile masks.
    /// Each bit set in each bitmask indicates whether the profile is supported, currently requested, or currently engaged, respectively.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML rejects the query,
    /// if the device does not support workload power profiles, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn workload_power_current_profiles(self) -> Result<WorkloadPowerCurrentProfiles> {
        let mut profiles = sys::nvmlWorkloadPowerProfileCurrentProfiles_t {
            version: struct_version::<sys::nvmlWorkloadPowerProfileCurrentProfiles_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(
                self.0,
                &raw mut profiles,
            ))?;
        }
        Ok(profiles.into())
    }

    /// Returns the current and pending DRAM Encryption modes for the device.
    ///
    /// For Blackwell or newer fully supported devices.
    /// Only applicable to devices that support DRAM Encryption.
    /// Requires [`InforomObject::Den`] version 1.0 or higher.
    ///
    /// Changing DRAM Encryption modes requires a reboot.
    /// The "pending" DRAM Encryption mode refers to the target mode following the next reboot.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML rejects the handle
    /// or outputs, if the device does not support DRAM encryption, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn dram_encryption_mode(self) -> Result<CurrentPending<DramEncryptionInfo>> {
        let mut current = sys::nvmlDramEncryptionInfo_t {
            version: struct_version::<sys::nvmlDramEncryptionInfo_t>(1),
            ..Default::default()
        };
        let mut pending = sys::nvmlDramEncryptionInfo_t {
            version: struct_version::<sys::nvmlDramEncryptionInfo_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDramEncryptionMode(
                self.0,
                &raw mut current,
                &raw mut pending,
            ))?;
        }
        Ok(CurrentPending {
            current: current.into(),
            pending: pending.into(),
        })
    }

    /// Sets the DRAM encryption mode for the device.
    ///
    /// For Kepler or newer fully supported devices.
    /// Only applicable to devices that support DRAM Encryption.
    /// Requires [`InforomObject::Den`] version 1.0 or higher.
    /// Requires root/admin permissions.
    ///
    /// The DRAM Encryption mode determines whether the GPU enables its DRAM Encryption support.
    ///
    /// This operation takes effect after the next reboot.
    ///
    /// See [`EnableState`] for details on available modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML rejects the handle
    /// or requested mode, if the device does not support DRAM encryption, if the
    /// current process lacks permission, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn set_dram_encryption_mode(self, dram_encryption: DramEncryptionInfo) -> Result<()> {
        let dram_encryption = sys::nvmlDramEncryptionInfo_t {
            version: struct_version::<sys::nvmlDramEncryptionInfo_t>(1),
            encryptionState: dram_encryption.encryption_state.into(),
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDramEncryptionMode(
                self.0,
                &raw const dram_encryption,
            ))
        }
    }

    /// Returns Confidential Computing protected and unprotected memory sizes.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not
    /// support Confidential Computing memory-size reporting, or if NVML has not
    /// been initialized.
    pub fn conf_compute_mem_size_info(self) -> Result<ConfComputeMemSizeInfo> {
        let mut info = sys::nvmlConfComputeMemSizeInfo_t::default();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetConfComputeMemSizeInfo(
                self.0,
                &raw mut info
            ))?;
        }
        Ok(info.into())
    }

    /// Returns Confidential Computing protected memory usage.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not
    /// support Confidential Computing protected-memory reporting, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_protected_memory_usage(self) -> Result<MemoryInfo> {
        let mut memory = sys::nvmlMemory_v2_t {
            version: struct_version::<sys::nvmlMemory_v2_t>(2),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetConfComputeProtectedMemoryUsage(
                self.0,
                (&raw mut memory).cast(),
            ))?;
        }
        Ok(memory.into())
    }

    /// Returns Confidential Computing GPU certificate details.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not
    /// support Confidential Computing certificate reporting, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_gpu_certificate(self) -> Result<ConfComputeGpuCertificate> {
        unsafe {
            let mut certificate = MaybeUninit::<sys::nvmlConfComputeGpuCertificate_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetConfComputeGpuCertificate(
                self.0,
                certificate.as_mut_ptr(),
            ))?;
            Ok(certificate.assume_init().into())
        }
    }

    /// Returns Confidential Computing GPU attestation report.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not
    /// support Confidential Computing attestation reports, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_gpu_attestation_report(self) -> Result<ConfComputeGpuAttestationReport> {
        unsafe {
            let mut report = MaybeUninit::<sys::nvmlConfComputeGpuAttestationReport_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetConfComputeGpuAttestationReport(
                self.0,
                report.as_mut_ptr(),
            ))?;
            Ok(report.assume_init().into())
        }
    }

    /// Returns the state of the device's NVLink for the specified link.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, or output, if
    /// the device does not support NVLink state reporting, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_state(self, link: u32) -> Result<EnableState> {
        let mut state = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkState(self.0, link, &raw mut state))?;
        }
        Ok(state.into())
    }

    /// Returns the version of the device's NVLink for the specified link.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, or output, if
    /// the device does not support NVLink version reporting, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_version(self, link: u32) -> Result<NvLinkVersion> {
        let mut version = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkVersion(
                self.0,
                link,
                &raw mut version
            ))?;
        }
        try_from_nvml_enum("nvlink version", version)
    }

    /// Returns the requested capability from the device's NVLink for the specified link.
    ///
    /// See [`NvLinkCapability`] for the supported capabilities.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, capability, or
    /// output, if the device does not support NVLink capability reporting, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_capability(self, link: u32, capability: NvLinkCapability) -> Result<bool> {
        let mut result = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkCapability(
                self.0,
                link,
                capability.into(),
                &raw mut result,
            ))?;
        }
        Ok(result != 0)
    }

    /// Returns the PCI information for the remote node on an NVLink link.
    ///
    /// `pci_subsystem_id` is not filled by this query and is indeterminate.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, or output, if
    /// the device does not support remote PCI reporting for NVLink, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_remote_pci_info(self, link: u32) -> Result<PciInfo> {
        unsafe {
            let mut pci = MaybeUninit::<sys::nvmlPciInfo_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetNvLinkRemotePciInfo_v2(
                self.0,
                link,
                pci.as_mut_ptr(),
            ))?;
            Ok(pci.assume_init().into())
        }
    }

    /// Returns the NVLink device type of the remote device connected over the given link.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, link index, or output, if NVLink is unsupported, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_remote_device_type(self, link: u32) -> Result<NvLinkRemoteDeviceType> {
        let mut device_type = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkRemoteDeviceType(
                self.0,
                link,
                &raw mut device_type,
            ))?;
        }
        Ok(device_type.into())
    }

    /// Returns the specified error counter value.
    ///
    /// See [`NvLinkErrorCounter`] for available counters.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, link index, counter, or
    /// output, if the device does not support NVLink error counters, if NVML has
    /// not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_error_counter(self, link: u32, counter: NvLinkErrorCounter) -> Result<u64> {
        let mut value = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkErrorCounter(
                self.0,
                link,
                counter.into(),
                &raw mut value,
            ))?;
        }
        Ok(value)
    }

    /// Returns the supported NVLink reduced bandwidth modes of the device.
    ///
    /// For Blackwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the handle or output, or if the device
    /// does not support NVLink bandwidth modes.
    pub fn nvlink_supported_bw_modes(self) -> Result<NvLinkSupportedBwModes> {
        let mut modes = sys::nvmlNvlinkSupportedBwModes_t {
            version: struct_version::<sys::nvmlNvlinkSupportedBwModes_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvlinkSupportedBwModes(
                self.0,
                &raw mut modes
            ))?;
        }
        Ok(modes.into())
    }

    /// Returns the NVLink reduced bandwidth mode for the device.
    ///
    /// For Blackwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if NVML rejects the handle or output, or if the device
    /// does not support NVLink bandwidth mode reporting.
    pub fn nvlink_bw_mode(self) -> Result<NvLinkBwMode> {
        let mut mode = sys::nvmlNvlinkGetBwMode_t {
            version: struct_version::<sys::nvmlNvlinkGetBwMode_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvlinkBwMode(self.0, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Query NVLINK information associated with this device.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if NVML rejects the handle
    /// or output, if the device does not support NVLink info reporting, if NVML
    /// has not been initialized, or if NVML reports an unexpected failure.
    pub fn nvlink_info(self) -> Result<NvLinkInfo> {
        let mut info = sys::nvmlNvLinkInfo_t {
            version: struct_version::<sys::nvmlNvLinkInfo_t>(2),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNvLinkInfo(self.0, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Returns the active frame buffer capture sessions statistics for the given device.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the stats
    /// output, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn fbc_stats(self) -> Result<FbcStats> {
        unsafe {
            let mut stats = MaybeUninit::<sys::nvmlFBCStats_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetFBCStats(self.0, stats.as_mut_ptr()))?;
            Ok(stats.assume_init().into())
        }
    }

    /// Returns information about active frame buffer capture sessions on a target device.
    ///
    /// This wrapper queries the required session count first, then returns the active FBC sessions as a [`Vec`].
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// `h_resolution`, `v_resolution`, `average_fps`, and `average_latency` may be zero if no new frames were captured since the session started.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the FBC session list
    /// changes while the wrapper is fetching it, if NVML reports an invalid
    /// session count, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn fbc_sessions(self) -> Result<Vec<FbcSessionInfo>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetFBCSessions(self.as_raw(), &raw mut count, ptr::null_mut() as _)
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut sessions = vec![sys::nvmlFBCSessionInfo_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFBCSessions(
                self.as_raw(),
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }
        sessions.truncate(count as usize);
        Ok(sessions.into_iter().map(Into::into).collect())
    }

    /// Returns the common ancestor for two devices.
    ///
    /// For all products.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects either handle or the output, if topology
    /// discovery is not supported on this device or OS, or if NVML fails during
    /// topology discovery.
    pub fn topology_common_ancestor(self, other: Self) -> Result<TopologyLevel> {
        let mut level = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_SYSTEM;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTopologyCommonAncestor(
                self.0,
                other.0,
                &raw mut level,
            ))?;
        }
        Ok(level.into())
    }

    /// Returns the set of GPUs nearest to this device at a specific interconnectivity level.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, topology level, count, or
    /// output buffer, if topology discovery is not supported on this device or
    /// OS, or if NVML fails during topology discovery.
    pub fn topology_nearest_gpus(self, level: TopologyLevel) -> Result<Vec<Device>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetTopologyNearestGpus(
                self.0,
                level.into(),
                &raw mut count,
                ptr::null_mut(),
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut devices = vec![ptr::null_mut(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTopologyNearestGpus(
                self.0,
                level.into(),
                &raw mut count,
                devices.as_mut_ptr(),
            ))?;
        }
        devices.truncate(count as usize);
        Ok(devices.into_iter().map(Self).collect())
    }

    /// Returns the status for a P2P capability between this device and `other`.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects either device handle, the capability, or
    /// output, or if NVML reports an unexpected failure.
    pub fn p2p_status(self, other: Self, capability: P2pCapabilityIndex) -> Result<P2pStatus> {
        let mut status = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_UNKNOWN;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetP2PStatus(
                self.0,
                other.0,
                capability.into(),
                &raw mut status,
            ))?;
        }
        Ok(status.into())
    }

    /// Check if the GPU devices are on the same physical board.
    ///
    /// For all fully supported products.
    ///
    /// # Errors
    ///
    /// Returns an error if either device is inaccessible, if NVML rejects either
    /// handle or output, if the check is unsupported, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn on_same_board(self, other: Self) -> Result<bool> {
        let mut on_same_board = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceOnSameBoard(
                self.0,
                other.0,
                &raw mut on_same_board,
            ))?;
        }
        Ok(on_same_board != 0)
    }

    pub fn compute_running_processes(self) -> Result<Vec<ProcessInfo>> {
        query_process_info_list(self, sys::nvmlDeviceGetComputeRunningProcesses)
    }

    /// Returns information about running processes on a device for the input context.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// Returns information only about running processes, such as CUDA applications with active contexts.
    ///
    /// This wrapper performs the NVML size query internally and returns the matching process details as a [`Vec`].
    ///
    /// The `used_gpu_memory` field is all of the memory used by the application.
    /// The `used_gpu_cc_protected_memory` field is all of the protected memory used by the application.
    ///
    /// Keep in mind that information returned by this call is dynamic and the number of elements might change in time.
    /// The wrapper retries internally if NVML reports that more space is needed because new processes were spawned.
    ///
    /// In MIG mode, a physical device handle reports aggregate information only if the caller has appropriate privileges.
    /// Per-instance information can be queried by using specific MIG device handles.
    /// Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
    /// Protected memory usage is currently not available in MIG mode or on Windows.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the process list
    /// changes while the wrapper is fetching it, if NVML rejects the device,
    /// process mode, or output, if the query is unsupported, if the current
    /// process lacks permission, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn running_process_details(self, mode: ProcessMode) -> Result<Vec<ProcessDetail>> {
        let mut list = sys::nvmlProcessDetailList_t {
            version: struct_version::<sys::nvmlProcessDetailList_t>(1),
            mode: mode.into(),
            ..Default::default()
        };
        let status = unsafe { sys::nvmlDeviceGetRunningProcessDetailList(self.0, &raw mut list) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && list.numProcArrayEntries == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut entries =
            vec![sys::nvmlProcessDetail_v1_t::default(); list.numProcArrayEntries as usize];
        list.procArray = entries.as_mut_ptr();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRunningProcessDetailList(
                self.0,
                &raw mut list
            ))?;
        }
        entries.truncate(list.numProcArrayEntries as usize);
        Ok(entries.into_iter().map(Into::into).collect())
    }

    pub fn graphics_running_processes(self) -> Result<Vec<ProcessInfo>> {
        query_process_info_list(self, sys::nvmlDeviceGetGraphicsRunningProcesses)
    }

    pub fn mps_compute_running_processes(self) -> Result<Vec<ProcessInfo>> {
        query_process_info_list(self, sys::nvmlDeviceGetMPSComputeRunningProcesses)
    }

    /// Returns the current utilization and process ID.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running.
    /// This wrapper returns utilization values as a [`Vec`] of process samples.
    /// One utilization sample structure is returned per running process that had
    /// non-zero utilization during the last sample period.
    /// It includes the CPU timestamp at which the samples were recorded.
    /// Individual utilization values are returned as unsigned integer values.
    /// If no valid sample entries are found since `last_seen_timestamp`, [`Status::NotFound`] is returned.
    ///
    /// This wrapper performs the NVML size query internally and retries with the required capacity.
    ///
    /// On success, NVML reports the number of process utilization sample structures that were actually written.
    /// This may differ from a previously read value as instances are created or destroyed.
    ///
    /// `last_seen_timestamp` represents the CPU timestamp in microseconds at which utilization samples were last read.
    /// Use `0` to read utilization based on all samples maintained by the driver's internal sample buffer.
    /// Use a timestamp returned by a previous query to read utilization since that query.
    ///
    /// On MIG-enabled GPUs, querying process utilization is not currently supported.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the
    /// handle, utilization buffer, or sampling timestamp, if the device does not
    /// support process utilization sampling, if NVML has not been initialized,
    /// or if NVML reports an unexpected failure. Missing sample entries are
    /// returned as an empty list.
    pub fn process_utilization(
        self,
        last_seen_timestamp: u64,
    ) -> Result<Vec<ProcessUtilizationSample>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlDeviceGetProcessUtilization(
                self.0,
                ptr::null_mut(),
                &raw mut count,
                last_seen_timestamp,
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status == sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut samples = vec![sys::nvmlProcessUtilizationSample_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetProcessUtilization(
                self.0,
                samples.as_mut_ptr(),
                &raw mut count,
                last_seen_timestamp,
            ))?;
        }
        samples.truncate(count as usize);
        Ok(samples.into_iter().map(Into::into).collect())
    }

    /// Returns the recent utilization and process ID for all running processes.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder, jpeg decoder, OFA (Optical Flow Accelerator) for all running processes.
    /// This wrapper returns utilization values as a [`Vec`] of process utilization records.
    /// One utilization sample structure is returned per running process that had
    /// non-zero utilization during the last sample period.
    /// It includes the CPU timestamp at which the samples were recorded.
    /// Individual utilization values are returned as unsigned integer values.
    ///
    /// This wrapper performs the NVML size query internally and retries with the required capacity.
    ///
    /// On success, NVML reports the number of process utilization info structures that were actually written.
    /// This may differ from a previously read value as instances are created or destroyed.
    ///
    /// `last_seen_timestamp` represents the CPU timestamp in microseconds at which utilization samples were last read.
    /// Use `0` to read utilization based on all samples maintained by the driver's internal sample buffer.
    /// Use a timestamp returned by a previous query to read utilization since that query.
    ///
    /// On MIG-enabled GPUs, querying process utilization is not currently supported.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the
    /// request layout, if the device is inaccessible, if the process list
    /// changes while the wrapper is fetching it, if NVML rejects the query, if
    /// the device does not support process utilization sampling, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure. Missing
    /// sample entries are returned as an empty list.
    pub fn processes_utilization(
        self,
        last_seen_timestamp: u64,
    ) -> Result<Vec<ProcessUtilizationInfo>> {
        let mut info = sys::nvmlProcessesUtilizationInfo_t {
            version: struct_version::<sys::nvmlProcessesUtilizationInfo_t>(1),
            lastSeenTimeStamp: last_seen_timestamp,
            processSamplesCount: 0,
            ..Default::default()
        };
        let status = unsafe { sys::nvmlDeviceGetProcessesUtilizationInfo(self.0, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.processSamplesCount == 0 {
            return Ok(Vec::new());
        }
        if status == sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut entries = vec![
            sys::nvmlProcessUtilizationInfo_v1_t::default();
            info.processSamplesCount as usize
        ];
        info.procUtilArray = entries.as_mut_ptr();
        unsafe {
            try_ffi!(sys::nvmlDeviceGetProcessesUtilizationInfo(
                self.0,
                &raw mut info
            ))?;
        }
        entries.truncate(info.processSamplesCount as usize);
        Ok(entries.into_iter().map(Into::into).collect())
    }

    /// Queries the state of per process accounting mode.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// See [`Device::accounting_stats`] for the reported accounting metrics.
    /// See [`sys::nvmlDeviceSetAccountingMode`].
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does
    /// not support accounting mode, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn accounting_mode(self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAccountingMode(self.0, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Queries process's accounting stats.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// Accounting stats capture GPU utilization and other statistics across the lifetime of a process.
    /// Accounting stats can be queried during the lifetime of the process and after its termination.
    /// The reported running time remains 0 while the process is still running and is updated to the actual duration after termination.
    /// Accounting stats are kept in a circular buffer, newly created processes overwrite information about old processes.
    ///
    /// The returned value includes the per-process accounting metrics exposed by NVML.
    /// Use [`Device::accounting_pids`] to list processes with available stats.
    ///
    /// * Accounting mode must be enabled.
    ///   See [`Device::accounting_mode`].
    /// * Only compute and graphics application stats can be queried.
    ///   Monitoring application stats cannot be queried because they do not contribute to GPU utilization.
    /// * With a PID collision, only stats for the latest process to terminate are reported.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, process ID, or output, if no
    /// accounting stats exist for the process, if accounting is unsupported or
    /// disabled, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn accounting_stats(self, pid: crate::types::Pid) -> Result<AccountingStats> {
        unsafe {
            let mut stats = MaybeUninit::<sys::nvmlAccountingStats_t>::uninit();
            try_ffi!(sys::nvmlDeviceGetAccountingStats(
                self.0,
                pid.0,
                stats.as_mut_ptr()
            ))?;
            Ok(stats.assume_init().into())
        }
    }

    /// Returns processes that have accounting stats.
    /// Returned processes can be running or terminated.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// This wrapper queries the process count internally.
    /// It returns an empty list when no accounting processes are available.
    ///
    /// See [`Device::accounting_stats`] for the per-process metrics.
    ///
    /// With a PID collision, some processes may not be accessible before the circular buffer is full.
    ///
    /// # Errors
    ///
    /// Returns an error if the accounting PID list changes while the wrapper is
    /// fetching it, if NVML rejects the handle or count output, if accounting is
    /// unsupported or disabled, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn accounting_pids(self) -> Result<Vec<crate::types::Pid>> {
        let mut count = 0;
        let status =
            unsafe { sys::nvmlDeviceGetAccountingPids(self.0, &raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut pids = vec![0u32; count as usize];
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAccountingPids(
                self.0,
                &raw mut count,
                pids.as_mut_ptr(),
            ))?;
        }
        pids.truncate(count as usize);
        Ok(pids.into_iter().map(crate::types::Pid).collect())
    }

    /// Returns the number of processes that the circular buffer with accounting pids can hold.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// This is the maximum number of processes with stored accounting information before
    /// information about the oldest processes is overwritten by newer process information.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if accounting is
    /// unsupported or disabled, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn accounting_buffer_size(self) -> Result<u32> {
        let mut size = 0;
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAccountingBufferSize(
                self.0,
                &raw mut size
            ))?;
        }
        Ok(size)
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
        FieldValue::SignedInt(value) => u64::try_from(value).map_err(|_| Error::NegativeValue {
            name: "field sample".into(),
            value: i64::from(value),
        })?,
        FieldValue::SignedLongLong(value) => {
            u64::try_from(value).map_err(|_| Error::NegativeValue {
                name: "field sample".into(),
                value,
            })?
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
