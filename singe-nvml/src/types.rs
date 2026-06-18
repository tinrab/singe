#![allow(deprecated)]

use std::{cmp::max, mem};

use crate::{device::Device, gpu_instance::GpuInstance};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{
    copy_string_to_c_chars, impl_enum_conversion, string_from_c_chars, string_from_c_ptr,
    try_enum_from_raw,
};
use singe_nvml_sys as sys;

use crate::{
    error::{Error, Result},
    utility::{mask255_bits, option_u64_from_not_available},
};

pub(crate) fn try_from_nvml_enum<T>(name: &'static str, value: u32) -> Result<T>
where
    T: TryFrom<u32>,
{
    try_enum_from_raw(name, value).map_err(|error| Error::UnknownEnumValue {
        name: error.name.into(),
        value: error.value,
    })
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct InitFlags: u32 {
        const NO_GPUS = sys::NVML_INIT_FLAG_NO_GPUS;
        const NO_ATTACH = sys::NVML_INIT_FLAG_NO_ATTACH;
        const FORCE_INIT = sys::NVML_INIT_FLAG_FORCE_INIT;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventTypes: u64 {
        const NONE = sys::nvmlEventTypeNone as u64;
        const SINGLE_BIT_ECC_ERROR = sys::nvmlEventTypeSingleBitEccError as u64;
        const DOUBLE_BIT_ECC_ERROR = sys::nvmlEventTypeDoubleBitEccError as u64;
        const PSTATE = sys::nvmlEventTypePState as u64;
        const XID_CRITICAL_ERROR = sys::nvmlEventTypeXidCriticalError as u64;
        const CLOCK = sys::nvmlEventTypeClock as u64;
        const POWER_SOURCE_CHANGE = sys::nvmlEventTypePowerSourceChange as u64;
        const MIG_CONFIG_CHANGE = sys::nvmlEventMigConfigChange as u64;
        const SINGLE_BIT_ECC_ERROR_STORM = sys::nvmlEventTypeSingleBitEccErrorStorm as u64;
        const DRAM_RETIREMENT_EVENT = sys::nvmlEventTypeDramRetirementEvent as u64;
        const DRAM_RETIREMENT_FAILURE = sys::nvmlEventTypeDramRetirementFailure as u64;
        const NON_FATAL_POISON_ERROR = sys::nvmlEventTypeNonFatalPoisonError as u64;
        const FATAL_POISON_ERROR = sys::nvmlEventTypeFatalPoisonError as u64;
        const GPU_UNAVAILABLE_ERROR = sys::nvmlEventTypeGpuUnavailableError as u64;
        const GPU_RECOVERY_ACTION = sys::nvmlEventTypeGpuRecoveryAction as u64;
        const ALL = sys::nvmlEventTypeAll as u64;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SystemEventTypes: u64 {
        const GPU_DRIVER_UNBIND = sys::nvmlSystemEventTypeGpuDriverUnbind as u64;
        const GPU_DRIVER_BIND = sys::nvmlSystemEventTypeGpuDriverBind as u64;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DeviceCapabilities: u32 {
        const EGM = sys::NVML_DEV_CAP_EGM;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DriverModelFlags: u32 {
        const DEFAULT = sys::nvmlFlagDefault;
        const FORCE = sys::nvmlFlagForce;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GpuInstanceProfileCapabilities: u32 {
        const P2P = sys::NVML_GPU_INSTANCE_PROFILE_CAPS_P2P;
        const GFX = sys::NVML_GPU_INSTANCE_PROFILE_CAPS_GFX;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ComputeInstanceProfileCapabilities: u32 {
        const GFX = sys::NVML_COMPUTE_INSTANCE_PROFILE_CAPS_GFX;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NvLinkPacketTypes: u32 {
        const NOP = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_NOP as _;
        const READ = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_READ as _;
        const WRITE = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_WRITE as _;
        const RATOM = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RATOM as _;
        const NRATOM = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_NRATOM as _;
        const FLUSH = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_FLUSH as _;
        const RESPDATA = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA as _;
        const RESPNODATA = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA as _;
        const ALL = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_ALL as _;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaDriverVersion {
    pub raw: i32,
    pub major: i32,
    pub minor: i32,
}

impl CudaDriverVersion {
    pub const fn from_raw(raw: i32) -> Self {
        Self {
            raw,
            major: raw / 1000,
            minor: (raw % 1000) / 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pid(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldId(pub u32);

impl FieldId {
    pub const ECC_CURRENT: Self = Self(sys::NVML_FI_DEV_ECC_CURRENT);
    pub const ECC_PENDING: Self = Self(sys::NVML_FI_DEV_ECC_PENDING);
    pub const RETIRED_SBE: Self = Self(sys::NVML_FI_DEV_RETIRED_SBE);
    pub const RETIRED_DBE: Self = Self(sys::NVML_FI_DEV_RETIRED_DBE);
    pub const RETIRED_PENDING: Self = Self(sys::NVML_FI_DEV_RETIRED_PENDING);
    pub const MEMORY_TEMP: Self = Self(sys::NVML_FI_DEV_MEMORY_TEMP);
    pub const TOTAL_ENERGY_CONSUMPTION: Self = Self(sys::NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION);
    pub const PCIE_REPLAY_COUNTER: Self = Self(sys::NVML_FI_DEV_PCIE_REPLAY_COUNTER);
    pub const REMAPPED_CORRECTED_ROWS: Self = Self(sys::NVML_FI_DEV_REMAPPED_COR);
    pub const REMAPPED_UNCORRECTED_ROWS: Self = Self(sys::NVML_FI_DEV_REMAPPED_UNC);
    pub const REMAPPED_PENDING: Self = Self(sys::NVML_FI_DEV_REMAPPED_PENDING);
    pub const REMAPPED_FAILURE: Self = Self(sys::NVML_FI_DEV_REMAPPED_FAILURE);
    pub const POWER_AVERAGE: Self = Self(sys::NVML_FI_DEV_POWER_AVERAGE);
    pub const POWER_INSTANT: Self = Self(sys::NVML_FI_DEV_POWER_INSTANT);
    pub const POWER_MIN_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_MIN_LIMIT);
    pub const POWER_MAX_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_MAX_LIMIT);
    pub const POWER_DEFAULT_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_DEFAULT_LIMIT);
    pub const POWER_CURRENT_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_CURRENT_LIMIT);
    pub const ENERGY: Self = Self(sys::NVML_FI_DEV_ENERGY);
    pub const POWER_REQUESTED_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_REQUESTED_LIMIT);
    pub const PCIE_TX_BYTES: Self = Self(sys::NVML_FI_DEV_PCIE_COUNT_TX_BYTES);
    pub const PCIE_RX_BYTES: Self = Self(sys::NVML_FI_DEV_PCIE_COUNT_RX_BYTES);
    pub const NVLINK_THROUGHPUT_DATA_TX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX);
    pub const NVLINK_THROUGHPUT_DATA_RX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX);
    pub const NVLINK_THROUGHPUT_RAW_TX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX);
    pub const NVLINK_THROUGHPUT_RAW_RX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX);
    pub const PERF_POLICY_POWER: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_POWER);
    pub const PERF_POLICY_THERMAL: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_THERMAL);
    pub const PERF_POLICY_SYNC_BOOST: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_SYNC_BOOST);
    pub const PERF_POLICY_BOARD_LIMIT: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT);
    pub const PERF_POLICY_LOW_UTILIZATION: Self =
        Self(sys::NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION);
    pub const PERF_POLICY_RELIABILITY: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_RELIABILITY);
    pub const PERF_POLICY_TOTAL_APP_CLOCKS: Self =
        Self(sys::NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS);
    pub const PERF_POLICY_TOTAL_BASE_CLOCKS: Self =
        Self(sys::NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS);
    pub const TEMPERATURE_SHUTDOWN_TLIMIT: Self =
        Self(sys::NVML_FI_DEV_TEMPERATURE_SHUTDOWN_TLIMIT);
    pub const TEMPERATURE_SLOWDOWN_TLIMIT: Self =
        Self(sys::NVML_FI_DEV_TEMPERATURE_SLOWDOWN_TLIMIT);
    pub const TEMPERATURE_MEMORY_MAX_TLIMIT: Self =
        Self(sys::NVML_FI_DEV_TEMPERATURE_MEM_MAX_TLIMIT);
    pub const TEMPERATURE_GPU_MAX_TLIMIT: Self = Self(sys::NVML_FI_DEV_TEMPERATURE_GPU_MAX_TLIMIT);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpmMetricId(pub u32);

impl GpmMetricId {
    pub const GRAPHICS_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GRAPHICS_UTIL as _);
    pub const SM_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_UTIL as _);
    pub const SM_OCCUPANCY: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_OCCUPANCY as _);
    pub const INTEGER_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_INTEGER_UTIL as _);
    pub const ANY_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_ANY_TENSOR_UTIL as _);
    pub const DFMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DFMA_TENSOR_UTIL as _);
    pub const HMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HMMA_TENSOR_UTIL as _);
    pub const IMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_IMMA_TENSOR_UTIL as _);
    pub const DRAM_BW_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DRAM_BW_UTIL as _);
    pub const FP64_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP64_UTIL as _);
    pub const FP32_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP32_UTIL as _);
    pub const FP16_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP16_UTIL as _);
    pub const PCIE_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PCIE_TX_PER_SEC as _);
    pub const PCIE_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PCIE_RX_PER_SEC as _);
    pub const NVLINK_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC as _);
    pub const NVLINK_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC as _);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuInstanceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuPlacementId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuVersion {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuVersionRange {
    pub supported: VgpuVersion,
    pub current: VgpuVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuGuestInfoState {
    Uninitialized =
        sys::nvmlVgpuGuestInfoState_t::NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED as _,
    Initialized =
        sys::nvmlVgpuGuestInfoState_t::NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED as _,
}

impl_enum_conversion!(u32, sys::nvmlVgpuGuestInfoState_t, VgpuGuestInfoState);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VgpuVmCompatibility: u32 {
        const NONE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_NONE as u32;
        const COLD = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_COLD as u32;
        const HIBERNATE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_HIBERNATE as u32;
        const SLEEP = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_SLEEP as u32;
        const LIVE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_LIVE as u32;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VgpuPgpuCompatibilityLimitCode: u32 {
        const NONE = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_NONE as u32;
        const HOST_DRIVER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER as u32;
        const GUEST_DRIVER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER as u32;
        const GPU = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_GPU as u32;
        const OTHER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_OTHER as u32;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuCompatibility {
    pub vm_compatibility: VgpuVmCompatibility,
    pub limit_code: VgpuPgpuCompatibilityLimitCode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VgpuMetadata {
    pub version: u32,
    pub revision: u32,
    pub guest_info_state: VgpuGuestInfoState,
    pub guest_driver_version: String,
    pub host_driver_version: String,
    pub virtualization_caps: u32,
    pub guest_vgpu_version: u32,
    pub opaque_data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PgpuMetadata {
    pub version: u32,
    pub revision: u32,
    pub host_driver_version: String,
    pub virtualization_caps: u32,
    pub host_supported_vgpu_range: VgpuVersion,
    pub opaque_data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuPlacementMode {
    Heterogeneous = sys::NVML_VGPU_PGPU_HETEROGENEOUS_MODE,
    Homogeneous = sys::NVML_VGPU_PGPU_HOMOGENEOUS_MODE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeDeviceId {
    pub device_id: u64,
    pub subsystem_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeResolution {
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeBar1Info {
    pub bar1_size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldQuery {
    pub field: FieldId,
    pub scope_id: u32,
}

impl FieldQuery {
    pub const fn new(field: FieldId) -> Self {
        Self { field, scope_id: 0 }
    }

    pub const fn with_scope(field: FieldId, scope_id: u32) -> Self {
        Self { field, scope_id }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EnableState {
    Disabled = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED as _,
    Enabled = sys::nvmlEnableState_t::NVML_FEATURE_ENABLED as _,
}

impl_enum_conversion!(u32, sys::nvmlEnableState_t, EnableState);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Brand {
    Unknown,
    Quadro,
    Tesla,
    Nvs,
    Grid,
    GeForce,
    Titan,
    NvidiaVApps,
    NvidiaVpc,
    NvidiaVcs,
    NvidiaVws,
    NvidiaCloudGaming,
    QuadroRtx,
    NvidiaRtx,
    Nvidia,
    GeForceRtx,
    TitanRtx,
    Other(u32),
}

impl Brand {
    pub const fn from_raw(value: u32) -> Self {
        match value {
            value if value == sys::nvmlBrandType_t::NVML_BRAND_UNKNOWN as u32 => Self::Unknown,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_QUADRO as u32 => Self::Quadro,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_TESLA as u32 => Self::Tesla,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVS as u32 => Self::Nvs,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_GRID as u32 => Self::Grid,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_GEFORCE as u32 => Self::GeForce,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_TITAN as u32 => Self::Titan,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VAPPS as u32 => {
                Self::NvidiaVApps
            }
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VPC as u32 => Self::NvidiaVpc,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VCS as u32 => Self::NvidiaVcs,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VWS as u32 => Self::NvidiaVws,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_CLOUD_GAMING as u32 => {
                Self::NvidiaCloudGaming
            }
            value if value == sys::nvmlBrandType_t::NVML_BRAND_QUADRO_RTX as u32 => Self::QuadroRtx,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_RTX as u32 => Self::NvidiaRtx,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA as u32 => Self::Nvidia,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_GEFORCE_RTX as u32 => {
                Self::GeForceRtx
            }
            value if value == sys::nvmlBrandType_t::NVML_BRAND_TITAN_RTX as u32 => Self::TitanRtx,
            raw => Self::Other(raw),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TemperatureSensor {
    Gpu = sys::nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU as _,
}

impl_enum_conversion!(u32, sys::nvmlTemperatureSensors_t, TemperatureSensor);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TemperatureThreshold {
    Shutdown = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_SHUTDOWN as _,
    Slowdown = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_SLOWDOWN as _,
    MemoryMax = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_MEM_MAX as _,
    GpuMax = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_GPU_MAX as _,
    AcousticMin = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN as _,
    AcousticCurrent =
        sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR as _,
    AcousticMax = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX as _,
    GpuCurrent = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_GPS_CURR as _,
}

impl_enum_conversion!(u32, sys::nvmlTemperatureThresholds_t, TemperatureThreshold);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeMode {
    Default = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_DEFAULT as _,
    ExclusiveThread = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_EXCLUSIVE_THREAD as _,
    Prohibited = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_PROHIBITED as _,
    ExclusiveProcess = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_EXCLUSIVE_PROCESS as _,
}

impl_enum_conversion!(u32, sys::nvmlComputeMode_t, ComputeMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Architecture {
    Kepler,
    Maxwell,
    Pascal,
    Volta,
    Turing,
    Ampere,
    Ada,
    Hopper,
    Blackwell,
    Other(u32),
}

impl Architecture {
    pub const fn from_raw(value: u32) -> Self {
        match value {
            sys::NVML_DEVICE_ARCH_KEPLER => Self::Kepler,
            sys::NVML_DEVICE_ARCH_MAXWELL => Self::Maxwell,
            sys::NVML_DEVICE_ARCH_PASCAL => Self::Pascal,
            sys::NVML_DEVICE_ARCH_VOLTA => Self::Volta,
            sys::NVML_DEVICE_ARCH_TURING => Self::Turing,
            sys::NVML_DEVICE_ARCH_AMPERE => Self::Ampere,
            sys::NVML_DEVICE_ARCH_ADA => Self::Ada,
            sys::NVML_DEVICE_ARCH_HOPPER => Self::Hopper,
            sys::NVML_DEVICE_ARCH_BLACKWELL => Self::Blackwell,
            raw => Self::Other(raw),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ClockType {
    Graphics = sys::nvmlClockType_t::NVML_CLOCK_GRAPHICS as _,
    Sm = sys::nvmlClockType_t::NVML_CLOCK_SM as _,
    Memory = sys::nvmlClockType_t::NVML_CLOCK_MEM as _,
    Video = sys::nvmlClockType_t::NVML_CLOCK_VIDEO as _,
}

impl_enum_conversion!(u32, sys::nvmlClockType_t, ClockType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkUtilizationCountUnits {
    Cycles = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_CYCLES as _,
    Packets = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_PACKETS as _,
    Bytes = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_BYTES as _,
    Reserved = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_RESERVED as _,
    Count = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_COUNT as _,
}

impl_enum_conversion!(
    u32,
    sys::nvmlNvLinkUtilizationCountUnits_t,
    NvLinkUtilizationCountUnits
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkCapability {
    P2pSupported = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_P2P_SUPPORTED as _,
    SysmemAccess = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_SYSMEM_ACCESS as _,
    P2pAtomics = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_P2P_ATOMICS as _,
    SysmemAtomics = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_SYSMEM_ATOMICS as _,
    SliBridge = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_SLI_BRIDGE as _,
    Valid = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_VALID as _,
    Count = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlNvLinkCapability_t, NvLinkCapability);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkErrorCounter {
    DlReplay = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_REPLAY as _,
    DlRecovery = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_RECOVERY as _,
    DlCrcFlit = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_CRC_FLIT as _,
    DlCrcData = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_CRC_DATA as _,
    DlEccData = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_ECC_DATA as _,
    Count = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlNvLinkErrorCounter_t, NvLinkErrorCounter);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkRemoteDeviceType {
    Gpu = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_GPU as _,
    IbmNpu = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_IBMNPU as _,
    Switch = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_SWITCH as _,
    Unknown = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlIntNvLinkDeviceType_t, NvLinkRemoteDeviceType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkVersion {
    Invalid = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_INVALID as _,
    V1_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_1_0 as _,
    V2_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_2_0 as _,
    V2_2 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_2_2 as _,
    V3_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_3_0 as _,
    V3_1 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_3_1 as _,
    V4_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_4_0 as _,
    V5_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_5_0 as _,
}

impl_enum_conversion!(u32, sys::nvmlNvlinkVersion_t, NvLinkVersion);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PowerProfileType {
    MaxP = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX_P as _,
    MaxQ = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX_Q as _,
    Compute = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_COMPUTE as _,
    MemoryBound = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MEMORY_BOUND as _,
    Network = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_NETWORK as _,
    Balanced = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_BALANCED as _,
    LlmInference = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_LLM_INFERENCE as _,
    LlmTraining = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_LLM_TRAINING as _,
    Rbm = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_RBM as _,
    Dcpcie = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_DCPCIE as _,
    HmmaSparse = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_HMMA_SPARSE as _,
    HmmaDense = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_HMMA_DENSE as _,
    SyncBalanced = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_SYNC_BALANCED as _,
    Hpc = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_HPC as _,
    Mig = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MIG as _,
    Max = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX as _,
}

impl_enum_conversion!(u32, sys::nvmlPowerProfileType_t, PowerProfileType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PowerMizerMode {
    Adaptive = sys::NVML_POWER_MIZER_MODE_ADAPTIVE,
    PreferMaximumPerformance = sys::NVML_POWER_MIZER_MODE_PREFER_MAXIMUM_PERFORMANCE,
    Auto = sys::NVML_POWER_MIZER_MODE_AUTO,
    PreferConsistentPerformance = sys::NVML_POWER_MIZER_MODE_PREFER_CONSISTENT_PERFORMANCE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PowerMizerModes {
    pub current_mode: u32,
    pub mode: u32,
    pub supported_modes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadPowerProfileInfo {
    pub profile_id: u32,
    pub priority: u32,
    pub conflicting_profiles: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadPowerProfilesInfo {
    pub supported_profiles: Vec<u32>,
    pub profiles: Vec<WorkloadPowerProfileInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadPowerCurrentProfiles {
    pub supported_profiles: Vec<u32>,
    pub requested_profiles: Vec<u32>,
    pub enforced_profiles: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DramEncryptionInfo {
    pub encryption_state: EnableState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeSystemCaps {
    pub cpu_caps: u32,
    pub gpus_caps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeSystemState {
    pub environment: u32,
    pub cc_feature: u32,
    pub dev_tools_mode: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeSystemSettings {
    pub environment: u32,
    pub cc_feature: u32,
    pub dev_tools_mode: u32,
    pub multi_gpu_mode: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeMemSizeInfo {
    pub protected_mem_size_kib: u64,
    pub unprotected_mem_size_kib: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfComputeGpuCertificate {
    pub cert_chain: Vec<u8>,
    pub attestation_cert_chain: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfComputeGpuAttestationReport {
    pub cec_attestation_report_present: bool,
    pub nonce: [u8; 32],
    pub attestation_report: Vec<u8>,
    pub cec_attestation_report: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeKeyRotationThreshold {
    pub attacker_advantage: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ClockId {
    Current = sys::nvmlClockId_t::NVML_CLOCK_ID_CURRENT as _,
    #[deprecated]
    AppClockTarget = sys::nvmlClockId_t::NVML_CLOCK_ID_APP_CLOCK_TARGET as _,
    #[deprecated]
    AppClockDefault = sys::nvmlClockId_t::NVML_CLOCK_ID_APP_CLOCK_DEFAULT as _,
    CustomerBoostMax = sys::nvmlClockId_t::NVML_CLOCK_ID_CUSTOMER_BOOST_MAX as _,
}

impl_enum_conversion!(u32, sys::nvmlClockId_t, ClockId);

/// Represents the thermal sensor targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ThermalTarget {
    None = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_NONE as _,
    /// GPU core temperature requires NvPhysicalGpuHandle.
    Gpu = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_GPU as _,
    /// GPU memory temperature requires NvPhysicalGpuHandle.
    Memory = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_MEMORY as _,
    /// GPU power supply temperature requires NvPhysicalGpuHandle.
    PowerSupply = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_POWER_SUPPLY as _,
    /// GPU board ambient temperature requires NvPhysicalGpuHandle.
    Board = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_BOARD as _,
    /// Visual Computing Device Board temperature requires NvVisualComputingDeviceHandle.
    VcdBoard = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_VCD_BOARD as _,
    /// Visual Computing Device Inlet temperature requires NvVisualComputingDeviceHandle.
    VcdInlet = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_VCD_INLET as _,
    /// Visual Computing Device Outlet temperature requires NvVisualComputingDeviceHandle.
    VcdOutlet = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_VCD_OUTLET as _,
    All = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_ALL as _,
    Unknown = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_UNKNOWN as _,
}

impl_enum_conversion!(i32, sys::nvmlThermalTarget_t, ThermalTarget);

/// Represents the thermal sensor controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ThermalController {
    None = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_NONE as _,
    GpuInternal = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_GPU_INTERNAL as _,
    Adm1032 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADM1032 as _,
    Adt7461 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADT7461 as _,
    Max6649 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_MAX6649 as _,
    Max1617 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_MAX1617 as _,
    Lm99 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_LM99 as _,
    Lm89 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_LM89 as _,
    Lm64 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_LM64 as _,
    G781 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_G781 as _,
    Adt7473 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADT7473 as _,
    Sbmax6649 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_SBMAX6649 as _,
    VbiosEvt = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_VBIOSEVT as _,
    Os = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_OS as _,
    NvsysconCanoas = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS as _,
    NvsysconE551 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_NVSYSCON_E551 as _,
    Max6649r = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_MAX6649R as _,
    Adt7473s = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADT7473S as _,
    Unknown = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_UNKNOWN as _,
}

impl_enum_conversion!(i32, sys::nvmlThermalController_t, ThermalController);

/// Enum to represent device addressing mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DeviceAddressingMode {
    /// No active mode.
    None = sys::nvmlDeviceAddressingModeType_t::NVML_DEVICE_ADDRESSING_MODE_NONE as _,
    /// Heterogeneous Memory Management mode.
    Hmm = sys::nvmlDeviceAddressingModeType_t::NVML_DEVICE_ADDRESSING_MODE_HMM as _,
    /// Address Translation Services mode.
    Ats = sys::nvmlDeviceAddressingModeType_t::NVML_DEVICE_ADDRESSING_MODE_ATS as _,
}

impl_enum_conversion!(
    u32,
    sys::nvmlDeviceAddressingModeType_t,
    DeviceAddressingMode
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BridgeChipType {
    Plx = sys::nvmlBridgeChipType_t::NVML_BRIDGE_CHIP_PLX as _,
    Bro4 = sys::nvmlBridgeChipType_t::NVML_BRIDGE_CHIP_BRO4 as _,
}

impl_enum_conversion!(u32, sys::nvmlBridgeChipType_t, BridgeChipType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PerformanceState {
    P0 = sys::nvmlPstates_t::NVML_PSTATE_0 as _,
    P1 = sys::nvmlPstates_t::NVML_PSTATE_1 as _,
    P2 = sys::nvmlPstates_t::NVML_PSTATE_2 as _,
    P3 = sys::nvmlPstates_t::NVML_PSTATE_3 as _,
    P4 = sys::nvmlPstates_t::NVML_PSTATE_4 as _,
    P5 = sys::nvmlPstates_t::NVML_PSTATE_5 as _,
    P6 = sys::nvmlPstates_t::NVML_PSTATE_6 as _,
    P7 = sys::nvmlPstates_t::NVML_PSTATE_7 as _,
    P8 = sys::nvmlPstates_t::NVML_PSTATE_8 as _,
    P9 = sys::nvmlPstates_t::NVML_PSTATE_9 as _,
    P10 = sys::nvmlPstates_t::NVML_PSTATE_10 as _,
    P11 = sys::nvmlPstates_t::NVML_PSTATE_11 as _,
    P12 = sys::nvmlPstates_t::NVML_PSTATE_12 as _,
    P13 = sys::nvmlPstates_t::NVML_PSTATE_13 as _,
    P14 = sys::nvmlPstates_t::NVML_PSTATE_14 as _,
    P15 = sys::nvmlPstates_t::NVML_PSTATE_15 as _,
    Unknown = sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlPstates_t, PerformanceState);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PcieUtilCounter {
    TxBytes = sys::nvmlPcieUtilCounter_t::NVML_PCIE_UTIL_TX_BYTES as _,
    RxBytes = sys::nvmlPcieUtilCounter_t::NVML_PCIE_UTIL_RX_BYTES as _,
}

impl_enum_conversion!(u32, sys::nvmlPcieUtilCounter_t, PcieUtilCounter);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TopologyLevel {
    Internal = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_INTERNAL as _,
    Single = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_SINGLE as _,
    Multiple = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_MULTIPLE as _,
    HostBridge = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_HOSTBRIDGE as _,
    Node = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_NODE as _,
    System = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_SYSTEM as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuTopologyLevel_t, TopologyLevel);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum P2pCapabilityIndex {
    Read = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_READ as _,
    Write = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_WRITE as _,
    Nvlink = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_NVLINK as _,
    Atomics = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_ATOMICS as _,
    Prop = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_PROP as _,
    Unknown = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuP2PCapsIndex_t, P2pCapabilityIndex);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum P2pStatus {
    Ok = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_OK as _,
    ChipsetNotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED as _,
    GpuNotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_GPU_NOT_SUPPORTED as _,
    IoHubNotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED as _,
    DisabledByRegKey = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_DISABLED_BY_REGKEY as _,
    NotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_NOT_SUPPORTED as _,
    Unknown = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuP2PStatus_t, P2pStatus);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum InforomObject {
    Oem = sys::nvmlInforomObject_t::NVML_INFOROM_OEM as _,
    Ecc = sys::nvmlInforomObject_t::NVML_INFOROM_ECC as _,
    Power = sys::nvmlInforomObject_t::NVML_INFOROM_POWER as _,
    Den = sys::nvmlInforomObject_t::NVML_INFOROM_DEN as _,
}

impl_enum_conversion!(u32, sys::nvmlInforomObject_t, InforomObject);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryErrorType {
    Corrected = sys::nvmlMemoryErrorType_t::NVML_MEMORY_ERROR_TYPE_CORRECTED as _,
    Uncorrected = sys::nvmlMemoryErrorType_t::NVML_MEMORY_ERROR_TYPE_UNCORRECTED as _,
}

impl_enum_conversion!(u32, sys::nvmlMemoryErrorType_t, MemoryErrorType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EccCounterType {
    Volatile = sys::nvmlEccCounterType_t::NVML_VOLATILE_ECC as _,
    Aggregate = sys::nvmlEccCounterType_t::NVML_AGGREGATE_ECC as _,
}

impl_enum_conversion!(u32, sys::nvmlEccCounterType_t, EccCounterType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryLocation {
    L1Cache = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_L1_CACHE as _,
    L2Cache = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_L2_CACHE as _,
    Dram = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_DRAM as _,
    RegisterFile = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_REGISTER_FILE as _,
    TextureMemory = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_TEXTURE_MEMORY as _,
    TextureShm = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_TEXTURE_SHM as _,
    Cbu = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_CBU as _,
    Sram = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_SRAM as _,
}

impl_enum_conversion!(u32, sys::nvmlMemoryLocation_t, MemoryLocation);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PerfPolicyType {
    Power = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_POWER as _,
    Thermal = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_THERMAL as _,
    SyncBoost = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_SYNC_BOOST as _,
    BoardLimit = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_BOARD_LIMIT as _,
    LowUtilization = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_LOW_UTILIZATION as _,
    Reliability = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_RELIABILITY as _,
    TotalAppClocks = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_TOTAL_APP_CLOCKS as _,
    TotalBaseClocks = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_TOTAL_BASE_CLOCKS as _,
}

impl_enum_conversion!(u32, sys::nvmlPerfPolicyType_t, PerfPolicyType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PageRetirementCause {
    MultipleSingleBitEccErrors =
        sys::nvmlPageRetirementCause_t::NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS
            as _,
    DoubleBitEccError =
        sys::nvmlPageRetirementCause_t::NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR as _,
}

impl_enum_conversion!(u32, sys::nvmlPageRetirementCause_t, PageRetirementCause);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FieldValueType {
    Double = sys::nvmlValueType_t::NVML_VALUE_TYPE_DOUBLE as _,
    UnsignedInt = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_INT as _,
    UnsignedLong = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_LONG as _,
    UnsignedLongLong = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_LONG_LONG as _,
    SignedLongLong = sys::nvmlValueType_t::NVML_VALUE_TYPE_SIGNED_LONG_LONG as _,
    SignedInt = sys::nvmlValueType_t::NVML_VALUE_TYPE_SIGNED_INT as _,
    UnsignedShort = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_SHORT as _,
}

impl_enum_conversion!(u32, sys::nvmlValueType_t, FieldValueType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SamplingType {
    TotalPower = sys::nvmlSamplingType_t::NVML_TOTAL_POWER_SAMPLES as _,
    GpuUtilization = sys::nvmlSamplingType_t::NVML_GPU_UTILIZATION_SAMPLES as _,
    MemoryUtilization = sys::nvmlSamplingType_t::NVML_MEMORY_UTILIZATION_SAMPLES as _,
    EncoderUtilization = sys::nvmlSamplingType_t::NVML_ENC_UTILIZATION_SAMPLES as _,
    DecoderUtilization = sys::nvmlSamplingType_t::NVML_DEC_UTILIZATION_SAMPLES as _,
    ProcessorClock = sys::nvmlSamplingType_t::NVML_PROCESSOR_CLK_SAMPLES as _,
    MemoryClock = sys::nvmlSamplingType_t::NVML_MEMORY_CLK_SAMPLES as _,
    ModulePower = sys::nvmlSamplingType_t::NVML_MODULE_POWER_SAMPLES as _,
    JpgUtilization = sys::nvmlSamplingType_t::NVML_JPG_UTILIZATION_SAMPLES as _,
    OfaUtilization = sys::nvmlSamplingType_t::NVML_OFA_UTILIZATION_SAMPLES as _,
}

impl_enum_conversion!(u32, sys::nvmlSamplingType_t, SamplingType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ProcessMode {
    Compute = 0,
    Graphics = 1,
    MpsCompute = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AffinityScope {
    Node = sys::NVML_AFFINITY_SCOPE_NODE,
    Socket = sys::NVML_AFFINITY_SCOPE_SOCKET,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BusType {
    Unknown,
    Pci,
    Pcie,
    Fpci,
    Agp,
    Other(u32),
}

impl BusType {
    pub const fn from_raw(value: u32) -> Self {
        match value {
            sys::NVML_BUS_TYPE_UNKNOWN => Self::Unknown,
            sys::NVML_BUS_TYPE_PCI => Self::Pci,
            sys::NVML_BUS_TYPE_PCIE => Self::Pcie,
            sys::NVML_BUS_TYPE_FPCI => Self::Fpci,
            sys::NVML_BUS_TYPE_AGP => Self::Agp,
            raw => Self::Other(raw),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PowerSource {
    Ac,
    Battery,
    Undersized,
    Other(u32),
}

impl PowerSource {
    pub const fn from_raw(value: u32) -> Self {
        match value {
            sys::NVML_POWER_SOURCE_AC => Self::Ac,
            sys::NVML_POWER_SOURCE_BATTERY => Self::Battery,
            sys::NVML_POWER_SOURCE_UNDERSIZED => Self::Undersized,
            raw => Self::Other(raw),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EncoderType {
    H264 = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_H264 as _,
    Hevc = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_HEVC as _,
    Av1 = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_AV1 as _,
    Unknown = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlEncoderType_t, EncoderType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum RestrictedApi {
    #[deprecated]
    SetApplicationsClocks =
        sys::nvmlRestrictedAPI_t::NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS as _,
    SetAutoBoostedClocks =
        sys::nvmlRestrictedAPI_t::NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS as _,
    Count = sys::nvmlRestrictedAPI_t::NVML_RESTRICTED_API_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlRestrictedAPI_t, RestrictedApi);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AdaptiveClockInfoStatus {
    Disabled = sys::NVML_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED,
    Enabled = sys::NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FanPolicy {
    TemperatureContinuousSw = sys::NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW,
    Manual = sys::NVML_FAN_POLICY_MANUAL,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FanState {
    Normal = sys::nvmlFanState_t::NVML_FAN_NORMAL as _,
    Failed = sys::nvmlFanState_t::NVML_FAN_FAILED as _,
}

impl_enum_conversion!(u32, sys::nvmlFanState_t, FanState);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CoolerControl {
    None = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_NONE as _,
    Toggle = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_TOGGLE as _,
    Variable = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_VARIABLE as _,
    Count = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlCoolerControl_t, CoolerControl);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CoolerTarget {
    None = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_NONE as _,
    Gpu = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_GPU as _,
    Memory = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_MEMORY as _,
    PowerSupply = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_POWER_SUPPLY as _,
    GpuRelated = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_GPU_RELATED as _,
}

impl_enum_conversion!(u32, sys::nvmlCoolerTarget_t, CoolerTarget);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FbcSessionType {
    Unknown = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_UNKNOWN as _,
    ToSys = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_TOSYS as _,
    Cuda = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_CUDA as _,
    Vid = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_VID as _,
    HwEnc = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_HWENC as _,
}

impl_enum_conversion!(u32, sys::nvmlFBCSessionType_t, FbcSessionType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DriverModel {
    Wddm = sys::nvmlDriverModel_t::NVML_DRIVER_WDDM as _,
    #[deprecated]
    Wdm = sys::nvmlDriverModel_t::NVML_DRIVER_WDM as _,
    Mcdm = sys::nvmlDriverModel_t::NVML_DRIVER_MCDM as _,
}

impl_enum_conversion!(u32, sys::nvmlDriverModel_t, DriverModel);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuOperationMode {
    AllOn = sys::nvmlGpuOperationMode_t::NVML_GOM_ALL_ON as _,
    Compute = sys::nvmlGpuOperationMode_t::NVML_GOM_COMPUTE as _,
    LowDp = sys::nvmlGpuOperationMode_t::NVML_GOM_LOW_DP as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuOperationMode_t, GpuOperationMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MigMode {
    Disabled = sys::NVML_DEVICE_MIG_DISABLE,
    Enabled = sys::NVML_DEVICE_MIG_ENABLE,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigModeActivation {
    pub status: Result<()>,
}

impl MigModeActivation {
    pub fn from_raw(status: sys::nvmlReturn_t) -> Self {
        Self {
            status: if status == sys::nvmlReturn_t::NVML_SUCCESS {
                Ok(())
            } else {
                Err(status.into())
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VirtualizationMode {
    None,
    Passthrough,
    Vgpu,
    HostVgpu,
    HostVsga,
    Other(u32),
}

impl VirtualizationMode {
    pub const fn from_raw(value: u32) -> Self {
        match value {
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_NONE
                        as u32 =>
            {
                Self::None
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH
                        as u32 =>
            {
                Self::Passthrough
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_VGPU
                        as u32 =>
            {
                Self::Vgpu
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU
                        as u32 =>
            {
                Self::HostVgpu
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA
                        as u32 =>
            {
                Self::HostVsga
            }
            raw => Self::Other(raw),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum HostVgpuMode {
    NonSriov = sys::nvmlHostVgpuMode_t::NVML_HOST_VGPU_MODE_NON_SRIOV as _,
    Sriov = sys::nvmlHostVgpuMode_t::NVML_HOST_VGPU_MODE_SRIOV as _,
}

impl_enum_conversion!(u32, sys::nvmlHostVgpuMode_t, HostVgpuMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuDriverCapability {
    HeterogeneousMultiVgpu =
        sys::nvmlVgpuDriverCapability_t::NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU as _,
    WarmUpdate = sys::nvmlVgpuDriverCapability_t::NVML_VGPU_DRIVER_CAP_WARM_UPDATE as _,
}

impl_enum_conversion!(u32, sys::nvmlVgpuDriverCapability_t, VgpuDriverCapability);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuVmIdType {
    DomainId = sys::nvmlVgpuVmIdType_t::NVML_VGPU_VM_ID_DOMAIN_ID as _,
    Uuid = sys::nvmlVgpuVmIdType_t::NVML_VGPU_VM_ID_UUID as _,
}

impl_enum_conversion!(u32, sys::nvmlVgpuVmIdType_t, VgpuVmIdType);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VgpuVmId {
    pub value: String,
    pub kind: VgpuVmIdType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuLicenseExpiry {
    pub year: u32,
    pub month: u16,
    pub day: u16,
    pub hour: u16,
    pub minute: u16,
    pub second: u16,
    pub status: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuLicenseInfo {
    pub is_licensed: bool,
    pub expiry: VgpuLicenseExpiry,
    pub current_state: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuRuntimeState {
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuSchedulerPolicy {
    BestEffort = sys::NVML_VGPU_SCHEDULER_POLICY_BEST_EFFORT,
    EqualShare = sys::NVML_VGPU_SCHEDULER_POLICY_EQUAL_SHARE,
    FixedShare = sys::NVML_VGPU_SCHEDULER_POLICY_FIXED_SHARE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuSchedulerArrMode {
    Default = sys::NVML_VGPU_SCHEDULER_ARR_DEFAULT,
    Disabled = sys::NVML_VGPU_SCHEDULER_ARR_DISABLE,
    Enabled = sys::NVML_VGPU_SCHEDULER_ARR_ENABLE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuSchedulerEngine {
    Graphics = sys::NVML_VGPU_SCHEDULER_ENGINE_TYPE_GRAPHICS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerParams {
    pub timeslice: u32,
    pub avg_factor: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerState {
    pub engine: VgpuSchedulerEngine,
    pub policy: VgpuSchedulerPolicy,
    pub arr_mode: VgpuSchedulerArrMode,
    pub params: VgpuSchedulerParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerLogEntry {
    pub timestamp: u64,
    pub time_run_total: u64,
    pub time_run: u64,
    pub sw_runlist_id: u32,
    pub target_time_slice: u64,
    pub cumulative_preemption_time: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerLog {
    pub state: VgpuSchedulerState,
    pub entries: Vec<VgpuSchedulerLogEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceVgpuCapability(sys::nvmlDeviceVgpuCapability_t);

impl From<DeviceVgpuCapability> for sys::nvmlDeviceVgpuCapability_t {
    fn from(value: DeviceVgpuCapability) -> Self {
        value.0
    }
}

impl DeviceVgpuCapability {
    pub const DEVICE_STREAMING: Self =
        Self(sys::nvmlDeviceVgpuCapability_t::NVML_DEVICE_VGPU_CAP_DEVICE_STREAMING);
    pub const WARM_UPDATE: Self =
        Self(sys::nvmlDeviceVgpuCapability_t::NVML_DEVICE_VGPU_CAP_WARM_UPDATE);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeCapability(sys::nvmlVgpuCapability_t);

impl From<VgpuTypeCapability> for sys::nvmlVgpuCapability_t {
    fn from(value: VgpuTypeCapability) -> Self {
        value.0
    }
}

impl VgpuTypeCapability {
    pub const NVLINK_P2P: Self = Self(sys::nvmlVgpuCapability_t::NVML_VGPU_CAP_NVLINK_P2P);
    pub const GPUDIRECT: Self = Self(sys::nvmlVgpuCapability_t::NVML_VGPU_CAP_GPUDIRECT);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeInstanceEngineProfile {
    Shared = sys::NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED,
    Count = sys::NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrentPending<T> {
    pub current: T,
    pub pending: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GspFirmwareMode {
    pub enabled: bool,
    pub default_mode: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AutoBoostClocks {
    pub enabled: EnableState,
    pub default_enabled: EnableState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    pub major: i32,
    pub minor: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Utilization {
    pub gpu: u32,
    pub memory: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryInfo {
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PciInfo {
    pub bus_id_legacy: String,
    pub bus_id: String,
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub pci_device_id: u32,
    pub pci_subsystem_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PciInfoExt {
    pub bus_id: String,
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub pci_device_id: u32,
    pub pci_subsystem_id: u32,
    pub base_class: u32,
    pub sub_class: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExcludedDeviceInfo {
    pub pci_info: PciInfo,
    pub uuid: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HwbcEntry {
    pub hwbc_id: u32,
    pub firmware_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpmSupport {
    pub is_supported_device: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpmMetric {
    pub metric_id: GpmMetricId,
    pub result: Result<f64>,
    pub short_name: String,
    pub long_name: String,
    pub unit: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum LedColor {
    Green = sys::nvmlLedColor_t::NVML_LED_COLOR_GREEN as _,
    Amber = sys::nvmlLedColor_t::NVML_LED_COLOR_AMBER as _,
}

impl_enum_conversion!(u32, sys::nvmlLedColor_t, LedColor);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LedState {
    pub cause: String,
    pub color: LedColor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnitInfo {
    pub name: String,
    pub id: String,
    pub serial: String,
    pub firmware_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PsuInfo {
    pub state: String,
    pub current: u32,
    pub voltage: u32,
    pub power: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitFanInfo {
    pub speed: u32,
    pub state: FanState,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnitFanSpeeds {
    pub fans: Vec<UnitFanInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum UnitTemperatureType {
    Intake = 0,
    Exhaust = 1,
    Board = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitTemperature {
    pub kind: UnitTemperatureType,
    pub temperature: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PowerLimits {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[deprecated]
pub struct EccErrorCounts {
    pub l1_cache: u64,
    pub l2_cache: u64,
    pub device_memory: u64,
    pub register_file: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViolationTime {
    pub reference_time: u64,
    pub violation_time: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetiredPage {
    pub address: u64,
    pub timestamp: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemappedRows {
    pub corrected: u32,
    pub uncorrected: u32,
    pub pending: bool,
    pub failure_occurred: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowRemapperHistogram {
    pub max: u32,
    pub high: u32,
    pub partial: u32,
    pub low: u32,
    pub none: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum FieldValue {
    Double(f64),
    UnsignedInt(u32),
    UnsignedLong(u64),
    UnsignedLongLong(u64),
    SignedLongLong(i64),
    SignedInt(i32),
    UnsignedShort(u16),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldSample {
    pub field: FieldId,
    pub scope_id: u32,
    pub timestamp: i64,
    pub latency_usec: i64,
    pub result: Result<FieldValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccountingStats {
    pub gpu_utilization: u32,
    pub memory_utilization: u32,
    pub max_memory_usage: u64,
    pub time: u64,
    pub start_time: u64,
    pub is_running: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub timestamp: u64,
    pub value: FieldValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    pub value_type: FieldValueType,
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessDetail {
    pub pid: Pid,
    pub used_gpu_memory: Option<u64>,
    pub gpu_instance_id: u32,
    pub compute_instance_id: u32,
    pub used_gpu_cc_protected_memory: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventData {
    pub device: Option<Device>,
    pub event_types: EventTypes,
    pub event_data: u64,
    pub gpu_instance_id: u32,
    pub compute_instance_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemEventData {
    pub event_types: SystemEventTypes,
    pub gpu_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bar1MemoryInfo {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UtilizationCounter {
    pub utilization: u32,
    pub sampling_period_us: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CurrentClockFreqs {
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PerformanceModes {
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockRange {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockRangeI32 {
    pub min: i32,
    pub max: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockOffset {
    pub kind: ClockType,
    pub pstate: PerformanceState,
    pub clock_offset_mhz: i32,
    pub min_clock_offset_mhz: i32,
    pub max_clock_offset_mhz: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MarginTemperature {
    pub value: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BbxFlushTime {
    pub timestamp: u64,
    pub duration_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceAttributes {
    pub multiprocessor_count: u32,
    pub shared_copy_engine_count: u32,
    pub shared_decoder_count: u32,
    pub shared_encoder_count: u32,
    pub shared_jpeg_count: u32,
    pub shared_ofa_count: u32,
    pub gpu_instance_slice_count: u32,
    pub compute_instance_slice_count: u32,
    pub memory_size_mb: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlatformInfo {
    pub ib_guid: [u8; 16],
    pub chassis_serial_number: [u8; 16],
    pub slot_number: u8,
    pub tray_index: u8,
    pub host_id: u8,
    pub peer_type: u8,
    pub module_id: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pdi {
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C2cModeInfo {
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClkMonFaultInfo {
    pub clock_api_domain: u32,
    pub clock_domain_fault_mask: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClkMonStatus {
    pub global_status: bool,
    pub faults: Vec<ClkMonFaultInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EccSramErrorStatus {
    pub aggregate_unc_parity: u64,
    pub aggregate_unc_secded: u64,
    pub aggregate_corrected: u64,
    pub volatile_unc_parity: u64,
    pub volatile_unc_secded: u64,
    pub volatile_corrected: u64,
    pub aggregate_unc_bucket_l2: u64,
    pub aggregate_unc_bucket_sm: u64,
    pub aggregate_unc_bucket_pcie: u64,
    pub aggregate_unc_bucket_mcu: u64,
    pub aggregate_unc_bucket_other: u64,
    pub threshold_exceeded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EccSramUniqueUncorrectedErrorEntry {
    pub unit: u32,
    pub location: u32,
    pub sublocation: u32,
    pub extlocation: u32,
    pub address: u32,
    pub is_parity: bool,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EccSramUniqueUncorrectedErrorCounts {
    pub entries: Vec<EccSramUniqueUncorrectedErrorEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThermalSensorSetting {
    pub controller: ThermalController,
    pub default_min_temp: i32,
    pub default_max_temp: i32,
    pub current_temp: i32,
    pub target: ThermalTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ThermalSettings {
    pub sensors: Vec<ThermalSensorSetting>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BridgeChipInfo {
    pub kind: BridgeChipType,
    pub firmware_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BridgeChipHierarchy {
    pub bridge_chips: Vec<BridgeChipInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepairStatus {
    pub channel_repair_pending: bool,
    pub tpc_repair_pending: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricState {
    NotSupported = sys::NVML_GPU_FABRIC_STATE_NOT_SUPPORTED,
    NotStarted = sys::NVML_GPU_FABRIC_STATE_NOT_STARTED,
    InProgress = sys::NVML_GPU_FABRIC_STATE_IN_PROGRESS,
    Completed = sys::NVML_GPU_FABRIC_STATE_COMPLETED,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricHealthSummary {
    NotSupported = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_NOT_SUPPORTED,
    Healthy = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_HEALTHY,
    Unhealthy = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_UNHEALTHY,
    LimitedCapacity = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_LIMITED_CAPACITY,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricBooleanState {
    NotSupported = sys::NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_NOT_SUPPORTED,
    True = sys::NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_TRUE,
    False = sys::NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricIncorrectConfiguration {
    NotSupported = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NOT_SUPPORTED,
    None = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NONE,
    IncorrectSysguid = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_SYSGUID,
    IncorrectChassisSerialNumber =
        sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_CHASSIS_SN,
    NoPartition = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NO_PARTITION,
    InsufficientNvlinks =
        sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INSUFFICIENT_NVLINKS,
    IncompatibleGpuFirmware =
        sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCOMPATIBLE_GPU_FW,
    InvalidLocation = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INVALID_LOCATION,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuFabricHealth {
    pub degraded_bandwidth: GpuFabricBooleanState,
    pub route_recovery: GpuFabricBooleanState,
    pub route_unhealthy: GpuFabricBooleanState,
    pub access_timeout_recovery: GpuFabricBooleanState,
    pub incorrect_configuration: GpuFabricIncorrectConfiguration,
    pub raw_mask: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuFabricInfo {
    pub cluster_uuid: [u8; sys::NVML_GPU_FABRIC_UUID_LEN as usize],
    pub status_code: u32,
    pub clique_id: u32,
    pub state: GpuFabricState,
    pub health: GpuFabricHealth,
    pub health_summary: GpuFabricHealthSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicPstateInfo {
    pub domain: u32,
    pub percentage: u32,
    pub increase_threshold: u32,
    pub decrease_threshold: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicPstatesInfo {
    pub flags: u32,
    pub utilization: Vec<DynamicPstateInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkUtilizationControl {
    pub units: NvLinkUtilizationCountUnits,
    pub packet_filter: NvLinkPacketTypes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkUtilizationCounter {
    pub rx: u64,
    pub tx: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkFirmwareVersion {
    pub ucode_type: u8,
    pub major: u32,
    pub minor: u32,
    pub sub_minor: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NvLinkInfo {
    pub nvle_enabled: bool,
    pub firmware_versions: Vec<NvLinkFirmwareVersion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkBwMode {
    pub is_best: bool,
    pub mode: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NvLinkSupportedBwModes {
    pub modes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderStats {
    pub session_count: u32,
    pub average_fps: u32,
    pub average_latency_us: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FanSpeedInfo {
    pub fan: u32,
    pub speed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MinMaxFanSpeed {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemperatureInfo {
    pub sensor: TemperatureSensor,
    pub temperature: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoolerInfo {
    pub index: u32,
    pub signal_type: CoolerControl,
    pub target: CoolerTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuInstanceProfileInfo {
    pub id: u32,
    pub slice_count: u32,
    pub instance_count: u32,
    pub multiprocessor_count: u32,
    pub copy_engine_count: u32,
    pub decoder_count: u32,
    pub encoder_count: u32,
    pub jpeg_count: u32,
    pub ofa_count: u32,
    pub memory_size_mb: u64,
    pub name: String,
    pub capabilities: GpuInstanceProfileCapabilities,
}

/// Information about a GPU instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuInstanceInfo {
    /// Parent device for the GPU instance.
    pub device: Device,
    /// GPU instance ID.
    pub id: u32,
    /// GPU instance profile ID.
    pub profile_id: u32,
    /// Placement of the GPU instance within the parent device.
    pub placement: GpuInstancePlacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuInstancePlacement {
    pub start: u32,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputeInstanceProfileInfo {
    pub id: u32,
    pub slice_count: u32,
    pub instance_count: u32,
    pub multiprocessor_count: u32,
    pub shared_copy_engine_count: u32,
    pub shared_decoder_count: u32,
    pub shared_encoder_count: u32,
    pub shared_jpeg_count: u32,
    pub shared_ofa_count: u32,
    pub name: String,
    pub capabilities: ComputeInstanceProfileCapabilities,
}

/// Information about a compute instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputeInstanceInfo {
    /// Parent device for the compute instance.
    pub device: Device,
    /// Parent GPU instance.
    pub gpu_instance: GpuInstance,
    /// Compute instance ID.
    pub id: u32,
    /// Compute instance profile ID.
    pub profile_id: u32,
    /// Placement of the compute instance within the parent GPU instance.
    pub placement: ComputeInstancePlacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeInstancePlacement {
    pub start: u32,
    pub size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderSessionInfo {
    pub session_id: u32,
    pub pid: Pid,
    pub vgpu_instance: u32,
    pub codec_type: EncoderType,
    pub horizontal_resolution: u32,
    pub vertical_resolution: u32,
    pub average_fps: u32,
    pub average_latency_us: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FbcStats {
    pub session_count: u32,
    pub average_fps: u32,
    pub average_latency_us: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FbcSessionInfo {
    pub session_id: u32,
    pub pid: Pid,
    pub vgpu_instance: u32,
    pub display_ordinal: u32,
    pub session_type: FbcSessionType,
    pub session_flags: u32,
    pub max_horizontal_resolution: u32,
    pub max_vertical_resolution: u32,
    pub horizontal_resolution: u32,
    pub vertical_resolution: u32,
    pub average_fps: u32,
    pub average_latency_us: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessInfo {
    pub pid: Pid,
    pub used_gpu_memory: Option<u64>,
    pub gpu_instance_id: u32,
    pub compute_instance_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessUtilizationSample {
    pub pid: Pid,
    pub timestamp: u64,
    pub sm_utilization: u32,
    pub memory_utilization: u32,
    pub encoder_utilization: u32,
    pub decoder_utilization: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessUtilizationInfo {
    pub timestamp: u64,
    pub pid: Pid,
    pub sm_utilization: u32,
    pub memory_utilization: u32,
    pub encoder_utilization: u32,
    pub decoder_utilization: u32,
    pub jpg_utilization: u32,
    pub ofa_utilization: u32,
}

impl From<sys::nvmlUtilization_t> for Utilization {
    fn from(value: sys::nvmlUtilization_t) -> Self {
        Self {
            gpu: value.gpu,
            memory: value.memory,
        }
    }
}

impl From<sys::nvmlMemory_v2_t> for MemoryInfo {
    fn from(value: sys::nvmlMemory_v2_t) -> Self {
        Self {
            total: value.total,
            reserved: value.reserved,
            free: value.free,
            used: value.used,
        }
    }
}

impl From<sys::nvmlPciInfo_t> for PciInfo {
    fn from(value: sys::nvmlPciInfo_t) -> Self {
        Self {
            bus_id_legacy: string_from_c_chars(&value.busIdLegacy),
            bus_id: string_from_c_chars(&value.busId),
            domain: value.domain,
            bus: value.bus,
            device: value.device,
            pci_device_id: value.pciDeviceId,
            pci_subsystem_id: value.pciSubSystemId,
        }
    }
}

impl From<sys::nvmlPciInfoExt_t> for PciInfoExt {
    fn from(value: sys::nvmlPciInfoExt_t) -> Self {
        Self {
            bus_id: string_from_c_chars(&value.busId),
            domain: value.domain,
            bus: value.bus,
            device: value.device,
            pci_device_id: value.pciDeviceId,
            pci_subsystem_id: value.pciSubSystemId,
            base_class: value.baseClass,
            sub_class: value.subClass,
        }
    }
}

impl From<sys::nvmlExcludedDeviceInfo_t> for ExcludedDeviceInfo {
    fn from(value: sys::nvmlExcludedDeviceInfo_t) -> Self {
        Self {
            pci_info: value.pciInfo.into(),
            uuid: string_from_c_chars(&value.uuid),
        }
    }
}

impl From<sys::nvmlHwbcEntry_t> for HwbcEntry {
    fn from(value: sys::nvmlHwbcEntry_t) -> Self {
        Self {
            hwbc_id: value.hwbcId,
            firmware_version: string_from_c_chars(&value.firmwareVersion),
        }
    }
}

impl From<sys::nvmlGpuInstanceProfileInfo_v3_t> for GpuInstanceProfileInfo {
    fn from(value: sys::nvmlGpuInstanceProfileInfo_v3_t) -> Self {
        Self {
            id: value.id,
            slice_count: value.sliceCount,
            instance_count: value.instanceCount,
            multiprocessor_count: value.multiprocessorCount,
            copy_engine_count: value.copyEngineCount,
            decoder_count: value.decoderCount,
            encoder_count: value.encoderCount,
            jpeg_count: value.jpegCount,
            ofa_count: value.ofaCount,
            memory_size_mb: value.memorySizeMB,
            name: string_from_c_chars(&value.name),
            capabilities: GpuInstanceProfileCapabilities::from_bits_retain(value.capabilities),
        }
    }
}

impl From<sys::nvmlGpuInstanceInfo_t> for GpuInstanceInfo {
    fn from(value: sys::nvmlGpuInstanceInfo_t) -> Self {
        Self {
            device: unsafe { Device::from_raw(value.device) },
            id: value.id,
            profile_id: value.profileId,
            placement: value.placement.into(),
        }
    }
}

impl From<sys::nvmlGpuInstancePlacement_t> for GpuInstancePlacement {
    fn from(value: sys::nvmlGpuInstancePlacement_t) -> Self {
        Self {
            start: value.start,
            size: value.size,
        }
    }
}

impl From<GpuInstancePlacement> for sys::nvmlGpuInstancePlacement_t {
    fn from(value: GpuInstancePlacement) -> Self {
        Self {
            start: value.start,
            size: value.size,
        }
    }
}

impl From<sys::nvmlComputeInstanceProfileInfo_v3_t> for ComputeInstanceProfileInfo {
    fn from(value: sys::nvmlComputeInstanceProfileInfo_v3_t) -> Self {
        Self {
            id: value.id,
            slice_count: value.sliceCount,
            instance_count: value.instanceCount,
            multiprocessor_count: value.multiprocessorCount,
            shared_copy_engine_count: value.sharedCopyEngineCount,
            shared_decoder_count: value.sharedDecoderCount,
            shared_encoder_count: value.sharedEncoderCount,
            shared_jpeg_count: value.sharedJpegCount,
            shared_ofa_count: value.sharedOfaCount,
            name: string_from_c_chars(&value.name),
            capabilities: ComputeInstanceProfileCapabilities::from_bits_retain(value.capabilities),
        }
    }
}

impl From<sys::nvmlComputeInstanceInfo_t> for ComputeInstanceInfo {
    fn from(value: sys::nvmlComputeInstanceInfo_t) -> Self {
        Self {
            device: unsafe { Device::from_raw(value.device) },
            gpu_instance: unsafe { GpuInstance::from_raw(value.gpuInstance) },
            id: value.id,
            profile_id: value.profileId,
            placement: value.placement.into(),
        }
    }
}

impl From<sys::nvmlComputeInstancePlacement_t> for ComputeInstancePlacement {
    fn from(value: sys::nvmlComputeInstancePlacement_t) -> Self {
        Self {
            start: value.start,
            size: value.size,
        }
    }
}

impl From<ComputeInstancePlacement> for sys::nvmlComputeInstancePlacement_t {
    fn from(value: ComputeInstancePlacement) -> Self {
        Self {
            start: value.start,
            size: value.size,
        }
    }
}

impl From<sys::nvmlVgpuPlacementId_t> for VgpuPlacementId {
    fn from(value: sys::nvmlVgpuPlacementId_t) -> Self {
        Self(value.placementId)
    }
}

impl From<sys::nvmlVgpuVersion_t> for VgpuVersion {
    fn from(value: sys::nvmlVgpuVersion_t) -> Self {
        Self {
            min: value.minVersion,
            max: value.maxVersion,
        }
    }
}

impl From<VgpuVersion> for sys::nvmlVgpuVersion_t {
    fn from(value: VgpuVersion) -> Self {
        Self {
            minVersion: value.min,
            maxVersion: value.max,
        }
    }
}

impl From<sys::nvmlVgpuPgpuCompatibility_t> for VgpuCompatibility {
    fn from(value: sys::nvmlVgpuPgpuCompatibility_t) -> Self {
        Self {
            vm_compatibility: VgpuVmCompatibility::from_bits_retain(
                value.vgpuVmCompatibility as u32,
            ),
            limit_code: VgpuPgpuCompatibilityLimitCode::from_bits_retain(
                value.compatibilityLimitCode as u32,
            ),
        }
    }
}

impl VgpuMetadata {
    pub(crate) fn from_raw(raw: &sys::nvmlVgpuMetadata_t, opaque_data: Vec<u8>) -> Self {
        Self {
            version: raw.version,
            revision: raw.revision,
            guest_info_state: raw.guestInfoState.into(),
            guest_driver_version: string_from_c_chars(&raw.guestDriverVersion),
            host_driver_version: string_from_c_chars(&raw.hostDriverVersion),
            virtualization_caps: raw.vgpuVirtualizationCaps,
            guest_vgpu_version: raw.guestVgpuVersion,
            opaque_data,
        }
    }

    pub(crate) fn encode_raw(&self) -> Vec<u8> {
        let opaque_offset = mem::offset_of!(sys::nvmlVgpuMetadata_t, opaqueData);
        let len = max(
            size_of::<sys::nvmlVgpuMetadata_t>(),
            opaque_offset.saturating_add(self.opaque_data.len()),
        );
        let mut raw = vec![0u8; len];
        let metadata = unsafe { &mut *raw.as_mut_ptr().cast::<sys::nvmlVgpuMetadata_t>() };
        metadata.version = self.version;
        metadata.revision = self.revision;
        metadata.guestInfoState = self.guest_info_state.into();
        metadata.vgpuVirtualizationCaps = self.virtualization_caps;
        metadata.guestVgpuVersion = self.guest_vgpu_version;
        metadata.opaqueDataSize = self.opaque_data.len() as u32;

        copy_string_to_c_chars(&mut metadata.guestDriverVersion, &self.guest_driver_version);
        copy_string_to_c_chars(&mut metadata.hostDriverVersion, &self.host_driver_version);
        raw[opaque_offset..opaque_offset + self.opaque_data.len()]
            .copy_from_slice(&self.opaque_data);
        raw
    }
}

impl PgpuMetadata {
    pub(crate) fn from_raw(raw: &sys::nvmlVgpuPgpuMetadata_t, opaque_data: Vec<u8>) -> Self {
        Self {
            version: raw.version,
            revision: raw.revision,
            host_driver_version: string_from_c_chars(&raw.hostDriverVersion),
            virtualization_caps: raw.pgpuVirtualizationCaps,
            host_supported_vgpu_range: raw.hostSupportedVgpuRange.into(),
            opaque_data,
        }
    }

    pub(crate) fn encode_raw(&self) -> Vec<u8> {
        let opaque_offset = mem::offset_of!(sys::nvmlVgpuPgpuMetadata_t, opaqueData);
        let len = max(
            size_of::<sys::nvmlVgpuPgpuMetadata_t>(),
            opaque_offset.saturating_add(self.opaque_data.len()),
        );
        let mut raw = vec![0u8; len];
        let metadata = unsafe { &mut *raw.as_mut_ptr().cast::<sys::nvmlVgpuPgpuMetadata_t>() };
        metadata.version = self.version;
        metadata.revision = self.revision;
        metadata.pgpuVirtualizationCaps = self.virtualization_caps;
        metadata.hostSupportedVgpuRange = self.host_supported_vgpu_range.into();
        metadata.opaqueDataSize = self.opaque_data.len() as u32;

        copy_string_to_c_chars(&mut metadata.hostDriverVersion, &self.host_driver_version);
        raw[opaque_offset..opaque_offset + self.opaque_data.len()]
            .copy_from_slice(&self.opaque_data);
        raw
    }
}

impl From<sys::nvmlVgpuSchedulerLogEntry_t> for VgpuSchedulerLogEntry {
    fn from(value: sys::nvmlVgpuSchedulerLogEntry_t) -> Self {
        Self {
            timestamp: value.timestamp,
            time_run_total: value.timeRunTotal,
            time_run: value.timeRun,
            sw_runlist_id: value.swRunlistId,
            target_time_slice: value.targetTimeSlice,
            cumulative_preemption_time: value.cumulativePreemptionTime,
        }
    }
}

impl From<sys::nvmlVgpuSchedulerLogEntry_v2_t> for VgpuSchedulerLogEntry {
    fn from(value: sys::nvmlVgpuSchedulerLogEntry_v2_t) -> Self {
        Self {
            timestamp: value.timestamp,
            time_run_total: value.timeRunTotal,
            time_run: value.timeRun,
            sw_runlist_id: value.swRunlistId,
            target_time_slice: value.targetTimeSlice,
            cumulative_preemption_time: value.cumulativePreemptionTime,
        }
    }
}

impl From<sys::nvmlVgpuLicenseExpiry_t> for VgpuLicenseExpiry {
    fn from(value: sys::nvmlVgpuLicenseExpiry_t) -> Self {
        Self {
            year: value.year,
            month: value.month,
            day: value.day,
            hour: value.hour,
            minute: value.min,
            second: value.sec,
            status: value.status,
        }
    }
}

impl From<sys::nvmlVgpuLicenseInfo_t> for VgpuLicenseInfo {
    fn from(value: sys::nvmlVgpuLicenseInfo_t) -> Self {
        Self {
            is_licensed: value.isLicensed != 0,
            expiry: value.licenseExpiry.into(),
            current_state: value.currentState,
        }
    }
}

impl From<sys::nvmlVgpuRuntimeState_t> for VgpuRuntimeState {
    fn from(value: sys::nvmlVgpuRuntimeState_t) -> Self {
        Self { size: value.size }
    }
}

impl VgpuSchedulerState {
    pub(crate) fn from_raw(engine: sys::nvmlVgpuSchedulerStateInfo_v2_t) -> Result<Self> {
        let avg_factor = (engine.avgFactor != 0).then_some(engine.avgFactor);
        let arr_mode = if avg_factor.is_some() {
            VgpuSchedulerArrMode::Enabled
        } else {
            VgpuSchedulerArrMode::Default
        };

        Ok(Self {
            engine: try_from_nvml_enum("vgpu scheduler engine", engine.engineId)?,
            policy: try_from_nvml_enum("vgpu scheduler policy", engine.schedulerPolicy)?,
            arr_mode,
            params: VgpuSchedulerParams {
                timeslice: engine.timeslice,
                avg_factor,
            },
        })
    }
}

impl VgpuSchedulerLog {
    pub(crate) fn from_raw(info: sys::nvmlVgpuSchedulerLogInfo_v2_t) -> Result<Self> {
        let avg_factor = (info.avgFactor != 0).then_some(info.avgFactor);
        let arr_mode = if avg_factor.is_some() {
            VgpuSchedulerArrMode::Enabled
        } else {
            VgpuSchedulerArrMode::Default
        };

        let entries = info.logEntries[..info.entriesCount as usize]
            .iter()
            .copied()
            .map(Into::into)
            .collect();

        Ok(Self {
            state: VgpuSchedulerState {
                engine: try_from_nvml_enum("vgpu scheduler engine", info.engineId)?,
                policy: try_from_nvml_enum("vgpu scheduler policy", info.schedulerPolicy)?,
                arr_mode,
                params: VgpuSchedulerParams {
                    timeslice: info.timeslice,
                    avg_factor,
                },
            },
            entries,
        })
    }
}

impl From<sys::nvmlLedState_t> for LedState {
    fn from(value: sys::nvmlLedState_t) -> Self {
        Self {
            cause: string_from_c_chars(&value.cause),
            color: value.color.into(),
        }
    }
}

impl From<sys::nvmlUnitInfo_t> for UnitInfo {
    fn from(value: sys::nvmlUnitInfo_t) -> Self {
        Self {
            name: string_from_c_chars(&value.name),
            id: string_from_c_chars(&value.id),
            serial: string_from_c_chars(&value.serial),
            firmware_version: string_from_c_chars(&value.firmwareVersion),
        }
    }
}

impl From<sys::nvmlPSUInfo_t> for PsuInfo {
    fn from(value: sys::nvmlPSUInfo_t) -> Self {
        Self {
            state: string_from_c_chars(&value.state),
            current: value.current,
            voltage: value.voltage,
            power: value.power,
        }
    }
}

impl From<sys::nvmlGpmSupport_t> for GpmSupport {
    fn from(value: sys::nvmlGpmSupport_t) -> Self {
        Self {
            is_supported_device: value.isSupportedDevice != 0,
        }
    }
}

impl From<sys::nvmlGpmMetric_t> for GpmMetric {
    fn from(value: sys::nvmlGpmMetric_t) -> Self {
        let result = if value.nvmlReturn == sys::nvmlReturn_t::NVML_SUCCESS {
            Ok(value.value)
        } else {
            Err(value.nvmlReturn.into())
        };

        Self {
            metric_id: GpmMetricId(value.metricId),
            result,
            short_name: unsafe { string_from_c_ptr(value.metricInfo.shortName) },
            long_name: unsafe { string_from_c_ptr(value.metricInfo.longName) },
            unit: unsafe { string_from_c_ptr(value.metricInfo.unit) },
        }
    }
}

impl From<sys::nvmlBAR1Memory_t> for Bar1MemoryInfo {
    fn from(value: sys::nvmlBAR1Memory_t) -> Self {
        Self {
            total: value.bar1Total,
            free: value.bar1Free,
            used: value.bar1Used,
        }
    }
}

impl From<sys::nvmlEccErrorCounts_t> for EccErrorCounts {
    fn from(value: sys::nvmlEccErrorCounts_t) -> Self {
        Self {
            l1_cache: value.l1Cache,
            l2_cache: value.l2Cache,
            device_memory: value.deviceMemory,
            register_file: value.registerFile,
        }
    }
}

impl From<sys::nvmlViolationTime_t> for ViolationTime {
    fn from(value: sys::nvmlViolationTime_t) -> Self {
        Self {
            reference_time: value.referenceTime,
            violation_time: value.violationTime,
        }
    }
}

impl From<sys::nvmlRowRemapperHistogramValues_t> for RowRemapperHistogram {
    fn from(value: sys::nvmlRowRemapperHistogramValues_t) -> Self {
        Self {
            max: value.max,
            high: value.high,
            partial: value.partial,
            low: value.low,
            none: value.none,
        }
    }
}

impl From<sys::nvmlAccountingStats_t> for AccountingStats {
    fn from(value: sys::nvmlAccountingStats_t) -> Self {
        Self {
            gpu_utilization: value.gpuUtilization,
            memory_utilization: value.memoryUtilization,
            max_memory_usage: value.maxMemoryUsage,
            time: value.time,
            start_time: value.startTime,
            is_running: value.isRunning != 0,
        }
    }
}

impl From<sys::nvmlDeviceCurrentClockFreqs_t> for CurrentClockFreqs {
    fn from(value: sys::nvmlDeviceCurrentClockFreqs_t) -> Self {
        Self {
            value: string_from_c_chars(&value.str_),
        }
    }
}

impl From<sys::nvmlDevicePerfModes_t> for PerformanceModes {
    fn from(value: sys::nvmlDevicePerfModes_t) -> Self {
        Self {
            value: string_from_c_chars(&value.str_),
        }
    }
}

impl From<sys::nvmlMarginTemperature_t> for MarginTemperature {
    fn from(value: sys::nvmlMarginTemperature_t) -> Self {
        Self {
            value: value.marginTemperature,
        }
    }
}

impl From<sys::nvmlDeviceAttributes_t> for DeviceAttributes {
    fn from(value: sys::nvmlDeviceAttributes_t) -> Self {
        Self {
            multiprocessor_count: value.multiprocessorCount,
            shared_copy_engine_count: value.sharedCopyEngineCount,
            shared_decoder_count: value.sharedDecoderCount,
            shared_encoder_count: value.sharedEncoderCount,
            shared_jpeg_count: value.sharedJpegCount,
            shared_ofa_count: value.sharedOfaCount,
            gpu_instance_slice_count: value.gpuInstanceSliceCount,
            compute_instance_slice_count: value.computeInstanceSliceCount,
            memory_size_mb: value.memorySizeMB,
        }
    }
}

impl From<sys::nvmlPlatformInfo_t> for PlatformInfo {
    fn from(value: sys::nvmlPlatformInfo_t) -> Self {
        Self {
            ib_guid: value.ibGuid,
            chassis_serial_number: value.chassisSerialNumber,
            slot_number: value.slotNumber,
            tray_index: value.trayIndex,
            host_id: value.hostId,
            peer_type: value.peerType,
            module_id: value.moduleId,
        }
    }
}

impl From<sys::nvmlPdi_t> for Pdi {
    fn from(value: sys::nvmlPdi_t) -> Self {
        Self { value: value.value }
    }
}

impl From<sys::nvmlBridgeChipInfo_t> for BridgeChipInfo {
    fn from(value: sys::nvmlBridgeChipInfo_t) -> Self {
        Self {
            kind: value.type_.into(),
            firmware_version: value.fwVersion,
        }
    }
}

impl From<sys::nvmlBridgeChipHierarchy_t> for BridgeChipHierarchy {
    fn from(value: sys::nvmlBridgeChipHierarchy_t) -> Self {
        Self {
            bridge_chips: value.bridgeChipInfo[..value.bridgeCount as usize]
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<sys::nvmlRepairStatus_t> for RepairStatus {
    fn from(value: sys::nvmlRepairStatus_t) -> Self {
        Self {
            channel_repair_pending: value.bChannelRepairPending != 0,
            tpc_repair_pending: value.bTpcRepairPending != 0,
        }
    }
}

impl From<sys::nvmlC2cModeInfo_v1_t> for C2cModeInfo {
    fn from(value: sys::nvmlC2cModeInfo_v1_t) -> Self {
        Self {
            enabled: value.isC2cEnabled != 0,
        }
    }
}

impl From<sys::nvmlClkMonFaultInfo_t> for ClkMonFaultInfo {
    fn from(value: sys::nvmlClkMonFaultInfo_t) -> Self {
        Self {
            clock_api_domain: value.clkApiDomain,
            clock_domain_fault_mask: value.clkDomainFaultMask,
        }
    }
}

impl From<sys::nvmlClkMonStatus_t> for ClkMonStatus {
    fn from(value: sys::nvmlClkMonStatus_t) -> Self {
        Self {
            global_status: value.bGlobalStatus != 0,
            faults: value.clkMonList[..value.clkMonListSize as usize]
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<sys::nvmlEccSramErrorStatus_t> for EccSramErrorStatus {
    fn from(value: sys::nvmlEccSramErrorStatus_t) -> Self {
        Self {
            aggregate_unc_parity: value.aggregateUncParity,
            aggregate_unc_secded: value.aggregateUncSecDed,
            aggregate_corrected: value.aggregateCor,
            volatile_unc_parity: value.volatileUncParity,
            volatile_unc_secded: value.volatileUncSecDed,
            volatile_corrected: value.volatileCor,
            aggregate_unc_bucket_l2: value.aggregateUncBucketL2,
            aggregate_unc_bucket_sm: value.aggregateUncBucketSm,
            aggregate_unc_bucket_pcie: value.aggregateUncBucketPcie,
            aggregate_unc_bucket_mcu: value.aggregateUncBucketMcu,
            aggregate_unc_bucket_other: value.aggregateUncBucketOther,
            threshold_exceeded: value.bThresholdExceeded != 0,
        }
    }
}

impl From<sys::nvmlEccSramUniqueUncorrectedErrorEntry_v1_t> for EccSramUniqueUncorrectedErrorEntry {
    fn from(value: sys::nvmlEccSramUniqueUncorrectedErrorEntry_v1_t) -> Self {
        Self {
            unit: value.unit,
            location: value.location,
            sublocation: value.sublocation,
            extlocation: value.extlocation,
            address: value.address,
            is_parity: value.isParity != 0,
            count: value.count,
        }
    }
}

impl From<sys::nvmlGpuThermalSettings_t__bindgen_ty_1> for ThermalSensorSetting {
    fn from(value: sys::nvmlGpuThermalSettings_t__bindgen_ty_1) -> Self {
        Self {
            controller: value.controller.into(),
            default_min_temp: value.defaultMinTemp,
            default_max_temp: value.defaultMaxTemp,
            current_temp: value.currentTemp,
            target: value.target.into(),
        }
    }
}

impl From<sys::nvmlGpuThermalSettings_t> for ThermalSettings {
    fn from(value: sys::nvmlGpuThermalSettings_t) -> Self {
        Self {
            sensors: value.sensor[..value.count as usize]
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<sys::nvmlClockOffset_t> for ClockOffset {
    fn from(value: sys::nvmlClockOffset_t) -> Self {
        Self {
            kind: value.type_.into(),
            pstate: value.pstate.into(),
            clock_offset_mhz: value.clockOffsetMHz,
            min_clock_offset_mhz: value.minClockOffsetMHz,
            max_clock_offset_mhz: value.maxClockOffsetMHz,
        }
    }
}

impl From<sys::nvmlNvLinkUtilizationControl_t> for NvLinkUtilizationControl {
    fn from(value: sys::nvmlNvLinkUtilizationControl_t) -> Self {
        Self {
            units: value.units.into(),
            packet_filter: NvLinkPacketTypes::from_bits_retain(value.pktfilter as u32),
        }
    }
}

impl From<sys::nvmlNvlinkFirmwareVersion_t> for NvLinkFirmwareVersion {
    fn from(value: sys::nvmlNvlinkFirmwareVersion_t) -> Self {
        Self {
            ucode_type: value.ucodeType,
            major: value.major,
            minor: value.minor,
            sub_minor: value.subMinor,
        }
    }
}

impl From<sys::nvmlNvLinkInfo_t> for NvLinkInfo {
    fn from(value: sys::nvmlNvLinkInfo_t) -> Self {
        Self {
            nvle_enabled: value.isNvleEnabled != 0,
            firmware_versions: value.firmwareInfo.firmwareVersion
                [..value.firmwareInfo.numValidEntries as usize]
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<sys::nvmlNvlinkGetBwMode_t> for NvLinkBwMode {
    fn from(value: sys::nvmlNvlinkGetBwMode_t) -> Self {
        Self {
            is_best: value.bIsBest != 0,
            mode: value.bwMode,
        }
    }
}

impl From<sys::nvmlNvlinkSupportedBwModes_t> for NvLinkSupportedBwModes {
    fn from(value: sys::nvmlNvlinkSupportedBwModes_t) -> Self {
        Self {
            modes: value.bwModes[..value.totalBwModes as usize].to_vec(),
        }
    }
}

impl From<sys::nvmlDevicePowerMizerModes_v1_t> for PowerMizerModes {
    fn from(value: sys::nvmlDevicePowerMizerModes_v1_t) -> Self {
        Self {
            current_mode: value.currentMode,
            mode: value.mode,
            supported_modes: value.supportedPowerMizerModes,
        }
    }
}

impl From<sys::nvmlWorkloadPowerProfileInfo_t> for WorkloadPowerProfileInfo {
    fn from(value: sys::nvmlWorkloadPowerProfileInfo_t) -> Self {
        Self {
            profile_id: value.profileId,
            priority: value.priority,
            conflicting_profiles: mask255_bits(value.conflictingMask),
        }
    }
}

impl From<sys::nvmlWorkloadPowerProfileProfilesInfo_t> for WorkloadPowerProfilesInfo {
    fn from(value: sys::nvmlWorkloadPowerProfileProfilesInfo_t) -> Self {
        Self {
            supported_profiles: mask255_bits(value.perfProfilesMask),
            profiles: value.perfProfile.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<sys::nvmlWorkloadPowerProfileCurrentProfiles_t> for WorkloadPowerCurrentProfiles {
    fn from(value: sys::nvmlWorkloadPowerProfileCurrentProfiles_t) -> Self {
        Self {
            supported_profiles: mask255_bits(value.perfProfilesMask),
            requested_profiles: mask255_bits(value.requestedProfilesMask),
            enforced_profiles: mask255_bits(value.enforcedProfilesMask),
        }
    }
}

impl From<sys::nvmlDramEncryptionInfo_t> for DramEncryptionInfo {
    fn from(value: sys::nvmlDramEncryptionInfo_t) -> Self {
        Self {
            encryption_state: value.encryptionState.into(),
        }
    }
}

impl From<sys::nvmlVgpuTypeBar1Info_t> for VgpuTypeBar1Info {
    fn from(value: sys::nvmlVgpuTypeBar1Info_t) -> Self {
        Self {
            bar1_size: value.bar1Size,
        }
    }
}

impl From<sys::nvmlConfComputeSystemCaps_t> for ConfComputeSystemCaps {
    fn from(value: sys::nvmlConfComputeSystemCaps_t) -> Self {
        Self {
            cpu_caps: value.cpuCaps,
            gpus_caps: value.gpusCaps,
        }
    }
}

impl From<sys::nvmlConfComputeSystemState_t> for ConfComputeSystemState {
    fn from(value: sys::nvmlConfComputeSystemState_t) -> Self {
        Self {
            environment: value.environment,
            cc_feature: value.ccFeature,
            dev_tools_mode: value.devToolsMode,
        }
    }
}

impl From<sys::nvmlSystemConfComputeSettings_t> for ConfComputeSystemSettings {
    fn from(value: sys::nvmlSystemConfComputeSettings_t) -> Self {
        Self {
            environment: value.environment,
            cc_feature: value.ccFeature,
            dev_tools_mode: value.devToolsMode,
            multi_gpu_mode: value.multiGpuMode,
        }
    }
}

impl From<sys::nvmlConfComputeMemSizeInfo_t> for ConfComputeMemSizeInfo {
    fn from(value: sys::nvmlConfComputeMemSizeInfo_t) -> Self {
        Self {
            protected_mem_size_kib: value.protectedMemSizeKib,
            unprotected_mem_size_kib: value.unprotectedMemSizeKib,
        }
    }
}

impl From<sys::nvmlConfComputeGpuCertificate_t> for ConfComputeGpuCertificate {
    fn from(value: sys::nvmlConfComputeGpuCertificate_t) -> Self {
        Self {
            cert_chain: value.certChain[..value.certChainSize as usize].to_vec(),
            attestation_cert_chain: value.attestationCertChain
                [..value.attestationCertChainSize as usize]
                .to_vec(),
        }
    }
}

impl From<sys::nvmlConfComputeGpuAttestationReport_t> for ConfComputeGpuAttestationReport {
    fn from(value: sys::nvmlConfComputeGpuAttestationReport_t) -> Self {
        Self {
            cec_attestation_report_present: value.isCecAttestationReportPresent != 0,
            nonce: value.nonce,
            attestation_report: value.attestationReport[..value.attestationReportSize as usize]
                .to_vec(),
            cec_attestation_report: value.cecAttestationReport
                [..value.cecAttestationReportSize as usize]
                .to_vec(),
        }
    }
}

impl From<sys::nvmlConfComputeGetKeyRotationThresholdInfo_t> for ConfComputeKeyRotationThreshold {
    fn from(value: sys::nvmlConfComputeGetKeyRotationThresholdInfo_t) -> Self {
        Self {
            attacker_advantage: value.attackerAdvantage,
        }
    }
}

impl GpuFabricInfo {
    pub(crate) fn from_raw(value: sys::nvmlGpuFabricInfoV_t) -> Result<Self> {
        Ok(Self {
            cluster_uuid: value.clusterUuid,
            status_code: value.status as u32,
            clique_id: value.cliqueId,
            state: try_from_nvml_enum("gpu fabric state", value.state as u32)?,
            health: GpuFabricHealth::from_raw(value.healthMask)?,
            health_summary: try_from_nvml_enum(
                "gpu fabric health summary",
                value.healthSummary as u32,
            )?,
        })
    }
}

impl From<sys::nvmlGpuDynamicPstatesInfo_t> for DynamicPstatesInfo {
    fn from(value: sys::nvmlGpuDynamicPstatesInfo_t) -> Self {
        Self {
            flags: value.flags,
            utilization: value
                .utilization
                .into_iter()
                .enumerate()
                .filter(|(_, entry)| entry.bIsPresent != 0)
                .map(|(domain, entry)| DynamicPstateInfo {
                    domain: domain as u32,
                    percentage: entry.percentage,
                    increase_threshold: entry.incThreshold,
                    decrease_threshold: entry.decThreshold,
                })
                .collect(),
        }
    }
}

impl GpuFabricHealth {
    fn from_raw(raw_mask: u32) -> Result<Self> {
        Ok(Self {
            degraded_bandwidth: decode_fabric_boolean_health(
                raw_mask,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_DEGRADED_BW,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_WIDTH_DEGRADED_BW,
            )?,
            route_recovery: decode_fabric_boolean_health(
                raw_mask,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ROUTE_RECOVERY,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_WIDTH_ROUTE_RECOVERY,
            )?,
            route_unhealthy: decode_fabric_boolean_health(
                raw_mask,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ROUTE_UNHEALTHY,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_WIDTH_ROUTE_UNHEALTHY,
            )?,
            access_timeout_recovery: decode_fabric_boolean_health(
                raw_mask,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_ACCESS_TIMEOUT_RECOVERY,
                sys::NVML_GPU_FABRIC_HEALTH_MASK_WIDTH_ACCESS_TIMEOUT_RECOVERY,
            )?,
            incorrect_configuration: try_from_nvml_enum(
                "gpu fabric incorrect configuration",
                (raw_mask >> sys::NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_INCORRECT_CONFIGURATION)
                    & sys::NVML_GPU_FABRIC_HEALTH_MASK_WIDTH_INCORRECT_CONFIGURATION,
            )?,
            raw_mask,
        })
    }
}

fn decode_fabric_boolean_health(
    raw_mask: u32,
    shift: u32,
    width: u32,
) -> Result<GpuFabricBooleanState> {
    try_from_nvml_enum("gpu fabric boolean state", (raw_mask >> shift) & width)
}

impl FieldValue {
    pub fn from_raw(value_type: sys::nvmlValueType_t, value: sys::nvmlValue_t) -> Result<Self> {
        Ok(match FieldValueType::try_from(u32::from(value_type)) {
            Ok(FieldValueType::Double) => Self::Double(unsafe { value.dVal }),
            Ok(FieldValueType::UnsignedInt) => Self::UnsignedInt(unsafe { value.uiVal }),
            Ok(FieldValueType::UnsignedLong) => Self::UnsignedLong(unsafe { value.ulVal }),
            Ok(FieldValueType::UnsignedLongLong) => Self::UnsignedLongLong(unsafe { value.ullVal }),
            Ok(FieldValueType::SignedLongLong) => Self::SignedLongLong(unsafe { value.sllVal }),
            Ok(FieldValueType::SignedInt) => Self::SignedInt(unsafe { value.siVal }),
            Ok(FieldValueType::UnsignedShort) => Self::UnsignedShort(unsafe { value.usVal }),
            Err(_) => return Err(Error::UnknownFieldValueType(u32::from(value_type))),
        })
    }
}

impl Sample {
    pub fn from_raw(value_type: sys::nvmlValueType_t, sample: sys::nvmlSample_t) -> Result<Self> {
        Ok(Self {
            timestamp: sample.timeStamp,
            value: FieldValue::from_raw(value_type, sample.sampleValue)?,
        })
    }
}

impl FieldSample {
    pub fn from_raw(value: sys::nvmlFieldValue_t) -> Self {
        let result = if value.nvmlReturn == sys::nvmlReturn_t::NVML_SUCCESS {
            FieldValue::from_raw(value.valueType, value.value)
        } else {
            Err(value.nvmlReturn.into())
        };

        Self {
            field: FieldId(value.fieldId),
            scope_id: value.scopeId,
            timestamp: value.timestamp,
            latency_usec: value.latencyUsec,
            result,
        }
    }
}

impl From<sys::nvmlEventData_t> for EventData {
    fn from(value: sys::nvmlEventData_t) -> Self {
        Self {
            device: (!value.device.is_null()).then_some(unsafe { Device::from_raw(value.device) }),
            event_types: EventTypes::from_bits_retain(value.eventType),
            event_data: value.eventData,
            gpu_instance_id: value.gpuInstanceId,
            compute_instance_id: value.computeInstanceId,
        }
    }
}

impl From<sys::nvmlSystemEventData_v1_t> for SystemEventData {
    fn from(value: sys::nvmlSystemEventData_v1_t) -> Self {
        Self {
            event_types: SystemEventTypes::from_bits_retain(value.eventType),
            gpu_id: value.gpuId,
        }
    }
}

impl From<sys::nvmlProcessInfo_t> for ProcessInfo {
    fn from(value: sys::nvmlProcessInfo_t) -> Self {
        Self {
            pid: Pid(value.pid),
            used_gpu_memory: option_u64_from_not_available(value.usedGpuMemory),
            gpu_instance_id: value.gpuInstanceId,
            compute_instance_id: value.computeInstanceId,
        }
    }
}

impl From<sys::nvmlProcessDetail_v1_t> for ProcessDetail {
    fn from(value: sys::nvmlProcessDetail_v1_t) -> Self {
        Self {
            pid: Pid(value.pid),
            used_gpu_memory: option_u64_from_not_available(value.usedGpuMemory),
            gpu_instance_id: value.gpuInstanceId,
            compute_instance_id: value.computeInstanceId,
            used_gpu_cc_protected_memory: option_u64_from_not_available(
                value.usedGpuCcProtectedMemory,
            ),
        }
    }
}

impl From<sys::nvmlProcessUtilizationSample_t> for ProcessUtilizationSample {
    fn from(value: sys::nvmlProcessUtilizationSample_t) -> Self {
        Self {
            pid: Pid(value.pid),
            timestamp: value.timeStamp,
            sm_utilization: value.smUtil,
            memory_utilization: value.memUtil,
            encoder_utilization: value.encUtil,
            decoder_utilization: value.decUtil,
        }
    }
}

impl From<sys::nvmlProcessUtilizationInfo_v1_t> for ProcessUtilizationInfo {
    fn from(value: sys::nvmlProcessUtilizationInfo_v1_t) -> Self {
        Self {
            timestamp: value.timeStamp,
            pid: Pid(value.pid),
            sm_utilization: value.smUtil,
            memory_utilization: value.memUtil,
            encoder_utilization: value.encUtil,
            decoder_utilization: value.decUtil,
            jpg_utilization: value.jpgUtil,
            ofa_utilization: value.ofaUtil,
        }
    }
}

impl From<sys::nvmlEncoderSessionInfo_t> for EncoderSessionInfo {
    fn from(value: sys::nvmlEncoderSessionInfo_t) -> Self {
        Self {
            session_id: value.sessionId,
            pid: Pid(value.pid),
            vgpu_instance: value.vgpuInstance,
            codec_type: value.codecType.into(),
            horizontal_resolution: value.hResolution,
            vertical_resolution: value.vResolution,
            average_fps: value.averageFps,
            average_latency_us: value.averageLatency,
        }
    }
}

impl From<sys::nvmlFBCStats_t> for FbcStats {
    fn from(value: sys::nvmlFBCStats_t) -> Self {
        Self {
            session_count: value.sessionsCount,
            average_fps: value.averageFPS,
            average_latency_us: value.averageLatency,
        }
    }
}

impl From<sys::nvmlFBCSessionInfo_t> for FbcSessionInfo {
    fn from(value: sys::nvmlFBCSessionInfo_t) -> Self {
        Self {
            session_id: value.sessionId,
            pid: Pid(value.pid),
            vgpu_instance: value.vgpuInstance,
            display_ordinal: value.displayOrdinal,
            session_type: value.sessionType.into(),
            session_flags: value.sessionFlags,
            max_horizontal_resolution: value.hMaxResolution,
            max_vertical_resolution: value.vMaxResolution,
            horizontal_resolution: value.hResolution,
            vertical_resolution: value.vResolution,
            average_fps: value.averageFPS,
            average_latency_us: value.averageLatency,
        }
    }
}
