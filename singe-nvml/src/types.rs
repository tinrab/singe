#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{
    cmp::max,
    fmt::{self, Display, Formatter},
    mem,
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{
    impl_enum_conversion, impl_enum_display, string_from_c_chars, string_from_c_ptr,
    try_enum_from_raw,
};
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result, Status},
    gpu_instance::GpuInstance,
    utility::{mask255_bits, option_u64_from_not_available, to_u32},
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
    /// Flags controlling NVML initialization and device attachment.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct InitFlags: u32 {
        /// Initialize the NVML Library even when no devices are found.
        const NO_GPUS = sys::NVML_INIT_FLAG_NO_GPUS;
        /// Initialize the NVML Library without attaching any discovered devices.
        const NO_ATTACH = sys::NVML_INIT_FLAG_NO_ATTACH;
        /// Force device initialization even when a previous nvmlInit was called with the NO_GPUS and NO_ATTACH flags.
        const FORCE_INIT = sys::NVML_INIT_FLAG_FORCE_INIT;
    }
}

impl Display for InitFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// GPU event types accepted by NVML event registration.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventTypes: u64 {
        /// Mask with no events.
        ///
        /// No event.
        const NONE = sys::nvmlEventTypeNone as u64;
        /// Event about single bit ECC errors.
        const SINGLE_BIT_ECC_ERROR = sys::nvmlEventTypeSingleBitEccError as u64;
        /// Event about double bit ECC errors.
        const DOUBLE_BIT_ECC_ERROR = sys::nvmlEventTypeDoubleBitEccError as u64;
        /// Event about PState changes.
        const PSTATE = sys::nvmlEventTypePState as u64;
        /// Event that Xid critical error occurred.
        ///
        /// XID critical error event.
        const XID_CRITICAL_ERROR = sys::nvmlEventTypeXidCriticalError as u64;
        /// Event about clock changes.
        ///
        /// Kepler only Clock change event.
        const CLOCK = sys::nvmlEventTypeClock as u64;
        /// Event about AC/Battery power source changes.
        ///
        /// Power source change event.
        const POWER_SOURCE_CHANGE = sys::nvmlEventTypePowerSourceChange as u64;
        /// Event about MIG configuration changes.
        ///
        /// MIG configuration change event.
        const MIG_CONFIG_CHANGE = sys::nvmlEventMigConfigChange as u64;
        /// Event about single bit ECC error storm.
        ///
        /// Single bit ECC error storm event.
        const SINGLE_BIT_ECC_ERROR_STORM = sys::nvmlEventTypeSingleBitEccErrorStorm as u64;
        /// Event about DRAM retirement event.
        ///
        /// DRAM retirement event.
        const DRAM_RETIREMENT_EVENT = sys::nvmlEventTypeDramRetirementEvent as u64;
        /// Event about DRAM retirement failure.
        ///
        /// DRAM retirement failure event.
        const DRAM_RETIREMENT_FAILURE = sys::nvmlEventTypeDramRetirementFailure as u64;
        /// Event for Non Fatal Poison.
        ///
        /// Non-fatal poison error event.
        const NON_FATAL_POISON_ERROR = sys::nvmlEventTypeNonFatalPoisonError as u64;
        /// Event for Fatal Poison.
        ///
        /// Fatal poison error event.
        const FATAL_POISON_ERROR = sys::nvmlEventTypeFatalPoisonError as u64;
        /// Event for GPU Unavailable.
        ///
        /// GPU unavailable error event.
        const GPU_UNAVAILABLE_ERROR = sys::nvmlEventTypeGpuUnavailableError as u64;
        /// Event for GPU Recovery Action.
        ///
        /// GPU recovery action event.
        const GPU_RECOVERY_ACTION = sys::nvmlEventTypeGpuRecoveryAction as u64;
        /// Mask of all events.
        ///
        /// Mask of all event types.
        const ALL = sys::nvmlEventTypeAll as u64;
    }
}

impl Display for EventTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// GPU driver binding and unbinding notifications.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SystemEventTypes: u64 {
        /// System Event for GPU Driver Unbind.
        ///
        /// Bitmask value of Driver Unbind System Event.
        const GPU_DRIVER_UNBIND = sys::nvmlSystemEventTypeGpuDriverUnbind as u64;
        /// Bitmask value of Driver Bind System Event.
        const GPU_DRIVER_BIND = sys::nvmlSystemEventTypeGpuDriverBind as u64;
    }
}

impl Display for SystemEventTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Device capability mask reported by NVML.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DeviceCapabilities: u32 {
        /// Device capability mask bit indicating extended GPU memory support.
        const EGM = sys::NVML_DEV_CAP_EGM;
    }
}

impl Display for DeviceCapabilities {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Flags controlling a Windows driver-model change.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DriverModelFlags: u32 {
        /// Generic flag used to specify the default behavior of some functions. See description of particular functions for details.
        ///
        /// Default flag. Generic flag used to force some behavior. See description of particular functions for details.
        const DEFAULT = sys::nvmlFlagDefault;
        /// Force flag.
        const FORCE = sys::nvmlFlagForce;
    }
}

impl Display for DriverModelFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Capabilities supported by a MIG GPU instance profile.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GpuInstanceProfileCapabilities: u32 {
        /// MIG GPU instance profile capability.
        ///
        /// Bit field values representing MIG profile capabilities [nvmlGpuInstanceProfileInfo_v3_t::capabilities](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlGpuInstanceProfileInfo__v3__t.html#structnvmlgpuinstanceprofileinfo__v3__t_1ad9e76b079f0e392344ed74621f6083f0) Peer-to-Peer support.
        const P2P = sys::NVML_GPU_INSTANCE_PROFILE_CAPS_P2P;
        /// GFX support.
        const GFX = sys::NVML_GPU_INSTANCE_PROFILE_CAPS_GFX;
    }
}

impl Display for GpuInstanceProfileCapabilities {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Capabilities supported by a MIG compute instance profile.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ComputeInstanceProfileCapabilities: u32 {
        /// MIG compute instance profile capability.
        ///
        /// Bit field values representing MIG profile capabilities [nvmlComputeInstanceProfileInfo_v3_t::capabilities](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlComputeInstanceProfileInfo__v3__t.html#structnvmlcomputeinstanceprofileinfo__v3__t_1a9cfd266f4c46066bde91f575655ca162).
        const GFX = sys::NVML_COMPUTE_INSTANCE_PROFILE_CAPS_GFX;
    }
}

impl Display for ComputeInstanceProfileCapabilities {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Enum to represent the NvLink utilization counter packet types to count \*\* this is ONLY applicable with the units as packets or bytes \*\* as specified in `nvmlNvLinkUtilizationCountUnits_t` \*\* all packet filter descriptions are target GPU centric \*\* these can be "OR'd" together.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NvLinkPacketTypes: u32 {
        /// no operation packets
        const NOP = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_NOP as _;
        /// read packets
        const READ = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_READ as _;
        /// write packets
        const WRITE = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_WRITE as _;
        /// reduction atomic requests
        const RATOM = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RATOM as _;
        /// non-reduction atomic requests
        const NRATOM = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_NRATOM as _;
        /// flush requests
        const FLUSH = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_FLUSH as _;
        /// responses with data
        const RESPDATA = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA as _;
        /// responses without data
        const RESPNODATA = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA as _;
        /// all packets
        const ALL = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_ALL as _;
    }
}

impl Display for NvLinkPacketTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// CUDA driver API version in native integer form and separate major and minor components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaDriverVersion {
    /// Native version integer, encoded as `1000 * major + 10 * minor`.
    pub raw: i32,
    /// CUDA driver API major version.
    pub major: i32,
    /// CUDA driver API minor version.
    pub minor: i32,
}

impl CudaDriverVersion {
    /// Decodes a native CUDA driver version integer without querying the driver.
    ///
    /// The native encoding is `1000 * major + 10 * minor`.
    #[must_use]
    pub const fn from_raw(raw: i32) -> Self {
        Self {
            raw,
            major: raw / 1000,
            minor: (raw % 1000) / 10,
        }
    }
}

/// Operating-system process identifier used by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pid(
    /// Native integer identifier.
    pub u32,
);

/// Native identifier of an NVML telemetry field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldId(
    /// Native integer identifier.
    pub u32,
);

impl FieldId {
    /// `NVLink` State - one of `NVML_NVLINK_STATE`_\* values.
    pub const NVLINK_GET_STATE: Self = Self(sys::NVML_FI_DEV_NVLINK_GET_STATE);

    /// GPU Recovery action. See [nvmlDeviceGpuRecoveryAction_t](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlVgpuStructs.html#group__nvmlvgpustructs_1ga9951f808183aa36957a968358dd791fb).
    pub const GET_GPU_RECOVERY_ACTION: Self = Self(sys::NVML_FI_DEV_GET_GPU_RECOVERY_ACTION);

    /// Number of groups with no spare bankremap availability.
    pub const BANK_REMAPPER_HISTOGRAM_NONE: Self =
        Self(sys::NVML_FI_DEV_BANK_REMAPPER_HISTOGRAM_NONE);

    /// Number of groups with full bank remap availability.
    pub const BANK_REMAPPER_HISTOGRAM_MAX: Self =
        Self(sys::NVML_FI_DEV_BANK_REMAPPER_HISTOGRAM_MAX);

    /// If any banks are pending remapping. 1=yes 0=no.
    pub const PENDING_BANK_REMAPPING: Self = Self(sys::NVML_FI_DEV_PENDING_BANK_REMAPPING);

    /// Number of inactive bank remappings.
    pub const INACTIVE_BANK_REMAPPINGS: Self = Self(sys::NVML_FI_DEV_INACTIVE_BANK_REMAPPINGS);

    /// Number of active bank remappings.
    pub const ACTIVE_BANK_REMAPPINGS: Self = Self(sys::NVML_FI_DEV_ACTIVE_BANK_REMAPPINGS);

    /// Field Identifiers.
    ///
    /// All Identifiers pertain to a device. Each ID is only used once and is guaranteed never to change. Current ECC mode. 1=Active. 0=Inactive.
    pub const ECC_CURRENT: Self = Self(sys::NVML_FI_DEV_ECC_CURRENT);
    /// Pending ECC mode. 1=Active. 0=Inactive.
    pub const ECC_PENDING: Self = Self(sys::NVML_FI_DEV_ECC_PENDING);
    /// Number of retired pages because of single bit errors.
    pub const RETIRED_SBE: Self = Self(sys::NVML_FI_DEV_RETIRED_SBE);
    /// Number of retired pages because of double bit errors.
    pub const RETIRED_DBE: Self = Self(sys::NVML_FI_DEV_RETIRED_DBE);
    /// If any pages are pending retirement. 1=yes. 0=no.
    pub const RETIRED_PENDING: Self = Self(sys::NVML_FI_DEV_RETIRED_PENDING);
    /// Memory temperature for the device.
    pub const MEMORY_TEMP: Self = Self(sys::NVML_FI_DEV_MEMORY_TEMP);
    /// Total energy consumption for the GPU in mJ since the driver was last reloaded.
    pub const TOTAL_ENERGY_CONSUMPTION: Self = Self(sys::NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION);
    /// `PCIe` replay counter.
    pub const PCIE_REPLAY_COUNTER: Self = Self(sys::NVML_FI_DEV_PCIE_REPLAY_COUNTER);
    /// Number of remapped rows due to correctable errors.
    pub const REMAPPED_CORRECTED_ROWS: Self = Self(sys::NVML_FI_DEV_REMAPPED_COR);
    /// Number of remapped rows due to uncorrectable errors.
    pub const REMAPPED_UNCORRECTED_ROWS: Self = Self(sys::NVML_FI_DEV_REMAPPED_UNC);
    /// If any rows are pending remapping. 1=yes 0=no.
    pub const REMAPPED_PENDING: Self = Self(sys::NVML_FI_DEV_REMAPPED_PENDING);
    /// If any rows failed to be remapped 1=yes 0=no.
    pub const REMAPPED_FAILURE: Self = Self(sys::NVML_FI_DEV_REMAPPED_FAILURE);
    /// Retrieves power usage for this GPU in milliwatts.
    ///
    /// It is only available if power management mode is supported. See `nvmlDeviceGetPowerManagementMode` and `nvmlDeviceGetPowerUsage`.
    ///
    /// scopeId needs to be specified. It signifies: 0 - GPU Only Scope - Metrics for GPU are retrieved 1 - Module scope - Metrics for the module (e.g. CPU + GPU) are retrieved. 2 - GPU base scope - GPU base Power metrics are retrieved. Note: Only supports _`MIN_LIMIT`, _`MAX_LIMIT`, _`DEFAULT_LIMIT`, and _`CURRENT_LIMIT` fields. Note: CPU here refers to NVIDIA CPU (e.g. Grace). x86 or non-NVIDIA ARM is not supported GPU power averaged over 1 sec interval, supported on Ampere (except GA100) or newer architectures.
    pub const POWER_AVERAGE: Self = Self(sys::NVML_FI_DEV_POWER_AVERAGE);
    /// Current GPU power, supported on all architectures.
    pub const POWER_INSTANT: Self = Self(sys::NVML_FI_DEV_POWER_INSTANT);
    /// Minimum power limit in milliwatts.
    pub const POWER_MIN_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_MIN_LIMIT);
    /// Maximum power limit in milliwatts.
    pub const POWER_MAX_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_MAX_LIMIT);
    /// Default power limit in milliwatts (limit which device boots with).
    pub const POWER_DEFAULT_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_DEFAULT_LIMIT);
    /// Limit currently enforced in milliwatts (This includes other limits set elsewhere. E.g. Out-of-band).
    pub const POWER_CURRENT_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_CURRENT_LIMIT);
    /// Total energy consumption (in mJ) since the driver was last reloaded. Same as `NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION` for the GPU.
    pub const ENERGY: Self = Self(sys::NVML_FI_DEV_ENERGY);
    /// Power limit requested by NVML or any other userspace client.
    pub const POWER_REQUESTED_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_REQUESTED_LIMIT);
    /// `PCIe` transmit bytes. Value can be wrapped.
    pub const PCIE_TX_BYTES: Self = Self(sys::NVML_FI_DEV_PCIE_COUNT_TX_BYTES);
    /// `PCIe` receive bytes. Value can be wrapped.
    pub const PCIE_RX_BYTES: Self = Self(sys::NVML_FI_DEV_PCIE_COUNT_RX_BYTES);
    /// `NVLink` throughput counters field values.
    ///
    /// Link ID needs to be specified in the scopeId field in [nvmlFieldValue_t](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlFieldValue__t.html#structnvmlfieldvalue__t). A scopeId of `UINT_MAX` returns aggregate value summed up across all links for the specified counter type in fieldId. `NVLink` TX Data throughput in KiB.
    pub const NVLINK_THROUGHPUT_DATA_TX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX);
    /// `NVLink` RX Data throughput in KiB.
    pub const NVLINK_THROUGHPUT_DATA_RX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX);
    /// `NVLink` TX Data + protocol overhead in KiB.
    pub const NVLINK_THROUGHPUT_RAW_TX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX);
    /// `NVLink` RX Data + protocol overhead in KiB.
    pub const NVLINK_THROUGHPUT_RAW_RX: Self = Self(sys::NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX);
    /// Perf Policy Counter for Power Policy.
    pub const PERF_POLICY_POWER: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_POWER);
    /// Perf Policy Counter for Thermal Policy.
    pub const PERF_POLICY_THERMAL: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_THERMAL);
    /// Perf Policy Counter for Sync boost Policy.
    pub const PERF_POLICY_SYNC_BOOST: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_SYNC_BOOST);
    /// Perf Policy Counter for Board Limit.
    pub const PERF_POLICY_BOARD_LIMIT: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT);

    /// Perf Policy Counter for Low GPU Utilization Policy.
    pub const PERF_POLICY_LOW_UTILIZATION: Self =
        Self(sys::NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION);

    /// Perf Policy Counter for Reliability Policy.
    pub const PERF_POLICY_RELIABILITY: Self = Self(sys::NVML_FI_DEV_PERF_POLICY_RELIABILITY);

    /// Perf Policy Counter for Total App Clock Policy.
    pub const PERF_POLICY_TOTAL_APP_CLOCKS: Self =
        Self(sys::NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS);

    /// Perf Policy Counter for Total Base Clocks Policy.
    pub const PERF_POLICY_TOTAL_BASE_CLOCKS: Self =
        Self(sys::NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS);

    /// GPU T.Limit temperature thresholds in degree Celsius.
    ///
    /// These fields are supported on Ada and later architectures and supersedes `nvmlDeviceGetTemperatureThreshold`. T.Limit temperature after which GPU may shut down for HW protection.
    pub const TEMPERATURE_SHUTDOWN_TLIMIT: Self =
        Self(sys::NVML_FI_DEV_TEMPERATURE_SHUTDOWN_TLIMIT);

    /// T.Limit temperature after which GPU may begin HW slowdown.
    pub const TEMPERATURE_SLOWDOWN_TLIMIT: Self =
        Self(sys::NVML_FI_DEV_TEMPERATURE_SLOWDOWN_TLIMIT);

    /// T.Limit temperature after which GPU may begin SW slowdown due to memory temperature.
    pub const TEMPERATURE_MEMORY_MAX_TLIMIT: Self =
        Self(sys::NVML_FI_DEV_TEMPERATURE_MEM_MAX_TLIMIT);

    /// T.Limit temperature after which GPU may be throttled below base clock.
    pub const TEMPERATURE_GPU_MAX_TLIMIT: Self = Self(sys::NVML_FI_DEV_TEMPERATURE_GPU_MAX_TLIMIT);
}

/// Native GPU Performance Monitoring metric identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpmMetricId(
    /// Native integer identifier.
    pub u32,
);

impl GpmMetricId {
    /// `NvLink` write bandwidth for link 71 in MiB/sec.
    pub const NVLINK_L71_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 71 in MiB/sec.
    pub const NVLINK_L71_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 70 in MiB/sec.
    pub const NVLINK_L70_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 70 in MiB/sec.
    pub const NVLINK_L70_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 69 in MiB/sec.
    pub const NVLINK_L69_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 69 in MiB/sec.
    pub const NVLINK_L69_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 68 in MiB/sec.
    pub const NVLINK_L68_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 68 in MiB/sec.
    pub const NVLINK_L68_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 67 in MiB/sec.
    pub const NVLINK_L67_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 67 in MiB/sec.
    pub const NVLINK_L67_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 66 in MiB/sec.
    pub const NVLINK_L66_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 66 in MiB/sec.
    pub const NVLINK_L66_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 65 in MiB/sec.
    pub const NVLINK_L65_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 65 in MiB/sec.
    pub const NVLINK_L65_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 64 in MiB/sec.
    pub const NVLINK_L64_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 64 in MiB/sec.
    pub const NVLINK_L64_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 63 in MiB/sec.
    pub const NVLINK_L63_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 63 in MiB/sec.
    pub const NVLINK_L63_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 62 in MiB/sec.
    pub const NVLINK_L62_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 62 in MiB/sec.
    pub const NVLINK_L62_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 61 in MiB/sec.
    pub const NVLINK_L61_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 61 in MiB/sec.
    pub const NVLINK_L61_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 60 in MiB/sec.
    pub const NVLINK_L60_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 60 in MiB/sec.
    pub const NVLINK_L60_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 59 in MiB/sec.
    pub const NVLINK_L59_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 59 in MiB/sec.
    pub const NVLINK_L59_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 58 in MiB/sec.
    pub const NVLINK_L58_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 58 in MiB/sec.
    pub const NVLINK_L58_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 57 in MiB/sec.
    pub const NVLINK_L57_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 57 in MiB/sec.
    pub const NVLINK_L57_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 56 in MiB/sec.
    pub const NVLINK_L56_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 56 in MiB/sec.
    pub const NVLINK_L56_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 55 in MiB/sec.
    pub const NVLINK_L55_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 55 in MiB/sec.
    pub const NVLINK_L55_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 54 in MiB/sec.
    pub const NVLINK_L54_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 54 in MiB/sec.
    pub const NVLINK_L54_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 53 in MiB/sec.
    pub const NVLINK_L53_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 53 in MiB/sec.
    pub const NVLINK_L53_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 52 in MiB/sec.
    pub const NVLINK_L52_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 52 in MiB/sec.
    pub const NVLINK_L52_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 51 in MiB/sec.
    pub const NVLINK_L51_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 51 in MiB/sec.
    pub const NVLINK_L51_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 50 in MiB/sec.
    pub const NVLINK_L50_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 50 in MiB/sec.
    pub const NVLINK_L50_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 49 in MiB/sec.
    pub const NVLINK_L49_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 49 in MiB/sec.
    pub const NVLINK_L49_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 48 in MiB/sec.
    pub const NVLINK_L48_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 48 in MiB/sec.
    pub const NVLINK_L48_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 47 in MiB/sec.
    pub const NVLINK_L47_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 47 in MiB/sec.
    pub const NVLINK_L47_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 46 in MiB/sec.
    pub const NVLINK_L46_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 46 in MiB/sec.
    pub const NVLINK_L46_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 45 in MiB/sec.
    pub const NVLINK_L45_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 45 in MiB/sec.
    pub const NVLINK_L45_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 44 in MiB/sec.
    pub const NVLINK_L44_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 44 in MiB/sec.
    pub const NVLINK_L44_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 43 in MiB/sec.
    pub const NVLINK_L43_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 43 in MiB/sec.
    pub const NVLINK_L43_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 42 in MiB/sec.
    pub const NVLINK_L42_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 42 in MiB/sec.
    pub const NVLINK_L42_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 41 in MiB/sec.
    pub const NVLINK_L41_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 41 in MiB/sec.
    pub const NVLINK_L41_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 40 in MiB/sec.
    pub const NVLINK_L40_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 40 in MiB/sec.
    pub const NVLINK_L40_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 39 in MiB/sec.
    pub const NVLINK_L39_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 39 in MiB/sec.
    pub const NVLINK_L39_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 38 in MiB/sec.
    pub const NVLINK_L38_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 38 in MiB/sec.
    pub const NVLINK_L38_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 37 in MiB/sec.
    pub const NVLINK_L37_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 37 in MiB/sec.
    pub const NVLINK_L37_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 36 in MiB/sec.
    pub const NVLINK_L36_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 36 in MiB/sec.
    pub const NVLINK_L36_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_RX_PER_SEC as _);

    /// `NvLink` write for link 71 in bytes since reboot.
    pub const NVLINK_L71_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_TX as _);

    /// `NvLink` read for link 71 in bytes since reboot.
    pub const NVLINK_L71_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_RX as _);

    /// `NvLink` write for link 70 in bytes since reboot.
    pub const NVLINK_L70_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_TX as _);

    /// `NvLink` read for link 70 in bytes since reboot.
    pub const NVLINK_L70_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_RX as _);

    /// `NvLink` write for link 69 in bytes since reboot.
    pub const NVLINK_L69_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_TX as _);

    /// `NvLink` read for link 69 in bytes since reboot.
    pub const NVLINK_L69_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_RX as _);

    /// `NvLink` write for link 68 in bytes since reboot.
    pub const NVLINK_L68_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_TX as _);

    /// `NvLink` read for link 68 in bytes since reboot.
    pub const NVLINK_L68_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_RX as _);

    /// `NvLink` write for link 67 in bytes since reboot.
    pub const NVLINK_L67_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_TX as _);

    /// `NvLink` read for link 67 in bytes since reboot.
    pub const NVLINK_L67_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_RX as _);

    /// `NvLink` write for link 66 in bytes since reboot.
    pub const NVLINK_L66_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_TX as _);

    /// `NvLink` read for link 66 in bytes since reboot.
    pub const NVLINK_L66_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_RX as _);

    /// `NvLink` write for link 65 in bytes since reboot.
    pub const NVLINK_L65_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_TX as _);

    /// `NvLink` read for link 65 in bytes since reboot.
    pub const NVLINK_L65_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_RX as _);

    /// `NvLink` write for link 64 in bytes since reboot.
    pub const NVLINK_L64_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_TX as _);

    /// `NvLink` read for link 64 in bytes since reboot.
    pub const NVLINK_L64_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_RX as _);

    /// `NvLink` write for link 63 in bytes since reboot.
    pub const NVLINK_L63_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_TX as _);

    /// `NvLink` read for link 63 in bytes since reboot.
    pub const NVLINK_L63_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_RX as _);

    /// `NvLink` write for link 62 in bytes since reboot.
    pub const NVLINK_L62_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_TX as _);

    /// `NvLink` read for link 62 in bytes since reboot.
    pub const NVLINK_L62_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_RX as _);

    /// `NvLink` write for link 61 in bytes since reboot.
    pub const NVLINK_L61_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_TX as _);

    /// `NvLink` read for link 61 in bytes since reboot.
    pub const NVLINK_L61_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_RX as _);

    /// `NvLink` write for link 60 in bytes since reboot.
    pub const NVLINK_L60_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_TX as _);

    /// `NvLink` read for link 60 in bytes since reboot.
    pub const NVLINK_L60_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_RX as _);

    /// `NvLink` write for link 59 in bytes since reboot.
    pub const NVLINK_L59_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_TX as _);

    /// `NvLink` read for link 59 in bytes since reboot.
    pub const NVLINK_L59_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_RX as _);

    /// `NvLink` write for link 58 in bytes since reboot.
    pub const NVLINK_L58_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_TX as _);

    /// `NvLink` read for link 58 in bytes since reboot.
    pub const NVLINK_L58_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_RX as _);

    /// `NvLink` write for link 57 in bytes since reboot.
    pub const NVLINK_L57_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_TX as _);

    /// `NvLink` read for link 57 in bytes since reboot.
    pub const NVLINK_L57_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_RX as _);

    /// `NvLink` write for link 56 in bytes since reboot.
    pub const NVLINK_L56_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_TX as _);

    /// `NvLink` read for link 56 in bytes since reboot.
    pub const NVLINK_L56_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_RX as _);

    /// `NvLink` write for link 55 in bytes since reboot.
    pub const NVLINK_L55_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_TX as _);

    /// `NvLink` read for link 55 in bytes since reboot.
    pub const NVLINK_L55_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_RX as _);

    /// `NvLink` write for link 54 in bytes since reboot.
    pub const NVLINK_L54_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_TX as _);

    /// `NvLink` read for link 54 in bytes since reboot.
    pub const NVLINK_L54_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_RX as _);

    /// `NvLink` write for link 53 in bytes since reboot.
    pub const NVLINK_L53_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_TX as _);

    /// `NvLink` read for link 53 in bytes since reboot.
    pub const NVLINK_L53_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_RX as _);

    /// `NvLink` write for link 52 in bytes since reboot.
    pub const NVLINK_L52_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_TX as _);

    /// `NvLink` read for link 52 in bytes since reboot.
    pub const NVLINK_L52_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_RX as _);

    /// `NvLink` write for link 51 in bytes since reboot.
    pub const NVLINK_L51_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_TX as _);

    /// `NvLink` read for link 51 in bytes since reboot.
    pub const NVLINK_L51_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_RX as _);

    /// `NvLink` write for link 50 in bytes since reboot.
    pub const NVLINK_L50_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_TX as _);

    /// `NvLink` read for link 50 in bytes since reboot.
    pub const NVLINK_L50_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_RX as _);

    /// `NvLink` write for link 49 in bytes since reboot.
    pub const NVLINK_L49_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_TX as _);

    /// `NvLink` read for link 49 in bytes since reboot.
    pub const NVLINK_L49_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_RX as _);

    /// `NvLink` write for link 48 in bytes since reboot.
    pub const NVLINK_L48_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_TX as _);

    /// `NvLink` read for link 48 in bytes since reboot.
    pub const NVLINK_L48_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_RX as _);

    /// `NvLink` write for link 47 in bytes since reboot.
    pub const NVLINK_L47_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_TX as _);

    /// `NvLink` read for link 47 in bytes since reboot.
    pub const NVLINK_L47_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_RX as _);

    /// `NvLink` write for link 46 in bytes since reboot.
    pub const NVLINK_L46_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_TX as _);

    /// `NvLink` read for link 46 in bytes since reboot.
    pub const NVLINK_L46_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_RX as _);

    /// `NvLink` write for link 45 in bytes since reboot.
    pub const NVLINK_L45_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_TX as _);

    /// `NvLink` read for link 45 in bytes since reboot.
    pub const NVLINK_L45_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_RX as _);

    /// `NvLink` write for link 44 in bytes since reboot.
    pub const NVLINK_L44_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_TX as _);

    /// `NvLink` read for link 44 in bytes since reboot.
    pub const NVLINK_L44_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_RX as _);

    /// `NvLink` write for link 43 in bytes since reboot.
    pub const NVLINK_L43_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_TX as _);

    /// `NvLink` read for link 43 in bytes since reboot.
    pub const NVLINK_L43_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_RX as _);

    /// `NvLink` write for link 42 in bytes since reboot.
    pub const NVLINK_L42_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_TX as _);

    /// `NvLink` read for link 42 in bytes since reboot.
    pub const NVLINK_L42_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_RX as _);

    /// `NvLink` write for link 41 in bytes since reboot.
    pub const NVLINK_L41_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_TX as _);

    /// `NvLink` read for link 41 in bytes since reboot.
    pub const NVLINK_L41_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_RX as _);

    /// `NvLink` write for link 40 in bytes since reboot.
    pub const NVLINK_L40_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_TX as _);

    /// `NvLink` read for link 40 in bytes since reboot.
    pub const NVLINK_L40_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_RX as _);

    /// `NvLink` write for link 39 in bytes since reboot.
    pub const NVLINK_L39_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_TX as _);

    /// `NvLink` read for link 39 in bytes since reboot.
    pub const NVLINK_L39_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_RX as _);

    /// `NvLink` write for link 38 in bytes since reboot.
    pub const NVLINK_L38_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_TX as _);

    /// `NvLink` read for link 38 in bytes since reboot.
    pub const NVLINK_L38_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_RX as _);

    /// `NvLink` write for link 37 in bytes since reboot.
    pub const NVLINK_L37_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_TX as _);

    /// `NvLink` read for link 37 in bytes since reboot.
    pub const NVLINK_L37_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_RX as _);

    /// `NvLink` write for link 36 in bytes since reboot.
    pub const NVLINK_L36_TX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_TX as _);

    /// `NvLink` read for link 36 in bytes since reboot.
    pub const NVLINK_L36_RX: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_RX as _);

    /// The GPU's SM FP16 activity since reboot.
    pub const FP16_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP16_CYCLES_ACTIVE as _);

    /// The GPU's SM FP32 activity since reboot.
    pub const FP32_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP32_CYCLES_ACTIVE as _);

    /// The GPU's SM FP64 activity since reboot.
    pub const FP64_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP64_CYCLES_ACTIVE as _);

    /// The GPU's SM integer activity since reboot.
    pub const INTEGER_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_INTEGER_CYCLES_ACTIVE as _);

    /// The GPU's SM DFMA tensor activity since reboot.
    pub const DFMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DFMA_CYCLES_ACTIVE as _);

    /// The GPU's SM IMMA tensor activity since reboot.
    pub const IMMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_IMMA_CYCLES_ACTIVE as _);

    /// The GPU's SM HMMA tensor activity since reboot.
    pub const HMMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HMMA_CYCLES_ACTIVE as _);

    /// The GPU's SM DMMA tensor activity since reboot.
    pub const DMMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DMMA_CYCLES_ACTIVE as _);

    /// The GPU's SM MMA tensor activity since reboot.
    pub const MMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_MMA_CYCLES_ACTIVE as _);

    /// The GPU's SM activity since reboot.
    pub const SM_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_CYCLES_ACTIVE as _);

    /// The GPU's SM cycles elapsed since reboot.
    pub const SM_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_CYCLES_ELAPSED as _);

    /// `NvLink` write bandwidth for link 35 in MiB/sec.
    pub const NVLINK_L35_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L35_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 35 in MiB/sec.
    pub const NVLINK_L35_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L35_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 34 in MiB/sec.
    pub const NVLINK_L34_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L34_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 34 in MiB/sec.
    pub const NVLINK_L34_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L34_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 33 in MiB/sec.
    pub const NVLINK_L33_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L33_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 33 in MiB/sec.
    pub const NVLINK_L33_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L33_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 32 in MiB/sec.
    pub const NVLINK_L32_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L32_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 32 in MiB/sec.
    pub const NVLINK_L32_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L32_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 31 in MiB/sec.
    pub const NVLINK_L31_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L31_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 31 in MiB/sec.
    pub const NVLINK_L31_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L31_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 30 in MiB/sec.
    pub const NVLINK_L30_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L30_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 30 in MiB/sec.
    pub const NVLINK_L30_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L30_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 29 in MiB/sec.
    pub const NVLINK_L29_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L29_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 29 in MiB/sec.
    pub const NVLINK_L29_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L29_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 28 in MiB/sec.
    pub const NVLINK_L28_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L28_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 28 in MiB/sec.
    pub const NVLINK_L28_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L28_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 27 in MiB/sec.
    pub const NVLINK_L27_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L27_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 27 in MiB/sec.
    pub const NVLINK_L27_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L27_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 26 in MiB/sec.
    pub const NVLINK_L26_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L26_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 26 in MiB/sec.
    pub const NVLINK_L26_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L26_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 25 in MiB/sec.
    pub const NVLINK_L25_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L25_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 25 in MiB/sec.
    pub const NVLINK_L25_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L25_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 24 in MiB/sec.
    pub const NVLINK_L24_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L24_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 24 in MiB/sec.
    pub const NVLINK_L24_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L24_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 23 in MiB/sec.
    pub const NVLINK_L23_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L23_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 23 in MiB/sec.
    pub const NVLINK_L23_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L23_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 22 in MiB/sec.
    pub const NVLINK_L22_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L22_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 22 in MiB/sec.
    pub const NVLINK_L22_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L22_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 21 in MiB/sec.
    pub const NVLINK_L21_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L21_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 21 in MiB/sec.
    pub const NVLINK_L21_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L21_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 20 in MiB/sec.
    pub const NVLINK_L20_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L20_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 20 in MiB/sec.
    pub const NVLINK_L20_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L20_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 19 in MiB/sec.
    pub const NVLINK_L19_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L19_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 19 in MiB/sec.
    pub const NVLINK_L19_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L19_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 18 in MiB/sec.
    pub const NVLINK_L18_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L18_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 18 in MiB/sec.
    pub const NVLINK_L18_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L18_RX_PER_SEC as _);

    /// Percentage of time GR engine 7 context switches were active. 0.0 - 100.0.
    pub const GR7_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 7.
    pub const GR7_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 7.
    pub const GR7_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 7.
    pub const GR7_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 7.
    pub const GR7_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_CYCLES_ELAPSED as _);

    /// Percentage of time GR engine 6 context switches were active. 0.0 - 100.0.
    pub const GR6_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 6.
    pub const GR6_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 6.
    pub const GR6_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 6.
    pub const GR6_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 6.
    pub const GR6_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_CYCLES_ELAPSED as _);

    /// Percentage of time GR engine 5 context switches were active. 0.0 - 100.0.
    pub const GR5_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 5.
    pub const GR5_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 5.
    pub const GR5_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 5.
    pub const GR5_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 5.
    pub const GR5_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_CYCLES_ELAPSED as _);

    /// Percentage of time GR engine 4 context switches were active. 0.0 - 100.0.
    pub const GR4_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 4.
    pub const GR4_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 4.
    pub const GR4_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 4.
    pub const GR4_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 4.
    pub const GR4_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_CYCLES_ELAPSED as _);

    /// Percentage of time GR engine 3 context switches were active. 0.0 - 100.0.
    pub const GR3_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 3.
    pub const GR3_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 3.
    pub const GR3_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 3.
    pub const GR3_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 3.
    pub const GR3_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_CYCLES_ELAPSED as _);

    /// Percentage of time GR engine 2 context switches were active. 0.0 - 100.0.
    pub const GR2_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 2.
    pub const GR2_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 2.
    pub const GR2_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 2.
    pub const GR2_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 2.
    pub const GR2_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_CYCLES_ELAPSED as _);

    /// Percentage of time GR engine 1 context switches were active. 0.0 - 100.0.
    pub const GR1_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 1.
    pub const GR1_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 1.
    pub const GR1_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 1.
    pub const GR1_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 1.
    pub const GR1_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_CYCLES_ELAPSED as _);

    /// Percentage of time GR engine 0 context switches were active. 0.0 - 100.0.
    pub const GR0_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_ACTIVE_PCT as _);

    /// Average context switch cycles per request for GR engine 0.
    pub const GR0_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_CYCLES_PER_REQ as _);

    /// Number of context switch requests for GR engine 0.
    pub const GR0_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_REQUESTS as _);

    /// Active context switch cycles for GR engine 0.
    pub const GR0_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_CYCLES_ACTIVE as _);

    /// Total context switch cycles elapsed for GR engine 0.
    pub const GR0_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_CYCLES_ELAPSED as _);

    /// Percent utilization of NVENC 3. 0.0 - 100.0.
    pub const NVENC_3_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_3_UTIL as _);

    /// Percent utilization of NVENC 2. 0.0 - 100.0.
    pub const NVENC_2_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_2_UTIL as _);

    /// Percent utilization of NVENC 1. 0.0 - 100.0.
    pub const NVENC_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_1_UTIL as _);

    /// Percent utilization of NVENC 0. 0.0 - 100.0.
    pub const NVENC_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_0_UTIL as _);

    /// Percentage of DRAM cache misses. 0.0 - 100.0.
    pub const DRAM_CACHE_MISS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DRAM_CACHE_MISS as _);

    /// Percentage of DRAM cache hits. 0.0 - 100.0.
    pub const DRAM_CACHE_HIT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DRAM_CACHE_HIT as _);

    /// Percentage of peer memory cache misses. 0.0 - 100.0.
    pub const PEERMEM_CACHE_MISS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PEERMEM_CACHE_MISS as _);

    /// Percentage of peer memory cache hits. 0.0 - 100.0.
    pub const PEERMEM_CACHE_HIT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PEERMEM_CACHE_HIT as _);

    /// Percentage of host memory cache misses. 0.0 - 100.0.
    pub const HOSTMEM_CACHE_MISS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HOSTMEM_CACHE_MISS as _);

    /// Percentage of host memory cache hits. 0.0 - 100.0.
    pub const HOSTMEM_CACHE_HIT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HOSTMEM_CACHE_HIT as _);

    /// C2C link 13 data receive bandwidth in MiB/sec.
    pub const C2C_LINK13_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_DATA_RX_PER_SEC as _);

    /// C2C link 13 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK13_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_DATA_TX_PER_SEC as _);

    /// C2C link 13 total receive bandwidth in MiB/sec.
    pub const C2C_LINK13_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_TOTAL_RX_PER_SEC as _);

    /// C2C link 13 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK13_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_TOTAL_TX_PER_SEC as _);

    /// C2C link 12 data receive bandwidth in MiB/sec.
    pub const C2C_LINK12_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_DATA_RX_PER_SEC as _);

    /// C2C link 12 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK12_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_DATA_TX_PER_SEC as _);

    /// C2C link 12 total receive bandwidth in MiB/sec.
    pub const C2C_LINK12_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_TOTAL_RX_PER_SEC as _);

    /// C2C link 12 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK12_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_TOTAL_TX_PER_SEC as _);

    /// C2C link 11 data receive bandwidth in MiB/sec.
    pub const C2C_LINK11_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_DATA_RX_PER_SEC as _);

    /// C2C link 11 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK11_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_DATA_TX_PER_SEC as _);

    /// C2C link 11 total receive bandwidth in MiB/sec.
    pub const C2C_LINK11_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_TOTAL_RX_PER_SEC as _);

    /// C2C link 11 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK11_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_TOTAL_TX_PER_SEC as _);

    /// C2C link 10 data receive bandwidth in MiB/sec.
    pub const C2C_LINK10_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_DATA_RX_PER_SEC as _);

    /// C2C link 10 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK10_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_DATA_TX_PER_SEC as _);

    /// C2C link 10 total receive bandwidth in MiB/sec.
    pub const C2C_LINK10_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_TOTAL_RX_PER_SEC as _);

    /// C2C link 10 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK10_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_TOTAL_TX_PER_SEC as _);

    /// C2C link 9 data receive bandwidth in MiB/sec.
    pub const C2C_LINK9_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_DATA_RX_PER_SEC as _);

    /// C2C link 9 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK9_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_DATA_TX_PER_SEC as _);

    /// C2C link 9 total receive bandwidth in MiB/sec.
    pub const C2C_LINK9_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_TOTAL_RX_PER_SEC as _);

    /// C2C link 9 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK9_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_TOTAL_TX_PER_SEC as _);

    /// C2C link 8 data receive bandwidth in MiB/sec.
    pub const C2C_LINK8_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_DATA_RX_PER_SEC as _);

    /// C2C link 8 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK8_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_DATA_TX_PER_SEC as _);

    /// C2C link 8 total receive bandwidth in MiB/sec.
    pub const C2C_LINK8_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_TOTAL_RX_PER_SEC as _);

    /// C2C link 8 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK8_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_TOTAL_TX_PER_SEC as _);

    /// C2C link 7 data receive bandwidth in MiB/sec.
    pub const C2C_LINK7_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_DATA_RX_PER_SEC as _);

    /// C2C link 7 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK7_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_DATA_TX_PER_SEC as _);

    /// C2C link 7 total receive bandwidth in MiB/sec.
    pub const C2C_LINK7_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_TOTAL_RX_PER_SEC as _);

    /// C2C link 7 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK7_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_TOTAL_TX_PER_SEC as _);

    /// C2C link 6 data receive bandwidth in MiB/sec.
    pub const C2C_LINK6_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_DATA_RX_PER_SEC as _);

    /// C2C link 6 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK6_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_DATA_TX_PER_SEC as _);

    /// C2C link 6 total receive bandwidth in MiB/sec.
    pub const C2C_LINK6_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_TOTAL_RX_PER_SEC as _);

    /// C2C link 6 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK6_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_TOTAL_TX_PER_SEC as _);

    /// C2C link 5 data receive bandwidth in MiB/sec.
    pub const C2C_LINK5_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_DATA_RX_PER_SEC as _);

    /// C2C link 5 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK5_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_DATA_TX_PER_SEC as _);

    /// C2C link 5 total receive bandwidth in MiB/sec.
    pub const C2C_LINK5_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_TOTAL_RX_PER_SEC as _);

    /// C2C link 5 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK5_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_TOTAL_TX_PER_SEC as _);

    /// C2C link 4 data receive bandwidth in MiB/sec.
    pub const C2C_LINK4_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_DATA_RX_PER_SEC as _);

    /// C2C link 4 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK4_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_DATA_TX_PER_SEC as _);

    /// C2C link 4 total receive bandwidth in MiB/sec.
    pub const C2C_LINK4_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_TOTAL_RX_PER_SEC as _);

    /// C2C link 4 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK4_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_TOTAL_TX_PER_SEC as _);

    /// C2C link 3 data receive bandwidth in MiB/sec.
    pub const C2C_LINK3_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_DATA_RX_PER_SEC as _);

    /// C2C link 3 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK3_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_DATA_TX_PER_SEC as _);

    /// C2C link 3 total receive bandwidth in MiB/sec.
    pub const C2C_LINK3_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_TOTAL_RX_PER_SEC as _);

    /// C2C link 3 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK3_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_TOTAL_TX_PER_SEC as _);

    /// C2C link 2 data receive bandwidth in MiB/sec.
    pub const C2C_LINK2_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_DATA_RX_PER_SEC as _);

    /// C2C link 2 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK2_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_DATA_TX_PER_SEC as _);

    /// C2C link 2 total receive bandwidth in MiB/sec.
    pub const C2C_LINK2_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_TOTAL_RX_PER_SEC as _);

    /// C2C link 2 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK2_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_TOTAL_TX_PER_SEC as _);

    /// C2C link 1 data receive bandwidth in MiB/sec.
    pub const C2C_LINK1_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_DATA_RX_PER_SEC as _);

    /// C2C link 1 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK1_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_DATA_TX_PER_SEC as _);

    /// C2C link 1 total receive bandwidth in MiB/sec.
    pub const C2C_LINK1_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_TOTAL_RX_PER_SEC as _);

    /// C2C link 1 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK1_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_TOTAL_TX_PER_SEC as _);

    /// C2C link 0 data receive bandwidth in MiB/sec.
    pub const C2C_LINK0_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_DATA_RX_PER_SEC as _);

    /// C2C link 0 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK0_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_DATA_TX_PER_SEC as _);

    /// C2C link 0 total receive bandwidth in MiB/sec.
    pub const C2C_LINK0_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_TOTAL_RX_PER_SEC as _);

    /// C2C link 0 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK0_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_TOTAL_TX_PER_SEC as _);

    /// C2C data receive bandwidth in MiB/sec.
    pub const C2C_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_DATA_RX_PER_SEC as _);

    /// C2C data transmit bandwidth in MiB/sec.
    pub const C2C_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_DATA_TX_PER_SEC as _);

    /// C2C total receive bandwidth in MiB/sec.
    pub const C2C_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_TOTAL_RX_PER_SEC as _);

    /// C2C total transmit bandwidth in MiB/sec.
    pub const C2C_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_TOTAL_TX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 17 in MiB/sec.
    pub const NVLINK_L17_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L17_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 17 in MiB/sec.
    pub const NVLINK_L17_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L17_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 16 in MiB/sec.
    pub const NVLINK_L16_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L16_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 16 in MiB/sec.
    pub const NVLINK_L16_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L16_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 15 in MiB/sec.
    pub const NVLINK_L15_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L15_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 15 in MiB/sec.
    pub const NVLINK_L15_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L15_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 14 in MiB/sec.
    pub const NVLINK_L14_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L14_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 14 in MiB/sec.
    pub const NVLINK_L14_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L14_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 13 in MiB/sec.
    pub const NVLINK_L13_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L13_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 13 in MiB/sec.
    pub const NVLINK_L13_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L13_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 12 in MiB/sec.
    pub const NVLINK_L12_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L12_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 12 in MiB/sec.
    pub const NVLINK_L12_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L12_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 11 in MiB/sec.
    pub const NVLINK_L11_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L11_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 11 in MiB/sec.
    pub const NVLINK_L11_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L11_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 10 in MiB/sec.
    pub const NVLINK_L10_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L10_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 10 in MiB/sec.
    pub const NVLINK_L10_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L10_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 9 in MiB/sec.
    pub const NVLINK_L9_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L9_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 9 in MiB/sec.
    pub const NVLINK_L9_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L9_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 8 in MiB/sec.
    pub const NVLINK_L8_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L8_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 8 in MiB/sec.
    pub const NVLINK_L8_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L8_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 7 in MiB/sec.
    pub const NVLINK_L7_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L7_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 7 in MiB/sec.
    pub const NVLINK_L7_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L7_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 6 in MiB/sec.
    pub const NVLINK_L6_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L6_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 6 in MiB/sec.
    pub const NVLINK_L6_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L6_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 5 in MiB/sec.
    pub const NVLINK_L5_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L5_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 5 in MiB/sec.
    pub const NVLINK_L5_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L5_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 4 in MiB/sec.
    pub const NVLINK_L4_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L4_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 4 in MiB/sec.
    pub const NVLINK_L4_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L4_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 3 in MiB/sec.
    pub const NVLINK_L3_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L3_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 3 in MiB/sec.
    pub const NVLINK_L3_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L3_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 2 in MiB/sec.
    pub const NVLINK_L2_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L2_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 2 in MiB/sec.
    pub const NVLINK_L2_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L2_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 1 in MiB/sec.
    pub const NVLINK_L1_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L1_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 1 in MiB/sec.
    pub const NVLINK_L1_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L1_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for link 0 in MiB/sec.
    pub const NVLINK_L0_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L0_TX_PER_SEC as _);

    /// `NvLink` read bandwidth for link 0 in MiB/sec.
    pub const NVLINK_L0_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L0_RX_PER_SEC as _);

    /// Percent utilization of NVOFA 1. 0.0 - 100.0.
    pub const NVOFA_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVOFA_1_UTIL as _);

    /// Percent utilization of NVOFA 0. 0.0 - 100.0.
    pub const NVOFA_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVOFA_0_UTIL as _);

    /// Percent utilization of NVJPG 7. 0.0 - 100.0.
    pub const NVJPG_7_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_7_UTIL as _);

    /// Percent utilization of NVJPG 6. 0.0 - 100.0.
    pub const NVJPG_6_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_6_UTIL as _);

    /// Percent utilization of NVJPG 5. 0.0 - 100.0.
    pub const NVJPG_5_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_5_UTIL as _);

    /// Percent utilization of NVJPG 4. 0.0 - 100.0.
    pub const NVJPG_4_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_4_UTIL as _);

    /// Percent utilization of NVJPG 3. 0.0 - 100.0.
    pub const NVJPG_3_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_3_UTIL as _);

    /// Percent utilization of NVJPG 2. 0.0 - 100.0.
    pub const NVJPG_2_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_2_UTIL as _);

    /// Percent utilization of NVJPG 1. 0.0 - 100.0.
    pub const NVJPG_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_1_UTIL as _);

    /// Percent utilization of NVJPG 0. 0.0 - 100.0.
    pub const NVJPG_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_0_UTIL as _);

    /// Percent utilization of NVDEC 7. 0.0 - 100.0.
    pub const NVDEC_7_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_7_UTIL as _);

    /// Percent utilization of NVDEC 6. 0.0 - 100.0.
    pub const NVDEC_6_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_6_UTIL as _);

    /// Percent utilization of NVDEC 5. 0.0 - 100.0.
    pub const NVDEC_5_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_5_UTIL as _);

    /// Percent utilization of NVDEC 4. 0.0 - 100.0.
    pub const NVDEC_4_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_4_UTIL as _);

    /// Percent utilization of NVDEC 3. 0.0 - 100.0.
    pub const NVDEC_3_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_3_UTIL as _);

    /// Percent utilization of NVDEC 2. 0.0 - 100.0.
    pub const NVDEC_2_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_2_UTIL as _);

    /// Percent utilization of NVDEC 1. 0.0 - 100.0.
    pub const NVDEC_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_1_UTIL as _);

    /// Percent utilization of NVDEC 0. 0.0 - 100.0.
    pub const NVDEC_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_0_UTIL as _);

    /// Percentage of time the GPU's SMs were doing DMMA tensor operations. 0.0 - 100.0.
    pub const DMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DMMA_TENSOR_UTIL as _);

    /// Percentage of time any compute/graphics app was active on the GPU. 0.0 - 100.0.
    pub const GRAPHICS_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GRAPHICS_UTIL as _);

    /// Percentage of SMs that were busy. 0.0 - 100.0.
    pub const SM_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_UTIL as _);
    /// Percentage of warps that were active vs theoretical maximum. 0.0 - 100.0.
    pub const SM_OCCUPANCY: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_OCCUPANCY as _);
    /// Percentage of time the GPU's SMs were doing integer operations. 0.0 - 100.0.
    pub const INTEGER_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_INTEGER_UTIL as _);

    /// Percentage of time the GPU's SMs were doing ANY tensor operations. 0.0 - 100.0.
    pub const ANY_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_ANY_TENSOR_UTIL as _);

    /// Percentage of time the GPU's SMs were doing DFMA tensor operations. 0.0 - 100.0.
    pub const DFMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DFMA_TENSOR_UTIL as _);

    /// Percentage of time the GPU's SMs were doing HMMA tensor operations. 0.0 - 100.0.
    pub const HMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HMMA_TENSOR_UTIL as _);

    /// Percentage of time the GPU's SMs were doing IMMA tensor operations. 0.0 - 100.0.
    pub const IMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_IMMA_TENSOR_UTIL as _);

    /// Percentage of DRAM bw used vs theoretical maximum. 0.0 - 100.0 \*/.
    pub const DRAM_BW_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DRAM_BW_UTIL as _);
    /// Percentage of time the GPU's SMs were doing non-tensor FP64 math. 0.0 - 100.0.
    pub const FP64_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP64_UTIL as _);
    /// Percentage of time the GPU's SMs were doing non-tensor FP32 math. 0.0 - 100.0.
    pub const FP32_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP32_UTIL as _);
    /// Percentage of time the GPU's SMs were doing non-tensor FP16 math. 0.0 - 100.0.
    pub const FP16_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP16_UTIL as _);

    /// `PCIe` traffic from this GPU in MiB/sec.
    pub const PCIE_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PCIE_TX_PER_SEC as _);

    /// `PCIe` traffic to this GPU in MiB/sec.
    pub const PCIE_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PCIE_RX_PER_SEC as _);

    /// `NvLink` read bandwidth for all links in MiB/sec.
    pub const NVLINK_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC as _);

    /// `NvLink` write bandwidth for all links in MiB/sec.
    pub const NVLINK_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC as _);
}

/// Native identifier of a virtual GPU type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeId(
    /// Native integer identifier.
    pub u32,
);

/// Native identifier of a virtual GPU instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuInstanceId(
    /// Native integer identifier.
    pub u32,
);

/// Native identifier of a placement available to a virtual GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuPlacementId(
    /// Native integer identifier.
    pub u32,
);

/// Minimum and maximum vGPU versions reported by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuVersion {
    /// Minimum vGPU version.
    pub min: u32,
    /// Maximum vGPU version.
    pub max: u32,
}

/// Supported and currently selected vGPU version ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuVersionRange {
    /// Version range supported by the installed vGPU software.
    pub supported: VgpuVersion,
    /// Version range currently selected for use.
    pub current: VgpuVersion,
}

/// vGPU GUEST info state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuGuestInfoState {
    /// Guest-dependent fields uninitialized.
    Uninitialized =
        sys::nvmlVgpuGuestInfoState_t::NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED as _,
    /// Guest-dependent fields initialized.
    Initialized =
        sys::nvmlVgpuGuestInfoState_t::NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED as _,
}

impl_enum_conversion!(u32, sys::nvmlVgpuGuestInfoState_t, VgpuGuestInfoState);

bitflags::bitflags! {
    /// vGPU VM compatibility codes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VgpuVmCompatibility: u32 {
        /// vGPU is not runnable.
        const NONE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_NONE as u32;
        /// vGPU is runnable from a cold / powered-off state (ACPI S5).
        const COLD = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_COLD as u32;
        /// vGPU is runnable from a hibernated state (ACPI S4).
        const HIBERNATE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_HIBERNATE as u32;
        /// vGPU is runnable from a sleeped state (ACPI S3).
        const SLEEP = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_SLEEP as u32;
        /// vGPU is runnable from a live/paused (ACPI S0).
        const LIVE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_LIVE as u32;
    }
}

impl Display for VgpuVmCompatibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// vGPU-pGPU compatibility limit codes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VgpuPgpuCompatibilityLimitCode: u32 {
        /// Compatibility is not limited.
        const NONE = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_NONE as u32;
        /// ompatibility is limited by host driver version.
        const HOST_DRIVER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER as u32;
        /// Compatibility is limited by guest driver version.
        const GUEST_DRIVER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER as u32;
        /// Compatibility is limited by GPU hardware.
        const GPU = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_GPU as u32;
        /// Compatibility is limited by an undefined factor.
        const OTHER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_OTHER as u32;
    }
}

impl Display for VgpuPgpuCompatibilityLimitCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// VM compatibility flags and physical-GPU compatibility limits for a vGPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuCompatibility {
    /// VM compatibility flags returned by NVML.
    pub vm_compatibility: VgpuVmCompatibility,
    /// Conditions limiting compatibility with the physical GPU.
    pub limit_code: VgpuPgpuCompatibilityLimitCode,
}

/// Owned virtual GPU metadata used for compatibility checks.
///
/// Opaque native data remains available for serialization back to NVML.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VgpuMetadata {
    /// Native metadata structure version.
    pub version: u32,
    /// Revision of the native metadata payload.
    pub revision: u32,
    /// Availability and state of guest metadata.
    pub guest_info_state: VgpuGuestInfoState,
    /// Driver version reported by the guest.
    pub guest_driver_version: String,
    /// Driver version reported by the host.
    pub host_driver_version: String,
    /// Native virtualization capability mask.
    pub virtualization_caps: u32,
    /// vGPU software version reported by the guest.
    pub guest_vgpu_version: u32,
    /// Opaque NVML metadata bytes preserved for compatibility checks.
    pub opaque_data: Vec<u8>,
}

/// Owned physical GPU metadata used for vGPU compatibility checks.
///
/// Opaque native data remains available for serialization back to NVML.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PgpuMetadata {
    /// Native metadata structure version.
    pub version: u32,
    /// Revision of the native metadata payload.
    pub revision: u32,
    /// Driver version reported by the physical GPU's host.
    pub host_driver_version: String,
    /// Native virtualization capability mask.
    pub virtualization_caps: u32,
    /// vGPU version range supported by the host.
    pub host_supported_vgpu_range: VgpuVersion,
    /// Opaque NVML metadata bytes preserved for compatibility checks.
    pub opaque_data: Vec<u8>,
}

/// Policy for mixing vGPU types when selecting placements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuPlacementMode {
    /// Macros to indicate the vGPU mode of the GPU.
    ///
    /// Heterogeneous vGPU mode.
    Heterogeneous = sys::NVML_VGPU_PGPU_HETEROGENEOUS_MODE,
    /// Homogeneous vGPU mode.
    Homogeneous = sys::NVML_VGPU_PGPU_HOMOGENEOUS_MODE,
}

/// PCI device and subsystem identifiers exposed by a vGPU type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeDeviceId {
    /// PCI device identifier exposed by the vGPU type.
    pub device_id: u64,
    /// PCI subsystem identifier exposed by the vGPU type.
    pub subsystem_id: u64,
}

/// Maximum display resolution for a vGPU type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeResolution {
    /// Maximum horizontal display resolution in pixels.
    pub x: u32,
    /// Maximum vertical display resolution in pixels.
    pub y: u32,
}

/// BAR1 aperture size required by a vGPU type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeBar1Info {
    /// BAR1 size in megabytes.
    pub bar1_size: u64,
}

/// Telemetry field identifier and the scope in which to query it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldQuery {
    /// Telemetry field to query.
    pub field: FieldId,
    /// Scope interpreted by the selected field, such as a link index.
    pub scope_id: u32,
}

impl FieldQuery {
    /// Creates a field query with scope selector zero.
    #[must_use]
    pub const fn new(field: FieldId) -> Self {
        Self { field, scope_id: 0 }
    }

    /// Creates a query with an explicit field-specific scope selector.
    #[must_use]
    pub const fn with_scope(field: FieldId, scope_id: u32) -> Self {
        Self { field, scope_id }
    }
}

/// Generic enable/disable enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EnableState {
    /// Feature disabled.
    Disabled = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED as _,
    /// Feature enabled.
    Enabled = sys::nvmlEnableState_t::NVML_FEATURE_ENABLED as _,
}

impl_enum_conversion!(u32, sys::nvmlEnableState_t, EnableState);

/// Product brand reported by NVML.
///
/// [`Self::Other`] preserves unrecognized native values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Brand {
    /// NVIDIA RTX Virtual Game Dev.
    NvidiaVgamedev,
    /// NVIDIA NPU.
    NvidiaNpu,
    /// Deprecated NVIDIA Deep Learning Accelerator.
    NvidiaDla,
    /// Identifies the unknown product brand.
    Unknown,
    /// Identifies the Quadro product brand.
    Quadro,
    /// Identifies the Tesla product brand.
    Tesla,
    /// Identifies the NVS product brand.
    Nvs,
    /// Legacy GRID brand, no longer reported by NVML.
    Grid,
    /// Identifies the `GeForce` product brand.
    GeForce,
    /// Identifies the TITAN product brand.
    Titan,
    /// NVIDIA Virtual Applications.
    NvidiaVApps,
    /// NVIDIA Virtual PC.
    NvidiaVpc,
    /// NVIDIA vGPU for Compute.
    NvidiaVcs,
    /// NVIDIA RTX Virtual Workstation.
    NvidiaVws,
    /// NVIDIA Cloud Gaming.
    NvidiaCloudGaming,
    /// Identifies the Quadro RTX product brand.
    QuadroRtx,
    /// Identifies the NVIDIA RTX product brand.
    NvidiaRtx,
    /// Identifies the NVIDIA product brand.
    Nvidia,
    /// Unused native `GeForce` RTX brand value.
    GeForceRtx,
    /// Unused native TITAN RTX brand value.
    TitanRtx,
    /// Unrecognized native product brand code, preserved without conversion.
    Other(u32),
}

impl Brand {
    /// Decodes a native product brand while preserving unrecognized codes in [`Self::Other`].
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        match value {
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VGAMEDEV as u32 => {
                Self::NvidiaVgamedev
            }
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_NPU as u32 => Self::NvidiaNpu,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_DLA as u32 => Self::NvidiaDla,
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

/// Temperature sensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TemperatureSensor {
    /// Temperature from the hottest part of the GPU die.
    GpuMax = sys::nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU_MAX as _,
    /// Temperature sensor for the GPU die.
    Gpu = sys::nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU as _,
}

impl_enum_conversion!(u32, sys::nvmlTemperatureSensors_t, TemperatureSensor);

/// Temperature thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TemperatureThreshold {
    /// Temperature at which the GPU will shut down for HW protection.
    Shutdown = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_SHUTDOWN as _,
    /// Temperature at which the GPU will begin HW slowdown.
    Slowdown = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_SLOWDOWN as _,
    /// Memory Temperature at which the GPU will begin SW slowdown.
    MemoryMax = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_MEM_MAX as _,
    /// GPU Temperature at which the GPU can be throttled below base clock.
    GpuMax = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_GPU_MAX as _,
    /// Minimum GPU Temperature that can be set as acoustic threshold.
    AcousticMin = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN as _,
    /// Current temperature that is set as acoustic threshold.
    AcousticCurrent =
        sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR as _,
    /// Maximum GPU temperature that can be set as acoustic threshold.
    AcousticMax = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX as _,
    /// Current temperature that is set as gps threshold.
    GpuCurrent = sys::nvmlTemperatureThresholds_t::NVML_TEMPERATURE_THRESHOLD_GPS_CURR as _,
}

impl_enum_conversion!(u32, sys::nvmlTemperatureThresholds_t, TemperatureThreshold);

/// Compute mode.
///
/// `nvmlComputeMode_t::NVML_COMPUTEMODE_EXCLUSIVE_PROCESS` was added in CUDA 4.0. Earlier CUDA versions supported a single exclusive mode, which is equivalent to `nvmlComputeMode_t::NVML_COMPUTEMODE_EXCLUSIVE_THREAD` in CUDA 4.0 and beyond.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeMode {
    /// Default compute mode , multiple contexts per device.
    Default = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_DEFAULT as _,
    /// Support Removed.
    ExclusiveThread = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_EXCLUSIVE_THREAD as _,
    /// Compute-prohibited mode , no contexts per device.
    Prohibited = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_PROHIBITED as _,
    /// Compute-exclusive-process mode , only one context per device, usable from multiple threads at a time.
    ExclusiveProcess = sys::nvmlComputeMode_t::NVML_COMPUTEMODE_EXCLUSIVE_PROCESS as _,
}

impl_enum_conversion!(u32, sys::nvmlComputeMode_t, ComputeMode);

/// GPU or accelerator architecture reported by NVML.
///
/// [`Self::Other`] preserves unrecognized native values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Architecture {
    /// Npu3.
    Npu3,
    /// Dla2.
    Dla2,
    /// Dla.
    Dla,
    /// NVIDIA Kepler architecture.
    Kepler,
    /// NVIDIA Maxwell architecture.
    Maxwell,
    /// NVIDIA Pascal architecture.
    Pascal,
    /// NVIDIA Volta architecture.
    Volta,
    /// NVIDIA Turing architecture.
    Turing,
    /// NVIDIA Ampere architecture.
    Ampere,
    /// NVIDIA Ada architecture.
    Ada,
    /// NVIDIA Hopper architecture.
    Hopper,
    /// NVIDIA Blackwell architecture.
    Blackwell,
    /// Unrecognized native device architecture code, preserved without conversion.
    Other(u32),
}

impl Architecture {
    /// Decodes a native device architecture while preserving unrecognized codes in [`Self::Other`].
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        match value {
            value if value == sys::NVML_DEVICE_ARCH_NPU3 => Self::Npu3,
            value if value == sys::NVML_DEVICE_ARCH_DLA2 => Self::Dla2,
            value if value == sys::NVML_DEVICE_ARCH_DLA => Self::Dla,
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

/// Clock types.
///
/// All speeds are in Mhz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ClockType {
    /// Graphics clock domain.
    Graphics = sys::nvmlClockType_t::NVML_CLOCK_GRAPHICS as _,
    /// SM clock domain.
    Sm = sys::nvmlClockType_t::NVML_CLOCK_SM as _,
    /// Memory clock domain.
    Memory = sys::nvmlClockType_t::NVML_CLOCK_MEM as _,
    /// Video encoder/decoder clock domain.
    Video = sys::nvmlClockType_t::NVML_CLOCK_VIDEO as _,
}

impl_enum_conversion!(u32, sys::nvmlClockType_t, ClockType);

/// Enum to represent the `NvLink` utilization counter packet units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkUtilizationCountUnits {
    /// count by cycles
    Cycles = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_CYCLES as _,
    /// count by packets
    Packets = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_PACKETS as _,
    /// count by bytes
    Bytes = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_BYTES as _,
    /// count reserved for internal use
    Reserved = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_RESERVED as _,
    /// this must be last
    Count = sys::nvmlNvLinkUtilizationCountUnits_t::NVML_NVLINK_COUNTER_UNIT_COUNT as _,
}

impl_enum_conversion!(
    u32,
    sys::nvmlNvLinkUtilizationCountUnits_t,
    NvLinkUtilizationCountUnits
);

/// Enum to represent `NvLink` queryable capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkCapability {
    /// P2P over `NVLink` is supported
    P2pSupported = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_P2P_SUPPORTED as _,
    /// Access to system memory is supported
    SysmemAccess = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_SYSMEM_ACCESS as _,
    /// P2P atomics are supported
    P2pAtomics = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_P2P_ATOMICS as _,
    /// System memory atomics are supported
    SysmemAtomics = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_SYSMEM_ATOMICS as _,
    /// SLI is supported over this link
    SliBridge = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_SLI_BRIDGE as _,
    /// Link is supported on this device
    Valid = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_VALID as _,
    /// should be last
    Count = sys::nvmlNvLinkCapability_t::NVML_NVLINK_CAP_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlNvLinkCapability_t, NvLinkCapability);

/// Enum to represent `NvLink` queryable error counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkErrorCounter {
    /// Data link transmit replay error counter
    DlReplay = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_REPLAY as _,
    /// Data link transmit recovery error counter
    DlRecovery = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_RECOVERY as _,
    /// Data link receive flow control digit CRC error counter
    DlCrcFlit = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_CRC_FLIT as _,
    /// Data link receive data CRC error counter
    DlCrcData = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_CRC_DATA as _,
    /// Data link receive data ECC error counter
    DlEccData = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_DL_ECC_DATA as _,
    /// this must be last
    Count = sys::nvmlNvLinkErrorCounter_t::NVML_NVLINK_ERROR_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlNvLinkErrorCounter_t, NvLinkErrorCounter);

/// Enum to represent `NvLink`'s remote device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkRemoteDeviceType {
    /// The remote `NVLink` endpoint is a GPU.
    Gpu = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_GPU as _,
    /// The remote `NVLink` endpoint is an IBM NPU.
    IbmNpu = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_IBMNPU as _,
    /// The remote `NVLink` endpoint is a switch.
    Switch = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_SWITCH as _,
    /// The remote `NVLink` endpoint type is unknown.
    Unknown = sys::nvmlIntNvLinkDeviceType_t::NVML_NVLINK_DEVICE_TYPE_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlIntNvLinkDeviceType_t, NvLinkRemoteDeviceType);

/// Represents Nvlink Version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum NvLinkVersion {
    /// `NVLink` version is invalid.
    Invalid = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_INVALID as _,
    /// `NVLink` Version 1.0.
    V1_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_1_0 as _,
    /// `NVLink` Version 2.0.
    V2_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_2_0 as _,
    /// `NVLink` Version 2.2.
    V2_2 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_2_2 as _,
    /// `NVLink` Version 3.0.
    V3_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_3_0 as _,
    /// `NVLink` Version 3.1.
    V3_1 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_3_1 as _,
    /// `NVLink` Version 4.0.
    V4_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_4_0 as _,
    /// `NVLink` Version 5.0.
    V5_0 = sys::nvmlNvlinkVersion_t::NVML_NVLINK_VERSION_5_0 as _,
}

impl_enum_conversion!(u32, sys::nvmlNvlinkVersion_t, NvLinkVersion);

/// Workload power profile identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PowerProfileType {
    /// `LowPriorityBackground`.
    LowPriorityBackground =
        sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_LOW_PRIORITY_BACKGROUND as _,
    /// `NetworkBound`.
    NetworkBound = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_NETWORK_BOUND as _,
    /// Training.
    Training = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_TRAINING as _,
    /// `MediumThroughputInference`.
    MediumThroughputInference =
        sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MEDIUM_THROUGHPUT_INFERENCE as _,
    /// `HighThroughputInference`.
    HighThroughputInference =
        sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_HIGH_THROUGHPUT_INFERENCE as _,
    /// `LowLatencyInference`.
    LowLatencyInference =
        sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_LOW_LATENCY_INFERENCE as _,
    /// Inference.
    Inference = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_INFERENCE as _,
    /// `MaxQ3`.
    MaxQ3 = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX_Q_3 as _,
    /// `MaxQ2`.
    MaxQ2 = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX_Q_2 as _,
    /// `MaxQ1`.
    MaxQ1 = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX_Q_1 as _,
    /// Identifies the native maximum-performance workload power profile. The identifier selects a bit in workload profile masks.
    MaxP = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX_P as _,
    /// Identifies the native Max-Q workload power profile. The identifier selects a bit in workload profile masks.
    MaxQ = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX_Q as _,
    /// Identifies the native compute workload power profile. The identifier selects a bit in workload profile masks.
    Compute = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_COMPUTE as _,
    /// Identifies the native memory-bound workload power profile. The identifier selects a bit in workload profile masks.
    MemoryBound = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MEMORY_BOUND as _,
    /// Identifies the native network workload power profile. The identifier selects a bit in workload profile masks.
    Network = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_NETWORK as _,
    /// Identifies the native balanced workload power profile. The identifier selects a bit in workload profile masks.
    Balanced = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_BALANCED as _,
    /// Identifies the native large-language-model inference workload power profile. The identifier selects a bit in workload profile masks.
    LlmInference = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_LLM_INFERENCE as _,
    /// Identifies the native large-language-model training workload power profile. The identifier selects a bit in workload profile masks.
    LlmTraining = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_LLM_TRAINING as _,
    /// Identifies the native RBM workload power profile. The identifier selects a bit in workload profile masks.
    Rbm = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_RBM as _,
    /// Identifies the native DCPCIE workload power profile. The identifier selects a bit in workload profile masks.
    Dcpcie = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_DCPCIE as _,
    /// Identifies the native sparse HMMA workload power profile. The identifier selects a bit in workload profile masks.
    HmmaSparse = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_HMMA_SPARSE as _,
    /// Identifies the native dense HMMA workload power profile. The identifier selects a bit in workload profile masks.
    HmmaDense = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_HMMA_DENSE as _,
    /// Identifies the native synchronized balanced workload power profile. The identifier selects a bit in workload profile masks.
    SyncBalanced = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_SYNC_BALANCED as _,
    /// Identifies the native high-performance-computing workload power profile. The identifier selects a bit in workload profile masks.
    Hpc = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_HPC as _,
    /// Identifies the native MIG workload power profile. The identifier selects a bit in workload profile masks.
    Mig = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MIG as _,
    /// One past the last workload power profile identifier defined by this header; used for bounds checks.
    Max = sys::nvmlPowerProfileType_t::NVML_POWER_PROFILE_MAX as _,
}

impl_enum_conversion!(u32, sys::nvmlPowerProfileType_t, PowerProfileType);

/// `PowerMizer` policy for device performance and power use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PowerMizerMode {
    /// Device powerMizer modes.
    ///
    /// adjust GPU clocks based on GPU utilization.
    Adaptive = sys::NVML_POWER_MIZER_MODE_ADAPTIVE,
    /// raise GPU clocks to favor maximum performance, to the extent that thermal and other constraints allow.
    PreferMaximumPerformance = sys::NVML_POWER_MIZER_MODE_PREFER_MAXIMUM_PERFORMANCE,
    /// `PowerMizer` mode is driver controlled.
    Auto = sys::NVML_POWER_MIZER_MODE_AUTO,
    /// lock to GPU base clocks.
    PreferConsistentPerformance = sys::NVML_POWER_MIZER_MODE_PREFER_CONSISTENT_PERFORMANCE,
}

/// Current, requested, and supported `PowerMizer` modes for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PowerMizerModes {
    /// Current `PowerMizer` mode reported by the device.
    pub current_mode: u32,
    /// `PowerMizer` mode requested when configuring the device.
    pub mode: u32,
    /// Bit mask of supported `PowerMizer` modes, with bit `1 << mode` set for each supported mode.
    pub supported_modes: u32,
}

/// Identity, priority, and incompatibilities of a workload power profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadPowerProfileInfo {
    /// Performance-profile identifier, such as a compute, memory, or Max-Q profile.
    pub profile_id: u32,
    /// Priority of the profile.
    pub priority: u32,
    /// Identifiers of workload power profiles that cannot be enabled alongside this profile.
    pub conflicting_profiles: Vec<u32>,
}

/// Supported workload power-profile identifiers and their properties.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadPowerProfilesInfo {
    /// Identifiers of the supported workload power profiles.
    pub supported_profiles: Vec<u32>,
    /// Properties of the reported workload power profiles.
    pub profiles: Vec<WorkloadPowerProfileInfo>,
}

/// Supported, requested, and enforced workload power-profile identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadPowerCurrentProfiles {
    /// Identifiers of profiles supported by the device.
    pub supported_profiles: Vec<u32>,
    /// Identifiers of profiles requested by the application.
    pub requested_profiles: Vec<u32>,
    /// Identifiers of profiles currently enforced by the device.
    pub enforced_profiles: Vec<u32>,
}

/// DRAM encryption state reported by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DramEncryptionInfo {
    /// DRAM encryption enablement state.
    pub encryption_state: EnableState,
}

/// Native confidential-computing capability codes for system CPUs and GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeSystemCaps {
    /// Native confidential-computing CPU capability code.
    pub cpu_caps: u32,
    /// Native confidential-computing GPU capability code.
    pub gpus_caps: u32,
}

/// Native confidential-computing environment, feature, and developer-tools state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeSystemState {
    /// Native confidential-computing environment code.
    pub environment: u32,
    /// Native confidential-computing feature state.
    pub cc_feature: u32,
    /// Native developer-tools mode state.
    pub dev_tools_mode: u32,
}

/// Confidential-computing settings, including the multi-GPU mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeSystemSettings {
    /// Native confidential-computing environment code.
    pub environment: u32,
    /// Native confidential-computing feature state.
    pub cc_feature: u32,
    /// Native developer-tools mode state.
    pub dev_tools_mode: u32,
    /// Native multi-GPU confidential-computing mode.
    pub multi_gpu_mode: u32,
}

/// Protected and unprotected confidential-computing memory sizes in kibibytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeMemSizeInfo {
    /// Protected memory size in kibibytes.
    pub protected_mem_size_kib: u64,
    /// Unprotected memory size in kibibytes.
    pub unprotected_mem_size_kib: u64,
}

/// Owned GPU certificate and attestation certificate chains.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfComputeGpuCertificate {
    /// GPU certificate-chain bytes.
    pub cert_chain: Vec<u8>,
    /// Attestation certificate-chain bytes.
    pub attestation_cert_chain: Vec<u8>,
}

/// GPU confidential-computing attestation data and the request nonce.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfComputeGpuAttestationReport {
    /// Whether a CEC attestation report is present.
    pub cec_attestation_report_present: bool,
    /// 32-byte nonce supplied for the attestation request.
    pub nonce: [u8; 32],
    /// GPU attestation report bytes.
    pub attestation_report: Vec<u8>,
    /// CEC attestation report bytes.
    pub cec_attestation_report: Vec<u8>,
}

/// Attacker-advantage threshold used by the native confidential-computing key-rotation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfComputeKeyRotationThreshold {
    /// Maximum attacker-advantage threshold used by the native key-rotation policy.
    pub attacker_advantage: u64,
}

/// Clock Ids.
///
/// These are used in combination with `nvmlClockType_t` to specify a single clock value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ClockId {
    /// Current actual clock value.
    Current = sys::nvmlClockId_t::NVML_CLOCK_ID_CURRENT as _,
    /// Target application clock. Deprecated, do not use.
    #[deprecated]
    AppClockTarget = sys::nvmlClockId_t::NVML_CLOCK_ID_APP_CLOCK_TARGET as _,
    /// Default application clock target Deprecated, do not use.
    #[deprecated]
    AppClockDefault = sys::nvmlClockId_t::NVML_CLOCK_ID_APP_CLOCK_DEFAULT as _,
    /// OEM-defined maximum clock rate.
    CustomerBoostMax = sys::nvmlClockId_t::NVML_CLOCK_ID_CUSTOMER_BOOST_MAX as _,
}

impl_enum_conversion!(u32, sys::nvmlClockId_t, ClockId);

/// Represents the thermal sensor targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ThermalTarget {
    /// No thermal target is selected.
    None = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_NONE as _,
    /// GPU core temperature requires `NvPhysicalGpuHandle`.
    Gpu = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_GPU as _,
    /// GPU memory temperature requires `NvPhysicalGpuHandle`.
    Memory = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_MEMORY as _,
    /// GPU power supply temperature requires `NvPhysicalGpuHandle`.
    PowerSupply = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_POWER_SUPPLY as _,
    /// GPU board ambient temperature requires `NvPhysicalGpuHandle`.
    Board = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_BOARD as _,
    /// Visual Computing Device Board temperature requires `NvVisualComputingDeviceHandle`.
    VcdBoard = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_VCD_BOARD as _,
    /// Visual Computing Device Inlet temperature requires `NvVisualComputingDeviceHandle`.
    VcdInlet = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_VCD_INLET as _,
    /// Visual Computing Device Outlet temperature requires `NvVisualComputingDeviceHandle`.
    VcdOutlet = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_VCD_OUTLET as _,
    /// Selects all thermal targets.
    All = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_ALL as _,
    /// The thermal target is unknown.
    Unknown = sys::nvmlThermalTarget_t::NVML_THERMAL_TARGET_UNKNOWN as _,
}

impl_enum_conversion!(i32, sys::nvmlThermalTarget_t, ThermalTarget);

/// Represents the thermal sensor controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ThermalController {
    /// No thermal controller is present.
    None = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_NONE as _,
    /// The GPU internal thermal controller.
    GpuInternal = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_GPU_INTERNAL as _,
    /// An ADM1032 thermal controller.
    Adm1032 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADM1032 as _,
    /// An ADT7461 thermal controller.
    Adt7461 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADT7461 as _,
    /// A MAX6649 thermal controller.
    Max6649 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_MAX6649 as _,
    /// A MAX1617 thermal controller.
    Max1617 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_MAX1617 as _,
    /// An LM99 thermal controller.
    Lm99 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_LM99 as _,
    /// An LM89 thermal controller.
    Lm89 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_LM89 as _,
    /// An LM64 thermal controller.
    Lm64 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_LM64 as _,
    /// A G781 thermal controller.
    G781 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_G781 as _,
    /// An ADT7473 thermal controller.
    Adt7473 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADT7473 as _,
    /// An SBMAX6649 thermal controller.
    Sbmax6649 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_SBMAX6649 as _,
    /// The VBIOS event thermal controller.
    VbiosEvt = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_VBIOSEVT as _,
    /// The operating-system thermal controller.
    Os = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_OS as _,
    /// The Canoas NV system controller.
    NvsysconCanoas = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS as _,
    /// The E551 NV system controller.
    NvsysconE551 = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_NVSYSCON_E551 as _,
    /// A MAX6649R thermal controller.
    Max6649r = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_MAX6649R as _,
    /// An ADT7473S thermal controller.
    Adt7473s = sys::nvmlThermalController_t::NVML_THERMAL_CONTROLLER_ADT7473S as _,
    /// The thermal controller is unknown.
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

/// Enum to represent type of bridge chip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BridgeChipType {
    /// Identifies a PLX bridge chip.
    Plx = sys::nvmlBridgeChipType_t::NVML_BRIDGE_CHIP_PLX as _,
    /// Identifies a BRO4 bridge chip.
    Bro4 = sys::nvmlBridgeChipType_t::NVML_BRIDGE_CHIP_BRO4 as _,
}

impl_enum_conversion!(u32, sys::nvmlBridgeChipType_t, BridgeChipType);

/// Allowed `PStates`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PerformanceState {
    /// Performance state 0 , Maximum Performance.
    P0 = sys::nvmlPstates_t::NVML_PSTATE_0 as _,
    /// Performance state 1.
    P1 = sys::nvmlPstates_t::NVML_PSTATE_1 as _,
    /// Performance state 2.
    P2 = sys::nvmlPstates_t::NVML_PSTATE_2 as _,
    /// Performance state 3.
    P3 = sys::nvmlPstates_t::NVML_PSTATE_3 as _,
    /// Performance state 4.
    P4 = sys::nvmlPstates_t::NVML_PSTATE_4 as _,
    /// Performance state 5.
    P5 = sys::nvmlPstates_t::NVML_PSTATE_5 as _,
    /// Performance state 6.
    P6 = sys::nvmlPstates_t::NVML_PSTATE_6 as _,
    /// Performance state 7.
    P7 = sys::nvmlPstates_t::NVML_PSTATE_7 as _,
    /// Performance state 8.
    P8 = sys::nvmlPstates_t::NVML_PSTATE_8 as _,
    /// Performance state 9.
    P9 = sys::nvmlPstates_t::NVML_PSTATE_9 as _,
    /// Performance state 10.
    P10 = sys::nvmlPstates_t::NVML_PSTATE_10 as _,
    /// Performance state 11.
    P11 = sys::nvmlPstates_t::NVML_PSTATE_11 as _,
    /// Performance state 12.
    P12 = sys::nvmlPstates_t::NVML_PSTATE_12 as _,
    /// Performance state 13.
    P13 = sys::nvmlPstates_t::NVML_PSTATE_13 as _,
    /// Performance state 14.
    P14 = sys::nvmlPstates_t::NVML_PSTATE_14 as _,
    /// Performance state 15 , Minimum Performance.
    P15 = sys::nvmlPstates_t::NVML_PSTATE_15 as _,
    /// Unknown performance state.
    Unknown = sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlPstates_t, PerformanceState);

/// Represents the queryable `PCIe` utilization counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PcieUtilCounter {
    /// 1KB granularity
    TxBytes = sys::nvmlPcieUtilCounter_t::NVML_PCIE_UTIL_TX_BYTES as _,
    /// 1KB granularity
    RxBytes = sys::nvmlPcieUtilCounter_t::NVML_PCIE_UTIL_RX_BYTES as _,
}

impl_enum_conversion!(u32, sys::nvmlPcieUtilCounter_t, PcieUtilCounter);

/// Represents level relationships within a system between two GPUs The enums are spaced to allow for future relationships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TopologyLevel {
    /// e.g. Tesla K80
    Internal = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_INTERNAL as _,
    /// all devices that only need traverse a single `PCIe` switch
    Single = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_SINGLE as _,
    /// all devices that need not traverse a host bridge
    Multiple = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_MULTIPLE as _,
    /// all devices that are connected to the same host bridge
    HostBridge = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_HOSTBRIDGE as _,
    /// all devices that are connected to the same NUMA node but possibly multiple host bridges
    Node = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_NODE as _,
    /// all devices in the system
    System = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_SYSTEM as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuTopologyLevel_t, TopologyLevel);

/// P2P Capability Index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum P2pCapabilityIndex {
    /// Selects peer-to-peer read capability for a P2P status query.
    Read = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_READ as _,
    /// Selects peer-to-peer write capability for a P2P status query.
    Write = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_WRITE as _,
    /// Selects `NVLink` peer-to-peer capability for a P2P status query.
    Nvlink = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_NVLINK as _,
    /// Selects peer-to-peer atomic-operation capability for a P2P status query.
    Atomics = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_ATOMICS as _,
    /// Deprecated name for the PCI peer-to-peer capability selector.
    Prop = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_PROP as _,
    /// Unknown peer-to-peer capability index.
    Unknown = sys::nvmlGpuP2PCapsIndex_t::NVML_P2P_CAPS_INDEX_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuP2PCapsIndex_t, P2pCapabilityIndex);

/// P2P Capability Index Status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum P2pStatus {
    /// The requested peer-to-peer capability is available.
    Ok = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_OK as _,
    /// The chipset does not support the requested peer-to-peer operation.
    ChipsetNotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED as _,
    /// The GPU does not support the requested peer-to-peer capability.
    GpuNotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_GPU_NOT_SUPPORTED as _,
    /// The I/O hub topology does not support the requested peer-to-peer capability.
    IoHubNotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED as _,
    /// The requested peer-to-peer capability is disabled by registry configuration.
    DisabledByRegKey = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_DISABLED_BY_REGKEY as _,
    /// The requested peer-to-peer capability is unsupported.
    NotSupported = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_NOT_SUPPORTED as _,
    /// The status of the requested peer-to-peer capability is unknown.
    Unknown = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuP2PStatus_t, P2pStatus);

/// Available infoROM objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum InforomObject {
    /// An object defined by OEM.
    Oem = sys::nvmlInforomObject_t::NVML_INFOROM_OEM as _,
    /// The ECC object determining the level of ECC support.
    Ecc = sys::nvmlInforomObject_t::NVML_INFOROM_ECC as _,
    /// The power management object.
    Power = sys::nvmlInforomObject_t::NVML_INFOROM_POWER as _,
    /// DRAM Encryption object.
    Den = sys::nvmlInforomObject_t::NVML_INFOROM_DEN as _,
}

impl_enum_conversion!(u32, sys::nvmlInforomObject_t, InforomObject);

/// Memory error types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryErrorType {
    /// A memory error that was corrected.
    ///
    /// For ECC errors, these are single bit errors For Texture memory, these are errors fixed by resend.
    Corrected = sys::nvmlMemoryErrorType_t::NVML_MEMORY_ERROR_TYPE_CORRECTED as _,
    /// A memory error that was not corrected.
    ///
    /// For ECC errors, these are double bit errors For Texture memory, these are errors where the resend fails.
    Uncorrected = sys::nvmlMemoryErrorType_t::NVML_MEMORY_ERROR_TYPE_UNCORRECTED as _,
}

impl_enum_conversion!(u32, sys::nvmlMemoryErrorType_t, MemoryErrorType);

/// ECC counter types.
///
/// Note: Volatile counts are reset each time the driver loads. On Windows this is once per boot. On Linux this can be more frequent. On Linux the driver unloads when no active clients exist. If persistence mode is enabled or there is always a driver client active (e.g. X11), then Linux also sees per-boot behavior. If not, volatile counts are reset each time a compute app is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EccCounterType {
    /// Volatile counts are reset each time the driver loads.
    Volatile = sys::nvmlEccCounterType_t::NVML_VOLATILE_ECC as _,
    /// Aggregate counts persist across reboots (i.e. for the lifetime of the device).
    Aggregate = sys::nvmlEccCounterType_t::NVML_AGGREGATE_ECC as _,
}

impl_enum_conversion!(u32, sys::nvmlEccCounterType_t, EccCounterType);

/// See `nvmlDeviceGetMemoryErrorCounter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryLocation {
    /// GPU L1 Cache.
    L1Cache = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_L1_CACHE as _,
    /// GPU L2 Cache.
    L2Cache = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_L2_CACHE as _,
    /// Turing+ DRAM.
    Dram = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_DRAM as _,
    /// GPU Register File.
    RegisterFile = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_REGISTER_FILE as _,
    /// GPU Texture Memory.
    TextureMemory = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_TEXTURE_MEMORY as _,
    /// Shared memory.
    TextureShm = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_TEXTURE_SHM as _,
    /// CBU.
    Cbu = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_CBU as _,
    /// Turing+ SRAM.
    Sram = sys::nvmlMemoryLocation_t::NVML_MEMORY_LOCATION_SRAM as _,
}

impl_enum_conversion!(u32, sys::nvmlMemoryLocation_t, MemoryLocation);

/// Represents type of perf policy for which violation times can be queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PerfPolicyType {
    /// How long did power violations cause the GPU to be below application clocks.
    Power = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_POWER as _,
    /// How long did thermal violations cause the GPU to be below application clocks.
    Thermal = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_THERMAL as _,
    /// How long did sync boost cause the GPU to be below application clocks.
    SyncBoost = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_SYNC_BOOST as _,
    /// How long did the board limit cause the GPU to be below application clocks.
    BoardLimit = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_BOARD_LIMIT as _,
    /// How long did low utilization cause the GPU to be below application clocks.
    LowUtilization = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_LOW_UTILIZATION as _,
    /// How long did the board reliability limit cause the GPU to be below application clocks.
    Reliability = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_RELIABILITY as _,
    /// Total time the GPU was held below application clocks by any limiter (0 - 5 above).
    TotalAppClocks = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_TOTAL_APP_CLOCKS as _,
    /// Total time the GPU was held below base clocks.
    TotalBaseClocks = sys::nvmlPerfPolicyType_t::NVML_PERF_POLICY_TOTAL_BASE_CLOCKS as _,
}

impl_enum_conversion!(u32, sys::nvmlPerfPolicyType_t, PerfPolicyType);

/// Causes for page retirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PageRetirementCause {
    /// Page was retired due to multiple single bit ECC error.
    MultipleSingleBitEccErrors =
        sys::nvmlPageRetirementCause_t::NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS
            as _,
    /// Page was retired due to double bit ECC error.
    DoubleBitEccError =
        sys::nvmlPageRetirementCause_t::NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR as _,
}

impl_enum_conversion!(u32, sys::nvmlPageRetirementCause_t, PageRetirementCause);

/// Represents the type for sample value returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FieldValueType {
    /// Double-precision floating-point value
    Double = sys::nvmlValueType_t::NVML_VALUE_TYPE_DOUBLE as _,
    /// Unsigned 32-bit integer value
    UnsignedInt = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_INT as _,
    /// Unsigned long integer value
    UnsignedLong = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_LONG as _,
    /// Unsigned 64-bit integer value
    UnsignedLongLong = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_LONG_LONG as _,
    /// Signed 64-bit integer value
    SignedLongLong = sys::nvmlValueType_t::NVML_VALUE_TYPE_SIGNED_LONG_LONG as _,
    /// Signed 32-bit integer value
    SignedInt = sys::nvmlValueType_t::NVML_VALUE_TYPE_SIGNED_INT as _,
    /// Unsigned 16-bit integer value
    UnsignedShort = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_SHORT as _,
}

impl_enum_conversion!(u32, sys::nvmlValueType_t, FieldValueType);

/// Represents Type of Sampling Event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SamplingType {
    /// To represent total power drawn by GPU.
    TotalPower = sys::nvmlSamplingType_t::NVML_TOTAL_POWER_SAMPLES as _,
    /// To represent percent of time during which one or more kernels was executing on the GPU.
    GpuUtilization = sys::nvmlSamplingType_t::NVML_GPU_UTILIZATION_SAMPLES as _,
    /// To represent percent of time during which global (device) memory was being read or written.
    MemoryUtilization = sys::nvmlSamplingType_t::NVML_MEMORY_UTILIZATION_SAMPLES as _,
    /// To represent percent of time during which NVENC remains busy.
    EncoderUtilization = sys::nvmlSamplingType_t::NVML_ENC_UTILIZATION_SAMPLES as _,
    /// To represent percent of time during which NVDEC remains busy.
    DecoderUtilization = sys::nvmlSamplingType_t::NVML_DEC_UTILIZATION_SAMPLES as _,
    /// To represent processor clock samples.
    ProcessorClock = sys::nvmlSamplingType_t::NVML_PROCESSOR_CLK_SAMPLES as _,
    /// To represent memory clock samples.
    MemoryClock = sys::nvmlSamplingType_t::NVML_MEMORY_CLK_SAMPLES as _,
    /// To represent module power samples for total module starting Grace Hopper.
    ModulePower = sys::nvmlSamplingType_t::NVML_MODULE_POWER_SAMPLES as _,
    /// To represent percent of time during which NVJPG remains busy.
    JpgUtilization = sys::nvmlSamplingType_t::NVML_JPG_UTILIZATION_SAMPLES as _,
    /// To represent percent of time during which NVOFA remains busy.
    OfaUtilization = sys::nvmlSamplingType_t::NVML_OFA_UTILIZATION_SAMPLES as _,
}

impl_enum_conversion!(u32, sys::nvmlSamplingType_t, SamplingType);

/// Enum to represent process mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ProcessMode {
    /// Processes with a compute context.
    Compute = sys::nvmlProcessMode_enum::NVML_PROCESS_MODE_COMPUTE as _,
    /// Processes with a graphics context.
    Graphics = sys::nvmlProcessMode_enum::NVML_PROCESS_MODE_GRAPHICS as _,
    /// Processes with a MPS (Multi-Process Service) compute context.
    MpsCompute = sys::nvmlProcessMode_enum::NVML_PROCESS_MODE_MPS as _,
}

impl_enum_display!(ProcessMode, {
    ProcessMode::Compute => "NVML_PROCESS_MODE_COMPUTE",
    ProcessMode::Graphics => "NVML_PROCESS_MODE_GRAPHICS",
    ProcessMode::MpsCompute => "NVML_PROCESS_MODE_MPS",
});

/// CPU grouping used for device-affinity queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AffinityScope {
    /// Scope of NUMA node for affinity queries.
    Node = sys::NVML_AFFINITY_SCOPE_NODE,
    /// Scope of processor socket for affinity queries.
    Socket = sys::NVML_AFFINITY_SCOPE_SOCKET,
}

/// Bus connection type reported by NVML.
///
/// [`Self::Other`] preserves unrecognized native values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BusType {
    /// Unknown bus type.
    Unknown,
    /// PCI bus.
    Pci,
    /// PCI-Express bus.
    Pcie,
    /// FPCI bus.
    Fpci,
    /// AGP bus.
    Agp,
    /// Unrecognized native bus type code, preserved without conversion.
    Other(u32),
}

impl BusType {
    /// Decodes a native bus type while preserving unrecognized codes in [`Self::Other`].
    #[must_use]
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

/// Power source reported for a device.
///
/// [`Self::Other`] preserves unrecognized native values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PowerSource {
    /// AC power source.
    Ac,
    /// Battery power source.
    Battery,
    /// Power source with insufficient capacity for the device.
    Undersized,
    /// Unrecognized native power source code, preserved without conversion.
    Other(u32),
}

impl PowerSource {
    /// Decodes a native power source while preserving unrecognized codes in [`Self::Other`].
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        match value {
            sys::NVML_POWER_SOURCE_AC => Self::Ac,
            sys::NVML_POWER_SOURCE_BATTERY => Self::Battery,
            sys::NVML_POWER_SOURCE_UNDERSIZED => Self::Undersized,
            raw => Self::Other(raw),
        }
    }
}

/// Represents type of encoder for capacity can be queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EncoderType {
    /// H264 encoder.
    H264 = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_H264 as _,
    /// HEVC encoder.
    Hevc = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_HEVC as _,
    /// AV1 encoder.
    Av1 = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_AV1 as _,
    /// Unknown encoder.
    Unknown = sys::nvmlEncoderType_t::NVML_ENCODER_QUERY_UNKNOWN as _,
}

impl_enum_conversion!(u32, sys::nvmlEncoderType_t, EncoderType);

/// API types that allow changes to default permission restrictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum RestrictedApi {
    /// APIs that change application clocks, see nvmlDeviceSetApplicationsClocks and see nvmlDeviceResetApplicationsClocks. Deprecated, keeping definition for backward compatibility.
    #[deprecated]
    SetApplicationsClocks =
        sys::nvmlRestrictedAPI_t::NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS as _,
    /// APIs that enable/disable Auto Boosted clocks see nvmlDeviceSetAutoBoostedClocksEnabled.
    SetAutoBoostedClocks =
        sys::nvmlRestrictedAPI_t::NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS as _,
    /// Keep this last
    Count = sys::nvmlRestrictedAPI_t::NVML_RESTRICTED_API_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlRestrictedAPI_t, RestrictedApi);

/// Adaptive-clocking enablement reported by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AdaptiveClockInfoStatus {
    /// Adaptive clocking status.
    ///
    /// Adaptive clocking is disabled.
    Disabled = sys::NVML_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED,
    /// Adaptive clocking is enabled.
    Enabled = sys::NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED,
}

/// Fan speed control policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FanPolicy {
    /// Device Power Modes.
    ///
    /// Device Fan control policy Temperature-controlled fan policy.
    TemperatureContinuousSw = sys::NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW,
    /// Manual fan control policy.
    Manual = sys::NVML_FAN_POLICY_MANUAL,
}

/// Fan state enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FanState {
    /// Fan is working properly.
    Normal = sys::nvmlFanState_t::NVML_FAN_NORMAL as _,
    /// Fan has failed.
    Failed = sys::nvmlFanState_t::NVML_FAN_FAILED as _,
}

impl_enum_conversion!(u32, sys::nvmlFanState_t, FanState);

/// Cooler control type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CoolerControl {
    /// This cooler has no control signal.
    None = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_NONE as _,
    /// This cooler can only be toggled either ON or OFF (eg a switch).
    Toggle = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_TOGGLE as _,
    /// This cooler's level can be adjusted from some minimum to some maximum (eg a knob).
    Variable = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_VARIABLE as _,
    /// Keep this last
    Count = sys::nvmlCoolerControl_t::NVML_THERMAL_COOLER_SIGNAL_COUNT as _,
}

impl_enum_conversion!(u32, sys::nvmlCoolerControl_t, CoolerControl);

/// Cooler's target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CoolerTarget {
    /// This cooler cools nothing.
    None = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_NONE as _,
    /// This cooler can cool the GPU.
    Gpu = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_GPU as _,
    /// This cooler can cool the memory.
    Memory = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_MEMORY as _,
    /// This cooler can cool the power supply.
    PowerSupply = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_POWER_SUPPLY as _,
    /// This cooler cools all of the components related to its target gpu. `GPU_RELATED` = GPU | MEMORY | `POWER_SUPPLY`.
    GpuRelated = sys::nvmlCoolerTarget_t::NVML_THERMAL_COOLER_TARGET_GPU_RELATED as _,
}

impl_enum_conversion!(u32, sys::nvmlCoolerTarget_t, CoolerTarget);

/// Represents frame buffer capture session type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FbcSessionType {
    /// Unknown.
    Unknown = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_UNKNOWN as _,
    /// `ToSys`.
    ToSys = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_TOSYS as _,
    /// CUDA.
    Cuda = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_CUDA as _,
    /// Vid.
    Vid = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_VID as _,
    /// `HEnc`.
    HwEnc = sys::nvmlFBCSessionType_t::NVML_FBC_SESSION_TYPE_HWENC as _,
}

impl_enum_conversion!(u32, sys::nvmlFBCSessionType_t, FbcSessionType);

/// Driver models.
///
/// Windows only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DriverModel {
    /// WDDM driver model , GPU treated as a display device.
    Wddm = sys::nvmlDriverModel_t::NVML_DRIVER_WDDM as _,
    /// WDM (TCC) model (deprecated) , GPU treated as a generic compute device.
    #[deprecated]
    Wdm = sys::nvmlDriverModel_t::NVML_DRIVER_WDM as _,
    /// MCDM driver model , GPU treated as a Microsoft compute device.
    Mcdm = sys::nvmlDriverModel_t::NVML_DRIVER_MCDM as _,
}

impl_enum_conversion!(u32, sys::nvmlDriverModel_t, DriverModel);

/// GPU Operation Mode.
///
/// GOM allows to reduce power usage and optimize GPU throughput by disabling GPU features.
///
/// Each GOM is designed to meet specific user needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuOperationMode {
    /// Everything is enabled and running at full speed.
    AllOn = sys::nvmlGpuOperationMode_t::NVML_GOM_ALL_ON as _,
    /// Designed for running only compute tasks. Graphics operations are not allowed.
    Compute = sys::nvmlGpuOperationMode_t::NVML_GOM_COMPUTE as _,
    /// Designed for running graphics applications that don't require high bandwidth double precision.
    LowDp = sys::nvmlGpuOperationMode_t::NVML_GOM_LOW_DP as _,
}

impl_enum_conversion!(u32, sys::nvmlGpuOperationMode_t, GpuOperationMode);

/// Multi-instance GPU partitioning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MigMode {
    /// Disable Multi Instance GPU mode.
    ///
    /// Disable Multi Instance GPU mode.
    Disabled = sys::NVML_DEVICE_MIG_DISABLE,
    /// Enable Multi Instance GPU mode.
    ///
    /// Enable Multi Instance GPU mode.
    Enabled = sys::NVML_DEVICE_MIG_ENABLE,
}

/// Outcome of activating a requested MIG mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigModeActivation {
    /// Success or the native error that prevented immediate mode activation.
    pub status: Result<()>,
}

impl MigModeActivation {
    /// Wraps a native activation status, preserving activation failures in [`Self::status`].
    #[must_use]
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

/// GPU virtualization mode reported by NVML.
///
/// [`Self::Other`] preserves unrecognized native values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VirtualizationMode {
    /// Bare-metal GPU without GPU virtualization.
    None,
    /// GPU assigned directly to a virtual machine.
    Passthrough,
    /// Virtual GPU exposed inside a virtual machine.
    Vgpu,
    /// GPU managed by a VGX hypervisor in vGPU mode.
    HostVgpu,
    /// GPU managed by a VGX hypervisor in vSGA mode.
    HostVsga,
    /// Unrecognized native GPU virtualization mode code, preserved without conversion.
    Other(u32),
}

impl VirtualizationMode {
    /// Decodes a native GPU virtualization mode while preserving unrecognized codes in [`Self::Other`].
    #[must_use]
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

/// Host vGPU modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum HostVgpuMode {
    /// Non SR-IOV mode.
    NonSriov = sys::nvmlHostVgpuMode_t::NVML_HOST_VGPU_MODE_NON_SRIOV as _,
    /// SR-IOV mode.
    Sriov = sys::nvmlHostVgpuMode_t::NVML_HOST_VGPU_MODE_SRIOV as _,
}

impl_enum_conversion!(u32, sys::nvmlHostVgpuMode_t, HostVgpuMode);

/// vGPU driver queryable capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuDriverCapability {
    /// Supports mixing of different vGPU profiles within one guest VM.
    HeterogeneousMultiVgpu =
        sys::nvmlVgpuDriverCapability_t::NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU as _,
    /// Supports FSR and warm update of vGPU host driver without terminating the running guest VM.
    WarmUpdate = sys::nvmlVgpuDriverCapability_t::NVML_VGPU_DRIVER_CAP_WARM_UPDATE as _,
}

impl_enum_conversion!(u32, sys::nvmlVgpuDriverCapability_t, VgpuDriverCapability);

/// Types of VM identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuVmIdType {
    /// VM ID represents DOMAIN ID.
    DomainId = sys::nvmlVgpuVmIdType_t::NVML_VGPU_VM_ID_DOMAIN_ID as _,
    /// VM ID represents UUID.
    Uuid = sys::nvmlVgpuVmIdType_t::NVML_VGPU_VM_ID_UUID as _,
}

impl_enum_conversion!(u32, sys::nvmlVgpuVmIdType_t, VgpuVmIdType);

/// Virtual-machine identifier and its NVML identifier kind.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VgpuVmId {
    /// Virtual-machine identifier in the format selected by `kind`.
    pub value: String,
    /// Native kind of the virtual-machine identifier.
    pub kind: VgpuVmIdType,
}

/// Calendar expiry time and expiry status of a vGPU license.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuLicenseExpiry {
    /// Year of license expiry.
    pub year: u32,
    /// Month of license expiry.
    pub month: u16,
    /// Day of license expiry.
    pub day: u16,
    /// Hour of license expiry.
    pub hour: u16,
    /// Minutes of license expiry.
    pub minute: u16,
    /// Seconds of license expiry.
    pub second: u16,
    /// License expiry status.
    pub status: u8,
}

/// vGPU licensing state and license expiry information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuLicenseInfo {
    /// Whether the vGPU instance has an active license.
    pub is_licensed: bool,
    /// License expiry time and status.
    pub expiry: VgpuLicenseExpiry,
    /// Current license state.
    pub current_state: u32,
}

/// Reported size of a vGPU instance's runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuRuntimeState {
    /// The runtime state size of the vGPU instance.
    pub size: u64,
}

/// Policy for sharing GPU execution time among vGPU instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuSchedulerPolicy {
    /// Best effort scheduler policy.
    BestEffort = sys::NVML_VGPU_SCHEDULER_POLICY_BEST_EFFORT,
    /// Equal share scheduler policy.
    EqualShare = sys::NVML_VGPU_SCHEDULER_POLICY_EQUAL_SHARE,
    /// Fixed share scheduler policy.
    FixedShare = sys::NVML_VGPU_SCHEDULER_POLICY_FIXED_SHARE,
}

/// Legacy adaptive round-robin mode for vGPU scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuSchedulerArrMode {
    /// Default Adaptive Round Robin mode.
    Default = sys::NVML_VGPU_SCHEDULER_ARR_DEFAULT,
    /// Disable Adaptive Round Robin mode.
    Disabled = sys::NVML_VGPU_SCHEDULER_ARR_DISABLE,
    /// Enable Adaptive Round Robin mode.
    Enabled = sys::NVML_VGPU_SCHEDULER_ARR_ENABLE,
}

/// Engine whose vGPU scheduling state is queried or configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum VgpuSchedulerEngine {
    /// vGPU scheduler engine types A GPU or GI may support a subset of engines
    ///
    /// Graphics engine.
    Graphics = sys::NVML_VGPU_SCHEDULER_ENGINE_TYPE_GRAPHICS,
}

/// vGPU scheduling time slice and optional adaptive round-robin averaging factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerParams {
    /// Configured or default software-runlist time slice in nanoseconds, or zero when scheduling is inactive.
    pub timeslice: u32,
    /// Adaptive round-robin time-slice compensation factor, when reported.
    pub avg_factor: Option<u32>,
}

/// vGPU scheduling engine, policy, adaptive round-robin mode, and parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerState {
    /// Engine whose scheduling state is reported.
    pub engine: VgpuSchedulerEngine,
    /// GPU-time sharing policy.
    pub policy: VgpuSchedulerPolicy,
    /// Adaptive round-robin scheduling mode.
    pub arr_mode: VgpuSchedulerArrMode,
    /// Time-slice and averaging parameters.
    pub params: VgpuSchedulerParams,
}

/// Timing information for a preempted vGPU software runlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerLogEntry {
    /// Timestamp in ns when this software runlist was preempted.
    pub timestamp: u64,
    /// Total time in ns this software runlist has run.
    pub time_run_total: u64,
    /// Time in ns this software runlist ran before preemption.
    pub time_run: u64,
    /// Software runlist Id.
    pub sw_runlist_id: u32,
    /// The actual timeslice after deduction.
    pub target_time_slice: u64,
    /// Preemption time in ns for this SW runlist.
    pub cumulative_preemption_time: u64,
}

/// Scheduler state and recent vGPU software-runlist timing records.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VgpuSchedulerLog {
    /// Scheduler state associated with the log.
    pub state: VgpuSchedulerState,
    /// Software-runlist timing records reported by the scheduler.
    pub entries: Vec<VgpuSchedulerLogEntry>,
}

/// Capability selector for querying vGPU support on a physical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceVgpuCapability(sys::nvmlDeviceVgpuCapability_t);

impl From<DeviceVgpuCapability> for sys::nvmlDeviceVgpuCapability_t {
    fn from(value: DeviceVgpuCapability) -> Self {
        value.0
    }
}

impl DeviceVgpuCapability {
    /// Query whether the vGPU profiles on the GPU supports migration data streaming.
    pub const DEVICE_STREAMING: Self =
        Self(sys::nvmlDeviceVgpuCapability_t::NVML_DEVICE_VGPU_CAP_DEVICE_STREAMING);

    /// Query whether the GPU supports FSR and warm update.
    pub const WARM_UPDATE: Self =
        Self(sys::nvmlDeviceVgpuCapability_t::NVML_DEVICE_VGPU_CAP_WARM_UPDATE);
}

/// Capability selector for querying a vGPU type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VgpuTypeCapability(sys::nvmlVgpuCapability_t);

impl From<VgpuTypeCapability> for sys::nvmlVgpuCapability_t {
    fn from(value: VgpuTypeCapability) -> Self {
        value.0
    }
}

impl VgpuTypeCapability {
    /// P2P over `NVLink` is supported.
    pub const NVLINK_P2P: Self = Self(sys::nvmlVgpuCapability_t::NVML_VGPU_CAP_NVLINK_P2P);
    /// `GPUDirect` capability is supported.
    pub const GPUDIRECT: Self = Self(sys::nvmlVgpuCapability_t::NVML_VGPU_CAP_GPUDIRECT);
}

/// Engine-sharing profile for a MIG compute instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeInstanceEngineProfile {
    /// All the engines except multiprocessors would be shared.
    Shared = sys::NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED,
    /// Number of engine profiles.
    Count = sys::NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT,
}

/// Current and pending values of a device setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrentPending<T> {
    /// Value currently in effect.
    pub current: T,
    /// Value pending activation.
    pub pending: T,
}

/// Current and default GPU System Processor firmware enablement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GspFirmwareMode {
    /// Whether GPU System Processor firmware is currently enabled.
    pub enabled: bool,
    /// Whether GPU System Processor firmware is enabled by default.
    pub default_mode: bool,
}

/// Current and default automatic boost-clock policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AutoBoostClocks {
    /// Current automatic boost-clock policy.
    pub enabled: EnableState,
    /// Default automatic boost-clock policy.
    pub default_enabled: EnableState,
}

/// CUDA compute capability reported as major and minor components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    /// CUDA compute-capability major version.
    pub major: i32,
    /// CUDA compute-capability minor version.
    pub minor: i32,
}

/// GPU execution and memory activity over a device sampling period.
///
/// The sampling period varies by product, from one second to one sixth of a second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Utilization {
    /// Percent of time over the past sample period during which one or more kernels was executing on the GPU.
    pub gpu: u32,
    /// Percent of time over the past sample period during which global (device) memory was being read or written.
    pub memory: u32,
}

/// Physical GPU memory totals, including system reservations and current allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryInfo {
    /// Total physical device memory (in bytes).
    pub total: u64,
    /// Device memory (in bytes) reserved for system use (driver or firmware).
    pub reserved: u64,
    /// Unallocated device memory (in bytes).
    pub free: u64,
    /// Allocated device memory (in bytes).
    pub used: u64,
}

/// PCI address and device identifiers for a GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PciInfo {
    /// PCI address in the legacy bus identifier format.
    pub bus_id_legacy: String,
    /// PCI address in the current bus identifier format.
    pub bus_id: String,
    /// The PCI domain on which the device's bus resides, 0 to 0xffffffff.
    pub domain: u32,
    /// The bus on which the device resides, 0 to 0xff.
    pub bus: u32,
    /// The device's id on the bus, 0 to 31.
    pub device: u32,
    /// The combined 16-bit device id and 16-bit vendor id.
    pub pci_device_id: u32,
    /// The 32-bit Sub System Device ID.
    pub pci_subsystem_id: u32,
}

/// PCI address, device identifiers, and class codes for a GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PciInfoExt {
    /// PCI address in the current bus identifier format.
    pub bus_id: String,
    /// The PCI domain on which the device's bus resides, 0 to 0xffffffff.
    pub domain: u32,
    /// The bus on which the device resides, 0 to 0xff.
    pub bus: u32,
    /// The device's id on the bus, 0 to 31.
    pub device: u32,
    /// The combined 16-bit device id and 16-bit vendor id.
    pub pci_device_id: u32,
    /// The 32-bit Sub System Device ID.
    pub pci_subsystem_id: u32,
    /// The 8-bit PCI base class code.
    pub base_class: u32,
    /// The 8-bit PCI sub class code.
    pub sub_class: u32,
}

/// PCI identity and UUID of a device excluded from normal NVML enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExcludedDeviceInfo {
    /// PCI identity of the excluded device.
    pub pci_info: PciInfo,
    /// UUID of the excluded device.
    pub uuid: String,
}

/// HWBC identifier and firmware version reported by NVML.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HwbcEntry {
    /// Native HWBC identifier.
    pub hwbc_id: u32,
    /// Firmware version reported for the HWBC.
    pub firmware_version: String,
}

/// GPU Performance Monitoring support reported for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpmSupport {
    /// Whether the device supports GPU Performance Monitoring.
    pub is_supported_device: bool,
}

/// GPU Performance Monitoring metric result, names, and unit label.
#[derive(Debug, Clone, PartialEq)]
pub struct GpmMetric {
    /// Metric identifier requested from NVML.
    pub metric_id: GpmMetricId,
    /// Metric value or the per-metric NVML error.
    pub result: Result<f64>,
    /// Short metric name reported by NVML.
    pub short_name: String,
    /// Long metric name reported by NVML.
    pub long_name: String,
    /// Unit label associated with the metric value.
    pub unit: String,
}

/// Led color enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum LedColor {
    /// GREEN, indicates good health.
    Green = sys::nvmlLedColor_t::NVML_LED_COLOR_GREEN as _,
    /// AMBER, indicates problem.
    Amber = sys::nvmlLedColor_t::NVML_LED_COLOR_AMBER as _,
}

impl_enum_conversion!(u32, sys::nvmlLedColor_t, LedColor);

/// Unit LED color and the reported cause of an amber indication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LedState {
    /// Cause of an amber LED indication, when NVML supplies one.
    pub cause: String,
    /// Current unit LED color.
    pub color: LedColor,
}

/// Name, identifiers, and firmware version of a multi-GPU unit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnitInfo {
    /// Unit name reported by NVML.
    pub name: String,
    /// Unit identifier reported by NVML.
    pub id: String,
    /// Unit serial number.
    pub serial: String,
    /// Unit firmware version.
    pub firmware_version: String,
}

/// Power-supply state and electrical measurements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PsuInfo {
    /// Power-supply state reported by NVML.
    pub state: String,
    /// PSU current (A).
    pub current: u32,
    /// PSU voltage (V).
    pub voltage: u32,
    /// PSU power draw (W).
    pub power: u32,
}

/// Unit fan speed and operating state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitFanInfo {
    /// Fan speed in revolutions per minute.
    pub speed: u32,
    /// Fan operating state.
    pub state: FanState,
}

/// Fan measurements for a multi-GPU unit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnitFanSpeeds {
    /// Speed and state of each reported unit fan.
    pub fans: Vec<UnitFanInfo>,
}

/// Temperature sensor location in a multi-GPU unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum UnitTemperatureType {
    /// Unit intake-air temperature.
    Intake = 0,
    /// Unit exhaust-air temperature.
    Exhaust = 1,
    /// Unit board temperature.
    Board = 2,
}

// NVML has no named constants for unit temperature sensors.
impl_enum_display!(UnitTemperatureType, {
    UnitTemperatureType::Intake => "intake",
    UnitTemperatureType::Exhaust => "exhaust",
    UnitTemperatureType::Board => "board",
});

/// Unit temperature reading and its sensor location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitTemperature {
    /// Sensor location in the unit.
    pub kind: UnitTemperatureType,
    /// Temperature in degrees Celsius.
    pub temperature: u32,
}

/// Minimum and maximum device power limits in milliwatts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PowerLimits {
    /// Minimum allowed power limit in milliwatts.
    pub min: u32,
    /// Maximum allowed power limit in milliwatts.
    pub max: u32,
}

/// ECC error counts grouped by GPU storage location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[deprecated]
pub struct EccErrorCounts {
    /// L1 cache errors.
    pub l1_cache: u64,
    /// L2 cache errors.
    pub l2_cache: u64,
    /// Device memory errors.
    pub device_memory: u64,
    /// Register file errors.
    pub register_file: u64,
}

/// Performance-policy violation duration and its CPU reference timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViolationTime {
    /// CPU reference timestamp in microseconds.
    pub reference_time: u64,
    /// Time spent violating the performance policy, in nanoseconds.
    pub violation_time: u64,
}

/// Hardware address and optional retirement timestamp of a retired GPU memory page.
///
/// The address is a physical page address, not a CUDA virtual address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetiredPage {
    /// Retired page's physical hardware address.
    pub address: u64,
    /// Native page-retirement timestamp, when available.
    pub timestamp: Option<u64>,
}

/// Correctable and uncorrectable row-remapping counts and pending or failed remapping state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemappedRows {
    /// Rows remapped because of correctable errors.
    pub corrected: u32,
    /// Rows remapped because of uncorrectable errors.
    pub uncorrected: u32,
    /// Whether row remapping is pending.
    pub pending: bool,
    /// Whether a row-remapping failure occurred.
    pub failure_occurred: bool,
}

/// Memory-bank counts grouped by remaining row-remapping capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowRemapperHistogram {
    /// Memory banks with maximum row-remapping capacity.
    pub max: u32,
    /// Memory banks with high row-remapping capacity.
    pub high: u32,
    /// Memory banks with partial row-remapping capacity.
    pub partial: u32,
    /// Memory banks with low row-remapping capacity.
    pub low: u32,
    /// Memory banks with no row-remapping capacity remaining.
    pub none: u32,
}

/// Typed scalar value decoded from an NVML telemetry union.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum FieldValue {
    /// Double-precision floating-point value.
    Double(f64),
    /// Native unsigned-int value, stored as `u32`.
    UnsignedInt(u32),
    /// Native unsigned-long value, stored as `u64`.
    UnsignedLong(u64),
    /// Native unsigned-long-long value, stored as `u64`.
    UnsignedLongLong(u64),
    /// Native signed-long-long value, stored as `i64`.
    SignedLongLong(i64),
    /// Native signed-int value, stored as `i32`.
    SignedInt(i32),
    /// Native unsigned-short value, stored as `u16`.
    UnsignedShort(u16),
}

/// Timestamped result of one NVML telemetry field query.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSample {
    /// Telemetry field that produced the sample.
    pub field: FieldId,
    /// Scope used when querying the field.
    pub scope_id: u32,
    /// CPU sample timestamp in microseconds.
    pub timestamp: i64,
    /// Field-query latency in microseconds.
    pub latency_usec: i64,
    /// Decoded field value or the error reported for that field.
    pub result: Result<FieldValue>,
}

/// GPU utilization, memory use, and timing statistics for a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccountingStats {
    /// Percent of time over the process's lifetime during which one or more kernels was executing on the GPU. Utilization stats just like returned by [nvmlDeviceGetUtilizationRates](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html#group__nvmldevicequeries_1ga30e0516c86aa5f98196c73f05c77d4df) but for the life time of a process (not just the last sample period). Set to `NVML_VALUE_NOT_AVAILABLE` if nvmlDeviceGetUtilizationRates is not supported.
    pub gpu_utilization: u32,
    /// Percent of time over the process's lifetime during which global (device) memory was being read or written. Set to `NVML_VALUE_NOT_AVAILABLE` if nvmlDeviceGetUtilizationRates is not supported.
    pub memory_utilization: u32,
    /// Maximum total memory in bytes that was ever allocated by the process. Set to `NVML_VALUE_NOT_AVAILABLE` if nvmlProcessInfo_t->usedGpuMemory is not supported.
    pub max_memory_usage: u64,
    /// Amount of time in ms during which the compute context was active. The time is reported as 0 if the process is not terminated.
    pub time: u64,
    /// CPU Timestamp in usec representing start time for the process.
    pub start_time: u64,
    /// Whether the process is still running.
    pub is_running: bool,
}

/// Timestamped scalar sample returned by NVML.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// CPU sample timestamp in microseconds.
    pub timestamp: u64,
    /// Decoded scalar sample value.
    pub value: FieldValue,
}

/// Samples sharing one native scalar value type.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    /// Native scalar value type shared by the samples.
    pub value_type: FieldValueType,
    /// Timestamped samples returned by NVML.
    pub samples: Vec<Sample>,
}

/// GPU and confidential-computing memory use and instance identifiers for a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessDetail {
    /// Process identifier.
    pub pid: Pid,
    /// Used GPU memory in bytes, or `None` when unavailable.
    pub used_gpu_memory: Option<u64>,
    /// GPU instance identifier, or `u32::MAX` when MIG is disabled.
    pub gpu_instance_id: u32,
    /// Compute instance identifier, or `u32::MAX` when MIG is disabled.
    pub compute_instance_id: u32,
    /// Used confidential-computing protected GPU memory in bytes, or `None` when unavailable.
    pub used_gpu_cc_protected_memory: Option<u64>,
}

pub use crate::operational_event::EventNotification;

/// Driver event types and the affected GPU's PCI identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemEventNotification {
    /// Driver event flags included in this notification.
    pub event_types: SystemEventTypes,
    /// `GpuId` in PCI format.
    pub gpu_id: u32,
}

/// Total, free, and used space in the GPU's BAR1 memory aperture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bar1MemoryInfo {
    /// Total BAR1 Memory (in bytes).
    pub total: u64,
    /// Unallocated BAR1 Memory (in bytes).
    pub free: u64,
    /// Allocated Used Memory (in bytes).
    pub used: u64,
}

/// Engine utilization percentage and the interval over which NVML measured it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UtilizationCounter {
    /// Percentage of time the engine was busy during the sample interval.
    pub utilization: u32,
    /// Sample interval in microseconds.
    pub sampling_period_us: u32,
}

/// Native text describing the current device clock frequencies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CurrentClockFreqs {
    /// Current-clock frequency text returned by NVML.
    pub value: String,
}

/// Native text describing device performance modes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PerformanceModes {
    /// Performance-mode text returned by NVML.
    pub value: String,
}

/// Minimum and maximum clock frequencies in megahertz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockRange {
    /// Minimum clock frequency in megahertz.
    pub min: u32,
    /// Maximum clock frequency in megahertz.
    pub max: u32,
}

/// Minimum and maximum signed clock offsets in megahertz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockRangeI32 {
    /// Minimum clock offset in megahertz.
    pub min: i32,
    /// Maximum clock offset in megahertz.
    pub max: i32,
}

/// Current and supported clock offsets for one clock domain and performance state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockOffset {
    /// Clock domain to which the offset applies.
    pub kind: ClockType,
    /// Performance state to which the offset applies.
    pub pstate: PerformanceState,
    /// Current clock offset in megahertz.
    pub clock_offset_mhz: i32,
    /// Minimum supported clock offset in megahertz.
    pub min_clock_offset_mhz: i32,
    /// Maximum supported clock offset in megahertz.
    pub max_clock_offset_mhz: i32,
}

/// Device temperature margin reported by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MarginTemperature {
    /// The margin temperature value.
    pub value: i32,
}

/// Native start timestamp and duration of the last BBX information-ROM flush.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BbxFlushTime {
    /// Native start timestamp of the last BBX flush.
    pub timestamp: u64,
    /// Duration of the last BBX flush in microseconds.
    pub duration_us: u64,
}

/// Hardware resource counts and memory size reported for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceAttributes {
    /// Streaming Multiprocessor count.
    pub multiprocessor_count: u32,
    /// Shared Copy Engine count.
    pub shared_copy_engine_count: u32,
    /// Shared Decoder Engine count.
    pub shared_decoder_count: u32,
    /// Shared Encoder Engine count.
    pub shared_encoder_count: u32,
    /// Shared JPEG Engine count.
    pub shared_jpeg_count: u32,
    /// Shared OFA Engine count.
    pub shared_ofa_count: u32,
    /// GPU instance slice count.
    pub gpu_instance_slice_count: u32,
    /// Compute instance slice count.
    pub compute_instance_slice_count: u32,
    /// Device memory size (in MiB).
    pub memory_size_mb: u64,
}

/// GPU placement and hardware identifiers reported by the platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlatformInfo {
    /// InfiniBand GUID reported by the platform. On Blackwell, bytes 8 through 15 are zero.
    pub ib_guid: [u8; 16],
    /// Serial number of the chassis containing this GPU (for Blackwell it is 13 bytes so indices 13-15 are zero).
    pub chassis_serial_number: [u8; 16],
    /// The slot number in the chassis containing this GPU (includes switches).
    pub slot_number: u8,
    /// The tray index within the compute slots in the chassis containing this GPU (does not include switches).
    pub tray_index: u8,
    /// Index of the node within the slot containing this GPU.
    pub host_id: u8,
    /// Platform-reported `NVLink` peer type, indicating whether a switch is present.
    pub peer_type: u8,
    /// ID of this GPU within the node.
    pub module_id: u8,
}

/// 64-bit PDI value reported by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pdi {
    /// 64-bit PDI value.
    pub value: u64,
}

/// C2C interconnect enablement reported by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C2cModeInfo {
    /// Whether the C2C interconnect is enabled.
    pub enabled: bool,
}

/// Clock-monitor domain and fault mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClkMonFaultInfo {
    /// The Domain which faulted.
    pub clock_api_domain: u32,
    /// Faults Information.
    pub clock_domain_fault_mask: u32,
}

/// Clock-monitor fault status and per-domain fault information.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClkMonStatus {
    /// Whether the clock monitor reports a fault.
    pub global_status: bool,
    /// Fault information for each reported clock domain.
    pub faults: Vec<ClkMonFaultInfo>,
}

/// Aggregate and volatile SRAM error counts and diagnostic-threshold status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EccSramErrorStatus {
    /// Aggregate uncorrectable parity error count.
    pub aggregate_unc_parity: u64,
    /// Aggregate uncorrectable SEC-DED error count.
    pub aggregate_unc_secded: u64,
    /// Aggregate correctable error count.
    pub aggregate_corrected: u64,
    /// Volatile uncorrectable parity error count.
    pub volatile_unc_parity: u64,
    /// Volatile uncorrectable SEC-DED error count.
    pub volatile_unc_secded: u64,
    /// Volatile correctable error count.
    pub volatile_corrected: u64,
    /// Aggregate uncorrectable error count for L2 cache bucket.
    pub aggregate_unc_bucket_l2: u64,
    /// Aggregate uncorrectable error count for SM bucket.
    pub aggregate_unc_bucket_sm: u64,
    /// Aggregate uncorrectable error count for PCIE bucket.
    pub aggregate_unc_bucket_pcie: u64,
    /// Aggregate uncorrectable error count for Microcontroller bucket.
    pub aggregate_unc_bucket_mcu: u64,
    /// Aggregate uncorrectable error count for Other bucket.
    pub aggregate_unc_bucket_other: u64,
    /// Whether SRAM errors exceed the field-diagnostics threshold.
    pub threshold_exceeded: bool,
}

/// Location and count of repeated uncorrected errors at one SRAM address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EccSramUniqueUncorrectedErrorEntry {
    /// The SRAM unit index.
    pub unit: u32,
    /// The error location within the SRAM unit.
    pub location: u32,
    /// The error sublocation within the SRAM unit.
    pub sublocation: u32,
    /// The error extlocation within the SRAM unit.
    pub extlocation: u32,
    /// The error address within the SRAM unit.
    pub address: u32,
    /// Whether the recorded SRAM error is a parity error.
    pub is_parity: bool,
    /// The error count at the same SRAM address.
    pub count: u32,
}

/// Unique uncorrected SRAM error locations and their occurrence counts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EccSramUniqueUncorrectedErrorCounts {
    /// Unique SRAM error locations and their occurrence counts.
    pub entries: Vec<EccSramUniqueUncorrectedErrorEntry>,
}

/// Thermal controller, target, and temperature readings for one sensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThermalSensorSetting {
    /// Controller supplying the sensor reading.
    pub controller: ThermalController,
    /// Default minimum sensor temperature in degrees Celsius.
    pub default_min_temp: i32,
    /// Default maximum sensor temperature in degrees Celsius.
    pub default_max_temp: i32,
    /// Current sensor temperature in degrees Celsius.
    pub current_temp: i32,
    /// Device component monitored by the sensor.
    pub target: ThermalTarget,
}

/// Thermal sensors reported for a device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ThermalSettings {
    /// Thermal sensor readings and their configuration.
    pub sensors: Vec<ThermalSensorSetting>,
}

/// Bridge-chip kind and firmware version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BridgeChipInfo {
    /// Type of bridge chip.
    pub kind: BridgeChipType,
    /// Firmware Version. 0=Version is unavailable.
    pub firmware_version: u32,
}

/// Bridge chips reported in the device connection hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BridgeChipHierarchy {
    /// Bridge chips in the reported connection hierarchy.
    pub bridge_chips: Vec<BridgeChipInfo>,
}

/// Pending memory-channel and TPC repair state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepairStatus {
    /// Whether memory-channel repair is pending.
    pub channel_repair_pending: bool,
    /// Whether TPC repair is pending.
    pub tpc_repair_pending: bool,
}

/// GPU registration state within an `NVLink` fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricState {
    /// Fabric Probe States.
    ///
    /// Fabric Probe State not supported.
    NotSupported = sys::NVML_GPU_FABRIC_STATE_NOT_SUPPORTED,
    /// Fabric Probe has not started.
    NotStarted = sys::NVML_GPU_FABRIC_STATE_NOT_STARTED,
    /// Fabric Probe in progress.
    InProgress = sys::NVML_GPU_FABRIC_STATE_IN_PROGRESS,
    /// Fabric Probe State completed.
    Completed = sys::NVML_GPU_FABRIC_STATE_COMPLETED,
}

/// Summary of the GPU fabric's health and available capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricHealthSummary {
    /// Fabric Health.
    ///
    /// Fabric Health Summary: Not supported.
    NotSupported = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_NOT_SUPPORTED,
    /// Fabric Health Summary: Healthy.
    Healthy = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_HEALTHY,
    /// Fabric Health Summary: Unhealthy.
    Unhealthy = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_UNHEALTHY,
    /// Fabric Health Summary: Limited Capacity.
    LimitedCapacity = sys::NVML_GPU_FABRIC_HEALTH_SUMMARY_LIMITED_CAPACITY,
}

/// Fabric-health flag with supported true, supported false, and unsupported states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricBooleanState {
    /// Fabric Degraded BW.
    ///
    /// Fabric Health Mask: Degraded Bandwidth not supported.
    NotSupported = sys::NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_NOT_SUPPORTED,
    /// Fabric Health Mask: Bandwidth degraded.
    True = sys::NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_TRUE,
    /// Fabric Health Mask: Bandwidth not degraded.
    False = sys::NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE,
}

/// Configuration issue reported by the GPU fabric-health mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GpuFabricIncorrectConfiguration {
    /// Fabric Incorrect Configuration.
    ///
    /// Fabric Health Mask: Incorrect Configuration not supported.
    NotSupported = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NOT_SUPPORTED,
    /// Fabric Health Mask: Correct Configuration.
    None = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NONE,
    /// Fabric Health Mask: Incorrect Configuration - `SysGUID`.
    IncorrectSysguid = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_SYSGUID,
    /// Fabric Health Mask: Incorrect Configuration - Chassis Serial Number.
    IncorrectChassisSerialNumber =
        sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_CHASSIS_SN,
    /// Fabric Health Mask: Incorrect Configuration - No Partition.
    NoPartition = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NO_PARTITION,
    /// Fabric Health Mask: Incorrect Configuration - Insufficient Nvlinks.
    InsufficientNvlinks =
        sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INSUFFICIENT_NVLINKS,
    /// Fabric Health Mask: Incorrect Configuration - Incompatible GPU Firmware.
    IncompatibleGpuFirmware =
        sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCOMPATIBLE_GPU_FW,
    /// Fabric Health Mask: Incorrect Configuration - Invalid Location.
    InvalidLocation = sys::NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INVALID_LOCATION,
}

/// Decoded fabric-health flags and their original native mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuFabricHealth {
    /// Connection state to the GPU Fabric Compressor.
    pub fabric_manager_state: GpuFabricManagerState,
    /// Whether the fabric reports degraded bandwidth.
    pub degraded_bandwidth: GpuFabricBooleanState,
    /// Whether route recovery is in progress.
    pub route_recovery: GpuFabricBooleanState,
    /// Whether the fabric reports unhealthy routes.
    pub route_unhealthy: GpuFabricBooleanState,
    /// Whether recovery from an access timeout is in progress.
    pub access_timeout_recovery: GpuFabricBooleanState,
    /// Reported fabric configuration issue.
    pub incorrect_configuration: GpuFabricIncorrectConfiguration,
    /// Original native fabric-health mask, including unrecognized bits.
    pub raw_mask: u32,
}

/// Fabric-cluster identity, membership, registration state, and health for a GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuFabricInfo {
    /// Fabric-cluster UUID bytes.
    pub cluster_uuid: [u8; sys::NVML_GPU_FABRIC_UUID_LEN as usize],
    /// Native status code associated with fabric registration.
    pub status_code: u32,
    /// Clique entries, sorted by ascending type then ascending ID.
    pub cliques: Vec<GpuFabricClique>,
    /// GPU fabric-registration state.
    pub state: GpuFabricState,
    /// Decoded fabric-health flags.
    pub health: GpuFabricHealth,
    /// Overall fabric health and capacity state.
    pub health_summary: GpuFabricHealthSummary,
}

/// Activity and performance-state transition thresholds for one utilization domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicPstateInfo {
    /// Native utilization-domain index.
    pub domain: u32,
    /// Percentage of time the domain was busy in the last one-second interval.
    pub percentage: u32,
    /// Utilization threshold that can trigger a performance-increasing state change.
    pub increase_threshold: u32,
    /// Utilization threshold that can trigger a performance-decreasing state change.
    pub decrease_threshold: u32,
}

/// Dynamic performance-state utilization measurements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicPstatesInfo {
    /// Reserved for future use.
    pub flags: u32,
    /// Utilization measurements for the reported performance-state domains.
    pub utilization: Vec<DynamicPstateInfo>,
}

/// Counting units and packet filters for an `NVLink` utilization counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkUtilizationControl {
    /// Unit in which the counter accumulates activity.
    pub units: NvLinkUtilizationCountUnits,
    /// Packet types selected for counting.
    pub packet_filter: NvLinkPacketTypes,
}

/// Receive and transmit values of an `NVLink` utilization counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkUtilizationCounter {
    /// Received activity in the units selected by the counter's control settings.
    pub rx: u64,
    /// Transmitted activity in the units selected by the counter's control settings.
    pub tx: u64,
}

/// Version of one `NVLink` firmware component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkFirmwareVersion {
    /// Native firmware microcode component identifier.
    pub ucode_type: u8,
    /// Firmware major version.
    pub major: u32,
    /// Firmware minor version.
    pub minor: u32,
    /// Firmware subminor version.
    pub sub_minor: u32,
}

/// NVLE enablement and `NVLink` firmware component versions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NvLinkInfo {
    /// Whether NVLE is enabled.
    pub nvle_enabled: bool,
    /// Reported firmware component versions.
    pub firmware_versions: Vec<NvLinkFirmwareVersion>,
}

/// Native `NVLink` bandwidth mode and whether it is the best available mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkBwMode {
    /// Whether the selected mode is the best available bandwidth mode.
    pub is_best: bool,
    /// Native bandwidth-mode identifier.
    pub mode: u8,
}

/// Native `NVLink` bandwidth-mode identifiers supported by a device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NvLinkSupportedBwModes {
    /// Supported native bandwidth-mode identifiers.
    pub modes: Vec<u8>,
}

/// Video encoder session count and aggregate throughput measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderStats {
    /// Number of active video encoder sessions.
    pub session_count: u32,
    /// Moving average of encoded frames per second.
    pub average_fps: u32,
    /// Moving average encoding latency in microseconds.
    pub average_latency_us: u32,
}

/// Device fan index and speed in revolutions per minute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FanSpeedInfo {
    /// Device fan index.
    pub fan: u32,
    /// Fan speed in revolutions per minute.
    pub speed: u32,
}

/// Minimum and maximum allowed device fan speeds as percentages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MinMaxFanSpeed {
    /// Minimum allowed fan speed as a percentage.
    pub min: u32,
    /// Maximum allowed fan speed as a percentage.
    pub max: u32,
}

/// Device temperature reading in degrees Celsius and its sensor selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemperatureInfo {
    /// Temperature sensor selection.
    pub sensor: TemperatureSensor,
    /// Reported temperature in degrees Celsius.
    pub temperature: i32,
}

/// Cooling-device index, control signal, and cooling target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoolerInfo {
    /// Cooling-device index.
    pub index: u32,
    /// Control signal used by the cooler.
    pub signal_type: CoolerControl,
    /// Device component cooled by this cooler.
    pub target: CoolerTarget,
}

/// Resource counts, memory size, and capabilities of a GPU instance profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuInstanceProfileInfo {
    /// Unique profile ID within the device.
    pub id: u32,
    /// GPU Slice count.
    pub slice_count: u32,
    /// GPU instance count.
    pub instance_count: u32,
    /// Streaming Multiprocessor count.
    pub multiprocessor_count: u32,
    /// Copy Engine count.
    pub copy_engine_count: u32,
    /// Decoder Engine count.
    pub decoder_count: u32,
    /// Encoder Engine count.
    pub encoder_count: u32,
    /// JPEG Engine count.
    pub jpeg_count: u32,
    /// OFA Engine count.
    pub ofa_count: u32,
    /// Memory size in `MBytes`.
    pub memory_size_mb: u64,
    /// Profile name reported by NVML.
    pub name: String,
    /// Capabilities supported by the GPU instance profile.
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

/// Memory-slice placement occupied by a GPU instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuInstancePlacement {
    /// Index of first occupied memory slice.
    pub start: u32,
    /// Number of memory slices occupied.
    pub size: u32,
}

/// Resource counts and capabilities of a compute instance profile within a GPU instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputeInstanceProfileInfo {
    /// Unique profile ID within the GPU instance.
    pub id: u32,
    /// GPU Slice count.
    pub slice_count: u32,
    /// Compute instance count.
    pub instance_count: u32,
    /// Streaming Multiprocessor count.
    pub multiprocessor_count: u32,
    /// Shared Copy Engine count.
    pub shared_copy_engine_count: u32,
    /// Shared Decoder Engine count.
    pub shared_decoder_count: u32,
    /// Shared Encoder Engine count.
    pub shared_encoder_count: u32,
    /// Shared JPEG Engine count.
    pub shared_jpeg_count: u32,
    /// Shared OFA Engine count.
    pub shared_ofa_count: u32,
    /// Profile name reported by NVML.
    pub name: String,
    /// Capabilities supported by the compute instance profile.
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

/// Compute-slice placement occupied by a compute instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeInstancePlacement {
    /// Index of first occupied compute slice.
    pub start: u32,
    /// Number of compute slices occupied.
    pub size: u32,
}

/// Video encoder session identity, codec, resolution, and throughput measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderSessionInfo {
    /// Unique session ID.
    pub session_id: u32,
    /// Process that owns the encoder session.
    pub pid: Pid,
    /// Owning vGPU instance ID (only valid on vGPU hosts, otherwise zero).
    pub vgpu_instance: u32,
    /// Video codec used by the session.
    pub codec_type: EncoderType,
    /// Current encode horizontal resolution.
    pub horizontal_resolution: u32,
    /// Current encode vertical resolution.
    pub vertical_resolution: u32,
    /// Moving average encode frames per second.
    pub average_fps: u32,
    /// Moving average encode latency in microseconds.
    pub average_latency_us: u32,
}

/// Frame-buffer capture session count and aggregate throughput measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FbcStats {
    /// Total no of sessions.
    pub session_count: u32,
    /// Moving average new frames captured per second.
    pub average_fps: u32,
    /// Moving average new frame capture latency in microseconds.
    pub average_latency_us: u32,
}

/// Frame-buffer capture session identity, geometry, and throughput measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FbcSessionInfo {
    /// Unique session ID.
    pub session_id: u32,
    /// Process that owns the capture session.
    pub pid: Pid,
    /// Owning vGPU instance ID (only valid on vGPU hosts, otherwise zero).
    pub vgpu_instance: u32,
    /// Display identifier.
    pub display_ordinal: u32,
    /// Capture session type.
    pub session_type: FbcSessionType,
    /// Session flags (one or more of `NVML_NVFBC_SESSION_FLAG_XXX`).
    pub session_flags: u32,
    /// Max horizontal resolution supported by the capture session.
    pub max_horizontal_resolution: u32,
    /// Max vertical resolution supported by the capture session.
    pub max_vertical_resolution: u32,
    /// Horizontal resolution requested by caller in capture call.
    pub horizontal_resolution: u32,
    /// Vertical resolution requested by caller in capture call.
    pub vertical_resolution: u32,
    /// Moving average new frames captured per second.
    pub average_fps: u32,
    /// Moving average new frame capture latency in microseconds.
    pub average_latency_us: u32,
}

/// GPU memory use and instance identifiers for a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessInfo {
    /// Process identifier.
    pub pid: Pid,
    /// Used GPU memory in bytes, or `None` when unavailable.
    pub used_gpu_memory: Option<u64>,
    /// GPU instance identifier, or `u32::MAX` when MIG is disabled.
    pub gpu_instance_id: u32,
    /// Compute instance identifier, or `u32::MAX` when MIG is disabled.
    pub compute_instance_id: u32,
}

/// Per-process engine utilization and a CPU sample timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessUtilizationSample {
    /// Process identifier.
    pub pid: Pid,
    /// CPU Timestamp in microseconds.
    pub timestamp: u64,
    /// SM (3D/Compute) Util Value.
    pub sm_utilization: u32,
    /// Frame Buffer Memory Util Value.
    pub memory_utilization: u32,
    /// Encoder Util Value.
    pub encoder_utilization: u32,
    /// Decoder Util Value.
    pub decoder_utilization: u32,
}

/// Per-process engine utilization, including JPEG and optical-flow engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessUtilizationInfo {
    /// CPU Timestamp in microseconds.
    pub timestamp: u64,
    /// Process identifier.
    pub pid: Pid,
    /// SM (3D/Compute) Util Value.
    pub sm_utilization: u32,
    /// Frame Buffer Memory Util Value.
    pub memory_utilization: u32,
    /// Encoder Util Value.
    pub encoder_utilization: u32,
    /// Decoder Util Value.
    pub decoder_utilization: u32,
    /// Jpeg Util Value.
    pub jpg_utilization: u32,
    /// Ofa Util Value.
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
            // SAFETY: NVML returned this device handle, which stays valid until NVML shuts down.
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
            // SAFETY: NVML returned this device handle, which stays valid until NVML shuts down.
            device: unsafe { Device::from_raw(value.device) },
            // SAFETY: NVML returned this device handle, which stays valid until NVML shuts down.
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

fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    let mut value = [0; size_of::<u32>()];
    value.copy_from_slice(&bytes[offset..offset + size_of::<u32>()]);
    u32::from_ne_bytes(value)
}

fn write_u32_at(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + size_of::<u32>()].copy_from_slice(&value.to_ne_bytes());
}

fn read_string_at(bytes: &[u8], offset: usize, capacity: usize) -> String {
    let bytes = &bytes[offset..offset + capacity];
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn write_string_at(bytes: &mut [u8], offset: usize, capacity: usize, value: &str) {
    let len = value.len().min(capacity.saturating_sub(1));
    bytes[offset..offset + len].copy_from_slice(&value.as_bytes()[..len]);
}

/// Splits a variable-length metadata blob into its fixed header and opaque data.
fn metadata_opaque_data(
    name: &'static str,
    raw: &[u8],
    header_size: usize,
    opaque_offset: usize,
    opaque_size_offset: usize,
) -> Result<Vec<u8>> {
    if raw.len() < header_size {
        return Err(Error::ShortOutput {
            name: name.into(),
            expected: header_size,
            actual: raw.len(),
        });
    }
    let opaque_size = read_u32_at(raw, opaque_size_offset) as usize;
    let opaque_end = opaque_offset.saturating_add(opaque_size);
    if raw.len() < opaque_end {
        return Err(Error::ShortOutput {
            name: name.into(),
            expected: opaque_end,
            actual: raw.len(),
        });
    }
    Ok(raw[opaque_offset..opaque_end].to_vec())
}

impl VgpuMetadata {
    /// Decodes a metadata blob written by `nvmlVgpuInstanceGetMetadata`.
    pub(crate) fn decode_raw(raw: &[u8]) -> Result<Self> {
        type Raw = sys::nvmlVgpuMetadata_t;
        let opaque_data = metadata_opaque_data(
            "vgpu metadata",
            raw,
            size_of::<Raw>(),
            mem::offset_of!(Raw, opaqueData),
            mem::offset_of!(Raw, opaqueDataSize),
        )?;
        Ok(Self {
            version: read_u32_at(raw, mem::offset_of!(Raw, version)),
            revision: read_u32_at(raw, mem::offset_of!(Raw, revision)),
            guest_info_state: try_from_nvml_enum(
                "guest info state",
                read_u32_at(raw, mem::offset_of!(Raw, guestInfoState)),
            )?,
            guest_driver_version: read_string_at(
                raw,
                mem::offset_of!(Raw, guestDriverVersion),
                sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize,
            ),
            host_driver_version: read_string_at(
                raw,
                mem::offset_of!(Raw, hostDriverVersion),
                sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize,
            ),
            virtualization_caps: read_u32_at(raw, mem::offset_of!(Raw, vgpuVirtualizationCaps)),
            guest_vgpu_version: read_u32_at(raw, mem::offset_of!(Raw, guestVgpuVersion)),
            opaque_data,
        })
    }

    pub(crate) fn encode_raw(&self) -> Result<Vec<u8>> {
        type Raw = sys::nvmlVgpuMetadata_t;
        let opaque_offset = mem::offset_of!(Raw, opaqueData);
        let len = max(
            size_of::<Raw>(),
            opaque_offset
                .checked_add(self.opaque_data.len())
                .ok_or_else(|| Error::out_of_range("NVML metadata length"))?,
        );
        let mut raw = vec![0u8; len];
        write_u32_at(&mut raw, mem::offset_of!(Raw, version), self.version);
        write_u32_at(&mut raw, mem::offset_of!(Raw, revision), self.revision);
        write_u32_at(
            &mut raw,
            mem::offset_of!(Raw, guestInfoState),
            self.guest_info_state.into(),
        );
        write_u32_at(
            &mut raw,
            mem::offset_of!(Raw, vgpuVirtualizationCaps),
            self.virtualization_caps,
        );
        write_u32_at(
            &mut raw,
            mem::offset_of!(Raw, guestVgpuVersion),
            self.guest_vgpu_version,
        );
        write_u32_at(
            &mut raw,
            mem::offset_of!(Raw, opaqueDataSize),
            to_u32(self.opaque_data.len(), "NVML metadata opaque size")?,
        );
        write_string_at(
            &mut raw,
            mem::offset_of!(Raw, guestDriverVersion),
            sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize,
            &self.guest_driver_version,
        );
        write_string_at(
            &mut raw,
            mem::offset_of!(Raw, hostDriverVersion),
            sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize,
            &self.host_driver_version,
        );
        raw[opaque_offset..opaque_offset + self.opaque_data.len()]
            .copy_from_slice(&self.opaque_data);
        Ok(raw)
    }
}

impl PgpuMetadata {
    /// Decodes a metadata blob written by `nvmlDeviceGetVgpuMetadata`.
    pub(crate) fn decode_raw(raw: &[u8]) -> Result<Self> {
        type Raw = sys::nvmlVgpuPgpuMetadata_t;
        let opaque_data = metadata_opaque_data(
            "pgpu metadata",
            raw,
            size_of::<Raw>(),
            mem::offset_of!(Raw, opaqueData),
            mem::offset_of!(Raw, opaqueDataSize),
        )?;
        let range_offset = mem::offset_of!(Raw, hostSupportedVgpuRange);
        Ok(Self {
            version: read_u32_at(raw, mem::offset_of!(Raw, version)),
            revision: read_u32_at(raw, mem::offset_of!(Raw, revision)),
            host_driver_version: read_string_at(
                raw,
                mem::offset_of!(Raw, hostDriverVersion),
                sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize,
            ),
            virtualization_caps: read_u32_at(raw, mem::offset_of!(Raw, pgpuVirtualizationCaps)),
            host_supported_vgpu_range: VgpuVersion {
                min: read_u32_at(
                    raw,
                    range_offset + mem::offset_of!(sys::nvmlVgpuVersion_t, minVersion),
                ),
                max: read_u32_at(
                    raw,
                    range_offset + mem::offset_of!(sys::nvmlVgpuVersion_t, maxVersion),
                ),
            },
            opaque_data,
        })
    }

    pub(crate) fn encode_raw(&self) -> Result<Vec<u8>> {
        type Raw = sys::nvmlVgpuPgpuMetadata_t;
        let opaque_offset = mem::offset_of!(Raw, opaqueData);
        let len = max(
            size_of::<Raw>(),
            opaque_offset
                .checked_add(self.opaque_data.len())
                .ok_or_else(|| Error::out_of_range("NVML metadata length"))?,
        );
        let range_offset = mem::offset_of!(Raw, hostSupportedVgpuRange);
        let mut raw = vec![0u8; len];
        write_u32_at(&mut raw, mem::offset_of!(Raw, version), self.version);
        write_u32_at(&mut raw, mem::offset_of!(Raw, revision), self.revision);
        write_u32_at(
            &mut raw,
            mem::offset_of!(Raw, pgpuVirtualizationCaps),
            self.virtualization_caps,
        );
        write_u32_at(
            &mut raw,
            range_offset + mem::offset_of!(sys::nvmlVgpuVersion_t, minVersion),
            self.host_supported_vgpu_range.min,
        );
        write_u32_at(
            &mut raw,
            range_offset + mem::offset_of!(sys::nvmlVgpuVersion_t, maxVersion),
            self.host_supported_vgpu_range.max,
        );
        write_u32_at(
            &mut raw,
            mem::offset_of!(Raw, opaqueDataSize),
            to_u32(self.opaque_data.len(), "NVML metadata opaque size")?,
        );
        write_string_at(
            &mut raw,
            mem::offset_of!(Raw, hostDriverVersion),
            sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize,
            &self.host_driver_version,
        );
        raw[opaque_offset..opaque_offset + self.opaque_data.len()]
            .copy_from_slice(&self.opaque_data);
        Ok(raw)
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
    pub(crate) fn from_raw(info: &sys::nvmlVgpuSchedulerLogInfo_v2_t) -> Result<Self> {
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
            // SAFETY: `string_from_c_ptr` accepts null, and non-null values are NUL-terminated strings owned by the library.
            short_name: unsafe { string_from_c_ptr(value.metricInfo.shortName) },
            // SAFETY: `string_from_c_ptr` accepts null, and non-null values are NUL-terminated strings owned by the library.
            long_name: unsafe { string_from_c_ptr(value.metricInfo.longName) },
            // SAFETY: `string_from_c_ptr` accepts null, and non-null values are NUL-terminated strings owned by the library.
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
    pub(crate) fn from_raw(value: &sys::nvmlGpuFabricInfo_v4_t) -> Result<Self> {
        Ok(Self {
            cluster_uuid: value.clusterUuid,
            status_code: value.status as u32,
            cliques: value
                .cliques
                .get(..value.numCliques as usize)
                .ok_or_else(|| Error::from(Status::InsufficientSize))?
                .iter()
                .map(|clique| GpuFabricClique {
                    kind: GpuFabricCliqueKind::from(clique.type_),
                    id: clique.id,
                })
                .collect(),
            state: try_from_nvml_enum("gpu fabric state", u32::from(value.state))?,
            health: GpuFabricHealth::from_raw(value.healthMask)?,
            health_summary: try_from_nvml_enum(
                "gpu fabric health summary",
                u32::from(value.healthSummary),
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
                    domain: u32::try_from(domain)
                        .expect("fixed NVML utilization domain index fits u32"),
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
            fabric_manager_state: GpuFabricManagerState::from(
                (raw_mask >> sys::NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_GFM_STATE)
                    & sys::NVML_GPU_FABRIC_HEALTH_MASK_WIDTH_GFM_STATE,
            ),
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
    /// Decodes a native scalar union using its value-type discriminator.
    ///
    /// # Errors
    ///
    /// Returns `Error::UnknownFieldValueType` if the native discriminator is not supported.
    pub fn from_raw(value_type: sys::nvmlValueType_t, value: sys::nvmlValue_t) -> Result<Self> {
        Ok(match FieldValueType::try_from(u32::from(value_type)) {
            // SAFETY: the type tag matched above selects this union field.
            Ok(FieldValueType::Double) => Self::Double(unsafe { value.dVal }),
            // SAFETY: the type tag matched above selects this union field.
            Ok(FieldValueType::UnsignedInt) => Self::UnsignedInt(unsafe { value.uiVal }),
            // SAFETY: the type tag matched above selects this union field.
            Ok(FieldValueType::UnsignedLong) => Self::UnsignedLong(unsafe { value.ulVal }),
            // SAFETY: the type tag matched above selects this union field.
            Ok(FieldValueType::UnsignedLongLong) => Self::UnsignedLongLong(unsafe { value.ullVal }),
            // SAFETY: the type tag matched above selects this union field.
            Ok(FieldValueType::SignedLongLong) => Self::SignedLongLong(unsafe { value.sllVal }),
            // SAFETY: the type tag matched above selects this union field.
            Ok(FieldValueType::SignedInt) => Self::SignedInt(unsafe { value.siVal }),
            // SAFETY: the type tag matched above selects this union field.
            Ok(FieldValueType::UnsignedShort) => Self::UnsignedShort(unsafe { value.usVal }),
            Err(_) => return Err(Error::UnknownFieldValueType(u32::from(value_type))),
        })
    }
}

impl Sample {
    /// Decodes a timestamped native sample using the supplied scalar value type.
    ///
    /// # Errors
    ///
    /// Returns `Error::UnknownFieldValueType` if the native discriminator is not supported.
    pub fn from_raw(value_type: sys::nvmlValueType_t, sample: sys::nvmlSample_t) -> Result<Self> {
        Ok(Self {
            timestamp: sample.timeStamp,
            value: FieldValue::from_raw(value_type, sample.sampleValue)?,
        })
    }
}

impl FieldSample {
    /// Decodes a native field sample and retains native or scalar-decoding errors in [`Self::result`].
    #[must_use]
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

impl From<sys::nvmlSystemEventData_v1_t> for SystemEventNotification {
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

/// GPU Fabric Compressor connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFabricManagerState {
    /// GPU Fabric Compressor connection reporting is not supported.
    NotSupported,
    /// GPU is connected to the GPU Fabric Compressor.
    Connected,
    /// GPU is disconnected from the GPU Fabric Compressor.
    Disconnected,
    /// Unrecognized connection state.
    Unknown(u32),
}

impl From<u32> for GpuFabricManagerState {
    fn from(value: u32) -> Self {
        match value {
            sys::NVML_GPU_FABRIC_HEALTH_MASK_GFM_STATE_NOT_SUPPORTED => Self::NotSupported,
            sys::NVML_GPU_FABRIC_HEALTH_MASK_GFM_STATE_CONNECTED => Self::Connected,
            sys::NVML_GPU_FABRIC_HEALTH_MASK_GFM_STATE_DISCONNECTED => Self::Disconnected,
            value => Self::Unknown(value),
        }
    }
}

/// Fabric operation type for a clique assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFabricCliqueKind {
    /// Unicast pointer operation.
    UnicastPointer,
    /// Multicast pointer operation.
    MulticastPointer,
    /// Unicast logical endpoint operation.
    UnicastLogicalEndpoint,
    /// Multicast logical endpoint operation.
    MulticastLogicalEndpoint,
    /// Unrecognized operation type.
    Unknown(u8),
}

impl From<u8> for GpuFabricCliqueKind {
    fn from(value: u8) -> Self {
        match u32::from(value) {
            sys::NVML_GPU_FABRIC_CLIQUE_TYPE_UNICAST_POINTER => Self::UnicastPointer,
            sys::NVML_GPU_FABRIC_CLIQUE_TYPE_MULTICAST_POINTER => Self::MulticastPointer,
            sys::NVML_GPU_FABRIC_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT => {
                Self::UnicastLogicalEndpoint
            }
            sys::NVML_GPU_FABRIC_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT => {
                Self::MulticastLogicalEndpoint
            }
            _ => Self::Unknown(value),
        }
    }
}

/// A single fabric operation type and clique ID assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuFabricClique {
    /// Classification of the clique within the GPU fabric.
    pub kind: GpuFabricCliqueKind,
    /// Clique ID assigned by the Fabric Compressor.
    pub id: u32,
}

bitflags::bitflags! {
    /// Bitmask of reasons that can affect GPU clocks.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ClockEventReasons: u64 {
        /// HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged.
        ///
        /// This is an indicator of:
        ///
        /// * temperature being too high
        /// * External Power Brake Assertion is triggered (e.g. by the system power supply)
        /// * Power draw is too high and Fast Trigger protection is reducing the clocks
        /// * May be also reported during PState or clock change
        ///   + This behavior may be removed in a later release.
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetTemperature`]
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetTemperatureThreshold`]
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetPowerUsage`] Hardware slowdown activated.
        const HW_SLOWDOWN = sys::nvmlClocksThrottleReasonHwSlowdown as u64;

        /// HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged.
        ///
        /// This is an indicator of:
        ///
        /// * temperature being too high
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetTemperature`]
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetTemperatureThreshold`]
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetPowerUsage`] Hardware thermal slowdown activated.
        const HW_THERMAL_SLOWDOWN = sys::nvmlClocksThrottleReasonHwThermalSlowdown as u64;

        /// HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged.
        ///
        /// This is an indicator of:
        ///
        /// * External Power Brake Assertion being triggered (e.g. by the system power supply)
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetTemperature`]
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetTemperatureThreshold`]
        ///
        /// See also
        ///
        /// [`nvmlDeviceGetPowerUsage`] Hardware power brake slowdown activated.
        const HW_POWER_BRAKE_SLOWDOWN = sys::nvmlClocksThrottleReasonHwPowerBrakeSlowdown as u64;

        /// Nothing is running on the GPU and the clocks are dropping to Idle state.
        const GPU_IDLE = sys::nvmlClocksEventReasonGpuIdle as u64;
        /// Clocks are set to application-specific values.
        const APPLICATIONS_CLOCKS_SETTING = sys::nvmlClocksEventReasonApplicationsClocksSetting as u64;
        /// The clocks have been optimized to ensure not to exceed currently set power limits.
        ///
        /// See also
        ///
        /// `nvmlDeviceGetPowerUsage`
        ///
        /// See also
        ///
        /// `nvmlDeviceSetPowerManagementLimit`
        ///
        /// See also
        ///
        /// `nvmlDeviceGetPowerManagementLimit` Software power cap activated.
        const SW_POWER_CAP = sys::nvmlClocksEventReasonSwPowerCap as u64;
        /// Sync Boost.
        ///
        /// This GPU has been added to a Sync boost group with nvidia-smi or DCGM in order to maximize performance per watt. All GPUs in the sync boost group will boost to the minimum possible clocks across the entire group. Look at the throttle reasons for other GPUs in the system to see why those GPUs are holding this one at lower clocks. Sync boost activated.
        const SYNC_BOOST = sys::nvmlClocksEventReasonSyncBoost as u64;
        /// SW Thermal Slowdown.
        ///
        /// The current clocks have been optimized to ensure the following is true:
        ///
        /// * Current GPU temperature does not exceed GPU Max Operating Temperature
        /// * Current memory temperature does not exceeed Memory Max Operating Temperature Software thermal slowdown activated.
        const SW_THERMAL_SLOWDOWN = sys::nvmlClocksEventReasonSwThermalSlowdown as u64;
        /// GPU clocks are limited by current setting of Display clocks.
        ///
        /// See also
        ///
        /// bug 1997531 Display clock setting limited.
        const DISPLAY_CLOCK_SETTING = sys::nvmlClocksEventReasonDisplayClockSetting as u64;
        /// Board limit.
        ///
        /// The board limit (operating) policy is currently limiting the GPU clocks. Board limit policy is limiting clocks.
        const BOARD_LIMIT = sys::nvmlClocksEventReasonBoardLimit as u64;
        /// Reliability.
        ///
        /// The reliability policy is currently limiting the GPU clocks. Reliability policy is limiting clocks.
        const RELIABILITY = sys::nvmlClocksEventReasonReliability as u64;
    }
}

/// GPU recovery action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuRecoveryAction(pub u32);

impl GpuRecoveryAction {
    /// No action needed.
    pub const NONE: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_NONE as _);

    /// Reset Gpu.
    pub const GPU_RESET: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_GPU_RESET as _);

    /// Reboot Node.
    pub const NODE_REBOOT: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_NODE_REBOOT as _);

    /// Drain P2P.
    pub const DRAIN_P2P: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_DRAIN_P2P as _);

    /// Drain P2P and Reset Gpu.
    pub const DRAIN_AND_RESET: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET as _);

    /// Recover IMEX Domain.
    pub const RECOVER_IMEX_DOMAIN: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN as _);

    /// Reset the GPU's `PCIe` bus.
    pub const BUS_RESET: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_BUS_RESET as _);

    /// Reboot the system.
    pub const SYSTEM_REBOOT: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT as _);
}

/// NVIDIA GRID license feature code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridLicenseFeatureCode(pub u32);

impl GridLicenseFeatureCode {
    /// Unknown.
    pub const UNKNOWN: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_UNKNOWN as _);

    /// Virtual GPU.
    pub const VGPU: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_VGPU as _);

    /// Nvidia RTX.
    pub const NVIDIA_RTX: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX as _);

    /// Gaming.
    pub const GAMING: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_GAMING as _);

    /// Compute.
    pub const COMPUTE: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_COMPUTE as _);

    /// vGameDev.
    pub const VGAMEDEV: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_VGAMEDEV as _);
}

/// `NVLink` state returned by the `NVLink` state field query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvLinkFieldState(pub u32);

impl NvLinkFieldState {
    /// `NVLink` is inactive.
    pub const INACTIVE: Self = Self(sys::NVML_NVLINK_STATE_INACTIVE);

    /// `NVLink` is active.
    pub const ACTIVE: Self = Self(sys::NVML_NVLINK_STATE_ACTIVE);

    /// `NVLink` is in sleep state.
    pub const SLEEP: Self = Self(sys::NVML_NVLINK_STATE_SLEEP);

    /// `NVLink` is active, but not usable for traffic.
    pub const ACTIVE_TRAFFIC_DISABLED: Self = Self(sys::NVML_NVLINK_STATE_ACTIVE_TRAFFIC_DISABLED);
}

impl_enum_display!(VgpuGuestInfoState, {
    VgpuGuestInfoState::Uninitialized => "NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED",
    VgpuGuestInfoState::Initialized => "NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED",
});

impl_enum_display!(VgpuPlacementMode, {
    VgpuPlacementMode::Heterogeneous => "NVML_VGPU_PGPU_HETEROGENEOUS_MODE",
    VgpuPlacementMode::Homogeneous => "NVML_VGPU_PGPU_HOMOGENEOUS_MODE",
});

impl_enum_display!(EnableState, {
    EnableState::Disabled => "NVML_FEATURE_DISABLED",
    EnableState::Enabled => "NVML_FEATURE_ENABLED",
});

impl_enum_display!(TemperatureSensor, {
    TemperatureSensor::GpuMax => "NVML_TEMPERATURE_GPU_MAX",
    TemperatureSensor::Gpu => "NVML_TEMPERATURE_GPU",
});

impl_enum_display!(TemperatureThreshold, {
    TemperatureThreshold::Shutdown => "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN",
    TemperatureThreshold::Slowdown => "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN",
    TemperatureThreshold::MemoryMax => "NVML_TEMPERATURE_THRESHOLD_MEM_MAX",
    TemperatureThreshold::GpuMax => "NVML_TEMPERATURE_THRESHOLD_GPU_MAX",
    TemperatureThreshold::AcousticMin => "NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN",
    TemperatureThreshold::AcousticCurrent => "NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR",
    TemperatureThreshold::AcousticMax => "NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX",
    TemperatureThreshold::GpuCurrent => "NVML_TEMPERATURE_THRESHOLD_GPS_CURR",
});

impl_enum_display!(ComputeMode, {
    ComputeMode::Default => "NVML_COMPUTEMODE_DEFAULT",
    ComputeMode::ExclusiveThread => "NVML_COMPUTEMODE_EXCLUSIVE_THREAD",
    ComputeMode::Prohibited => "NVML_COMPUTEMODE_PROHIBITED",
    ComputeMode::ExclusiveProcess => "NVML_COMPUTEMODE_EXCLUSIVE_PROCESS",
});

impl_enum_display!(ClockType, {
    ClockType::Graphics => "NVML_CLOCK_GRAPHICS",
    ClockType::Sm => "NVML_CLOCK_SM",
    ClockType::Memory => "NVML_CLOCK_MEM",
    ClockType::Video => "NVML_CLOCK_VIDEO",
});

impl_enum_display!(NvLinkUtilizationCountUnits, {
    NvLinkUtilizationCountUnits::Cycles => "NVML_NVLINK_COUNTER_UNIT_CYCLES",
    NvLinkUtilizationCountUnits::Packets => "NVML_NVLINK_COUNTER_UNIT_PACKETS",
    NvLinkUtilizationCountUnits::Bytes => "NVML_NVLINK_COUNTER_UNIT_BYTES",
    NvLinkUtilizationCountUnits::Reserved => "NVML_NVLINK_COUNTER_UNIT_RESERVED",
    NvLinkUtilizationCountUnits::Count => "NVML_NVLINK_COUNTER_UNIT_COUNT",
});

impl_enum_display!(NvLinkCapability, {
    NvLinkCapability::P2pSupported => "NVML_NVLINK_CAP_P2P_SUPPORTED",
    NvLinkCapability::SysmemAccess => "NVML_NVLINK_CAP_SYSMEM_ACCESS",
    NvLinkCapability::P2pAtomics => "NVML_NVLINK_CAP_P2P_ATOMICS",
    NvLinkCapability::SysmemAtomics => "NVML_NVLINK_CAP_SYSMEM_ATOMICS",
    NvLinkCapability::SliBridge => "NVML_NVLINK_CAP_SLI_BRIDGE",
    NvLinkCapability::Valid => "NVML_NVLINK_CAP_VALID",
    NvLinkCapability::Count => "NVML_NVLINK_CAP_COUNT",
});

impl_enum_display!(NvLinkErrorCounter, {
    NvLinkErrorCounter::DlReplay => "NVML_NVLINK_ERROR_DL_REPLAY",
    NvLinkErrorCounter::DlRecovery => "NVML_NVLINK_ERROR_DL_RECOVERY",
    NvLinkErrorCounter::DlCrcFlit => "NVML_NVLINK_ERROR_DL_CRC_FLIT",
    NvLinkErrorCounter::DlCrcData => "NVML_NVLINK_ERROR_DL_CRC_DATA",
    NvLinkErrorCounter::DlEccData => "NVML_NVLINK_ERROR_DL_ECC_DATA",
    NvLinkErrorCounter::Count => "NVML_NVLINK_ERROR_COUNT",
});

impl_enum_display!(NvLinkRemoteDeviceType, {
    NvLinkRemoteDeviceType::Gpu => "NVML_NVLINK_DEVICE_TYPE_GPU",
    NvLinkRemoteDeviceType::IbmNpu => "NVML_NVLINK_DEVICE_TYPE_IBMNPU",
    NvLinkRemoteDeviceType::Switch => "NVML_NVLINK_DEVICE_TYPE_SWITCH",
    NvLinkRemoteDeviceType::Unknown => "NVML_NVLINK_DEVICE_TYPE_UNKNOWN",
});

impl_enum_display!(NvLinkVersion, {
    NvLinkVersion::Invalid => "NVML_NVLINK_VERSION_INVALID",
    NvLinkVersion::V1_0 => "NVML_NVLINK_VERSION_1_0",
    NvLinkVersion::V2_0 => "NVML_NVLINK_VERSION_2_0",
    NvLinkVersion::V2_2 => "NVML_NVLINK_VERSION_2_2",
    NvLinkVersion::V3_0 => "NVML_NVLINK_VERSION_3_0",
    NvLinkVersion::V3_1 => "NVML_NVLINK_VERSION_3_1",
    NvLinkVersion::V4_0 => "NVML_NVLINK_VERSION_4_0",
    NvLinkVersion::V5_0 => "NVML_NVLINK_VERSION_5_0",
});

impl_enum_display!(PowerProfileType, {
    PowerProfileType::LowPriorityBackground => "NVML_POWER_PROFILE_LOW_PRIORITY_BACKGROUND",
    PowerProfileType::NetworkBound => "NVML_POWER_PROFILE_NETWORK_BOUND",
    PowerProfileType::Training => "NVML_POWER_PROFILE_TRAINING",
    PowerProfileType::MediumThroughputInference => "NVML_POWER_PROFILE_MEDIUM_THROUGHPUT_INFERENCE",
    PowerProfileType::HighThroughputInference => "NVML_POWER_PROFILE_HIGH_THROUGHPUT_INFERENCE",
    PowerProfileType::LowLatencyInference => "NVML_POWER_PROFILE_LOW_LATENCY_INFERENCE",
    PowerProfileType::Inference => "NVML_POWER_PROFILE_INFERENCE",
    PowerProfileType::MaxQ3 => "NVML_POWER_PROFILE_MAX_Q_3",
    PowerProfileType::MaxQ2 => "NVML_POWER_PROFILE_MAX_Q_2",
    PowerProfileType::MaxQ1 => "NVML_POWER_PROFILE_MAX_Q_1",
    PowerProfileType::MaxP => "NVML_POWER_PROFILE_MAX_P",
    PowerProfileType::MaxQ => "NVML_POWER_PROFILE_MAX_Q",
    PowerProfileType::Compute => "NVML_POWER_PROFILE_COMPUTE",
    PowerProfileType::MemoryBound => "NVML_POWER_PROFILE_MEMORY_BOUND",
    PowerProfileType::Network => "NVML_POWER_PROFILE_NETWORK",
    PowerProfileType::Balanced => "NVML_POWER_PROFILE_BALANCED",
    PowerProfileType::LlmInference => "NVML_POWER_PROFILE_LLM_INFERENCE",
    PowerProfileType::LlmTraining => "NVML_POWER_PROFILE_LLM_TRAINING",
    PowerProfileType::Rbm => "NVML_POWER_PROFILE_RBM",
    PowerProfileType::Dcpcie => "NVML_POWER_PROFILE_DCPCIE",
    PowerProfileType::HmmaSparse => "NVML_POWER_PROFILE_HMMA_SPARSE",
    PowerProfileType::HmmaDense => "NVML_POWER_PROFILE_HMMA_DENSE",
    PowerProfileType::SyncBalanced => "NVML_POWER_PROFILE_SYNC_BALANCED",
    PowerProfileType::Hpc => "NVML_POWER_PROFILE_HPC",
    PowerProfileType::Mig => "NVML_POWER_PROFILE_MIG",
    PowerProfileType::Max => "NVML_POWER_PROFILE_MAX",
});

impl_enum_display!(PowerMizerMode, {
    PowerMizerMode::Adaptive => "NVML_POWER_MIZER_MODE_ADAPTIVE",
    PowerMizerMode::PreferMaximumPerformance => "NVML_POWER_MIZER_MODE_PREFER_MAXIMUM_PERFORMANCE",
    PowerMizerMode::Auto => "NVML_POWER_MIZER_MODE_AUTO",
    PowerMizerMode::PreferConsistentPerformance => "NVML_POWER_MIZER_MODE_PREFER_CONSISTENT_PERFORMANCE",
});

impl_enum_display!(ClockId, {
    ClockId::Current => "NVML_CLOCK_ID_CURRENT",
    ClockId::AppClockTarget => "NVML_CLOCK_ID_APP_CLOCK_TARGET",
    ClockId::AppClockDefault => "NVML_CLOCK_ID_APP_CLOCK_DEFAULT",
    ClockId::CustomerBoostMax => "NVML_CLOCK_ID_CUSTOMER_BOOST_MAX",
});

impl_enum_display!(ThermalTarget, {
    ThermalTarget::None => "NVML_THERMAL_TARGET_NONE",
    ThermalTarget::Gpu => "NVML_THERMAL_TARGET_GPU",
    ThermalTarget::Memory => "NVML_THERMAL_TARGET_MEMORY",
    ThermalTarget::PowerSupply => "NVML_THERMAL_TARGET_POWER_SUPPLY",
    ThermalTarget::Board => "NVML_THERMAL_TARGET_BOARD",
    ThermalTarget::VcdBoard => "NVML_THERMAL_TARGET_VCD_BOARD",
    ThermalTarget::VcdInlet => "NVML_THERMAL_TARGET_VCD_INLET",
    ThermalTarget::VcdOutlet => "NVML_THERMAL_TARGET_VCD_OUTLET",
    ThermalTarget::All => "NVML_THERMAL_TARGET_ALL",
    ThermalTarget::Unknown => "NVML_THERMAL_TARGET_UNKNOWN",
});

impl_enum_display!(ThermalController, {
    ThermalController::None => "NVML_THERMAL_CONTROLLER_NONE",
    ThermalController::GpuInternal => "NVML_THERMAL_CONTROLLER_GPU_INTERNAL",
    ThermalController::Adm1032 => "NVML_THERMAL_CONTROLLER_ADM1032",
    ThermalController::Adt7461 => "NVML_THERMAL_CONTROLLER_ADT7461",
    ThermalController::Max6649 => "NVML_THERMAL_CONTROLLER_MAX6649",
    ThermalController::Max1617 => "NVML_THERMAL_CONTROLLER_MAX1617",
    ThermalController::Lm99 => "NVML_THERMAL_CONTROLLER_LM99",
    ThermalController::Lm89 => "NVML_THERMAL_CONTROLLER_LM89",
    ThermalController::Lm64 => "NVML_THERMAL_CONTROLLER_LM64",
    ThermalController::G781 => "NVML_THERMAL_CONTROLLER_G781",
    ThermalController::Adt7473 => "NVML_THERMAL_CONTROLLER_ADT7473",
    ThermalController::Sbmax6649 => "NVML_THERMAL_CONTROLLER_SBMAX6649",
    ThermalController::VbiosEvt => "NVML_THERMAL_CONTROLLER_VBIOSEVT",
    ThermalController::Os => "NVML_THERMAL_CONTROLLER_OS",
    ThermalController::NvsysconCanoas => "NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS",
    ThermalController::NvsysconE551 => "NVML_THERMAL_CONTROLLER_NVSYSCON_E551",
    ThermalController::Max6649r => "NVML_THERMAL_CONTROLLER_MAX6649R",
    ThermalController::Adt7473s => "NVML_THERMAL_CONTROLLER_ADT7473S",
    ThermalController::Unknown => "NVML_THERMAL_CONTROLLER_UNKNOWN",
});

impl_enum_display!(DeviceAddressingMode, {
    DeviceAddressingMode::None => "NVML_DEVICE_ADDRESSING_MODE_NONE",
    DeviceAddressingMode::Hmm => "NVML_DEVICE_ADDRESSING_MODE_HMM",
    DeviceAddressingMode::Ats => "NVML_DEVICE_ADDRESSING_MODE_ATS",
});

impl_enum_display!(BridgeChipType, {
    BridgeChipType::Plx => "NVML_BRIDGE_CHIP_PLX",
    BridgeChipType::Bro4 => "NVML_BRIDGE_CHIP_BRO4",
});

impl_enum_display!(PerformanceState, {
    PerformanceState::P0 => "NVML_PSTATE_0",
    PerformanceState::P1 => "NVML_PSTATE_1",
    PerformanceState::P2 => "NVML_PSTATE_2",
    PerformanceState::P3 => "NVML_PSTATE_3",
    PerformanceState::P4 => "NVML_PSTATE_4",
    PerformanceState::P5 => "NVML_PSTATE_5",
    PerformanceState::P6 => "NVML_PSTATE_6",
    PerformanceState::P7 => "NVML_PSTATE_7",
    PerformanceState::P8 => "NVML_PSTATE_8",
    PerformanceState::P9 => "NVML_PSTATE_9",
    PerformanceState::P10 => "NVML_PSTATE_10",
    PerformanceState::P11 => "NVML_PSTATE_11",
    PerformanceState::P12 => "NVML_PSTATE_12",
    PerformanceState::P13 => "NVML_PSTATE_13",
    PerformanceState::P14 => "NVML_PSTATE_14",
    PerformanceState::P15 => "NVML_PSTATE_15",
    PerformanceState::Unknown => "NVML_PSTATE_UNKNOWN",
});

impl_enum_display!(PcieUtilCounter, {
    PcieUtilCounter::TxBytes => "NVML_PCIE_UTIL_TX_BYTES",
    PcieUtilCounter::RxBytes => "NVML_PCIE_UTIL_RX_BYTES",
});

impl_enum_display!(TopologyLevel, {
    TopologyLevel::Internal => "NVML_TOPOLOGY_INTERNAL",
    TopologyLevel::Single => "NVML_TOPOLOGY_SINGLE",
    TopologyLevel::Multiple => "NVML_TOPOLOGY_MULTIPLE",
    TopologyLevel::HostBridge => "NVML_TOPOLOGY_HOSTBRIDGE",
    TopologyLevel::Node => "NVML_TOPOLOGY_NODE",
    TopologyLevel::System => "NVML_TOPOLOGY_SYSTEM",
});

impl_enum_display!(P2pCapabilityIndex, {
    P2pCapabilityIndex::Read => "NVML_P2P_CAPS_INDEX_READ",
    P2pCapabilityIndex::Write => "NVML_P2P_CAPS_INDEX_WRITE",
    P2pCapabilityIndex::Nvlink => "NVML_P2P_CAPS_INDEX_NVLINK",
    P2pCapabilityIndex::Atomics => "NVML_P2P_CAPS_INDEX_ATOMICS",
    P2pCapabilityIndex::Prop => "NVML_P2P_CAPS_INDEX_PROP",
    P2pCapabilityIndex::Unknown => "NVML_P2P_CAPS_INDEX_UNKNOWN",
});

impl_enum_display!(P2pStatus, {
    P2pStatus::Ok => "NVML_P2P_STATUS_OK",
    P2pStatus::ChipsetNotSupported => "NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED",
    P2pStatus::GpuNotSupported => "NVML_P2P_STATUS_GPU_NOT_SUPPORTED",
    P2pStatus::IoHubNotSupported => "NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED",
    P2pStatus::DisabledByRegKey => "NVML_P2P_STATUS_DISABLED_BY_REGKEY",
    P2pStatus::NotSupported => "NVML_P2P_STATUS_NOT_SUPPORTED",
    P2pStatus::Unknown => "NVML_P2P_STATUS_UNKNOWN",
});

impl_enum_display!(InforomObject, {
    InforomObject::Oem => "NVML_INFOROM_OEM",
    InforomObject::Ecc => "NVML_INFOROM_ECC",
    InforomObject::Power => "NVML_INFOROM_POWER",
    InforomObject::Den => "NVML_INFOROM_DEN",
});

impl_enum_display!(MemoryErrorType, {
    MemoryErrorType::Corrected => "NVML_MEMORY_ERROR_TYPE_CORRECTED",
    MemoryErrorType::Uncorrected => "NVML_MEMORY_ERROR_TYPE_UNCORRECTED",
});

impl_enum_display!(EccCounterType, {
    EccCounterType::Volatile => "NVML_VOLATILE_ECC",
    EccCounterType::Aggregate => "NVML_AGGREGATE_ECC",
});

impl_enum_display!(MemoryLocation, {
    MemoryLocation::L1Cache => "NVML_MEMORY_LOCATION_L1_CACHE",
    MemoryLocation::L2Cache => "NVML_MEMORY_LOCATION_L2_CACHE",
    MemoryLocation::Dram => "NVML_MEMORY_LOCATION_DRAM",
    MemoryLocation::RegisterFile => "NVML_MEMORY_LOCATION_REGISTER_FILE",
    MemoryLocation::TextureMemory => "NVML_MEMORY_LOCATION_TEXTURE_MEMORY",
    MemoryLocation::TextureShm => "NVML_MEMORY_LOCATION_TEXTURE_SHM",
    MemoryLocation::Cbu => "NVML_MEMORY_LOCATION_CBU",
    MemoryLocation::Sram => "NVML_MEMORY_LOCATION_SRAM",
});

impl_enum_display!(PerfPolicyType, {
    PerfPolicyType::Power => "NVML_PERF_POLICY_POWER",
    PerfPolicyType::Thermal => "NVML_PERF_POLICY_THERMAL",
    PerfPolicyType::SyncBoost => "NVML_PERF_POLICY_SYNC_BOOST",
    PerfPolicyType::BoardLimit => "NVML_PERF_POLICY_BOARD_LIMIT",
    PerfPolicyType::LowUtilization => "NVML_PERF_POLICY_LOW_UTILIZATION",
    PerfPolicyType::Reliability => "NVML_PERF_POLICY_RELIABILITY",
    PerfPolicyType::TotalAppClocks => "NVML_PERF_POLICY_TOTAL_APP_CLOCKS",
    PerfPolicyType::TotalBaseClocks => "NVML_PERF_POLICY_TOTAL_BASE_CLOCKS",
});

impl_enum_display!(PageRetirementCause, {
    PageRetirementCause::MultipleSingleBitEccErrors => "NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS",
    PageRetirementCause::DoubleBitEccError => "NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR",
});

impl_enum_display!(FieldValueType, {
    FieldValueType::Double => "NVML_VALUE_TYPE_DOUBLE",
    FieldValueType::UnsignedInt => "NVML_VALUE_TYPE_UNSIGNED_INT",
    FieldValueType::UnsignedLong => "NVML_VALUE_TYPE_UNSIGNED_LONG",
    FieldValueType::UnsignedLongLong => "NVML_VALUE_TYPE_UNSIGNED_LONG_LONG",
    FieldValueType::SignedLongLong => "NVML_VALUE_TYPE_SIGNED_LONG_LONG",
    FieldValueType::SignedInt => "NVML_VALUE_TYPE_SIGNED_INT",
    FieldValueType::UnsignedShort => "NVML_VALUE_TYPE_UNSIGNED_SHORT",
});

impl_enum_display!(SamplingType, {
    SamplingType::TotalPower => "NVML_TOTAL_POWER_SAMPLES",
    SamplingType::GpuUtilization => "NVML_GPU_UTILIZATION_SAMPLES",
    SamplingType::MemoryUtilization => "NVML_MEMORY_UTILIZATION_SAMPLES",
    SamplingType::EncoderUtilization => "NVML_ENC_UTILIZATION_SAMPLES",
    SamplingType::DecoderUtilization => "NVML_DEC_UTILIZATION_SAMPLES",
    SamplingType::ProcessorClock => "NVML_PROCESSOR_CLK_SAMPLES",
    SamplingType::MemoryClock => "NVML_MEMORY_CLK_SAMPLES",
    SamplingType::ModulePower => "NVML_MODULE_POWER_SAMPLES",
    SamplingType::JpgUtilization => "NVML_JPG_UTILIZATION_SAMPLES",
    SamplingType::OfaUtilization => "NVML_OFA_UTILIZATION_SAMPLES",
});

impl_enum_display!(AffinityScope, {
    AffinityScope::Node => "NVML_AFFINITY_SCOPE_NODE",
    AffinityScope::Socket => "NVML_AFFINITY_SCOPE_SOCKET",
});

impl_enum_display!(EncoderType, {
    EncoderType::H264 => "NVML_ENCODER_QUERY_H264",
    EncoderType::Hevc => "NVML_ENCODER_QUERY_HEVC",
    EncoderType::Av1 => "NVML_ENCODER_QUERY_AV1",
    EncoderType::Unknown => "NVML_ENCODER_QUERY_UNKNOWN",
});

impl_enum_display!(RestrictedApi, {
    RestrictedApi::SetApplicationsClocks => "NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS",
    RestrictedApi::SetAutoBoostedClocks => "NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS",
    RestrictedApi::Count => "NVML_RESTRICTED_API_COUNT",
});

impl_enum_display!(AdaptiveClockInfoStatus, {
    AdaptiveClockInfoStatus::Disabled => "NVML_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED",
    AdaptiveClockInfoStatus::Enabled => "NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED",
});

impl_enum_display!(FanPolicy, {
    FanPolicy::TemperatureContinuousSw => "NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW",
    FanPolicy::Manual => "NVML_FAN_POLICY_MANUAL",
});

impl_enum_display!(FanState, {
    FanState::Normal => "NVML_FAN_NORMAL",
    FanState::Failed => "NVML_FAN_FAILED",
});

impl_enum_display!(CoolerControl, {
    CoolerControl::None => "NVML_THERMAL_COOLER_SIGNAL_NONE",
    CoolerControl::Toggle => "NVML_THERMAL_COOLER_SIGNAL_TOGGLE",
    CoolerControl::Variable => "NVML_THERMAL_COOLER_SIGNAL_VARIABLE",
    CoolerControl::Count => "NVML_THERMAL_COOLER_SIGNAL_COUNT",
});

impl_enum_display!(CoolerTarget, {
    CoolerTarget::None => "NVML_THERMAL_COOLER_TARGET_NONE",
    CoolerTarget::Gpu => "NVML_THERMAL_COOLER_TARGET_GPU",
    CoolerTarget::Memory => "NVML_THERMAL_COOLER_TARGET_MEMORY",
    CoolerTarget::PowerSupply => "NVML_THERMAL_COOLER_TARGET_POWER_SUPPLY",
    CoolerTarget::GpuRelated => "NVML_THERMAL_COOLER_TARGET_GPU_RELATED",
});

impl_enum_display!(FbcSessionType, {
    FbcSessionType::Unknown => "NVML_FBC_SESSION_TYPE_UNKNOWN",
    FbcSessionType::ToSys => "NVML_FBC_SESSION_TYPE_TOSYS",
    FbcSessionType::Cuda => "NVML_FBC_SESSION_TYPE_CUDA",
    FbcSessionType::Vid => "NVML_FBC_SESSION_TYPE_VID",
    FbcSessionType::HwEnc => "NVML_FBC_SESSION_TYPE_HWENC",
});

impl_enum_display!(DriverModel, {
    DriverModel::Wddm => "NVML_DRIVER_WDDM",
    DriverModel::Wdm => "NVML_DRIVER_WDM",
    DriverModel::Mcdm => "NVML_DRIVER_MCDM",
});

impl_enum_display!(GpuOperationMode, {
    GpuOperationMode::AllOn => "NVML_GOM_ALL_ON",
    GpuOperationMode::Compute => "NVML_GOM_COMPUTE",
    GpuOperationMode::LowDp => "NVML_GOM_LOW_DP",
});

impl_enum_display!(MigMode, {
    MigMode::Disabled => "NVML_DEVICE_MIG_DISABLE",
    MigMode::Enabled => "NVML_DEVICE_MIG_ENABLE",
});

impl_enum_display!(HostVgpuMode, {
    HostVgpuMode::NonSriov => "NVML_HOST_VGPU_MODE_NON_SRIOV",
    HostVgpuMode::Sriov => "NVML_HOST_VGPU_MODE_SRIOV",
});

impl_enum_display!(VgpuDriverCapability, {
    VgpuDriverCapability::HeterogeneousMultiVgpu => "NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU",
    VgpuDriverCapability::WarmUpdate => "NVML_VGPU_DRIVER_CAP_WARM_UPDATE",
});

impl_enum_display!(VgpuVmIdType, {
    VgpuVmIdType::DomainId => "NVML_VGPU_VM_ID_DOMAIN_ID",
    VgpuVmIdType::Uuid => "NVML_VGPU_VM_ID_UUID",
});

impl_enum_display!(VgpuSchedulerPolicy, {
    VgpuSchedulerPolicy::BestEffort => "NVML_VGPU_SCHEDULER_POLICY_BEST_EFFORT",
    VgpuSchedulerPolicy::EqualShare => "NVML_VGPU_SCHEDULER_POLICY_EQUAL_SHARE",
    VgpuSchedulerPolicy::FixedShare => "NVML_VGPU_SCHEDULER_POLICY_FIXED_SHARE",
});

impl_enum_display!(VgpuSchedulerArrMode, {
    VgpuSchedulerArrMode::Default => "NVML_VGPU_SCHEDULER_ARR_DEFAULT",
    VgpuSchedulerArrMode::Disabled => "NVML_VGPU_SCHEDULER_ARR_DISABLE",
    VgpuSchedulerArrMode::Enabled => "NVML_VGPU_SCHEDULER_ARR_ENABLE",
});

impl_enum_display!(VgpuSchedulerEngine, {
    VgpuSchedulerEngine::Graphics => "NVML_VGPU_SCHEDULER_ENGINE_TYPE_GRAPHICS",
});

impl_enum_display!(ComputeInstanceEngineProfile, {
    ComputeInstanceEngineProfile::Shared => "NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED",
    ComputeInstanceEngineProfile::Count => "NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT",
});

impl_enum_display!(LedColor, {
    LedColor::Green => "NVML_LED_COLOR_GREEN",
    LedColor::Amber => "NVML_LED_COLOR_AMBER",
});

impl_enum_display!(GpuFabricState, {
    GpuFabricState::NotSupported => "NVML_GPU_FABRIC_STATE_NOT_SUPPORTED",
    GpuFabricState::NotStarted => "NVML_GPU_FABRIC_STATE_NOT_STARTED",
    GpuFabricState::InProgress => "NVML_GPU_FABRIC_STATE_IN_PROGRESS",
    GpuFabricState::Completed => "NVML_GPU_FABRIC_STATE_COMPLETED",
});

impl_enum_display!(GpuFabricHealthSummary, {
    GpuFabricHealthSummary::NotSupported => "NVML_GPU_FABRIC_HEALTH_SUMMARY_NOT_SUPPORTED",
    GpuFabricHealthSummary::Healthy => "NVML_GPU_FABRIC_HEALTH_SUMMARY_HEALTHY",
    GpuFabricHealthSummary::Unhealthy => "NVML_GPU_FABRIC_HEALTH_SUMMARY_UNHEALTHY",
    GpuFabricHealthSummary::LimitedCapacity => "NVML_GPU_FABRIC_HEALTH_SUMMARY_LIMITED_CAPACITY",
});

impl_enum_display!(GpuFabricBooleanState, {
    GpuFabricBooleanState::NotSupported => "NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_NOT_SUPPORTED",
    GpuFabricBooleanState::True => "NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_TRUE",
    GpuFabricBooleanState::False => "NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE",
});

impl_enum_display!(GpuFabricIncorrectConfiguration, {
    GpuFabricIncorrectConfiguration::NotSupported => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NOT_SUPPORTED",
    GpuFabricIncorrectConfiguration::None => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NONE",
    GpuFabricIncorrectConfiguration::IncorrectSysguid => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_SYSGUID",
    GpuFabricIncorrectConfiguration::IncorrectChassisSerialNumber => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_CHASSIS_SN",
    GpuFabricIncorrectConfiguration::NoPartition => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NO_PARTITION",
    GpuFabricIncorrectConfiguration::InsufficientNvlinks => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INSUFFICIENT_NVLINKS",
    GpuFabricIncorrectConfiguration::IncompatibleGpuFirmware => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCOMPATIBLE_GPU_FW",
    GpuFabricIncorrectConfiguration::InvalidLocation => "NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INVALID_LOCATION",
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgpu_metadata_round_trips_through_raw_bytes() {
        let metadata = VgpuMetadata {
            version: 1,
            revision: 2,
            guest_info_state: VgpuGuestInfoState::Initialized,
            guest_driver_version: "580.00".to_owned(),
            host_driver_version: "580.01".to_owned(),
            virtualization_caps: 3,
            guest_vgpu_version: 4,
            opaque_data: vec![1, 2, 3, 4, 5, 6, 7],
        };
        let raw = metadata.encode_raw().unwrap();
        assert_eq!(VgpuMetadata::decode_raw(&raw).unwrap(), metadata);
        assert!(matches!(
            VgpuMetadata::decode_raw(&raw[..raw.len() - 1]),
            Err(Error::ShortOutput { .. })
        ));
    }

    #[test]
    fn test_pgpu_metadata_round_trips_through_raw_bytes() {
        let metadata = PgpuMetadata {
            version: 1,
            revision: 2,
            host_driver_version: "580.01".to_owned(),
            virtualization_caps: 3,
            host_supported_vgpu_range: VgpuVersion { min: 5, max: 9 },
            opaque_data: vec![9; 12],
        };
        let raw = metadata.encode_raw().unwrap();
        assert_eq!(PgpuMetadata::decode_raw(&raw).unwrap(), metadata);
    }
}
