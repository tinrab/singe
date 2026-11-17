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
use singe_core::{impl_enum_display, string_from_c_chars, string_from_c_ptr, try_enum_from_raw};
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result},
    gpu_instance::GpuInstance,
    library::Library,
    utility::{reported_entries, to_u32, to_usize},
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

// `impl_enum_display!` cannot name the `Unknown(code)` variant of a `native_enum!` catch-all.
macro_rules! impl_unknown_enum_display {
    ($enum:ty, { $($variant:path => $name:expr),+ $(,)? } $(,)?) => {
        impl Display for $enum {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                match self {
                    $($variant => f.write_str($name),)+
                    Self::Unknown(code) => write!(f, "unknown value {code}"),
                }
            }
        }
    };
}

bitflags::bitflags! {
    /// Flags controlling NVML initialization and device attachment.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct InitFlags: u32 {
        /// Initialize the NVML Library even when no devices are found.
        const NO_GPUS = sys::NVML_INIT_FLAG_NO_GPUS;
        /// Initialize the NVML Library without attaching any discovered devices.
        const NO_ATTACH = sys::NVML_INIT_FLAG_NO_ATTACH;
        /// Initializes GPUs even when an earlier initialization used `NO_GPUS` and `NO_ATTACH`.
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
        /// No event types.
        const NONE = sys::nvmlEventTypeNone as u64;
        /// Event about single bit ECC errors.
        const SINGLE_BIT_ECC_ERROR = sys::nvmlEventTypeSingleBitEccError as u64;
        /// Event about double bit ECC errors.
        const DOUBLE_BIT_ECC_ERROR = sys::nvmlEventTypeDoubleBitEccError as u64;
        /// Event about PState changes.
        const PSTATE = sys::nvmlEventTypePState as u64;
        /// Critical Xid error.
        const XID_CRITICAL_ERROR = sys::nvmlEventTypeXidCriticalError as u64;
        /// Clock change, reported only on Kepler.
        const CLOCK = sys::nvmlEventTypeClock as u64;
        /// Change between AC and battery power.
        const POWER_SOURCE_CHANGE = sys::nvmlEventTypePowerSourceChange as u64;
        /// MIG configuration change.
        const MIG_CONFIG_CHANGE = sys::nvmlEventMigConfigChange as u64;
        /// Storm of single-bit ECC errors.
        const SINGLE_BIT_ECC_ERROR_STORM = sys::nvmlEventTypeSingleBitEccErrorStorm as u64;
        /// DRAM page retirement.
        const DRAM_RETIREMENT_EVENT = sys::nvmlEventTypeDramRetirementEvent as u64;
        /// Failed DRAM page retirement.
        const DRAM_RETIREMENT_FAILURE = sys::nvmlEventTypeDramRetirementFailure as u64;
        /// Non-fatal poison error.
        const NON_FATAL_POISON_ERROR = sys::nvmlEventTypeNonFatalPoisonError as u64;
        /// Fatal poison error.
        const FATAL_POISON_ERROR = sys::nvmlEventTypeFatalPoisonError as u64;
        /// GPU became unavailable.
        const GPU_UNAVAILABLE_ERROR = sys::nvmlEventTypeGpuUnavailableError as u64;
        /// GPU recovery action required.
        const GPU_RECOVERY_ACTION = sys::nvmlEventTypeGpuRecoveryAction as u64;
        /// Every event type.
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
        /// A GPU was unbound from the driver.
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
        /// Default behavior.
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
        /// The profile supports peer-to-peer transfers.
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
        /// The profile supports graphics.
        const GFX = sys::NVML_COMPUTE_INSTANCE_PROFILE_CAPS_GFX;
    }
}

impl Display for ComputeInstanceProfileCapabilities {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// `NVLink` packet types that a utilization counter counts, as seen by the target GPU.
    ///
    /// The filter applies only to counters whose unit is packets or bytes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NvLinkPacketTypes: u32 {
        /// No-operation packets.
        const NOP = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_NOP.0;
        /// Read packets.
        const READ = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_READ.0;
        /// Write packets.
        const WRITE = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_WRITE.0;
        /// Reduction atomic requests.
        const RATOM = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RATOM.0;
        /// Non-reduction atomic requests.
        const NRATOM = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_NRATOM.0;
        /// Flush requests.
        const FLUSH = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_FLUSH.0;
        /// Responses with data.
        const RESPDATA = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA.0;
        /// Responses without data.
        const RESPNODATA = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA.0;
        /// All packets.
        const ALL = sys::nvmlNvLinkUtilizationCountPktTypes_t::NVML_NVLINK_COUNTER_PKTFILTER_ALL.0;
    }
}

impl Display for NvLinkPacketTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
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
    /// `NVLink` link state.
    pub const NVLINK_GET_STATE: Self = Self(sys::NVML_FI_DEV_NVLINK_GET_STATE);

    /// Recovery action the GPU requires.
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
    /// All Identifiers pertain to a device.
    /// Each ID is only used once and is guaranteed never to change.
    /// Current ECC mode. 1=Active. 0=Inactive.
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

    /// GPU power averaged over one second, in milliwatts.
    ///
    /// Requires power management support and an Ampere or newer GPU other than GA100.
    /// The field scope selects the GPU (0), the module including an NVIDIA CPU such as Grace (1), or the GPU base power (2).
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

    /// Total energy consumption in millijoules since the driver was last loaded.
    pub const ENERGY: Self = Self(sys::NVML_FI_DEV_ENERGY);

    /// Power limit requested by NVML or any other userspace client.
    pub const POWER_REQUESTED_LIMIT: Self = Self(sys::NVML_FI_DEV_POWER_REQUESTED_LIMIT);

    /// `PCIe` transmit bytes.
    /// Value can be wrapped.
    pub const PCIE_TX_BYTES: Self = Self(sys::NVML_FI_DEV_PCIE_COUNT_TX_BYTES);

    /// `PCIe` receive bytes.
    /// Value can be wrapped.
    pub const PCIE_RX_BYTES: Self = Self(sys::NVML_FI_DEV_PCIE_COUNT_RX_BYTES);

    /// `NVLink` transmitted data throughput in KiB.
    ///
    /// The field scope selects the link, and `u32::MAX` sums all links.
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
    /// These fields require an Ada or newer GPU.
    /// T.Limit temperature after which GPU may shut down for HW protection.
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
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 71 in MiB/sec.
    pub const NVLINK_L71_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 70 in MiB/sec.
    pub const NVLINK_L70_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 70 in MiB/sec.
    pub const NVLINK_L70_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 69 in MiB/sec.
    pub const NVLINK_L69_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 69 in MiB/sec.
    pub const NVLINK_L69_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 68 in MiB/sec.
    pub const NVLINK_L68_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 68 in MiB/sec.
    pub const NVLINK_L68_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 67 in MiB/sec.
    pub const NVLINK_L67_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 67 in MiB/sec.
    pub const NVLINK_L67_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 66 in MiB/sec.
    pub const NVLINK_L66_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 66 in MiB/sec.
    pub const NVLINK_L66_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 65 in MiB/sec.
    pub const NVLINK_L65_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 65 in MiB/sec.
    pub const NVLINK_L65_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 64 in MiB/sec.
    pub const NVLINK_L64_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 64 in MiB/sec.
    pub const NVLINK_L64_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 63 in MiB/sec.
    pub const NVLINK_L63_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 63 in MiB/sec.
    pub const NVLINK_L63_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 62 in MiB/sec.
    pub const NVLINK_L62_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 62 in MiB/sec.
    pub const NVLINK_L62_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 61 in MiB/sec.
    pub const NVLINK_L61_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 61 in MiB/sec.
    pub const NVLINK_L61_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 60 in MiB/sec.
    pub const NVLINK_L60_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 60 in MiB/sec.
    pub const NVLINK_L60_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 59 in MiB/sec.
    pub const NVLINK_L59_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 59 in MiB/sec.
    pub const NVLINK_L59_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 58 in MiB/sec.
    pub const NVLINK_L58_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 58 in MiB/sec.
    pub const NVLINK_L58_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 57 in MiB/sec.
    pub const NVLINK_L57_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 57 in MiB/sec.
    pub const NVLINK_L57_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 56 in MiB/sec.
    pub const NVLINK_L56_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 56 in MiB/sec.
    pub const NVLINK_L56_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 55 in MiB/sec.
    pub const NVLINK_L55_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 55 in MiB/sec.
    pub const NVLINK_L55_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 54 in MiB/sec.
    pub const NVLINK_L54_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 54 in MiB/sec.
    pub const NVLINK_L54_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 53 in MiB/sec.
    pub const NVLINK_L53_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 53 in MiB/sec.
    pub const NVLINK_L53_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 52 in MiB/sec.
    pub const NVLINK_L52_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 52 in MiB/sec.
    pub const NVLINK_L52_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 51 in MiB/sec.
    pub const NVLINK_L51_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 51 in MiB/sec.
    pub const NVLINK_L51_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 50 in MiB/sec.
    pub const NVLINK_L50_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 50 in MiB/sec.
    pub const NVLINK_L50_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 49 in MiB/sec.
    pub const NVLINK_L49_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 49 in MiB/sec.
    pub const NVLINK_L49_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 48 in MiB/sec.
    pub const NVLINK_L48_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 48 in MiB/sec.
    pub const NVLINK_L48_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 47 in MiB/sec.
    pub const NVLINK_L47_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 47 in MiB/sec.
    pub const NVLINK_L47_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 46 in MiB/sec.
    pub const NVLINK_L46_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 46 in MiB/sec.
    pub const NVLINK_L46_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 45 in MiB/sec.
    pub const NVLINK_L45_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 45 in MiB/sec.
    pub const NVLINK_L45_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 44 in MiB/sec.
    pub const NVLINK_L44_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 44 in MiB/sec.
    pub const NVLINK_L44_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 43 in MiB/sec.
    pub const NVLINK_L43_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 43 in MiB/sec.
    pub const NVLINK_L43_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 42 in MiB/sec.
    pub const NVLINK_L42_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 42 in MiB/sec.
    pub const NVLINK_L42_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 41 in MiB/sec.
    pub const NVLINK_L41_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 41 in MiB/sec.
    pub const NVLINK_L41_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 40 in MiB/sec.
    pub const NVLINK_L40_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 40 in MiB/sec.
    pub const NVLINK_L40_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 39 in MiB/sec.
    pub const NVLINK_L39_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 39 in MiB/sec.
    pub const NVLINK_L39_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 38 in MiB/sec.
    pub const NVLINK_L38_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 38 in MiB/sec.
    pub const NVLINK_L38_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 37 in MiB/sec.
    pub const NVLINK_L37_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 37 in MiB/sec.
    pub const NVLINK_L37_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 36 in MiB/sec.
    pub const NVLINK_L36_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 36 in MiB/sec.
    pub const NVLINK_L36_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_RX_PER_SEC.0);

    /// `NvLink` write for link 71 in bytes since reboot.
    pub const NVLINK_L71_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_TX.0);

    /// `NvLink` read for link 71 in bytes since reboot.
    pub const NVLINK_L71_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L71_RX.0);

    /// `NvLink` write for link 70 in bytes since reboot.
    pub const NVLINK_L70_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_TX.0);

    /// `NvLink` read for link 70 in bytes since reboot.
    pub const NVLINK_L70_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L70_RX.0);

    /// `NvLink` write for link 69 in bytes since reboot.
    pub const NVLINK_L69_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_TX.0);

    /// `NvLink` read for link 69 in bytes since reboot.
    pub const NVLINK_L69_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L69_RX.0);

    /// `NvLink` write for link 68 in bytes since reboot.
    pub const NVLINK_L68_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_TX.0);

    /// `NvLink` read for link 68 in bytes since reboot.
    pub const NVLINK_L68_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L68_RX.0);

    /// `NvLink` write for link 67 in bytes since reboot.
    pub const NVLINK_L67_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_TX.0);

    /// `NvLink` read for link 67 in bytes since reboot.
    pub const NVLINK_L67_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L67_RX.0);

    /// `NvLink` write for link 66 in bytes since reboot.
    pub const NVLINK_L66_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_TX.0);

    /// `NvLink` read for link 66 in bytes since reboot.
    pub const NVLINK_L66_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L66_RX.0);

    /// `NvLink` write for link 65 in bytes since reboot.
    pub const NVLINK_L65_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_TX.0);

    /// `NvLink` read for link 65 in bytes since reboot.
    pub const NVLINK_L65_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L65_RX.0);

    /// `NvLink` write for link 64 in bytes since reboot.
    pub const NVLINK_L64_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_TX.0);

    /// `NvLink` read for link 64 in bytes since reboot.
    pub const NVLINK_L64_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L64_RX.0);

    /// `NvLink` write for link 63 in bytes since reboot.
    pub const NVLINK_L63_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_TX.0);

    /// `NvLink` read for link 63 in bytes since reboot.
    pub const NVLINK_L63_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L63_RX.0);

    /// `NvLink` write for link 62 in bytes since reboot.
    pub const NVLINK_L62_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_TX.0);

    /// `NvLink` read for link 62 in bytes since reboot.
    pub const NVLINK_L62_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L62_RX.0);

    /// `NvLink` write for link 61 in bytes since reboot.
    pub const NVLINK_L61_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_TX.0);

    /// `NvLink` read for link 61 in bytes since reboot.
    pub const NVLINK_L61_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L61_RX.0);

    /// `NvLink` write for link 60 in bytes since reboot.
    pub const NVLINK_L60_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_TX.0);

    /// `NvLink` read for link 60 in bytes since reboot.
    pub const NVLINK_L60_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L60_RX.0);

    /// `NvLink` write for link 59 in bytes since reboot.
    pub const NVLINK_L59_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_TX.0);

    /// `NvLink` read for link 59 in bytes since reboot.
    pub const NVLINK_L59_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L59_RX.0);

    /// `NvLink` write for link 58 in bytes since reboot.
    pub const NVLINK_L58_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_TX.0);

    /// `NvLink` read for link 58 in bytes since reboot.
    pub const NVLINK_L58_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L58_RX.0);

    /// `NvLink` write for link 57 in bytes since reboot.
    pub const NVLINK_L57_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_TX.0);

    /// `NvLink` read for link 57 in bytes since reboot.
    pub const NVLINK_L57_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L57_RX.0);

    /// `NvLink` write for link 56 in bytes since reboot.
    pub const NVLINK_L56_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_TX.0);

    /// `NvLink` read for link 56 in bytes since reboot.
    pub const NVLINK_L56_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L56_RX.0);

    /// `NvLink` write for link 55 in bytes since reboot.
    pub const NVLINK_L55_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_TX.0);

    /// `NvLink` read for link 55 in bytes since reboot.
    pub const NVLINK_L55_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L55_RX.0);

    /// `NvLink` write for link 54 in bytes since reboot.
    pub const NVLINK_L54_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_TX.0);

    /// `NvLink` read for link 54 in bytes since reboot.
    pub const NVLINK_L54_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L54_RX.0);

    /// `NvLink` write for link 53 in bytes since reboot.
    pub const NVLINK_L53_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_TX.0);

    /// `NvLink` read for link 53 in bytes since reboot.
    pub const NVLINK_L53_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L53_RX.0);

    /// `NvLink` write for link 52 in bytes since reboot.
    pub const NVLINK_L52_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_TX.0);

    /// `NvLink` read for link 52 in bytes since reboot.
    pub const NVLINK_L52_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L52_RX.0);

    /// `NvLink` write for link 51 in bytes since reboot.
    pub const NVLINK_L51_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_TX.0);

    /// `NvLink` read for link 51 in bytes since reboot.
    pub const NVLINK_L51_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L51_RX.0);

    /// `NvLink` write for link 50 in bytes since reboot.
    pub const NVLINK_L50_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_TX.0);

    /// `NvLink` read for link 50 in bytes since reboot.
    pub const NVLINK_L50_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L50_RX.0);

    /// `NvLink` write for link 49 in bytes since reboot.
    pub const NVLINK_L49_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_TX.0);

    /// `NvLink` read for link 49 in bytes since reboot.
    pub const NVLINK_L49_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L49_RX.0);

    /// `NvLink` write for link 48 in bytes since reboot.
    pub const NVLINK_L48_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_TX.0);

    /// `NvLink` read for link 48 in bytes since reboot.
    pub const NVLINK_L48_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L48_RX.0);

    /// `NvLink` write for link 47 in bytes since reboot.
    pub const NVLINK_L47_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_TX.0);

    /// `NvLink` read for link 47 in bytes since reboot.
    pub const NVLINK_L47_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L47_RX.0);

    /// `NvLink` write for link 46 in bytes since reboot.
    pub const NVLINK_L46_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_TX.0);

    /// `NvLink` read for link 46 in bytes since reboot.
    pub const NVLINK_L46_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L46_RX.0);

    /// `NvLink` write for link 45 in bytes since reboot.
    pub const NVLINK_L45_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_TX.0);

    /// `NvLink` read for link 45 in bytes since reboot.
    pub const NVLINK_L45_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L45_RX.0);

    /// `NvLink` write for link 44 in bytes since reboot.
    pub const NVLINK_L44_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_TX.0);

    /// `NvLink` read for link 44 in bytes since reboot.
    pub const NVLINK_L44_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L44_RX.0);

    /// `NvLink` write for link 43 in bytes since reboot.
    pub const NVLINK_L43_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_TX.0);

    /// `NvLink` read for link 43 in bytes since reboot.
    pub const NVLINK_L43_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L43_RX.0);

    /// `NvLink` write for link 42 in bytes since reboot.
    pub const NVLINK_L42_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_TX.0);

    /// `NvLink` read for link 42 in bytes since reboot.
    pub const NVLINK_L42_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L42_RX.0);

    /// `NvLink` write for link 41 in bytes since reboot.
    pub const NVLINK_L41_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_TX.0);

    /// `NvLink` read for link 41 in bytes since reboot.
    pub const NVLINK_L41_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L41_RX.0);

    /// `NvLink` write for link 40 in bytes since reboot.
    pub const NVLINK_L40_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_TX.0);

    /// `NvLink` read for link 40 in bytes since reboot.
    pub const NVLINK_L40_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L40_RX.0);

    /// `NvLink` write for link 39 in bytes since reboot.
    pub const NVLINK_L39_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_TX.0);

    /// `NvLink` read for link 39 in bytes since reboot.
    pub const NVLINK_L39_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L39_RX.0);

    /// `NvLink` write for link 38 in bytes since reboot.
    pub const NVLINK_L38_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_TX.0);

    /// `NvLink` read for link 38 in bytes since reboot.
    pub const NVLINK_L38_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L38_RX.0);

    /// `NvLink` write for link 37 in bytes since reboot.
    pub const NVLINK_L37_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_TX.0);

    /// `NvLink` read for link 37 in bytes since reboot.
    pub const NVLINK_L37_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L37_RX.0);

    /// `NvLink` write for link 36 in bytes since reboot.
    pub const NVLINK_L36_TX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_TX.0);

    /// `NvLink` read for link 36 in bytes since reboot.
    pub const NVLINK_L36_RX: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L36_RX.0);

    /// The GPU's SM FP16 activity since reboot.
    pub const FP16_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP16_CYCLES_ACTIVE.0);

    /// The GPU's SM FP32 activity since reboot.
    pub const FP32_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP32_CYCLES_ACTIVE.0);

    /// The GPU's SM FP64 activity since reboot.
    pub const FP64_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP64_CYCLES_ACTIVE.0);

    /// The GPU's SM integer activity since reboot.
    pub const INTEGER_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_INTEGER_CYCLES_ACTIVE.0);

    /// The GPU's SM DFMA tensor activity since reboot.
    pub const DFMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DFMA_CYCLES_ACTIVE.0);

    /// The GPU's SM IMMA tensor activity since reboot.
    pub const IMMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_IMMA_CYCLES_ACTIVE.0);

    /// The GPU's SM HMMA tensor activity since reboot.
    pub const HMMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HMMA_CYCLES_ACTIVE.0);

    /// The GPU's SM DMMA tensor activity since reboot.
    pub const DMMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DMMA_CYCLES_ACTIVE.0);

    /// The GPU's SM MMA tensor activity since reboot.
    pub const MMA_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_MMA_CYCLES_ACTIVE.0);

    /// The GPU's SM activity since reboot.
    pub const SM_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_CYCLES_ACTIVE.0);

    /// The GPU's SM cycles elapsed since reboot.
    pub const SM_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_CYCLES_ELAPSED.0);

    /// `NvLink` write bandwidth for link 35 in MiB/sec.
    pub const NVLINK_L35_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L35_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 35 in MiB/sec.
    pub const NVLINK_L35_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L35_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 34 in MiB/sec.
    pub const NVLINK_L34_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L34_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 34 in MiB/sec.
    pub const NVLINK_L34_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L34_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 33 in MiB/sec.
    pub const NVLINK_L33_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L33_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 33 in MiB/sec.
    pub const NVLINK_L33_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L33_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 32 in MiB/sec.
    pub const NVLINK_L32_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L32_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 32 in MiB/sec.
    pub const NVLINK_L32_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L32_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 31 in MiB/sec.
    pub const NVLINK_L31_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L31_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 31 in MiB/sec.
    pub const NVLINK_L31_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L31_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 30 in MiB/sec.
    pub const NVLINK_L30_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L30_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 30 in MiB/sec.
    pub const NVLINK_L30_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L30_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 29 in MiB/sec.
    pub const NVLINK_L29_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L29_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 29 in MiB/sec.
    pub const NVLINK_L29_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L29_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 28 in MiB/sec.
    pub const NVLINK_L28_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L28_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 28 in MiB/sec.
    pub const NVLINK_L28_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L28_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 27 in MiB/sec.
    pub const NVLINK_L27_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L27_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 27 in MiB/sec.
    pub const NVLINK_L27_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L27_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 26 in MiB/sec.
    pub const NVLINK_L26_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L26_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 26 in MiB/sec.
    pub const NVLINK_L26_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L26_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 25 in MiB/sec.
    pub const NVLINK_L25_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L25_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 25 in MiB/sec.
    pub const NVLINK_L25_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L25_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 24 in MiB/sec.
    pub const NVLINK_L24_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L24_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 24 in MiB/sec.
    pub const NVLINK_L24_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L24_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 23 in MiB/sec.
    pub const NVLINK_L23_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L23_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 23 in MiB/sec.
    pub const NVLINK_L23_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L23_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 22 in MiB/sec.
    pub const NVLINK_L22_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L22_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 22 in MiB/sec.
    pub const NVLINK_L22_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L22_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 21 in MiB/sec.
    pub const NVLINK_L21_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L21_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 21 in MiB/sec.
    pub const NVLINK_L21_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L21_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 20 in MiB/sec.
    pub const NVLINK_L20_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L20_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 20 in MiB/sec.
    pub const NVLINK_L20_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L20_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 19 in MiB/sec.
    pub const NVLINK_L19_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L19_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 19 in MiB/sec.
    pub const NVLINK_L19_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L19_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 18 in MiB/sec.
    pub const NVLINK_L18_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L18_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 18 in MiB/sec.
    pub const NVLINK_L18_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L18_RX_PER_SEC.0);

    /// Percentage of time GR engine 7 context switches were active. 0.0 - 100.0.
    pub const GR7_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 7.
    pub const GR7_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 7.
    pub const GR7_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 7.
    pub const GR7_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 7.
    pub const GR7_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR7_CTXSW_CYCLES_ELAPSED.0);

    /// Percentage of time GR engine 6 context switches were active. 0.0 - 100.0.
    pub const GR6_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 6.
    pub const GR6_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 6.
    pub const GR6_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 6.
    pub const GR6_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 6.
    pub const GR6_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR6_CTXSW_CYCLES_ELAPSED.0);

    /// Percentage of time GR engine 5 context switches were active. 0.0 - 100.0.
    pub const GR5_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 5.
    pub const GR5_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 5.
    pub const GR5_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 5.
    pub const GR5_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 5.
    pub const GR5_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR5_CTXSW_CYCLES_ELAPSED.0);

    /// Percentage of time GR engine 4 context switches were active. 0.0 - 100.0.
    pub const GR4_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 4.
    pub const GR4_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 4.
    pub const GR4_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 4.
    pub const GR4_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 4.
    pub const GR4_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR4_CTXSW_CYCLES_ELAPSED.0);

    /// Percentage of time GR engine 3 context switches were active. 0.0 - 100.0.
    pub const GR3_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 3.
    pub const GR3_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 3.
    pub const GR3_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 3.
    pub const GR3_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 3.
    pub const GR3_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR3_CTXSW_CYCLES_ELAPSED.0);

    /// Percentage of time GR engine 2 context switches were active. 0.0 - 100.0.
    pub const GR2_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 2.
    pub const GR2_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 2.
    pub const GR2_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 2.
    pub const GR2_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 2.
    pub const GR2_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR2_CTXSW_CYCLES_ELAPSED.0);

    /// Percentage of time GR engine 1 context switches were active. 0.0 - 100.0.
    pub const GR1_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 1.
    pub const GR1_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 1.
    pub const GR1_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 1.
    pub const GR1_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 1.
    pub const GR1_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR1_CTXSW_CYCLES_ELAPSED.0);

    /// Percentage of time GR engine 0 context switches were active. 0.0 - 100.0.
    pub const GR0_CTXSW_ACTIVE_PCT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_ACTIVE_PCT.0);

    /// Average context switch cycles per request for GR engine 0.
    pub const GR0_CTXSW_CYCLES_PER_REQ: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_CYCLES_PER_REQ.0);

    /// Number of context switch requests for GR engine 0.
    pub const GR0_CTXSW_REQUESTS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_REQUESTS.0);

    /// Active context switch cycles for GR engine 0.
    pub const GR0_CTXSW_CYCLES_ACTIVE: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_CYCLES_ACTIVE.0);

    /// Total context switch cycles elapsed for GR engine 0.
    pub const GR0_CTXSW_CYCLES_ELAPSED: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GR0_CTXSW_CYCLES_ELAPSED.0);

    /// Percent utilization of NVENC 3. 0.0 - 100.0.
    pub const NVENC_3_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_3_UTIL.0);

    /// Percent utilization of NVENC 2. 0.0 - 100.0.
    pub const NVENC_2_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_2_UTIL.0);

    /// Percent utilization of NVENC 1. 0.0 - 100.0.
    pub const NVENC_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_1_UTIL.0);

    /// Percent utilization of NVENC 0. 0.0 - 100.0.
    pub const NVENC_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVENC_0_UTIL.0);

    /// Percentage of DRAM cache misses. 0.0 - 100.0.
    pub const DRAM_CACHE_MISS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DRAM_CACHE_MISS.0);

    /// Percentage of DRAM cache hits. 0.0 - 100.0.
    pub const DRAM_CACHE_HIT: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DRAM_CACHE_HIT.0);

    /// Percentage of peer memory cache misses. 0.0 - 100.0.
    pub const PEERMEM_CACHE_MISS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PEERMEM_CACHE_MISS.0);

    /// Percentage of peer memory cache hits. 0.0 - 100.0.
    pub const PEERMEM_CACHE_HIT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PEERMEM_CACHE_HIT.0);

    /// Percentage of host memory cache misses. 0.0 - 100.0.
    pub const HOSTMEM_CACHE_MISS: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HOSTMEM_CACHE_MISS.0);

    /// Percentage of host memory cache hits. 0.0 - 100.0.
    pub const HOSTMEM_CACHE_HIT: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HOSTMEM_CACHE_HIT.0);

    /// C2C link 13 data receive bandwidth in MiB/sec.
    pub const C2C_LINK13_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_DATA_RX_PER_SEC.0);

    /// C2C link 13 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK13_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_DATA_TX_PER_SEC.0);

    /// C2C link 13 total receive bandwidth in MiB/sec.
    pub const C2C_LINK13_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_TOTAL_RX_PER_SEC.0);

    /// C2C link 13 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK13_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK13_TOTAL_TX_PER_SEC.0);

    /// C2C link 12 data receive bandwidth in MiB/sec.
    pub const C2C_LINK12_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_DATA_RX_PER_SEC.0);

    /// C2C link 12 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK12_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_DATA_TX_PER_SEC.0);

    /// C2C link 12 total receive bandwidth in MiB/sec.
    pub const C2C_LINK12_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_TOTAL_RX_PER_SEC.0);

    /// C2C link 12 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK12_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK12_TOTAL_TX_PER_SEC.0);

    /// C2C link 11 data receive bandwidth in MiB/sec.
    pub const C2C_LINK11_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_DATA_RX_PER_SEC.0);

    /// C2C link 11 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK11_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_DATA_TX_PER_SEC.0);

    /// C2C link 11 total receive bandwidth in MiB/sec.
    pub const C2C_LINK11_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_TOTAL_RX_PER_SEC.0);

    /// C2C link 11 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK11_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK11_TOTAL_TX_PER_SEC.0);

    /// C2C link 10 data receive bandwidth in MiB/sec.
    pub const C2C_LINK10_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_DATA_RX_PER_SEC.0);

    /// C2C link 10 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK10_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_DATA_TX_PER_SEC.0);

    /// C2C link 10 total receive bandwidth in MiB/sec.
    pub const C2C_LINK10_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_TOTAL_RX_PER_SEC.0);

    /// C2C link 10 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK10_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK10_TOTAL_TX_PER_SEC.0);

    /// C2C link 9 data receive bandwidth in MiB/sec.
    pub const C2C_LINK9_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_DATA_RX_PER_SEC.0);

    /// C2C link 9 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK9_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_DATA_TX_PER_SEC.0);

    /// C2C link 9 total receive bandwidth in MiB/sec.
    pub const C2C_LINK9_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_TOTAL_RX_PER_SEC.0);

    /// C2C link 9 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK9_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK9_TOTAL_TX_PER_SEC.0);

    /// C2C link 8 data receive bandwidth in MiB/sec.
    pub const C2C_LINK8_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_DATA_RX_PER_SEC.0);

    /// C2C link 8 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK8_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_DATA_TX_PER_SEC.0);

    /// C2C link 8 total receive bandwidth in MiB/sec.
    pub const C2C_LINK8_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_TOTAL_RX_PER_SEC.0);

    /// C2C link 8 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK8_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK8_TOTAL_TX_PER_SEC.0);

    /// C2C link 7 data receive bandwidth in MiB/sec.
    pub const C2C_LINK7_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_DATA_RX_PER_SEC.0);

    /// C2C link 7 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK7_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_DATA_TX_PER_SEC.0);

    /// C2C link 7 total receive bandwidth in MiB/sec.
    pub const C2C_LINK7_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_TOTAL_RX_PER_SEC.0);

    /// C2C link 7 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK7_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK7_TOTAL_TX_PER_SEC.0);

    /// C2C link 6 data receive bandwidth in MiB/sec.
    pub const C2C_LINK6_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_DATA_RX_PER_SEC.0);

    /// C2C link 6 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK6_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_DATA_TX_PER_SEC.0);

    /// C2C link 6 total receive bandwidth in MiB/sec.
    pub const C2C_LINK6_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_TOTAL_RX_PER_SEC.0);

    /// C2C link 6 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK6_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK6_TOTAL_TX_PER_SEC.0);

    /// C2C link 5 data receive bandwidth in MiB/sec.
    pub const C2C_LINK5_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_DATA_RX_PER_SEC.0);

    /// C2C link 5 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK5_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_DATA_TX_PER_SEC.0);

    /// C2C link 5 total receive bandwidth in MiB/sec.
    pub const C2C_LINK5_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_TOTAL_RX_PER_SEC.0);

    /// C2C link 5 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK5_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK5_TOTAL_TX_PER_SEC.0);

    /// C2C link 4 data receive bandwidth in MiB/sec.
    pub const C2C_LINK4_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_DATA_RX_PER_SEC.0);

    /// C2C link 4 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK4_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_DATA_TX_PER_SEC.0);

    /// C2C link 4 total receive bandwidth in MiB/sec.
    pub const C2C_LINK4_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_TOTAL_RX_PER_SEC.0);

    /// C2C link 4 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK4_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK4_TOTAL_TX_PER_SEC.0);

    /// C2C link 3 data receive bandwidth in MiB/sec.
    pub const C2C_LINK3_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_DATA_RX_PER_SEC.0);

    /// C2C link 3 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK3_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_DATA_TX_PER_SEC.0);

    /// C2C link 3 total receive bandwidth in MiB/sec.
    pub const C2C_LINK3_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_TOTAL_RX_PER_SEC.0);

    /// C2C link 3 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK3_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK3_TOTAL_TX_PER_SEC.0);

    /// C2C link 2 data receive bandwidth in MiB/sec.
    pub const C2C_LINK2_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_DATA_RX_PER_SEC.0);

    /// C2C link 2 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK2_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_DATA_TX_PER_SEC.0);

    /// C2C link 2 total receive bandwidth in MiB/sec.
    pub const C2C_LINK2_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_TOTAL_RX_PER_SEC.0);

    /// C2C link 2 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK2_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK2_TOTAL_TX_PER_SEC.0);

    /// C2C link 1 data receive bandwidth in MiB/sec.
    pub const C2C_LINK1_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_DATA_RX_PER_SEC.0);

    /// C2C link 1 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK1_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_DATA_TX_PER_SEC.0);

    /// C2C link 1 total receive bandwidth in MiB/sec.
    pub const C2C_LINK1_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_TOTAL_RX_PER_SEC.0);

    /// C2C link 1 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK1_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK1_TOTAL_TX_PER_SEC.0);

    /// C2C link 0 data receive bandwidth in MiB/sec.
    pub const C2C_LINK0_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_DATA_RX_PER_SEC.0);

    /// C2C link 0 data transmit bandwidth in MiB/sec.
    pub const C2C_LINK0_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_DATA_TX_PER_SEC.0);

    /// C2C link 0 total receive bandwidth in MiB/sec.
    pub const C2C_LINK0_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_TOTAL_RX_PER_SEC.0);

    /// C2C link 0 total transmit bandwidth in MiB/sec.
    pub const C2C_LINK0_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_LINK0_TOTAL_TX_PER_SEC.0);

    /// C2C data receive bandwidth in MiB/sec.
    pub const C2C_DATA_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_DATA_RX_PER_SEC.0);

    /// C2C data transmit bandwidth in MiB/sec.
    pub const C2C_DATA_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_DATA_TX_PER_SEC.0);

    /// C2C total receive bandwidth in MiB/sec.
    pub const C2C_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_TOTAL_RX_PER_SEC.0);

    /// C2C total transmit bandwidth in MiB/sec.
    pub const C2C_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_C2C_TOTAL_TX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 17 in MiB/sec.
    pub const NVLINK_L17_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L17_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 17 in MiB/sec.
    pub const NVLINK_L17_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L17_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 16 in MiB/sec.
    pub const NVLINK_L16_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L16_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 16 in MiB/sec.
    pub const NVLINK_L16_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L16_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 15 in MiB/sec.
    pub const NVLINK_L15_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L15_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 15 in MiB/sec.
    pub const NVLINK_L15_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L15_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 14 in MiB/sec.
    pub const NVLINK_L14_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L14_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 14 in MiB/sec.
    pub const NVLINK_L14_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L14_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 13 in MiB/sec.
    pub const NVLINK_L13_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L13_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 13 in MiB/sec.
    pub const NVLINK_L13_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L13_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 12 in MiB/sec.
    pub const NVLINK_L12_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L12_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 12 in MiB/sec.
    pub const NVLINK_L12_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L12_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 11 in MiB/sec.
    pub const NVLINK_L11_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L11_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 11 in MiB/sec.
    pub const NVLINK_L11_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L11_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 10 in MiB/sec.
    pub const NVLINK_L10_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L10_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 10 in MiB/sec.
    pub const NVLINK_L10_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L10_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 9 in MiB/sec.
    pub const NVLINK_L9_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L9_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 9 in MiB/sec.
    pub const NVLINK_L9_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L9_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 8 in MiB/sec.
    pub const NVLINK_L8_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L8_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 8 in MiB/sec.
    pub const NVLINK_L8_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L8_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 7 in MiB/sec.
    pub const NVLINK_L7_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L7_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 7 in MiB/sec.
    pub const NVLINK_L7_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L7_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 6 in MiB/sec.
    pub const NVLINK_L6_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L6_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 6 in MiB/sec.
    pub const NVLINK_L6_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L6_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 5 in MiB/sec.
    pub const NVLINK_L5_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L5_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 5 in MiB/sec.
    pub const NVLINK_L5_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L5_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 4 in MiB/sec.
    pub const NVLINK_L4_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L4_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 4 in MiB/sec.
    pub const NVLINK_L4_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L4_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 3 in MiB/sec.
    pub const NVLINK_L3_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L3_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 3 in MiB/sec.
    pub const NVLINK_L3_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L3_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 2 in MiB/sec.
    pub const NVLINK_L2_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L2_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 2 in MiB/sec.
    pub const NVLINK_L2_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L2_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 1 in MiB/sec.
    pub const NVLINK_L1_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L1_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 1 in MiB/sec.
    pub const NVLINK_L1_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L1_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for link 0 in MiB/sec.
    pub const NVLINK_L0_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L0_TX_PER_SEC.0);

    /// `NvLink` read bandwidth for link 0 in MiB/sec.
    pub const NVLINK_L0_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_L0_RX_PER_SEC.0);

    /// Percent utilization of NVOFA 1. 0.0 - 100.0.
    pub const NVOFA_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVOFA_1_UTIL.0);

    /// Percent utilization of NVOFA 0. 0.0 - 100.0.
    pub const NVOFA_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVOFA_0_UTIL.0);

    /// Percent utilization of NVJPG 7. 0.0 - 100.0.
    pub const NVJPG_7_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_7_UTIL.0);

    /// Percent utilization of NVJPG 6. 0.0 - 100.0.
    pub const NVJPG_6_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_6_UTIL.0);

    /// Percent utilization of NVJPG 5. 0.0 - 100.0.
    pub const NVJPG_5_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_5_UTIL.0);

    /// Percent utilization of NVJPG 4. 0.0 - 100.0.
    pub const NVJPG_4_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_4_UTIL.0);

    /// Percent utilization of NVJPG 3. 0.0 - 100.0.
    pub const NVJPG_3_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_3_UTIL.0);

    /// Percent utilization of NVJPG 2. 0.0 - 100.0.
    pub const NVJPG_2_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_2_UTIL.0);

    /// Percent utilization of NVJPG 1. 0.0 - 100.0.
    pub const NVJPG_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_1_UTIL.0);

    /// Percent utilization of NVJPG 0. 0.0 - 100.0.
    pub const NVJPG_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVJPG_0_UTIL.0);

    /// Percent utilization of NVDEC 7. 0.0 - 100.0.
    pub const NVDEC_7_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_7_UTIL.0);

    /// Percent utilization of NVDEC 6. 0.0 - 100.0.
    pub const NVDEC_6_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_6_UTIL.0);

    /// Percent utilization of NVDEC 5. 0.0 - 100.0.
    pub const NVDEC_5_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_5_UTIL.0);

    /// Percent utilization of NVDEC 4. 0.0 - 100.0.
    pub const NVDEC_4_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_4_UTIL.0);

    /// Percent utilization of NVDEC 3. 0.0 - 100.0.
    pub const NVDEC_3_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_3_UTIL.0);

    /// Percent utilization of NVDEC 2. 0.0 - 100.0.
    pub const NVDEC_2_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_2_UTIL.0);

    /// Percent utilization of NVDEC 1. 0.0 - 100.0.
    pub const NVDEC_1_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_1_UTIL.0);

    /// Percent utilization of NVDEC 0. 0.0 - 100.0.
    pub const NVDEC_0_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVDEC_0_UTIL.0);

    /// Percentage of time the GPU's SMs were doing DMMA tensor operations. 0.0 - 100.0.
    pub const DMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DMMA_TENSOR_UTIL.0);

    /// Percentage of time any compute/graphics app was active on the GPU. 0.0 - 100.0.
    pub const GRAPHICS_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_GRAPHICS_UTIL.0);

    /// Percentage of SMs that were busy. 0.0 - 100.0.
    pub const SM_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_UTIL.0);

    /// Percentage of warps that were active vs theoretical maximum. 0.0 - 100.0.
    pub const SM_OCCUPANCY: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_SM_OCCUPANCY.0);

    /// Percentage of time the GPU's SMs were doing integer operations. 0.0 - 100.0.
    pub const INTEGER_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_INTEGER_UTIL.0);

    /// Percentage of time the GPU's SMs were doing ANY tensor operations. 0.0 - 100.0.
    pub const ANY_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_ANY_TENSOR_UTIL.0);

    /// Percentage of time the GPU's SMs were doing DFMA tensor operations. 0.0 - 100.0.
    pub const DFMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DFMA_TENSOR_UTIL.0);

    /// Percentage of time the GPU's SMs were doing HMMA tensor operations. 0.0 - 100.0.
    pub const HMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_HMMA_TENSOR_UTIL.0);

    /// Percentage of time the GPU's SMs were doing IMMA tensor operations. 0.0 - 100.0.
    pub const IMMA_TENSOR_UTIL: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_IMMA_TENSOR_UTIL.0);

    /// Percentage of the theoretical DRAM bandwidth used, from 0.0 to 100.0.
    pub const DRAM_BW_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_DRAM_BW_UTIL.0);

    /// Percentage of time the GPU's SMs were doing non-tensor FP64 math. 0.0 - 100.0.
    pub const FP64_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP64_UTIL.0);

    /// Percentage of time the GPU's SMs were doing non-tensor FP32 math. 0.0 - 100.0.
    pub const FP32_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP32_UTIL.0);

    /// Percentage of time the GPU's SMs were doing non-tensor FP16 math. 0.0 - 100.0.
    pub const FP16_UTIL: Self = Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_FP16_UTIL.0);

    /// `PCIe` traffic from this GPU in MiB/sec.
    pub const PCIE_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PCIE_TX_PER_SEC.0);

    /// `PCIe` traffic to this GPU in MiB/sec.
    pub const PCIE_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_PCIE_RX_PER_SEC.0);

    /// `NvLink` read bandwidth for all links in MiB/sec.
    pub const NVLINK_TOTAL_RX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC.0);

    /// `NvLink` write bandwidth for all links in MiB/sec.
    pub const NVLINK_TOTAL_TX_PER_SEC: Self =
        Self(sys::nvmlGpmMetricId_t::NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC.0);
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

singe_core::native_enum! {
    /// vGPU GUEST info state.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum VgpuGuestInfoState: sys::nvmlVgpuGuestInfoState_t, unknown = Unknown(u32) {
        /// Guest-dependent fields uninitialized.
        Uninitialized = NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED,
        /// Guest-dependent fields initialized.
        Initialized = NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED,
    }
}

bitflags::bitflags! {
    /// vGPU VM compatibility codes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VgpuVmCompatibility: u32 {
        /// vGPU is not runnable.
        const NONE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_NONE.0;
        /// vGPU is runnable from a cold / powered-off state (ACPI S5).
        const COLD = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_COLD.0;
        /// vGPU is runnable from a hibernated state (ACPI S4).
        const HIBERNATE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_HIBERNATE.0;
        /// vGPU is runnable from a sleeped state (ACPI S3).
        const SLEEP = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_SLEEP.0;
        /// vGPU is runnable from a live/paused (ACPI S0).
        const LIVE = sys::nvmlVgpuVmCompatibility_t::NVML_VGPU_VM_COMPATIBILITY_LIVE.0;
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
        const NONE = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_NONE.0;
        /// Compatibility is limited by host driver version.
        const HOST_DRIVER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER.0;
        /// Compatibility is limited by guest driver version.
        const GUEST_DRIVER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER.0;
        /// Compatibility is limited by GPU hardware.
        const GPU = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_GPU.0;
        /// Compatibility is limited by an undefined factor.
        const OTHER = sys::nvmlVgpuPgpuCompatibilityLimitCode_t::NVML_VGPU_COMPATIBILITY_LIMIT_OTHER.0;
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
    /// vGPUs of different types share the GPU.
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

singe_core::native_enum! {
    /// Whether a feature is enabled.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum EnableState: sys::nvmlEnableState_t, unknown = Unknown(u32) {
        /// Feature disabled.
        Disabled = NVML_FEATURE_DISABLED,
        /// Feature enabled.
        Enabled = NVML_FEATURE_ENABLED,
    }
}

/// Product brand reported by NVML.
///
/// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Brand {
    /// NVIDIA RTX Virtual Game Dev.
    NvidiaVgamedev,
    /// NVIDIA NPU.
    NvidiaNpu,
    /// Deprecated NVIDIA Deep Learning Accelerator.
    NvidiaDla,
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
    Unknown(u32),
}

impl Brand {
    /// Decodes a native product brand while keeping unrecognized codes in [`Self::Unknown`].
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        match value {
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VGAMEDEV.0 => {
                Self::NvidiaVgamedev
            }
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_NPU.0 => Self::NvidiaNpu,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_DLA.0 => Self::NvidiaDla,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_QUADRO.0 => Self::Quadro,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_TESLA.0 => Self::Tesla,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVS.0 => Self::Nvs,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_GRID.0 => Self::Grid,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_GEFORCE.0 => Self::GeForce,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_TITAN.0 => Self::Titan,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VAPPS.0 => Self::NvidiaVApps,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VPC.0 => Self::NvidiaVpc,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VCS.0 => Self::NvidiaVcs,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_VWS.0 => Self::NvidiaVws,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_CLOUD_GAMING.0 => {
                Self::NvidiaCloudGaming
            }
            value if value == sys::nvmlBrandType_t::NVML_BRAND_QUADRO_RTX.0 => Self::QuadroRtx,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA_RTX.0 => Self::NvidiaRtx,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_NVIDIA.0 => Self::Nvidia,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_GEFORCE_RTX.0 => Self::GeForceRtx,
            value if value == sys::nvmlBrandType_t::NVML_BRAND_TITAN_RTX.0 => Self::TitanRtx,
            raw => Self::Unknown(raw),
        }
    }
}

singe_core::native_enum! {
    /// Temperature sensors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum TemperatureSensor: sys::nvmlTemperatureSensors_t, unknown = Unknown(u32) {
        /// Temperature from the hottest part of the GPU die.
        GpuMax = NVML_TEMPERATURE_GPU_MAX,
        /// Temperature sensor for the GPU die.
        Gpu = NVML_TEMPERATURE_GPU,
    }
}

singe_core::native_enum! {
    /// Temperature thresholds.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum TemperatureThreshold: sys::nvmlTemperatureThresholds_t {
        /// Temperature at which the GPU will shut down for HW protection.
        Shutdown = NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
        /// Temperature at which the GPU will begin HW slowdown.
        Slowdown = NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
        /// Memory Temperature at which the GPU will begin SW slowdown.
        MemoryMax = NVML_TEMPERATURE_THRESHOLD_MEM_MAX,
        /// GPU Temperature at which the GPU can be throttled below base clock.
        GpuMax = NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
        /// Minimum GPU Temperature that can be set as acoustic threshold.
        AcousticMin = NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN,
        /// Current temperature that is set as acoustic threshold.
        AcousticCurrent = NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR,
        /// Maximum GPU temperature that can be set as acoustic threshold.
        AcousticMax = NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX,
        /// Current temperature that is set as gps threshold.
        GpuCurrent = NVML_TEMPERATURE_THRESHOLD_GPS_CURR,
    }
}

singe_core::native_enum! {
    /// Compute mode, which limits how many CUDA contexts can use a GPU.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum ComputeMode: sys::nvmlComputeMode_t, unknown = Unknown(u32) {
        /// Default compute mode , multiple contexts per device.
        Default = NVML_COMPUTEMODE_DEFAULT,
        /// Support Removed.
        ExclusiveThread = NVML_COMPUTEMODE_EXCLUSIVE_THREAD,
        /// Compute-prohibited mode , no contexts per device.
        Prohibited = NVML_COMPUTEMODE_PROHIBITED,
        /// Compute-exclusive-process mode , only one context per device, usable from multiple threads at a time.
        ExclusiveProcess = NVML_COMPUTEMODE_EXCLUSIVE_PROCESS,
    }
}

/// GPU or accelerator architecture reported by NVML.
///
/// [`Self::Unknown`] holds values this crate does not name.
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
    Unknown(u32),
}

impl Architecture {
    /// Decodes a native device architecture while keeping unrecognized codes in [`Self::Unknown`].
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
            raw => Self::Unknown(raw),
        }
    }
}

singe_core::native_enum! {
    /// Clock types.
    ///
    /// All speeds are in Mhz.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum ClockType: sys::nvmlClockType_t, unknown = Unknown(u32) {
        /// Graphics clock domain.
        Graphics = NVML_CLOCK_GRAPHICS,
        /// SM clock domain.
        Sm = NVML_CLOCK_SM,
        /// Memory clock domain.
        Memory = NVML_CLOCK_MEM,
        /// Video encoder/decoder clock domain.
        Video = NVML_CLOCK_VIDEO,
    }
}

singe_core::native_enum! {
    /// Unit counted by an `NVLink` utilization counter.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum NvLinkUtilizationCountUnits: sys::nvmlNvLinkUtilizationCountUnits_t,
        unknown = Unknown(u32)
    {
        /// Counts cycles.
        Cycles = NVML_NVLINK_COUNTER_UNIT_CYCLES,
        /// Counts packets.
        Packets = NVML_NVLINK_COUNTER_UNIT_PACKETS,
        /// Counts bytes.
        Bytes = NVML_NVLINK_COUNTER_UNIT_BYTES,
        /// Reserved for internal use.
        Reserved = NVML_NVLINK_COUNTER_UNIT_RESERVED,
        /// Number of values in this enumeration.
        Count = NVML_NVLINK_COUNTER_UNIT_COUNT,
    }
}

singe_core::native_enum! {
    /// Queryable `NVLink` capability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum NvLinkCapability: sys::nvmlNvLinkCapability_t {
        /// Peer-to-peer transfers over `NVLink` are supported.
        P2pSupported = NVML_NVLINK_CAP_P2P_SUPPORTED,
        /// Access to system memory is supported.
        SysmemAccess = NVML_NVLINK_CAP_SYSMEM_ACCESS,
        /// Peer-to-peer atomics are supported.
        P2pAtomics = NVML_NVLINK_CAP_P2P_ATOMICS,
        /// System memory atomics are supported.
        SysmemAtomics = NVML_NVLINK_CAP_SYSMEM_ATOMICS,
        /// SLI is supported over this link.
        SliBridge = NVML_NVLINK_CAP_SLI_BRIDGE,
        /// The link is supported on this device.
        Valid = NVML_NVLINK_CAP_VALID,
        /// Number of values in this enumeration.
        Count = NVML_NVLINK_CAP_COUNT,
    }
}

singe_core::native_enum! {
    /// Queryable `NVLink` error counter.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum NvLinkErrorCounter: sys::nvmlNvLinkErrorCounter_t {
        /// Data link transmit replay errors.
        DlReplay = NVML_NVLINK_ERROR_DL_REPLAY,
        /// Data link transmit recovery errors.
        DlRecovery = NVML_NVLINK_ERROR_DL_RECOVERY,
        /// Data link receive flow control digit CRC errors.
        DlCrcFlit = NVML_NVLINK_ERROR_DL_CRC_FLIT,
        /// Data link receive data CRC errors.
        DlCrcData = NVML_NVLINK_ERROR_DL_CRC_DATA,
        /// Data link receive data ECC errors.
        DlEccData = NVML_NVLINK_ERROR_DL_ECC_DATA,
        /// Number of values in this enumeration.
        Count = NVML_NVLINK_ERROR_COUNT,
    }
}

singe_core::native_enum! {
    /// Type of the device at the remote end of an `NVLink`.
    ///
    /// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum NvLinkRemoteDeviceType: sys::nvmlIntNvLinkDeviceType_t, unknown = Unknown(u32) {
        /// The remote `NVLink` endpoint is a GPU.
        Gpu = NVML_NVLINK_DEVICE_TYPE_GPU,
        /// The remote `NVLink` endpoint is an IBM NPU.
        IbmNpu = NVML_NVLINK_DEVICE_TYPE_IBMNPU,
        /// The remote `NVLink` endpoint is a switch.
        Switch = NVML_NVLINK_DEVICE_TYPE_SWITCH,
    }
}

singe_core::native_enum! {
    /// `NVLink` version.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum NvLinkVersion: sys::nvmlNvlinkVersion_t, unknown = Unknown(u32) {
        /// `NVLink` version is invalid.
        Invalid = NVML_NVLINK_VERSION_INVALID,
        /// `NVLink` Version 1.0.
        V1_0 = NVML_NVLINK_VERSION_1_0,
        /// `NVLink` Version 2.0.
        V2_0 = NVML_NVLINK_VERSION_2_0,
        /// `NVLink` Version 2.2.
        V2_2 = NVML_NVLINK_VERSION_2_2,
        /// `NVLink` Version 3.0.
        V3_0 = NVML_NVLINK_VERSION_3_0,
        /// `NVLink` Version 3.1.
        V3_1 = NVML_NVLINK_VERSION_3_1,
        /// `NVLink` Version 4.0.
        V4_0 = NVML_NVLINK_VERSION_4_0,
        /// `NVLink` Version 5.0.
        V5_0 = NVML_NVLINK_VERSION_5_0,
    }
}

singe_core::native_enum! {
    /// Workload power profile identifiers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PowerProfileType: sys::nvmlPowerProfileType_t {
        /// `LowPriorityBackground`.
        LowPriorityBackground = NVML_POWER_PROFILE_LOW_PRIORITY_BACKGROUND,
        /// `NetworkBound`.
        NetworkBound = NVML_POWER_PROFILE_NETWORK_BOUND,
        /// Training.
        Training = NVML_POWER_PROFILE_TRAINING,
        /// `MediumThroughputInference`.
        MediumThroughputInference = NVML_POWER_PROFILE_MEDIUM_THROUGHPUT_INFERENCE,
        /// `HighThroughputInference`.
        HighThroughputInference = NVML_POWER_PROFILE_HIGH_THROUGHPUT_INFERENCE,
        /// `LowLatencyInference`.
        LowLatencyInference = NVML_POWER_PROFILE_LOW_LATENCY_INFERENCE,
        /// Inference.
        Inference = NVML_POWER_PROFILE_INFERENCE,
        /// `MaxQ3`.
        MaxQ3 = NVML_POWER_PROFILE_MAX_Q_3,
        /// `MaxQ2`.
        MaxQ2 = NVML_POWER_PROFILE_MAX_Q_2,
        /// `MaxQ1`.
        MaxQ1 = NVML_POWER_PROFILE_MAX_Q_1,
        /// Identifies the native maximum-performance workload power profile.
        /// The identifier selects a bit in workload profile masks.
        MaxP = NVML_POWER_PROFILE_MAX_P,
        /// Identifies the native Max-Q workload power profile.
        /// The identifier selects a bit in workload profile masks.
        MaxQ = NVML_POWER_PROFILE_MAX_Q,
        /// Identifies the native compute workload power profile.
        /// The identifier selects a bit in workload profile masks.
        Compute = NVML_POWER_PROFILE_COMPUTE,
        /// Identifies the native memory-bound workload power profile.
        /// The identifier selects a bit in workload profile masks.
        MemoryBound = NVML_POWER_PROFILE_MEMORY_BOUND,
        /// Identifies the native network workload power profile.
        /// The identifier selects a bit in workload profile masks.
        Network = NVML_POWER_PROFILE_NETWORK,
        /// Identifies the native balanced workload power profile.
        /// The identifier selects a bit in workload profile masks.
        Balanced = NVML_POWER_PROFILE_BALANCED,
        /// Identifies the native large-language-model inference workload power profile.
        /// The identifier selects a bit in workload profile masks.
        LlmInference = NVML_POWER_PROFILE_LLM_INFERENCE,
        /// Identifies the native large-language-model training workload power profile.
        /// The identifier selects a bit in workload profile masks.
        LlmTraining = NVML_POWER_PROFILE_LLM_TRAINING,
        /// Identifies the native RBM workload power profile.
        /// The identifier selects a bit in workload profile masks.
        Rbm = NVML_POWER_PROFILE_RBM,
        /// Identifies the native DCPCIE workload power profile.
        /// The identifier selects a bit in workload profile masks.
        Dcpcie = NVML_POWER_PROFILE_DCPCIE,
        /// Identifies the native sparse HMMA workload power profile.
        /// The identifier selects a bit in workload profile masks.
        HmmaSparse = NVML_POWER_PROFILE_HMMA_SPARSE,
        /// Identifies the native dense HMMA workload power profile.
        /// The identifier selects a bit in workload profile masks.
        HmmaDense = NVML_POWER_PROFILE_HMMA_DENSE,
        /// Identifies the native synchronized balanced workload power profile.
        /// The identifier selects a bit in workload profile masks.
        SyncBalanced = NVML_POWER_PROFILE_SYNC_BALANCED,
        /// Identifies the native high-performance-computing workload power profile.
        /// The identifier selects a bit in workload profile masks.
        Hpc = NVML_POWER_PROFILE_HPC,
        /// Identifies the native MIG workload power profile.
        /// The identifier selects a bit in workload profile masks.
        Mig = NVML_POWER_PROFILE_MIG,
        /// One past the last workload power profile identifier.
        Max = NVML_POWER_PROFILE_MAX,
    }
}

/// `PowerMizer` policy for device performance and power use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PowerMizerMode {
    /// Adjusts GPU clocks to GPU utilization.
    Adaptive = sys::NVML_POWER_MIZER_MODE_ADAPTIVE,
    /// Raises GPU clocks for maximum performance, as far as thermal and other limits allow.
    PreferMaximumPerformance = sys::NVML_POWER_MIZER_MODE_PREFER_MAXIMUM_PERFORMANCE,
    /// `PowerMizer` mode is driver controlled.
    Auto = sys::NVML_POWER_MIZER_MODE_AUTO,
    /// Locks the GPU to its base clocks.
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

singe_core::native_enum! {
    /// Clock setting to query, combined with a [`ClockType`] to select one clock value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ClockId: sys::nvmlClockId_t {
        /// Current actual clock value.
        Current = NVML_CLOCK_ID_CURRENT,
        /// Target application clock.
        /// Deprecated, do not use.
        #[deprecated]
        AppClockTarget = NVML_CLOCK_ID_APP_CLOCK_TARGET,
        /// Default application clock target Deprecated, do not use.
        #[deprecated]
        AppClockDefault = NVML_CLOCK_ID_APP_CLOCK_DEFAULT,
        /// OEM-defined maximum clock rate.
        CustomerBoostMax = NVML_CLOCK_ID_CUSTOMER_BOOST_MAX,
    }
}

singe_core::native_enum! {
    /// Component a thermal sensor measures.
    ///
    /// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum ThermalTarget: sys::nvmlThermalTarget_t, unknown = Unknown(i32) {
        /// No thermal target is selected.
        None = NVML_THERMAL_TARGET_NONE,
        /// GPU core temperature requires `NvPhysicalGpuHandle`.
        Gpu = NVML_THERMAL_TARGET_GPU,
        /// GPU memory temperature requires `NvPhysicalGpuHandle`.
        Memory = NVML_THERMAL_TARGET_MEMORY,
        /// GPU power supply temperature requires `NvPhysicalGpuHandle`.
        PowerSupply = NVML_THERMAL_TARGET_POWER_SUPPLY,
        /// GPU board ambient temperature requires `NvPhysicalGpuHandle`.
        Board = NVML_THERMAL_TARGET_BOARD,
        /// Visual Computing Device Board temperature requires `NvVisualComputingDeviceHandle`.
        VcdBoard = NVML_THERMAL_TARGET_VCD_BOARD,
        /// Visual Computing Device Inlet temperature requires `NvVisualComputingDeviceHandle`.
        VcdInlet = NVML_THERMAL_TARGET_VCD_INLET,
        /// Visual Computing Device Outlet temperature requires `NvVisualComputingDeviceHandle`.
        VcdOutlet = NVML_THERMAL_TARGET_VCD_OUTLET,
        /// Selects all thermal targets.
        All = NVML_THERMAL_TARGET_ALL,
    }
}

singe_core::native_enum! {
    /// Controller of a thermal sensor.
    ///
    /// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum ThermalController: sys::nvmlThermalController_t, unknown = Unknown(i32) {
        /// No thermal controller is present.
        None = NVML_THERMAL_CONTROLLER_NONE,
        /// The GPU internal thermal controller.
        GpuInternal = NVML_THERMAL_CONTROLLER_GPU_INTERNAL,
        /// An ADM1032 thermal controller.
        Adm1032 = NVML_THERMAL_CONTROLLER_ADM1032,
        /// An ADT7461 thermal controller.
        Adt7461 = NVML_THERMAL_CONTROLLER_ADT7461,
        /// A MAX6649 thermal controller.
        Max6649 = NVML_THERMAL_CONTROLLER_MAX6649,
        /// A MAX1617 thermal controller.
        Max1617 = NVML_THERMAL_CONTROLLER_MAX1617,
        /// An LM99 thermal controller.
        Lm99 = NVML_THERMAL_CONTROLLER_LM99,
        /// An LM89 thermal controller.
        Lm89 = NVML_THERMAL_CONTROLLER_LM89,
        /// An LM64 thermal controller.
        Lm64 = NVML_THERMAL_CONTROLLER_LM64,
        /// A G781 thermal controller.
        G781 = NVML_THERMAL_CONTROLLER_G781,
        /// An ADT7473 thermal controller.
        Adt7473 = NVML_THERMAL_CONTROLLER_ADT7473,
        /// An SBMAX6649 thermal controller.
        Sbmax6649 = NVML_THERMAL_CONTROLLER_SBMAX6649,
        /// The VBIOS event thermal controller.
        VbiosEvt = NVML_THERMAL_CONTROLLER_VBIOSEVT,
        /// The operating-system thermal controller.
        Os = NVML_THERMAL_CONTROLLER_OS,
        /// The Canoas NV system controller.
        NvsysconCanoas = NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS,
        /// The E551 NV system controller.
        NvsysconE551 = NVML_THERMAL_CONTROLLER_NVSYSCON_E551,
        /// A MAX6649R thermal controller.
        Max6649r = NVML_THERMAL_CONTROLLER_MAX6649R,
        /// An ADT7473S thermal controller.
        Adt7473s = NVML_THERMAL_CONTROLLER_ADT7473S,
    }
}

singe_core::native_enum! {
    /// Addressing mode between the GPU and system memory.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum DeviceAddressingMode: sys::nvmlDeviceAddressingModeType_t, unknown = Unknown(u32) {
        /// No active mode.
        None = NVML_DEVICE_ADDRESSING_MODE_NONE,
        /// Heterogeneous Memory Management mode.
        Hmm = NVML_DEVICE_ADDRESSING_MODE_HMM,
        /// Address Translation Services mode.
        Ats = NVML_DEVICE_ADDRESSING_MODE_ATS,
    }
}

singe_core::native_enum! {
    /// Type of a bridge chip.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum BridgeChipType: sys::nvmlBridgeChipType_t, unknown = Unknown(u32) {
        /// Identifies a PLX bridge chip.
        Plx = NVML_BRIDGE_CHIP_PLX,
        /// Identifies a BRO4 bridge chip.
        Bro4 = NVML_BRIDGE_CHIP_BRO4,
    }
}

singe_core::native_enum! {
    /// Allowed `PStates`.
    ///
    /// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum PerformanceState: sys::nvmlPstates_t, unknown = Unknown(u32) {
        /// Performance state 0 , Maximum Performance.
        P0 = NVML_PSTATE_0,
        /// Performance state 1.
        P1 = NVML_PSTATE_1,
        /// Performance state 2.
        P2 = NVML_PSTATE_2,
        /// Performance state 3.
        P3 = NVML_PSTATE_3,
        /// Performance state 4.
        P4 = NVML_PSTATE_4,
        /// Performance state 5.
        P5 = NVML_PSTATE_5,
        /// Performance state 6.
        P6 = NVML_PSTATE_6,
        /// Performance state 7.
        P7 = NVML_PSTATE_7,
        /// Performance state 8.
        P8 = NVML_PSTATE_8,
        /// Performance state 9.
        P9 = NVML_PSTATE_9,
        /// Performance state 10.
        P10 = NVML_PSTATE_10,
        /// Performance state 11.
        P11 = NVML_PSTATE_11,
        /// Performance state 12.
        P12 = NVML_PSTATE_12,
        /// Performance state 13.
        P13 = NVML_PSTATE_13,
        /// Performance state 14.
        P14 = NVML_PSTATE_14,
        /// Performance state 15 , Minimum Performance.
        P15 = NVML_PSTATE_15,
    }
}

singe_core::native_enum! {
    /// Queryable `PCIe` utilization counter.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PcieUtilCounter: sys::nvmlPcieUtilCounter_t {
        /// Transmitted bytes, in KiB units.
        TxBytes = NVML_PCIE_UTIL_TX_BYTES,
        /// Received bytes, in KiB units.
        RxBytes = NVML_PCIE_UTIL_RX_BYTES,
    }
}

singe_core::native_enum! {
    /// Closest common ancestor of two GPUs in the system topology.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum TopologyLevel: sys::nvmlGpuTopologyLevel_t, unknown = Unknown(u32) {
        /// GPUs on the same board, such as a Tesla K80.
        Internal = NVML_TOPOLOGY_INTERNAL,
        /// GPUs connected through a single `PCIe` switch.
        Single = NVML_TOPOLOGY_SINGLE,
        /// GPUs connected through multiple `PCIe` switches without a host bridge.
        Multiple = NVML_TOPOLOGY_MULTIPLE,
        /// GPUs connected to the same host bridge.
        HostBridge = NVML_TOPOLOGY_HOSTBRIDGE,
        /// GPUs on the same NUMA node, possibly behind different host bridges.
        Node = NVML_TOPOLOGY_NODE,
        /// Any GPUs in the system.
        System = NVML_TOPOLOGY_SYSTEM,
    }
}

singe_core::native_enum! {
    /// Peer-to-peer capability to query.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum P2pCapabilityIndex: sys::nvmlGpuP2PCapsIndex_t {
        /// Selects peer-to-peer read capability for a P2P status query.
        Read = NVML_P2P_CAPS_INDEX_READ,
        /// Selects peer-to-peer write capability for a P2P status query.
        Write = NVML_P2P_CAPS_INDEX_WRITE,
        /// Selects `NVLink` peer-to-peer capability for a P2P status query.
        Nvlink = NVML_P2P_CAPS_INDEX_NVLINK,
        /// Selects peer-to-peer atomic-operation capability for a P2P status query.
        Atomics = NVML_P2P_CAPS_INDEX_ATOMICS,
        /// Deprecated name for the PCI peer-to-peer capability selector.
        Prop = NVML_P2P_CAPS_INDEX_PROP,
        /// Unknown peer-to-peer capability index.
        Unknown = NVML_P2P_CAPS_INDEX_UNKNOWN,
    }
}

singe_core::native_enum! {
    /// Status of a peer-to-peer capability.
    ///
    /// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum P2pStatus: sys::nvmlGpuP2PStatus_t, unknown = Unknown(u32) {
        /// The requested peer-to-peer capability is available.
        Ok = NVML_P2P_STATUS_OK,
        /// The chipset does not support the requested peer-to-peer operation.
        ChipsetNotSupported = NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED,
        /// The GPU does not support the requested peer-to-peer capability.
        GpuNotSupported = NVML_P2P_STATUS_GPU_NOT_SUPPORTED,
        /// The I/O hub topology does not support the requested peer-to-peer capability.
        IoHubNotSupported = NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED,
        /// The requested peer-to-peer capability is disabled by registry configuration.
        DisabledByRegKey = NVML_P2P_STATUS_DISABLED_BY_REGKEY,
        /// The requested peer-to-peer capability is unsupported.
        NotSupported = NVML_P2P_STATUS_NOT_SUPPORTED,
    }
}

singe_core::native_enum! {
    /// Available infoROM objects.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum InforomObject: sys::nvmlInforomObject_t {
        /// An object defined by OEM.
        Oem = NVML_INFOROM_OEM,
        /// The ECC object determining the level of ECC support.
        Ecc = NVML_INFOROM_ECC,
        /// The power management object.
        Power = NVML_INFOROM_POWER,
        /// DRAM Encryption object.
        Den = NVML_INFOROM_DEN,
    }
}

singe_core::native_enum! {
    /// Memory error types.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryErrorType: sys::nvmlMemoryErrorType_t {
        /// A corrected memory error: a single-bit ECC error, or a texture memory error fixed by a resend.
        Corrected = NVML_MEMORY_ERROR_TYPE_CORRECTED,
        /// An uncorrected memory error: a double-bit ECC error, or a texture memory error whose resend failed.
        Uncorrected = NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
    }
}

singe_core::native_enum! {
    /// ECC counter types.
    ///
    /// Note: Volatile counts are reset each time the driver loads.
    /// On Windows this is once per boot.
    /// On Linux this can be more frequent.
    /// On Linux the driver unloads when no active clients exist.
    /// If persistence mode is enabled or there is always a driver client active (e.g. X11), then Linux also sees per-boot behavior.
    /// If not, volatile counts are reset each time a compute app is run.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum EccCounterType: sys::nvmlEccCounterType_t {
        /// Volatile counts are reset each time the driver loads.
        Volatile = NVML_VOLATILE_ECC,
        /// Aggregate counts persist across reboots (i.e. for the lifetime of the device).
        Aggregate = NVML_AGGREGATE_ECC,
    }
}

singe_core::native_enum! {
    /// Memory location whose ECC error counter is queried.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryLocation: sys::nvmlMemoryLocation_t {
        /// GPU L1 Cache.
        L1Cache = NVML_MEMORY_LOCATION_L1_CACHE,
        /// GPU L2 Cache.
        L2Cache = NVML_MEMORY_LOCATION_L2_CACHE,
        /// Turing+ DRAM.
        Dram = NVML_MEMORY_LOCATION_DRAM,
        /// GPU Register File.
        RegisterFile = NVML_MEMORY_LOCATION_REGISTER_FILE,
        /// GPU Texture Memory.
        TextureMemory = NVML_MEMORY_LOCATION_TEXTURE_MEMORY,
        /// Shared memory.
        TextureShm = NVML_MEMORY_LOCATION_TEXTURE_SHM,
        /// CBU.
        Cbu = NVML_MEMORY_LOCATION_CBU,
        /// Turing+ SRAM.
        Sram = NVML_MEMORY_LOCATION_SRAM,
    }
}

singe_core::native_enum! {
    /// Performance policy whose violation time can be queried.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PerfPolicyType: sys::nvmlPerfPolicyType_t {
        /// How long did power violations cause the GPU to be below application clocks.
        Power = NVML_PERF_POLICY_POWER,
        /// How long did thermal violations cause the GPU to be below application clocks.
        Thermal = NVML_PERF_POLICY_THERMAL,
        /// How long did sync boost cause the GPU to be below application clocks.
        SyncBoost = NVML_PERF_POLICY_SYNC_BOOST,
        /// How long did the board limit cause the GPU to be below application clocks.
        BoardLimit = NVML_PERF_POLICY_BOARD_LIMIT,
        /// How long did low utilization cause the GPU to be below application clocks.
        LowUtilization = NVML_PERF_POLICY_LOW_UTILIZATION,
        /// How long did the board reliability limit cause the GPU to be below application clocks.
        Reliability = NVML_PERF_POLICY_RELIABILITY,
        /// Total time the GPU was held below application clocks by any limiter (0 - 5 above).
        TotalAppClocks = NVML_PERF_POLICY_TOTAL_APP_CLOCKS,
        /// Total time the GPU was held below base clocks.
        TotalBaseClocks = NVML_PERF_POLICY_TOTAL_BASE_CLOCKS,
    }
}

singe_core::native_enum! {
    /// Causes for page retirement.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PageRetirementCause: sys::nvmlPageRetirementCause_t {
        /// Page was retired due to multiple single bit ECC error.
        MultipleSingleBitEccErrors = NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
        /// Page was retired due to double bit ECC error.
        DoubleBitEccError = NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
    }
}

singe_core::native_enum! {
    /// Type of a sample or field value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum FieldValueType: sys::nvmlValueType_t, unknown = Unknown(u32) {
        /// Double-precision floating-point value.
        Double = NVML_VALUE_TYPE_DOUBLE,
        /// Unsigned 32-bit integer value.
        UnsignedInt = NVML_VALUE_TYPE_UNSIGNED_INT,
        /// Unsigned long integer value.
        UnsignedLong = NVML_VALUE_TYPE_UNSIGNED_LONG,
        /// Unsigned 64-bit integer value.
        UnsignedLongLong = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG,
        /// Signed 64-bit integer value.
        SignedLongLong = NVML_VALUE_TYPE_SIGNED_LONG_LONG,
        /// Signed 32-bit integer value.
        SignedInt = NVML_VALUE_TYPE_SIGNED_INT,
        /// Unsigned 16-bit integer value.
        UnsignedShort = NVML_VALUE_TYPE_UNSIGNED_SHORT,
    }
}

singe_core::native_enum! {
    /// Quantity that a sample measures.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SamplingType: sys::nvmlSamplingType_t {
        /// To represent total power drawn by GPU.
        TotalPower = NVML_TOTAL_POWER_SAMPLES,
        /// To represent percent of time during which one or more kernels was executing on the GPU.
        GpuUtilization = NVML_GPU_UTILIZATION_SAMPLES,
        /// To represent percent of time during which global (device) memory was being read or written.
        MemoryUtilization = NVML_MEMORY_UTILIZATION_SAMPLES,
        /// To represent percent of time during which NVENC remains busy.
        EncoderUtilization = NVML_ENC_UTILIZATION_SAMPLES,
        /// To represent percent of time during which NVDEC remains busy.
        DecoderUtilization = NVML_DEC_UTILIZATION_SAMPLES,
        /// To represent processor clock samples.
        ProcessorClock = NVML_PROCESSOR_CLK_SAMPLES,
        /// To represent memory clock samples.
        MemoryClock = NVML_MEMORY_CLK_SAMPLES,
        /// To represent module power samples for total module starting Grace Hopper.
        ModulePower = NVML_MODULE_POWER_SAMPLES,
        /// To represent percent of time during which NVJPG remains busy.
        JpgUtilization = NVML_JPG_UTILIZATION_SAMPLES,
        /// To represent percent of time during which NVOFA remains busy.
        OfaUtilization = NVML_OFA_UTILIZATION_SAMPLES,
    }
}

/// Kind of work a process runs on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ProcessMode {
    /// Processes with a compute context.
    Compute = sys::nvmlProcessMode_enum::NVML_PROCESS_MODE_COMPUTE.0,
    /// Processes with a graphics context.
    Graphics = sys::nvmlProcessMode_enum::NVML_PROCESS_MODE_GRAPHICS.0,
    /// Processes with a MPS (Multi-Process Service) compute context.
    MpsCompute = sys::nvmlProcessMode_enum::NVML_PROCESS_MODE_MPS.0,
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
/// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BusType {
    /// PCI bus.
    Pci,
    /// PCI-Express bus.
    Pcie,
    /// FPCI bus.
    Fpci,
    /// AGP bus.
    Agp,
    /// Unrecognized native bus type code, preserved without conversion.
    Unknown(u32),
}

impl BusType {
    /// Decodes a native bus type while keeping unrecognized codes in [`Self::Unknown`].
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        match value {
            sys::NVML_BUS_TYPE_PCI => Self::Pci,
            sys::NVML_BUS_TYPE_PCIE => Self::Pcie,
            sys::NVML_BUS_TYPE_FPCI => Self::Fpci,
            sys::NVML_BUS_TYPE_AGP => Self::Agp,
            raw => Self::Unknown(raw),
        }
    }
}

/// Power source reported for a device.
///
/// [`Self::Unknown`] holds values this crate does not name.
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
    Unknown(u32),
}

impl PowerSource {
    /// Decodes a native power source while keeping unrecognized codes in [`Self::Unknown`].
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        match value {
            sys::NVML_POWER_SOURCE_AC => Self::Ac,
            sys::NVML_POWER_SOURCE_BATTERY => Self::Battery,
            sys::NVML_POWER_SOURCE_UNDERSIZED => Self::Undersized,
            raw => Self::Unknown(raw),
        }
    }
}

singe_core::native_enum! {
    /// Encoder type whose capacity can be queried.
    ///
    /// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum EncoderType: sys::nvmlEncoderType_t, unknown = Unknown(u32) {
        /// H264 encoder.
        H264 = NVML_ENCODER_QUERY_H264,
        /// HEVC encoder.
        Hevc = NVML_ENCODER_QUERY_HEVC,
        /// AV1 encoder.
        Av1 = NVML_ENCODER_QUERY_AV1,
    }
}

singe_core::native_enum! {
    /// API types that allow changes to default permission restrictions.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum RestrictedApi: sys::nvmlRestrictedAPI_t {
        /// APIs that change application clocks.
        #[deprecated]
        SetApplicationsClocks = NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
        /// APIs that enable or disable auto-boosted clocks.
        SetAutoBoostedClocks = NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS,
        /// Number of restricted API types.
        Count = NVML_RESTRICTED_API_COUNT,
    }
}

/// Adaptive-clocking enablement reported by NVML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AdaptiveClockInfoStatus {
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
    /// The driver controls the fan speed from the temperature.
    TemperatureContinuousSw = sys::NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW,
    /// Manual fan control policy.
    Manual = sys::NVML_FAN_POLICY_MANUAL,
}

singe_core::native_enum! {
    /// Health of a unit fan.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum FanState: sys::nvmlFanState_t, unknown = Unknown(u32) {
        /// Fan is working properly.
        Normal = NVML_FAN_NORMAL,
        /// Fan has failed.
        Failed = NVML_FAN_FAILED,
    }
}

singe_core::native_enum! {
    /// Cooler control type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum CoolerControl: sys::nvmlCoolerControl_t, unknown = Unknown(u32) {
        /// This cooler has no control signal.
        None = NVML_THERMAL_COOLER_SIGNAL_NONE,
        /// This cooler can only be toggled either ON or OFF (eg a switch).
        Toggle = NVML_THERMAL_COOLER_SIGNAL_TOGGLE,
        /// This cooler's level can be adjusted from some minimum to some maximum (eg a knob).
        Variable = NVML_THERMAL_COOLER_SIGNAL_VARIABLE,
        /// Number of values in this enumeration.
        Count = NVML_THERMAL_COOLER_SIGNAL_COUNT,
    }
}

singe_core::native_enum! {
    /// Cooler's target.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum CoolerTarget: sys::nvmlCoolerTarget_t, unknown = Unknown(u32) {
        /// This cooler cools nothing.
        None = NVML_THERMAL_COOLER_TARGET_NONE,
        /// This cooler can cool the GPU.
        Gpu = NVML_THERMAL_COOLER_TARGET_GPU,
        /// This cooler can cool the memory.
        Memory = NVML_THERMAL_COOLER_TARGET_MEMORY,
        /// This cooler can cool the power supply.
        PowerSupply = NVML_THERMAL_COOLER_TARGET_POWER_SUPPLY,
        /// This cooler cools all of the components related to its target gpu.
        /// `GPU_RELATED` = GPU | MEMORY | `POWER_SUPPLY`.
        GpuRelated = NVML_THERMAL_COOLER_TARGET_GPU_RELATED,
    }
}

singe_core::native_enum! {
    /// Frame buffer capture session type.
    ///
    /// [`Self::Unknown`] holds values this crate does not name, including the native unknown value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum FbcSessionType: sys::nvmlFBCSessionType_t, unknown = Unknown(u32) {
        /// `ToSys`.
        ToSys = NVML_FBC_SESSION_TYPE_TOSYS,
        /// CUDA.
        Cuda = NVML_FBC_SESSION_TYPE_CUDA,
        /// Vid.
        Vid = NVML_FBC_SESSION_TYPE_VID,
        /// `HEnc`.
        HwEnc = NVML_FBC_SESSION_TYPE_HWENC,
    }
}

singe_core::native_enum! {
    /// Driver models.
    ///
    /// Windows only.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum DriverModel: sys::nvmlDriverModel_t, unknown = Unknown(u32) {
        /// WDDM driver model , GPU treated as a display device.
        Wddm = NVML_DRIVER_WDDM,
        /// WDM (TCC) model (deprecated) , GPU treated as a generic compute device.
        #[deprecated]
        Wdm = NVML_DRIVER_WDM,
        /// MCDM driver model , GPU treated as a Microsoft compute device.
        Mcdm = NVML_DRIVER_MCDM,
    }
}

singe_core::native_enum! {
    /// GPU Operation Mode.
    ///
    /// GOM allows to reduce power usage and optimize GPU throughput by disabling GPU features.
    ///
    /// Each GOM is designed to meet specific user needs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum GpuOperationMode: sys::nvmlGpuOperationMode_t, unknown = Unknown(u32) {
        /// Everything is enabled and running at full speed.
        AllOn = NVML_GOM_ALL_ON,
        /// Designed for running only compute tasks.
        /// Graphics operations are not allowed.
        Compute = NVML_GOM_COMPUTE,
        /// Designed for running graphics applications that don't require high bandwidth double precision.
        LowDp = NVML_GOM_LOW_DP,
    }
}

/// Multi-instance GPU partitioning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MigMode {
    /// MIG mode is disabled.
    Disabled = sys::NVML_DEVICE_MIG_DISABLE,
    /// MIG mode is enabled.
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
/// [`Self::Unknown`] holds values this crate does not name.
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
    Unknown(u32),
}

impl VirtualizationMode {
    /// Decodes a native GPU virtualization mode while keeping unrecognized codes in [`Self::Unknown`].
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        match value {
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_NONE.0 =>
            {
                Self::None
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH.0 =>
            {
                Self::Passthrough
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_VGPU.0 =>
            {
                Self::Vgpu
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU.0 =>
            {
                Self::HostVgpu
            }
            value
                if value
                    == sys::nvmlGpuVirtualizationMode_t::NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA.0 =>
            {
                Self::HostVsga
            }
            raw => Self::Unknown(raw),
        }
    }
}

singe_core::native_enum! {
    /// Host vGPU modes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum HostVgpuMode: sys::nvmlHostVgpuMode_t, unknown = Unknown(u32) {
        /// Non SR-IOV mode.
        NonSriov = NVML_HOST_VGPU_MODE_NON_SRIOV,
        /// SR-IOV mode.
        Sriov = NVML_HOST_VGPU_MODE_SRIOV,
    }
}

singe_core::native_enum! {
    /// vGPU driver queryable capabilities.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum VgpuDriverCapability: sys::nvmlVgpuDriverCapability_t {
        /// Supports mixing of different vGPU profiles within one guest VM.
        HeterogeneousMultiVgpu = NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU,
        /// Supports FSR and warm update of vGPU host driver without terminating the running guest VM.
        WarmUpdate = NVML_VGPU_DRIVER_CAP_WARM_UPDATE,
    }
}

singe_core::native_enum! {
    /// Types of VM identifiers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum VgpuVmIdType: sys::nvmlVgpuVmIdType_t, unknown = Unknown(u32) {
        /// VM ID represents DOMAIN ID.
        DomainId = NVML_VGPU_VM_ID_DOMAIN_ID,
        /// VM ID represents UUID.
        Uuid = NVML_VGPU_VM_ID_UUID,
    }
}

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

singe_core::native_enum! {
    /// Color of a unit LED.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum LedColor: sys::nvmlLedColor_t, unknown = Unknown(u32) {
        /// GREEN, indicates good health.
        Green = NVML_LED_COLOR_GREEN,
        /// AMBER, indicates problem.
        Amber = NVML_LED_COLOR_AMBER,
    }
}

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
    /// Percent of the process's lifetime during which one or more kernels ran on the GPU.
    ///
    /// The value is `u32::MAX` when the GPU does not report utilization.
    pub gpu_utilization: u32,
    /// Percent of the process's lifetime during which device memory was read or written.
    ///
    /// The value is `u32::MAX` when the GPU does not report utilization.
    pub memory_utilization: u32,
    /// Maximum total memory in bytes that the process allocated.
    ///
    /// The value is `u64::MAX` when the GPU does not report process memory use.
    pub max_memory_usage: u64,
    /// Amount of time in ms during which the compute context was active.
    /// The time is reported as 0 if the process is not terminated.
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
    /// InfiniBand GUID reported by the platform.
    /// On Blackwell, bytes 8 through 15 are zero.
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
    /// Fabric probing is not supported.
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
    /// Fabric health is not supported.
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
    /// The flag is not supported.
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
    /// Configuration checks are not supported.
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
#[derive(Debug, PartialEq, Eq, Hash)]
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
    /// Native session flag bits.
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

impl GpuInstanceInfo {
    // The caller must pass information that NVML returned while `library`'s initialization was alive.
    pub(crate) unsafe fn from_raw(value: sys::nvmlGpuInstanceInfo_t, library: &Library) -> Self {
        Self {
            // SAFETY: the caller passes information NVML returned under `library`, which the `Device` retains.
            device: unsafe { Device::from_raw(value.device, library) },
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

impl ComputeInstanceInfo {
    // The caller must pass information that NVML returned while `library`'s initialization was alive.
    pub(crate) unsafe fn from_raw(
        value: sys::nvmlComputeInstanceInfo_t,
        library: &Library,
    ) -> Self {
        Self {
            // SAFETY: the caller passes information NVML returned under `library`, which the `Device` retains.
            device: unsafe { Device::from_raw(value.device, library) },
            // SAFETY: the caller passes information NVML returned under `library`, which the `GpuInstance` retains.
            gpu_instance: unsafe { GpuInstance::from_raw(value.gpuInstance, library) },
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
            vm_compatibility: VgpuVmCompatibility::from_bits_retain(value.vgpuVmCompatibility.0),
            limit_code: VgpuPgpuCompatibilityLimitCode::from_bits_retain(
                value.compatibilityLimitCode.0,
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

// Splits a variable-length metadata blob into its fixed header and opaque data.
fn metadata_opaque_data(
    name: &'static str,
    raw: &[u8],
    header_size: usize,
    opaque_offset: usize,
    opaque_size_offset: usize,
) -> Result<Vec<u8>> {
    if raw.len() < header_size {
        return Err(Error::BufferTooSmall {
            name: name.into(),
            required: header_size,
            actual: raw.len(),
        });
    }

    let opaque_size = to_usize(
        read_u32_at(raw, opaque_size_offset),
        "NVML opaque data size",
    )?;
    let opaque_end = opaque_offset.saturating_add(opaque_size);
    if raw.len() < opaque_end {
        return Err(Error::BufferTooSmall {
            name: name.into(),
            required: opaque_end,
            actual: raw.len(),
        });
    }
    Ok(raw[opaque_offset..opaque_end].to_vec())
}

impl VgpuMetadata {
    // Decodes a metadata blob written by `nvmlVgpuInstanceGetMetadata`.
    pub(crate) fn decode_raw(raw: &[u8]) -> Result<Self> {
        type Raw = sys::nvmlVgpuMetadata_t;
        let opaque_data = metadata_opaque_data(
            "vgpu metadata",
            raw,
            size_of::<Raw>(),
            mem::offset_of!(Raw, opaqueData),
            mem::offset_of!(Raw, opaqueDataSize),
        )?;

        let guest_info_state = read_u32_at(raw, mem::offset_of!(Raw, guestInfoState));
        Ok(Self {
            version: read_u32_at(raw, mem::offset_of!(Raw, version)),
            revision: read_u32_at(raw, mem::offset_of!(Raw, revision)),
            guest_info_state: sys::nvmlVgpuGuestInfoState_t(guest_info_state).into(),
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
            sys::nvmlVgpuGuestInfoState_t::from(self.guest_info_state).0,
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
    // Decodes a metadata blob written by `nvmlDeviceGetVgpuMetadata`.
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

        let entries = reported_entries(&info.logEntries, info.entriesCount)
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

impl GpmMetric {
    // The caller must pass a metric that NVML filled, whose name pointers are null or NUL-terminated strings NVML owns.
    pub(crate) unsafe fn from_raw(value: sys::nvmlGpmMetric_t) -> Self {
        let result = if value.nvmlReturn == sys::nvmlReturn_t::NVML_SUCCESS {
            Ok(value.value)
        } else {
            Err(value.nvmlReturn.into())
        };

        Self {
            metric_id: GpmMetricId(value.metricId),
            result,
            // SAFETY: the caller passes a metric NVML filled, whose `shortName` is null or a NUL-terminated string NVML owns.
            short_name: unsafe { string_from_c_ptr(value.metricInfo.shortName) },
            // SAFETY: the caller passes a metric NVML filled, whose `longName` is null or a NUL-terminated string NVML owns.
            long_name: unsafe { string_from_c_ptr(value.metricInfo.longName) },
            // SAFETY: the caller passes a metric NVML filled, whose `unit` is null or a NUL-terminated string NVML owns.
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
            bridge_chips: reported_entries(&value.bridgeChipInfo, value.bridgeCount)
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
            faults: reported_entries(&value.clkMonList, value.clkMonListSize)
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
            sensors: reported_entries(&value.sensor, value.count)
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
            packet_filter: NvLinkPacketTypes::from_bits_retain(value.pktfilter.0),
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
            firmware_versions: reported_entries(
                &value.firmwareInfo.firmwareVersion,
                value.firmwareInfo.numValidEntries,
            )
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
            modes: reported_entries(&value.bwModes, value.totalBwModes).to_vec(),
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
            cert_chain: reported_entries(&value.certChain, value.certChainSize).to_vec(),
            attestation_cert_chain: reported_entries(
                &value.attestationCertChain,
                value.attestationCertChainSize,
            )
            .to_vec(),
        }
    }
}

impl From<sys::nvmlConfComputeGpuAttestationReport_t> for ConfComputeGpuAttestationReport {
    fn from(value: sys::nvmlConfComputeGpuAttestationReport_t) -> Self {
        Self {
            cec_attestation_report_present: value.isCecAttestationReportPresent != 0,
            nonce: value.nonce,
            attestation_report: reported_entries(
                &value.attestationReport,
                value.attestationReportSize,
            )
            .to_vec(),
            cec_attestation_report: reported_entries(
                &value.cecAttestationReport,
                value.cecAttestationReportSize,
            )
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
            status_code: value.status.0,
            cliques: value
                .cliques
                .get(..to_usize(value.numCliques, "clique count")?)
                .ok_or_else(|| Error::out_of_range("clique count"))?
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
    /// Returns an error if `value_type` is not a known value type.
    pub fn from_raw(value_type: sys::nvmlValueType_t, value: sys::nvmlValue_t) -> Result<Self> {
        Ok(match FieldValueType::from(value_type) {
            // SAFETY: every `nvmlValue_t` field is a plain number valid for any bit pattern.
            FieldValueType::Double => Self::Double(unsafe { value.dVal }),
            // SAFETY: every `nvmlValue_t` field is a plain number valid for any bit pattern.
            FieldValueType::UnsignedInt => Self::UnsignedInt(unsafe { value.uiVal }),
            // SAFETY: every `nvmlValue_t` field is a plain number valid for any bit pattern.
            FieldValueType::UnsignedLong => Self::UnsignedLong(unsafe { value.ulVal }),
            // SAFETY: every `nvmlValue_t` field is a plain number valid for any bit pattern.
            FieldValueType::UnsignedLongLong => Self::UnsignedLongLong(unsafe { value.ullVal }),
            // SAFETY: every `nvmlValue_t` field is a plain number valid for any bit pattern.
            FieldValueType::SignedLongLong => Self::SignedLongLong(unsafe { value.sllVal }),
            // SAFETY: every `nvmlValue_t` field is a plain number valid for any bit pattern.
            FieldValueType::SignedInt => Self::SignedInt(unsafe { value.siVal }),
            // SAFETY: every `nvmlValue_t` field is a plain number valid for any bit pattern.
            FieldValueType::UnsignedShort => Self::UnsignedShort(unsafe { value.usVal }),
            FieldValueType::Unknown(code) => {
                return Err(Error::UnknownEnumValue {
                    name: "field value type".into(),
                    value: i64::from(code),
                });
            }
        })
    }
}

impl Sample {
    /// Decodes a timestamped native sample using the supplied scalar value type.
    ///
    /// # Errors
    ///
    /// Returns an error if `value_type` is not a known value type.
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
        /// Hardware slowdown cuts the core clocks by half or more.
        ///
        /// Causes include a high temperature, an external power brake from the power supply, power draw high enough to trip fast-trigger protection, and P-state or clock changes.
        const HW_SLOWDOWN = sys::nvmlClocksThrottleReasonHwSlowdown as u64;

        /// Hardware thermal slowdown cuts the core clocks by half or more because the temperature is too high.
        const HW_THERMAL_SLOWDOWN = sys::nvmlClocksThrottleReasonHwThermalSlowdown as u64;

        /// Hardware power brake slowdown cuts the core clocks by half or more because an external power brake, such as from the power supply, is asserted.
        const HW_POWER_BRAKE_SLOWDOWN = sys::nvmlClocksThrottleReasonHwPowerBrakeSlowdown as u64;

        /// Nothing is running on the GPU and the clocks are dropping to Idle state.
        const GPU_IDLE = sys::nvmlClocksEventReasonGpuIdle as u64;
        /// Clocks are set to application-specific values.
        const APPLICATIONS_CLOCKS_SETTING = sys::nvmlClocksEventReasonApplicationsClocksSetting as u64;
        /// The clocks are lowered to stay within the power limit.
        const SW_POWER_CAP = sys::nvmlClocksEventReasonSwPowerCap as u64;
        /// The GPU is in a sync boost group, which runs every member at the lowest clocks across the group.
        ///
        /// Other GPUs' clock event reasons show why they hold this one at lower clocks.
        const SYNC_BOOST = sys::nvmlClocksEventReasonSyncBoost as u64;
        /// The clocks are lowered to keep the GPU and memory below their maximum operating temperatures.
        const SW_THERMAL_SLOWDOWN = sys::nvmlClocksEventReasonSwThermalSlowdown as u64;
        /// The display clock setting limits the GPU clocks.
        const DISPLAY_CLOCK_SETTING = sys::nvmlClocksEventReasonDisplayClockSetting as u64;
        /// The board limit policy limits the GPU clocks.
        const BOARD_LIMIT = sys::nvmlClocksEventReasonBoardLimit as u64;
        /// The reliability policy limits the GPU clocks.
        const RELIABILITY = sys::nvmlClocksEventReasonReliability as u64;
    }
}

/// GPU recovery action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuRecoveryAction(pub u32);

impl GpuRecoveryAction {
    /// No action needed.
    pub const NONE: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_NONE.0);

    /// Reset Gpu.
    pub const GPU_RESET: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_GPU_RESET.0);

    /// Reboot Node.
    pub const NODE_REBOOT: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_NODE_REBOOT.0);

    /// Drain P2P.
    pub const DRAIN_P2P: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_DRAIN_P2P.0);

    /// Drain P2P and Reset Gpu.
    pub const DRAIN_AND_RESET: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET.0);

    /// Recover IMEX Domain.
    pub const RECOVER_IMEX_DOMAIN: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_RECOVER_IMEX_DOMAIN.0);

    /// Reset the GPU's `PCIe` bus.
    pub const BUS_RESET: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_BUS_RESET.0);

    /// Reboot the system.
    pub const SYSTEM_REBOOT: Self =
        Self(sys::nvmlDeviceGpuRecoveryAction_s::NVML_GPU_RECOVERY_ACTION_SYSTEM_REBOOT.0);
}

/// NVIDIA GRID license feature code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridLicenseFeatureCode(pub u32);

impl GridLicenseFeatureCode {
    /// Unknown.
    pub const UNKNOWN: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_UNKNOWN.0);

    /// Virtual GPU.
    pub const VGPU: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_VGPU.0);

    /// Nvidia RTX.
    pub const NVIDIA_RTX: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX.0);

    /// Gaming.
    pub const GAMING: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_GAMING.0);

    /// Compute.
    pub const COMPUTE: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_COMPUTE.0);

    /// vGameDev.
    pub const VGAMEDEV: Self =
        Self(sys::nvmlGridLicenseFeatureCode_t::NVML_GRID_LICENSE_FEATURE_CODE_VGAMEDEV.0);
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

impl_unknown_enum_display!(VgpuGuestInfoState, {
    VgpuGuestInfoState::Uninitialized => "NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED",
    VgpuGuestInfoState::Initialized => "NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED",
});

impl_enum_display!(VgpuPlacementMode, {
    VgpuPlacementMode::Heterogeneous => "NVML_VGPU_PGPU_HETEROGENEOUS_MODE",
    VgpuPlacementMode::Homogeneous => "NVML_VGPU_PGPU_HOMOGENEOUS_MODE",
});

impl_unknown_enum_display!(EnableState, {
    EnableState::Disabled => "NVML_FEATURE_DISABLED",
    EnableState::Enabled => "NVML_FEATURE_ENABLED",
});

impl_unknown_enum_display!(TemperatureSensor, {
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

impl_unknown_enum_display!(ComputeMode, {
    ComputeMode::Default => "NVML_COMPUTEMODE_DEFAULT",
    ComputeMode::ExclusiveThread => "NVML_COMPUTEMODE_EXCLUSIVE_THREAD",
    ComputeMode::Prohibited => "NVML_COMPUTEMODE_PROHIBITED",
    ComputeMode::ExclusiveProcess => "NVML_COMPUTEMODE_EXCLUSIVE_PROCESS",
});

impl_unknown_enum_display!(ClockType, {
    ClockType::Graphics => "NVML_CLOCK_GRAPHICS",
    ClockType::Sm => "NVML_CLOCK_SM",
    ClockType::Memory => "NVML_CLOCK_MEM",
    ClockType::Video => "NVML_CLOCK_VIDEO",
});

impl_unknown_enum_display!(NvLinkUtilizationCountUnits, {
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

impl_unknown_enum_display!(NvLinkRemoteDeviceType, {
    NvLinkRemoteDeviceType::Gpu => "NVML_NVLINK_DEVICE_TYPE_GPU",
    NvLinkRemoteDeviceType::IbmNpu => "NVML_NVLINK_DEVICE_TYPE_IBMNPU",
    NvLinkRemoteDeviceType::Switch => "NVML_NVLINK_DEVICE_TYPE_SWITCH",
});

impl_unknown_enum_display!(NvLinkVersion, {
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

impl_unknown_enum_display!(ThermalTarget, {
    ThermalTarget::None => "NVML_THERMAL_TARGET_NONE",
    ThermalTarget::Gpu => "NVML_THERMAL_TARGET_GPU",
    ThermalTarget::Memory => "NVML_THERMAL_TARGET_MEMORY",
    ThermalTarget::PowerSupply => "NVML_THERMAL_TARGET_POWER_SUPPLY",
    ThermalTarget::Board => "NVML_THERMAL_TARGET_BOARD",
    ThermalTarget::VcdBoard => "NVML_THERMAL_TARGET_VCD_BOARD",
    ThermalTarget::VcdInlet => "NVML_THERMAL_TARGET_VCD_INLET",
    ThermalTarget::VcdOutlet => "NVML_THERMAL_TARGET_VCD_OUTLET",
    ThermalTarget::All => "NVML_THERMAL_TARGET_ALL",
});

impl_unknown_enum_display!(ThermalController, {
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
});

impl_unknown_enum_display!(DeviceAddressingMode, {
    DeviceAddressingMode::None => "NVML_DEVICE_ADDRESSING_MODE_NONE",
    DeviceAddressingMode::Hmm => "NVML_DEVICE_ADDRESSING_MODE_HMM",
    DeviceAddressingMode::Ats => "NVML_DEVICE_ADDRESSING_MODE_ATS",
});

impl_unknown_enum_display!(BridgeChipType, {
    BridgeChipType::Plx => "NVML_BRIDGE_CHIP_PLX",
    BridgeChipType::Bro4 => "NVML_BRIDGE_CHIP_BRO4",
});

impl_unknown_enum_display!(PerformanceState, {
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
});

impl_enum_display!(PcieUtilCounter, {
    PcieUtilCounter::TxBytes => "NVML_PCIE_UTIL_TX_BYTES",
    PcieUtilCounter::RxBytes => "NVML_PCIE_UTIL_RX_BYTES",
});

impl_unknown_enum_display!(TopologyLevel, {
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

impl_unknown_enum_display!(P2pStatus, {
    P2pStatus::Ok => "NVML_P2P_STATUS_OK",
    P2pStatus::ChipsetNotSupported => "NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED",
    P2pStatus::GpuNotSupported => "NVML_P2P_STATUS_GPU_NOT_SUPPORTED",
    P2pStatus::IoHubNotSupported => "NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED",
    P2pStatus::DisabledByRegKey => "NVML_P2P_STATUS_DISABLED_BY_REGKEY",
    P2pStatus::NotSupported => "NVML_P2P_STATUS_NOT_SUPPORTED",
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

impl_unknown_enum_display!(FieldValueType, {
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

impl_unknown_enum_display!(EncoderType, {
    EncoderType::H264 => "NVML_ENCODER_QUERY_H264",
    EncoderType::Hevc => "NVML_ENCODER_QUERY_HEVC",
    EncoderType::Av1 => "NVML_ENCODER_QUERY_AV1",
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

impl_unknown_enum_display!(FanState, {
    FanState::Normal => "NVML_FAN_NORMAL",
    FanState::Failed => "NVML_FAN_FAILED",
});

impl_unknown_enum_display!(CoolerControl, {
    CoolerControl::None => "NVML_THERMAL_COOLER_SIGNAL_NONE",
    CoolerControl::Toggle => "NVML_THERMAL_COOLER_SIGNAL_TOGGLE",
    CoolerControl::Variable => "NVML_THERMAL_COOLER_SIGNAL_VARIABLE",
    CoolerControl::Count => "NVML_THERMAL_COOLER_SIGNAL_COUNT",
});

impl_unknown_enum_display!(CoolerTarget, {
    CoolerTarget::None => "NVML_THERMAL_COOLER_TARGET_NONE",
    CoolerTarget::Gpu => "NVML_THERMAL_COOLER_TARGET_GPU",
    CoolerTarget::Memory => "NVML_THERMAL_COOLER_TARGET_MEMORY",
    CoolerTarget::PowerSupply => "NVML_THERMAL_COOLER_TARGET_POWER_SUPPLY",
    CoolerTarget::GpuRelated => "NVML_THERMAL_COOLER_TARGET_GPU_RELATED",
});

impl_unknown_enum_display!(FbcSessionType, {
    FbcSessionType::ToSys => "NVML_FBC_SESSION_TYPE_TOSYS",
    FbcSessionType::Cuda => "NVML_FBC_SESSION_TYPE_CUDA",
    FbcSessionType::Vid => "NVML_FBC_SESSION_TYPE_VID",
    FbcSessionType::HwEnc => "NVML_FBC_SESSION_TYPE_HWENC",
});

impl_unknown_enum_display!(DriverModel, {
    DriverModel::Wddm => "NVML_DRIVER_WDDM",
    DriverModel::Wdm => "NVML_DRIVER_WDM",
    DriverModel::Mcdm => "NVML_DRIVER_MCDM",
});

impl_unknown_enum_display!(GpuOperationMode, {
    GpuOperationMode::AllOn => "NVML_GOM_ALL_ON",
    GpuOperationMode::Compute => "NVML_GOM_COMPUTE",
    GpuOperationMode::LowDp => "NVML_GOM_LOW_DP",
});

impl_enum_display!(MigMode, {
    MigMode::Disabled => "NVML_DEVICE_MIG_DISABLE",
    MigMode::Enabled => "NVML_DEVICE_MIG_ENABLE",
});

impl_unknown_enum_display!(HostVgpuMode, {
    HostVgpuMode::NonSriov => "NVML_HOST_VGPU_MODE_NON_SRIOV",
    HostVgpuMode::Sriov => "NVML_HOST_VGPU_MODE_SRIOV",
});

impl_enum_display!(VgpuDriverCapability, {
    VgpuDriverCapability::HeterogeneousMultiVgpu => "NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU",
    VgpuDriverCapability::WarmUpdate => "NVML_VGPU_DRIVER_CAP_WARM_UPDATE",
});

impl_unknown_enum_display!(VgpuVmIdType, {
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

impl_unknown_enum_display!(LedColor, {
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

fn option_u64_from_not_available(value: u64) -> Option<u64> {
    if value == i64::from(sys::NVML_VALUE_NOT_AVAILABLE).cast_unsigned() {
        None
    } else {
        Some(value)
    }
}

fn mask255_bits(mask: sys::nvmlMask255_t) -> Vec<u32> {
    mask.mask
        .into_iter()
        .enumerate()
        .flat_map(|(word_index, word)| {
            (0..u32::BITS).filter_map(move |bit_index| {
                let bit = 1u32 << bit_index;
                (word & bit != 0).then_some(
                    (u32::try_from(word_index).expect("fixed NVML mask word index fits u32"))
                        * u32::BITS
                        + bit_index,
                )
            })
        })
        .collect()
}

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
            Err(Error::BufferTooSmall { .. })
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
