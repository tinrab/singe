use std::{ffi::CStr, ptr};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result, Status},
    optional::{
        nvmlDeviceGetAdaptiveTgpModeInfo_v1, nvmlDeviceGetBankRemapperStatus_v1,
        nvmlDeviceGetMemoryLimits_v1, nvmlDeviceGetNvLinkTelemetrySamples_v1,
        nvmlDevicePerfMetricsGetSamples_v1, nvmlDeviceSetAdaptiveTgpMode_v1,
        nvmlDeviceSetMemoryLimits_v1, nvmlDeviceSetNvlinkBwModeAsync_v1,
    },
    to_u32, try_ffi,
    types::{EnableState, FieldValue, GridLicenseFeatureCode, VgpuLicenseExpiry},
    utility::struct_version,
};

/// Contains Adaptive TGP Mode state and telemetry information.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveTgpModeInfo {
    /// In-band enable requested or not requested.
    pub in_band_enable_request: EnableState,
    /// Feature allowed by out-of-band/admin or not allowed.
    pub feature_allowed_by_admin: EnableState,
    /// Out-of-band/admin override active or inactive.
    pub admin_override_enabled: EnableState,
    /// Enablement after arbitration: active or inactive.
    pub enablement_status: EnableState,
    /// Adjusted TGP limit in milliwatts, valid only when the feature is enabled.
    pub adjusted_limit_milliwatts: u32,
}

/// Maximum memory that can be allocated within a cgroup partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLimit {
    /// A limit expressed in bytes.
    Bytes(u64),
    /// No hard limit is set.
    Unlimited,
}

/// Current memory limits and memory used against those limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLimits {
    /// Amount of memory guaranteed before allocations may fail under memory pressure.
    pub soft_limit: u64,
    /// Maximum amount of memory that can be allocated.
    pub hard_limit: MemoryLimit,
    /// Currently used memory in bytes.
    pub current_used: u64,
}

/// Bank remapper status and spare-group histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankRemapperStatus {
    /// Number of active remappings.
    pub active_remappings: u32,
    /// Number of inactive remappings.
    pub inactive_remappings: u32,
    /// Whether there exists any pending bank remapping.
    pub pending: bool,
    /// Number of groups that have maximum spare.
    pub max_spare_group_count: u32,
    /// Number of groups that have no spare.
    pub no_spare_group_count: u32,
}

/// Requested reduced `NVLink` bandwidth mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvLinkBandwidthMode {
    /// Set to the best available bandwidth mode.
    Best,
    /// Requested bandwidth mode from the device's supported modes.
    Mode(u8),
}

/// Per-link `NVLink` telemetry sample type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvLinkTelemetryType {
    /// Raw TX flit counter for a single link.
    Transmit,
    /// Raw RX flit counter for a single link.
    Receive,
}

/// One link and metric requested in a telemetry batch.
#[derive(Debug, Clone, Copy)]
pub struct NvLinkTelemetryRequest {
    /// Link identifier whose telemetry is requested.
    pub link_id: u32,
    /// Transmit or receive telemetry to query.
    pub sample_type: NvLinkTelemetryType,
}

/// Historical samples or a per-slot NVML error.
#[derive(Debug)]
pub struct NvLinkTelemetrySamples {
    /// Request associated with these samples.
    pub request: NvLinkTelemetryRequest,
    /// Raw telemetry samples, or the error reported for this request.
    pub samples: Result<Vec<u64>>,
}

/// PRM counter input values.
#[derive(Debug, Clone, Copy)]
pub struct PrmCounterRequest {
    /// Counter ID.
    pub counter: PrmCounterId,
    /// Local port number.
    pub local_port: u32,
}

impl Device {
    /// Retrieves Adaptive TGP Mode state and telemetry for a GPU.
    ///
    /// For Rubin or newer devices. The adjusted base power is valid only when enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the device is inaccessible, or adaptive TGP reporting is unsupported. Also propagates NVML query failures.
    pub fn adaptive_tgp_mode_info(self) -> Result<AdaptiveTgpModeInfo> {
        let mut info = sys::nvmlAdaptiveTgpModeInfo_v1_t::default();
        // SAFETY: `nvmlDeviceGetAdaptiveTgpModeInfo_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlDeviceGetAdaptiveTgpModeInfo_v1(
                self.as_raw(),
                &raw mut info
            ))?;
        }
        Ok(AdaptiveTgpModeInfo {
            in_band_enable_request: info.inBandEnableRequest.into(),
            feature_allowed_by_admin: info.featureAllowedByAdmin.into(),
            admin_override_enabled: info.adminOverrideEnabled.into(),
            enablement_status: info.enablementStatus.into(),
            adjusted_limit_milliwatts: info.adjustedLimitMw,
        })
    }

    /// Requests to enable or disable Adaptive TGP Mode for a GPU.
    ///
    /// For Rubin or newer devices. Requires root/admin privileges. An out-of-band
    /// administrator policy may override the in-band request.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the device or mode is unsupported, or NVML denies permission to change adaptive TGP mode. Also propagates NVML configuration failures.
    pub fn set_adaptive_tgp_mode(self, mode: EnableState) -> Result<()> {
        // SAFETY: `nvmlDeviceSetAdaptiveTgpMode_v1` receives a live handle and a plain mode value.
        unsafe { try_ffi!(nvmlDeviceSetAdaptiveTgpMode_v1(self.as_raw(), mode.into())) }
    }

    /// Gets current memory limits and memory used for the specified cgroup partition.
    ///
    /// For Linux only. `cgroup` is the full path to the sysfs cgroup file name.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the cgroup namespace is invalid or unavailable, or memory-limit reporting is unsupported. Also propagates NVML query failures.
    pub fn memory_limits(self, cgroup: &CStr) -> Result<MemoryLimits> {
        let mut limits = sys::nvmlGetMemoryLimits_v1_t {
            nameSpace: cgroup.as_ptr(),
            ..Default::default()
        };
        // SAFETY: `nvmlDeviceGetMemoryLimits_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlDeviceGetMemoryLimits_v1(self.as_raw(), &raw mut limits))?;
        }
        Ok(MemoryLimits {
            soft_limit: limits.softLimit,
            hard_limit: if limits.hardLimit == (sys::NVML_DEVICE_MEMORY_LIMIT_MAX as u64) {
                MemoryLimit::Unlimited
            } else {
                MemoryLimit::Bytes(limits.hardLimit)
            },
            current_used: limits.currentUsed,
        })
    }

    /// Sets the memory limits of the device for a cgroup partition.
    ///
    /// For Linux only. Requires root/admin permissions. The soft limit is the
    /// guaranteed amount; the hard limit is the maximum allocation size.
    ///
    /// # Errors
    ///
    /// Returns `Status::InvalidArgument` if `soft_limit` exceeds the encoded hard limit. Also returns an error if the entry point is unavailable or NVML rejects the cgroup, limits, device, or caller permissions.
    pub fn set_memory_limits(
        self,
        cgroup: &CStr,
        soft_limit: u64,
        hard_limit: MemoryLimit,
    ) -> Result<()> {
        let hard_limit = match hard_limit {
            MemoryLimit::Bytes(value) => value,
            MemoryLimit::Unlimited => sys::NVML_DEVICE_MEMORY_LIMIT_MAX as u64,
        };
        if soft_limit > hard_limit {
            return Err(Status::InvalidArgument.into());
        }
        let mut limits = sys::nvmlSetMemoryLimits_v1_t {
            nameSpace: cgroup.as_ptr(),
            softLimit: soft_limit,
            hardLimit: hard_limit,
        };
        // SAFETY: `nvmlDeviceSetMemoryLimits_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe { try_ffi!(nvmlDeviceSetMemoryLimits_v1(self.as_raw(), &raw mut limits)) }
    }

    /// Clears limits by setting the soft limit to zero and hard limit to unlimited.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::set_memory_limits`], including an unavailable entry point, invalid cgroup namespace, or insufficient permissions.
    pub fn clear_memory_limits(self, cgroup: &CStr) -> Result<()> {
        self.set_memory_limits(cgroup, 0, MemoryLimit::Unlimited)
    }

    /// Gets bank remapper status for Rubin or newer devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the device is inaccessible, or bank-remapper reporting is unsupported. Also propagates NVML query failures.
    pub fn bank_remapper_status(self) -> Result<BankRemapperStatus> {
        let mut status = sys::nvmlEccBankRemapperStatus_v1_t::default();
        // SAFETY: `nvmlDeviceGetBankRemapperStatus_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlDeviceGetBankRemapperStatus_v1(
                self.as_raw(),
                &raw mut status
            ))?;
        }
        Ok(BankRemapperStatus {
            active_remappings: status.activeRemappings,
            inactive_remappings: status.inactiveRemappings,
            pending: status.bPending != 0,
            max_spare_group_count: status.histogram.maxSpareGroupCount,
            no_spare_group_count: status.histogram.noSpareGroupCount,
        })
    }

    /// Sets the `NVLink` Reduced Bandwidth Mode for the device.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested bandwidth mode is unsupported, the device is inaccessible, or NVML denies permission to configure `NVLink` bandwidth. Also propagates NVML configuration failures.
    pub fn set_nvlink_bandwidth_mode(self, mode: NvLinkBandwidthMode) -> Result<()> {
        let mut params = sys::nvmlNvlinkSetBwMode_t {
            version: struct_version::<sys::nvmlNvlinkSetBwMode_t>(1),
            bSetBest: u32::from(mode == NvLinkBandwidthMode::Best),
            bwMode: match mode {
                NvLinkBandwidthMode::Best => 0,
                NvLinkBandwidthMode::Mode(value) => value,
            },
        };
        // SAFETY: `nvmlDeviceSetNvlinkBwMode` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetNvlinkBwMode(
                self.as_raw(),
                &raw mut params
            ))
        }
    }

    /// Sets `NVLink` Reduced Bandwidth Mode asynchronously and returns the polling timeout in milliseconds.
    ///
    /// For Rubin or newer devices. Poll fabric information for the completed state
    /// to check whether the setting was applied.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point or rejects the requested mode, device, or caller permissions. This reports request-submission errors; the returned timeout describes subsequent polling.
    pub fn set_nvlink_bandwidth_mode_async(self, mode: NvLinkBandwidthMode) -> Result<u32> {
        let mut params = sys::nvmlNvlinkSetBwModeAsync_v1_t {
            bSetBest: u32::from(mode == NvLinkBandwidthMode::Best),
            bwMode: match mode {
                NvLinkBandwidthMode::Best => 0,
                NvLinkBandwidthMode::Mode(value) => value.into(),
            },
            ..Default::default()
        };
        // SAFETY: `nvmlDeviceSetNvlinkBwModeAsync_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlDeviceSetNvlinkBwModeAsync_v1(
                self.as_raw(),
                &raw mut params
            ))?;
        }
        Ok(params.asyncPollTimeoutMs)
    }

    /// Retrieves a batch of historical `NVLink` per-link telemetry samples.
    ///
    /// Samples are taken at approximately 100 ms intervals. Intended for periodic
    /// polling every roughly two seconds. For Rubin or newer devices.
    ///
    /// # Errors
    ///
    /// Returns an error if request or sample counts cannot fit in `u32`, the native entry point is unavailable, or a batch query fails. Per-link errors, including insufficient sample capacity, are retained in the corresponding returned record.
    pub fn nvlink_telemetry_samples(
        self,
        requests: &[NvLinkTelemetryRequest],
    ) -> Result<Vec<NvLinkTelemetrySamples>> {
        let count =
            u32::try_from(requests.len()).map_err(|_| Error::from(Status::InvalidArgument))?;
        let mut slots: Vec<_> = requests.iter().map(|request| sys::nvmlNvlinkTelemetrySample_v1_t {
            linkId: request.link_id,
            sampleType: match request.sample_type { NvLinkTelemetryType::Transmit => sys::nvmlNvlinkTelemetrySampleType_t::NVML_NVLINK_TELEMETRY_SAMPLE_TYPE_THROUGHPUT_RAW_TX as _, NvLinkTelemetryType::Receive => sys::nvmlNvlinkTelemetrySampleType_t::NVML_NVLINK_TELEMETRY_SAMPLE_TYPE_THROUGHPUT_RAW_RX as _ },
            ..Default::default()
        }).collect();
        let mut batch = sys::nvmlNvlinkTelemetrySamples_v1_t {
            telemetryCount: count,
            telemetrySamples: slots.as_mut_ptr(),
        };
        // SAFETY: `nvmlDeviceGetNvLinkTelemetrySamples_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlDeviceGetNvLinkTelemetrySamples_v1(
                self.as_raw(),
                &raw mut batch
            ))?;
        }
        let mut buffers: Vec<_> = slots
            .iter()
            .map(|slot| {
                if slot.nvmlReturn == sys::nvmlReturn_t::NVML_SUCCESS {
                    vec![0; slot.sampleCount as usize]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let initial_status: Vec<_> = slots.iter().map(|slot| slot.nvmlReturn).collect();
        for (slot, buffer) in slots.iter_mut().zip(&mut buffers) {
            slot.samples = if buffer.is_empty() {
                ptr::null_mut()
            } else {
                buffer.as_mut_ptr()
            };
            slot.sampleCount = to_u32(buffer.len(), "telemetry sample count")?;
        }
        // SAFETY: `nvmlDeviceGetNvLinkTelemetrySamples_v1` receives a live handle and a versioned local struct that outlives the call.
        unsafe {
            try_ffi!(nvmlDeviceGetNvLinkTelemetrySamples_v1(
                self.as_raw(),
                &raw mut batch
            ))?;
        }
        Ok(requests
            .iter()
            .copied()
            .zip(slots)
            .zip(buffers)
            .zip(initial_status)
            .map(|(((request, slot), mut samples), initial)| {
                let result = if initial != sys::nvmlReturn_t::NVML_SUCCESS {
                    Err(initial.into())
                } else if slot.nvmlReturn != sys::nvmlReturn_t::NVML_SUCCESS {
                    Err(slot.nvmlReturn.into())
                } else if slot.sampleCount as usize > samples.len() {
                    Err(Status::InsufficientSize.into())
                } else {
                    samples.truncate(slot.sampleCount as usize);
                    Ok(samples)
                };
                NvLinkTelemetrySamples {
                    request,
                    samples: result,
                }
            })
            .collect())
    }

    /// Reads a list of GPU PRM counters for Blackwell or newer devices on Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if the request count cannot fit in `u32` or NVML rejects the counter batch. Per-counter NVML errors and unrecognized scalar value types are retained in the returned inner results.
    pub fn read_prm_counters(
        self,
        requests: &[PrmCounterRequest],
    ) -> Result<Vec<Result<FieldValue>>> {
        let count =
            u32::try_from(requests.len()).map_err(|_| Error::from(Status::InvalidArgument))?;
        let mut counters: Vec<_> = requests
            .iter()
            .map(|request| sys::nvmlPRMCounter_v1_t {
                counterId: request.counter.0,
                inData: sys::nvmlPRMCounterInput_v1_t {
                    localPort: request.local_port,
                },
                ..Default::default()
            })
            .collect();
        let mut list = sys::nvmlPRMCounterList_v1_t {
            numCounters: count,
            counters: counters.as_mut_ptr(),
        };
        // SAFETY: `nvmlDeviceReadPRMCounters_v1` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvmlDeviceReadPRMCounters_v1(
                self.as_raw(),
                &raw mut list
            ))?;
        }
        Ok(counters
            .into_iter()
            .map(|counter| {
                let value = counter.counterValue;
                if value.status != sys::nvmlReturn_t::NVML_SUCCESS {
                    return Err(value.status.into());
                }
                FieldValue::from_raw(value.outputType, value.outputValue)
            })
            .collect())
    }
}

/// PRM counter identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrmCounterId(pub u32);

/// Power tuple containing power consumption in milliwatts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerTuple {
    /// Power consumption in milliwatts.
    pub pwrm_w: u32,
}

impl From<sys::nvmlPmgrPwrTuple_t> for PowerTuple {
    fn from(value: sys::nvmlPmgrPwrTuple_t) -> Self {
        Self {
            pwrm_w: value.pwrmW,
        }
    }
}

/// Metrics for a single power rail, including frequency and utilization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RailMetrics {
    /// Frequency in kilohertz.
    pub freqk_hz: u32,
    /// Utilization percentage (fixed-point).
    pub util_pct: u64,
}

impl From<sys::nvmlRailMetrics_t> for RailMetrics {
    fn from(value: sys::nvmlRailMetrics_t) -> Self {
        Self {
            freqk_hz: value.freqkHz,
            util_pct: value.utilPct,
        }
    }
}

/// Metrics for all core rails in the system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreRailMetrics {
    /// Array of core rail metrics.
    pub rails: [RailMetrics; 2],
}

impl From<sys::nvmlCoreRailMetrics_t> for CoreRailMetrics {
    fn from(value: sys::nvmlCoreRailMetrics_t) -> Self {
        Self {
            rails: value.rails.map(Into::into),
        }
    }
}

/// Performance metrics for DLPPM 1x power model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerModelMetricsDlppm1xPerf {
    /// Performance metric in milliseconds.
    pub perfms: u32,
}

impl From<sys::nvmlPwrModelMetricsDlppm1xPerf_t> for PowerModelMetricsDlppm1xPerf {
    fn from(value: sys::nvmlPwrModelMetricsDlppm1xPerf_t) -> Self {
        Self {
            perfms: value.perfms,
        }
    }
}

/// Complete power model metrics for DLPPM 1x, including rail metrics and TGP power.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerModelMetricsDlppm1x {
    /// Validity flag: non-zero if metrics are valid.
    pub valid: bool,
    /// Core rail metrics.
    pub core_rail: CoreRailMetrics,
    /// Fb rail metrics.
    pub fb_rail: RailMetrics,
    /// Total Graphics Power (TGP) in milliwatts.
    pub tgp_pwr_tuple: PowerTuple,
    /// Performance metrics.
    pub perf_metrics: PowerModelMetricsDlppm1xPerf,
}

impl From<sys::nvmlPwrModelMetricsDlppm1x_t> for PowerModelMetricsDlppm1x {
    fn from(value: sys::nvmlPwrModelMetricsDlppm1x_t) -> Self {
        Self {
            valid: value.bValid != 0,
            core_rail: value.coreRail.into(),
            fb_rail: value.fbRail.into(),
            tgp_pwr_tuple: value.tgpPwrTuple.into(),
            perf_metrics: value.perfMetrics.into(),
        }
    }
}

/// DRAMCLK estimates containing multiple estimated metrics for different DRAMCLK frequencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerModelMetricsDlppm1xDramclkEstimates {
    /// Array of estimated metrics for each inference loop.
    pub estimated_metrics: [PowerModelMetricsDlppm1x; 8],
    /// Number of valid entries in estimatedMetrics array.
    pub num_estimated_metrics: u8,
}

impl From<sys::nvmlPwrModelMetricsDlppm1xDramclkEstimates_t>
    for PowerModelMetricsDlppm1xDramclkEstimates
{
    fn from(value: sys::nvmlPwrModelMetricsDlppm1xDramclkEstimates_t) -> Self {
        Self {
            estimated_metrics: value.estimatedMetrics.map(Into::into),
            num_estimated_metrics: value.numEstimatedMetrics,
        }
    }
}

/// Observed metrics from the power model, including initial DRAMCLK estimates and current measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedMetrics {
    /// Initial DRAMCLK estimates for different scenarios.
    pub initial_dramclk_est: [PowerModelMetricsDlppm1xDramclkEstimates; 3],
    /// Validity flag: non-zero if observed metrics are valid.
    pub valid: bool,
    /// Observed core rail metrics.
    pub core_rail: CoreRailMetrics,
    /// Observed fb rail metrics.
    pub fb_rail: RailMetrics,
    /// Observed Total Graphics Power (TGP) in milliwatts.
    pub tgp_pwr_tuple: PowerTuple,
    /// Observed performance metrics.
    pub perf_metrics: PowerModelMetricsDlppm1xPerf,
}

impl From<sys::nvmlObservedMetrics_t> for ObservedMetrics {
    fn from(value: sys::nvmlObservedMetrics_t) -> Self {
        Self {
            initial_dramclk_est: value.initialDramclkEst.map(Into::into),
            valid: value.bValid != 0,
            core_rail: value.coreRail.into(),
            fb_rail: value.fbRail.into(),
            tgp_pwr_tuple: value.tgpPwrTuple.into(),
            perf_metrics: value.perfMetrics.into(),
        }
    }
}

/// Performance metrics sample for DLPPC 2x controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceMetricsDlppc2xSample {
    /// Observed metrics from the DLPPC 2x controller.
    pub observed_metrics: ObservedMetrics,
}

impl From<sys::nvmlPerfMetricsDlppc2xSample_t> for PerformanceMetricsDlppc2xSample {
    fn from(value: sys::nvmlPerfMetricsDlppc2xSample_t) -> Self {
        Self {
            observed_metrics: value.observedMetrics.into(),
        }
    }
}

/// Power model metrics sample for PFPP 1x, containing frequency inputs and estimated TGP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerModelMetricsSamplePfpp1x {
    /// Array of input frequencies in kilohertz for each domain.
    pub freqk_hz: [u32; 16],
    /// Estimated Total Graphics Power in milliwatts.
    pub est_tgp_pwrm_w: u32,
}

impl From<sys::nvmlPwrModelMetricsSamplePfpp1x_t> for PowerModelMetricsSamplePfpp1x {
    fn from(value: sys::nvmlPwrModelMetricsSamplePfpp1x_t) -> Self {
        Self {
            freqk_hz: value.freqkHz,
            est_tgp_pwrm_w: value.estTgpPwrmW,
        }
    }
}

/// Operating point for PFPP 1x power model, defining a frequency-power pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerModelOperatingPointPfpp1x {
    /// Operating frequency in kilohertz.
    pub freqk_hz: u32,
    /// Power consumption at this frequency in milliwatts.
    pub pwrm_w: u32,
}

impl From<sys::nvmlPwrModelOperatingPointPfpp1x_t> for PowerModelOperatingPointPfpp1x {
    fn from(value: sys::nvmlPwrModelOperatingPointPfpp1x_t) -> Self {
        Self {
            freqk_hz: value.freqkHz,
            pwrm_w: value.pwrmW,
        }
    }
}

/// Complete power model metrics for PFPP 1x, including estimated metrics and key operating points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerModelMetricsPfpp1x {
    /// Number of valid vf points.
    pub num_vf_points: u8,
    /// Array of estimated metrics for different operating points.
    pub estimated_metrics: [PowerModelMetricsSamplePfpp1x; 32],
    /// Validity flag: non-zero if metrics are valid.
    pub valid: bool,
    /// Operating point with maximum performance per watt.
    pub max_perf_per_watt_point: PowerModelOperatingPointPfpp1x,
    /// Operating point at maximum frequency and voltage.
    pub fmax_at_vmax_point: PowerModelOperatingPointPfpp1x,
    /// TGP headroom in milliwatts.
    pub tgp_headroomm_w: u32,
}

impl From<sys::nvmlPwrModelMetricsPfpp1x_t> for PowerModelMetricsPfpp1x {
    fn from(value: sys::nvmlPwrModelMetricsPfpp1x_t) -> Self {
        Self {
            num_vf_points: value.numVfPoints,
            estimated_metrics: value.estimatedMetrics.map(Into::into),
            valid: value.bValid != 0,
            max_perf_per_watt_point: value.maxPerfPerWattPoint.into(),
            fmax_at_vmax_point: value.fmaxAtVmaxPoint.into(),
            tgp_headroomm_w: value.tgpHeadroommW,
        }
    }
}

/// Performance metrics sample for PFPP 1x controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceMetricsPfpp1xSample {
    /// Estimated metrics from the PFPP 1x controller.
    pub estimated_metrics: PowerModelMetricsPfpp1x,
}

impl From<sys::nvmlPerfMetricsPfpp1xSample_t> for PerformanceMetricsPfpp1xSample {
    fn from(value: sys::nvmlPerfMetricsPfpp1xSample_t) -> Self {
        Self {
            estimated_metrics: value.estimatedMetrics.into(),
        }
    }
}

impl PrmCounterId {
    /// Sentinel.
    pub const NONE: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_NONE as _);

    /// PPCNT group 0x12, `link_down_events`.
    pub const PPCNT_PHYSICAL_LAYER_CTRS_LINK_DOWN_EVENTS: Self = Self(
        sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PHYSICAL_LAYER_CTRS_LINK_DOWN_EVENTS
            as _,
    );

    /// PPCNT group 0x12, `successful_recovery_events`.
    pub const PPCNT_PHYSICAL_LAYER_CTRS_SUCCESSFUL_RECOVERY_EVENTS: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PHYSICAL_LAYER_CTRS_SUCCESSFUL_RECOVERY_EVENTS as _);

    /// PPCNT group 0x1A, `total_successful_recovery_events`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_EVENTS: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_EVENTS as _);

    /// PPCNT group 0x1A, `time_since_last_recovery`.
    pub const PPCNT_RECOVERY_CTRS_TIME_SINCE_LAST_RECOVERY: Self = Self(
        sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_SINCE_LAST_RECOVERY
            as _,
    );

    /// PPCNT group 0x1A, `time_between_last_two_recoveries`.
    pub const PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES as _);

    /// PPCNT group 0x1A, `time_in_last_host_serdes_feq_recovery`.
    pub const PPCNT_RECOVERY_CTRS_TIME_IN_LAST_HOST_SERDES_FEQ_RECOVERY: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_IN_LAST_HOST_SERDES_FEQ_RECOVERY as _);

    /// PPCNT group 0x1A, `total_time_in_host_serdes_feq_recovery`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_TIME_IN_HOST_SERDES_FEQ_RECOVERY: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_TIME_IN_HOST_SERDES_FEQ_RECOVERY as _);

    /// PPCNT group 0x1A, `total_host_serdes_feq_recovery_count`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_RECOVERY_COUNT: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_RECOVERY_COUNT as _);

    /// PPCNT group 0x1A, `total_host_serdes_feq_successful_recovery_count`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_SUCCESSFUL_RECOVERY_COUNT: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_SUCCESSFUL_RECOVERY_COUNT as _);

    /// PPCNT group 0x1A, `last_host_serdes_feq_attempts_count`.
    pub const PPCNT_RECOVERY_CTRS_LAST_HOST_SERDES_FEQ_ATTEMPTS_COUNT: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_LAST_HOST_SERDES_FEQ_ATTEMPTS_COUNT as _);

    /// PPCNT group 0x1A, `last_successful_recovery_step_attempts`.
    pub const PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_STEP_ATTEMPTS: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_STEP_ATTEMPTS as _);

    /// PPCNT group 0x1A, `last_successful_recovery_time`.
    pub const PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_TIME: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_TIME as _);

    /// PPCNT group 0x1A, `total_successful_recovery_time`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_TIME: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_TIME as _);

    /// PPCNT group 0x20, `port_xmit_wait`.
    pub const PPCNT_PORTCOUNTERS_PORT_XMIT_WAIT: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PORTCOUNTERS_PORT_XMIT_WAIT as _);

    /// PPCNT group 0x22, `plr_rcv_codes`.
    pub const PPCNT_PLR_RCV_CODES: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODES as _);

    /// PPCNT group 0x22, `plr_rcv_code_err`.
    pub const PPCNT_PLR_RCV_CODE_ERR: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODE_ERR as _);

    /// PPCNT group 0x22, `plr_rcv_uncorrectable_code`.
    pub const PPCNT_PLR_RCV_UNCORRECTABLE_CODE: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_UNCORRECTABLE_CODE as _);

    /// PPCNT group 0x22, `plr_xmit_codes`.
    pub const PPCNT_PLR_XMIT_CODES: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_CODES as _);

    /// PPCNT group 0x22, `plr_xmit_retry_codes`.
    pub const PPCNT_PLR_XMIT_RETRY_CODES: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_CODES as _);

    /// PPCNT group 0x22, `plr_xmit_retry_events`.
    pub const PPCNT_PLR_XMIT_RETRY_EVENTS: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_EVENTS as _);

    /// PPCNT group 0x22, `plr_sync_events`.
    pub const PPCNT_PLR_SYNC_EVENTS: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_SYNC_EVENTS as _);

    /// PPRM, `oper_recovery`.
    pub const PPRM_OPER_RECOVERY: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPRM_OPER_RECOVERY as _);
}

/// Performance metrics sample from a controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceControllerSample {
    /// Observed metrics from the DLPPC 2x controller.
    Dlppc2x(Box<PerformanceMetricsDlppc2xSample>),
    /// Estimated metrics from the PFPP 1x controller.
    Pfpp1x(Box<PerformanceMetricsPfpp1xSample>),
    /// Controller type not recognized by this version of NVML.
    Unknown(u32),
}

/// A performance sample containing data from one or more controllers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceSample {
    /// Controller measurements decoded for this sample.
    pub controllers: Vec<PerformanceControllerSample>,
}

impl Device {
    /// Gets performance metric samples.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point or the native query fails. Returns `Status::InsufficientSize` if a reported sample or controller count exceeds its native array capacity.
    pub fn performance_metric_samples(self) -> Result<Vec<PerformanceSample>> {
        let mut data = Box::new(sys::nvmlPerfMetricsSamples_v1_t::default());
        // SAFETY: `nvmlDevicePerfMetricsGetSamples_v1` receives a live handle and a boxed, versioned struct that outlives the call.
        unsafe {
            try_ffi!(nvmlDevicePerfMetricsGetSamples_v1(
                self.as_raw(),
                &raw mut *data
            ))?;
        }
        let samples = data
            .samples
            .get(..data.numSamples as usize)
            .ok_or_else(|| Error::from(Status::InsufficientSize))?;
        samples
            .iter()
            .map(|sample| {
                let controllers = sample
                    .controllerData
                    .get(..sample.numControllerData as usize)
                    .ok_or_else(|| Error::from(Status::InsufficientSize))?;
                let controllers = controllers
                    .iter()
                    .map(|controller| match controller.controllerType {
                        sys::NVML_PERF_METRICS_CONTROLLER_TYPE_DLPPC_2X => {
                            PerformanceControllerSample::Dlppc2x(Box::new(
                                // SAFETY: the controller type matched above selects this union field.
                                unsafe { controller.data.dlppc2x }.into(),
                            ))
                        }
                        sys::NVML_PERF_METRICS_CONTROLLER_TYPE_PFPP_1X => {
                            PerformanceControllerSample::Pfpp1x(Box::new(
                                // SAFETY: the controller type matched above selects this union field.
                                unsafe { controller.data.pfpp1x }.into(),
                            ))
                        }
                        other => PerformanceControllerSample::Unknown(other),
                    })
                    .collect();
                Ok(PerformanceSample { controllers })
            })
            .collect()
    }
}

/// Structure containing vGPU software licensable feature information.
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

/// Structure to store vGPU software licensable features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridLicensableFeatures {
    /// Whether vGPU Software Licensing is supported on the system.
    pub supported: bool,
    /// Array of vGPU software licensable features.
    pub features: Vec<GridLicensableFeature>,
}

impl Device {
    /// Retrieve the vGPU Software licensable features.
    ///
    /// Identifies whether the system supports vGPU Software Licensing. If it
    /// does, returns the licensable features and their current license status.
    ///
    /// # Errors
    ///
    /// Propagates NVML license-feature query errors. Returns `Status::InsufficientSize` if the reported feature count exceeds the native array capacity.
    pub fn grid_licensable_features(self) -> Result<GridLicensableFeatures> {
        let mut raw = sys::nvmlGridLicensableFeatures_t::default();
        // SAFETY: `nvmlDeviceGetGridLicensableFeatures_v4` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGridLicensableFeatures_v4(
                self.as_raw(),
                &raw mut raw
            ))?;
        }
        let features = raw
            .gridLicensableFeatures
            .get(..raw.licensableFeaturesCount as usize)
            .ok_or_else(|| Error::from(Status::InsufficientSize))?
            .iter()
            .map(|feature| GridLicensableFeature {
                code: GridLicenseFeatureCode(feature.featureCode as u32),
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
