use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result, Status},
    try_ffi,
    types::{
        EnableState, FieldId, PerformanceState, PowerLimits, PowerMizerMode, PowerMizerModes,
        PowerSource, WorkloadPowerCurrentProfiles, WorkloadPowerProfilesInfo,
    },
    utility::{struct_version, to_usize},
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
    /// Returns power usage for this GPU and its associated circuitry, such as memory, in milliwatts.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
    /// On Ampere (except GA100) or newer GPUs, this returns power averaged over a one-second interval.
    /// On GA100 and older architectures, instantaneous power is returned.
    ///
    /// Query [`FieldId::POWER_AVERAGE`] on newer architectures or [`FieldId::POWER_INSTANT`] through [`Device::field_values`] for a specific kind of reading.
    ///
    /// It is only available if [`Device::power_management_mode`] reports power management as enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support power readings, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_usage(&self) -> Result<u32> {
        let mut power = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `power` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerUsage(self.raw, &raw mut power))?;
        }
        Ok(power)
    }

    /// Returns whether the device supports querying its current power limit.
    ///
    /// An unsupported power-limit field maps to [`EnableState::Disabled`].
    ///
    /// # Errors
    ///
    /// Propagates field-query errors, per-field decoding errors, and per-field NVML failures other than [`Status::NotSupported`].
    /// Also returns an error if the single-field query yields no entry.
    pub fn power_management_mode(&self) -> Result<EnableState> {
        let sample = self.single_field_value(FieldId::POWER_CURRENT_LIMIT, 0)?;
        match sample.result {
            Ok(_) => Ok(EnableState::Enabled),
            Err(Error::Nvml {
                code: Status::NotSupported,
            }) => Ok(EnableState::Disabled),
            Err(error) => Err(error),
        }
    }

    /// Returns the power management limit associated with this device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// The power limit defines the upper boundary for the card's power draw.
    /// If the card's total power draw reaches this limit the power management algorithm kicks in.
    ///
    /// This reading is only available if [`Device::power_management_mode`] reports power management as enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support power limits, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_management_limit(&self) -> Result<u32> {
        let mut limit = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `limit` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerManagementLimit(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or power limit, if the device does not support power-limit control, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_power_management_limit(&self, limit: u32) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the limit is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetPowerManagementLimit(self.raw, limit))?;
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support power limits, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_management_default_limit(&self) -> Result<u32> {
        let mut limit = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `limit` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerManagementDefaultLimit(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support power limits, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn enforced_power_limit(&self) -> Result<u32> {
        let mut limit = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `limit` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEnforcedPowerLimit(
                self.raw,
                &raw mut limit
            ))?;
        }
        Ok(limit)
    }

    /// Returns information about possible values of power management limits on this device.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or outputs, if the device does not support power-limit ranges, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_management_limit_constraints(&self) -> Result<PowerLimits> {
        let mut min = 0;
        let mut max = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `min` and `max` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerManagementLimitConstraints(
                self.raw,
                &raw mut min,
                &raw mut max,
            ))?;
        }
        Ok(PowerLimits { min, max })
    }

    /// Returns the current `PowerMizer` modes of this device.
    ///
    /// `PowerMizerMode` provides a hint to the driver as to how to manage the performance of the GPU.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support `PowerMizer` mode readings, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn power_mizer_mode(&self) -> Result<PowerMizerModes> {
        let mut modes = sys::nvmlDevicePowerMizerModes_v1_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `modes` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerMizerMode_v1(
                self.raw,
                &raw mut modes
            ))?;
        }
        Ok(modes.into())
    }

    /// Sets the `PowerMizer` mode of this device.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or mode, if the device does not support `PowerMizer` mode changes, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_power_mizer_mode(&self, mode: PowerMizerMode) -> Result<()> {
        let mut power_mizer = sys::nvmlDevicePowerMizerModes_v1_t {
            mode: mode.into(),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `power_mizer` is a local that outlives the call.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetPowerMizerMode_v1(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support energy readings, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn total_energy_consumption(&self) -> Result<u64> {
        let mut energy = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `energy` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTotalEnergyConsumption(
                self.raw,
                &raw mut energy
            ))?;
        }
        Ok(energy)
    }

    /// Returns the current device performance state.
    ///
    /// # Errors
    ///
    /// Propagates NVML query errors from [`Self::performance_state`], including an inaccessible device or unsupported performance-state reporting.
    pub fn power_state(&self) -> Result<PerformanceState> {
        self.performance_state()
    }

    /// Returns the device power source.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support power-source reporting, or if NVML has not been initialized.
    pub fn power_source(&self) -> Result<PowerSource> {
        let mut power_source = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `power_source` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPowerSource(
                self.raw,
                &raw mut power_source
            ))?;
        }
        Ok(PowerSource::from_raw(power_source))
    }

    /// Returns performance profiles information.
    ///
    /// For Blackwell or newer fully supported devices.
    /// Returns the workload power profile information reported by NVML.
    /// The `perf_profiles_mask` field has a bit set for each supported mode index.
    /// Each supported mode has a corresponding entry in the profile array containing the profile ID, the priority of this mode, and a conflicting mask.
    /// Lower priority values indicate higher priority, and each bit set in the conflicting mask corresponds to a different profile that cannot be used with the given profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML reports that the output layout is too small, if NVML rejects the query, if the device does not support workload power profiles, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn workload_power_profiles_info(&self) -> Result<WorkloadPowerProfilesInfo> {
        let mut info = sys::nvmlWorkloadPowerProfileProfilesInfo_t {
            version: struct_version::<sys::nvmlWorkloadPowerProfileProfilesInfo_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceWorkloadPowerProfileGetProfilesInfo(
                self.raw,
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
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML rejects the query, if the device does not support workload power profiles, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn workload_power_current_profiles(&self) -> Result<WorkloadPowerCurrentProfiles> {
        let mut profiles = sys::nvmlWorkloadPowerProfileCurrentProfiles_t {
            version: struct_version::<sys::nvmlWorkloadPowerProfileCurrentProfiles_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `profiles` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(
                self.raw,
                &raw mut profiles,
            ))?;
        }
        Ok(profiles.into())
    }

    /// Returns the Adaptive TGP Mode state and telemetry of this device.
    ///
    /// For Rubin or newer devices.
    /// The adjusted base power is valid only when enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the device is inaccessible, or adaptive TGP reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn adaptive_tgp_mode_info(&self) -> Result<AdaptiveTgpModeInfo> {
        let mut info = sys::nvmlAdaptiveTgpModeInfo_v1_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAdaptiveTgpModeInfo_v1(
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

    /// Requests enabling or disabling Adaptive TGP Mode on this device.
    ///
    /// For Rubin or newer devices.
    /// Requires root/admin privileges.
    /// An out-of-band administrator policy may override the in-band request.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the device or mode is unsupported, or NVML denies permission to change adaptive TGP mode.
    /// Also propagates NVML configuration failures.
    pub fn set_adaptive_tgp_mode(&self, mode: EnableState) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the mode is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetAdaptiveTgpMode_v1(
                self.as_raw(),
                mode.into()
            ))
        }
    }

    /// Returns the performance metric samples of this device.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point or the native query fails.
    /// Returns an error if a reported sample or controller count exceeds its native array capacity.
    pub fn performance_metric_samples(&self) -> Result<Vec<PerformanceSample>> {
        let mut data = Box::new(sys::nvmlPerfMetricsSamples_v1_t::default());
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `data` is a writable boxed struct that outlives the call.
        unsafe {
            try_ffi!(sys::nvmlDevicePerfMetricsGetSamples_v1(
                self.as_raw(),
                &raw mut *data
            ))?;
        }

        let samples = data
            .samples
            .get(..to_usize(data.numSamples, "performance sample count")?)
            .ok_or_else(|| Error::out_of_range("performance sample count"))?;
        samples
            .iter()
            .map(|sample| {
                let controllers = sample
                    .controllerData
                    .get(..to_usize(sample.numControllerData, "controller sample count")?)
                    .ok_or_else(|| Error::out_of_range("controller sample count"))?;
                let controllers = controllers
                    .iter()
                    .map(|controller| match controller.controllerType {
                        sys::NVML_PERF_METRICS_CONTROLLER_TYPE_DLPPC_2X => {
                            PerformanceControllerSample::Dlppc2x(Box::new(
                                // SAFETY: `controllerType` is `NVML_PERF_METRICS_CONTROLLER_TYPE_DLPPC_2X`, which NVML documents as selecting the `dlppc2x` field.
                                unsafe { controller.data.dlppc2x }.into(),
                            ))
                        }
                        sys::NVML_PERF_METRICS_CONTROLLER_TYPE_PFPP_1X => {
                            PerformanceControllerSample::Pfpp1x(Box::new(
                                // SAFETY: `controllerType` is `NVML_PERF_METRICS_CONTROLLER_TYPE_PFPP_1X`, which NVML documents as selecting the `pfpp1x` field.
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
