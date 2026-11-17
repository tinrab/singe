use std::ptr;

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    library::GpmSample,
    try_ffi,
    types::{
        EnableState, FieldQuery, FieldSample, FieldValue, GpmSupport, Sample, Samples, SamplingType,
    },
    utility::{to_i32, to_u32, to_usize},
};

/// PRM counter input values.
#[derive(Debug, Clone, Copy)]
pub struct PrmCounterRequest {
    /// Counter ID.
    pub counter: PrmCounterId,
    /// Local port number.
    pub local_port: u32,
}

/// PRM counter identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrmCounterId(pub u32);

impl PrmCounterId {
    /// Requests no counter.
    pub const NONE: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_NONE.0);

    /// PPCNT group 0x12, `link_down_events`.
    pub const PPCNT_PHYSICAL_LAYER_CTRS_LINK_DOWN_EVENTS: Self = Self(
        sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PHYSICAL_LAYER_CTRS_LINK_DOWN_EVENTS.0,
    );

    /// PPCNT group 0x12, `successful_recovery_events`.
    pub const PPCNT_PHYSICAL_LAYER_CTRS_SUCCESSFUL_RECOVERY_EVENTS: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PHYSICAL_LAYER_CTRS_SUCCESSFUL_RECOVERY_EVENTS.0);

    /// PPCNT group 0x1A, `total_successful_recovery_events`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_EVENTS: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_EVENTS.0);

    /// PPCNT group 0x1A, `time_since_last_recovery`.
    pub const PPCNT_RECOVERY_CTRS_TIME_SINCE_LAST_RECOVERY: Self = Self(
        sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_SINCE_LAST_RECOVERY.0,
    );

    /// PPCNT group 0x1A, `time_between_last_two_recoveries`.
    pub const PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES.0);

    /// PPCNT group 0x1A, `time_in_last_host_serdes_feq_recovery`.
    pub const PPCNT_RECOVERY_CTRS_TIME_IN_LAST_HOST_SERDES_FEQ_RECOVERY: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_IN_LAST_HOST_SERDES_FEQ_RECOVERY.0);

    /// PPCNT group 0x1A, `total_time_in_host_serdes_feq_recovery`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_TIME_IN_HOST_SERDES_FEQ_RECOVERY: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_TIME_IN_HOST_SERDES_FEQ_RECOVERY.0);

    /// PPCNT group 0x1A, `total_host_serdes_feq_recovery_count`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_RECOVERY_COUNT: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_RECOVERY_COUNT.0);

    /// PPCNT group 0x1A, `total_host_serdes_feq_successful_recovery_count`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_SUCCESSFUL_RECOVERY_COUNT: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_HOST_SERDES_FEQ_SUCCESSFUL_RECOVERY_COUNT.0);

    /// PPCNT group 0x1A, `last_host_serdes_feq_attempts_count`.
    pub const PPCNT_RECOVERY_CTRS_LAST_HOST_SERDES_FEQ_ATTEMPTS_COUNT: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_LAST_HOST_SERDES_FEQ_ATTEMPTS_COUNT.0);

    /// PPCNT group 0x1A, `last_successful_recovery_step_attempts`.
    pub const PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_STEP_ATTEMPTS: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_STEP_ATTEMPTS.0);

    /// PPCNT group 0x1A, `last_successful_recovery_time`.
    pub const PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_TIME: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_LAST_SUCCESSFUL_RECOVERY_TIME.0);

    /// PPCNT group 0x1A, `total_successful_recovery_time`.
    pub const PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_TIME: Self = Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TOTAL_SUCCESSFUL_RECOVERY_TIME.0);

    /// PPCNT group 0x20, `port_xmit_wait`.
    pub const PPCNT_PORTCOUNTERS_PORT_XMIT_WAIT: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PORTCOUNTERS_PORT_XMIT_WAIT.0);

    /// PPCNT group 0x22, `plr_rcv_codes`.
    pub const PPCNT_PLR_RCV_CODES: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODES.0);

    /// PPCNT group 0x22, `plr_rcv_code_err`.
    pub const PPCNT_PLR_RCV_CODE_ERR: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODE_ERR.0);

    /// PPCNT group 0x22, `plr_rcv_uncorrectable_code`.
    pub const PPCNT_PLR_RCV_UNCORRECTABLE_CODE: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_UNCORRECTABLE_CODE.0);

    /// PPCNT group 0x22, `plr_xmit_codes`.
    pub const PPCNT_PLR_XMIT_CODES: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_CODES.0);

    /// PPCNT group 0x22, `plr_xmit_retry_codes`.
    pub const PPCNT_PLR_XMIT_RETRY_CODES: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_CODES.0);

    /// PPCNT group 0x22, `plr_xmit_retry_events`.
    pub const PPCNT_PLR_XMIT_RETRY_EVENTS: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_EVENTS.0);

    /// PPCNT group 0x22, `plr_sync_events`.
    pub const PPCNT_PLR_SYNC_EVENTS: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPCNT_PLR_SYNC_EVENTS.0);

    /// PPRM, `oper_recovery`.
    pub const PPRM_OPER_RECOVERY: Self =
        Self(sys::nvmlPRMCounterId_t::NVML_PRM_COUNTER_ID_PPRM_OPER_RECOVERY.0);
}

impl Device {
    /// Returns whether this device supports GPM.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// This supports device handles and MIG device handles.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or cannot query GPM support.
    pub fn gpm_support(&self) -> Result<GpmSupport> {
        let mut support = sys::nvmlGpmSupport_t {
            version: sys::NVML_GPM_SUPPORT_VERSION,
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `support` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlGpmQueryDeviceSupport(self.raw, &raw mut support))?;
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
    /// Returns an error if NVML rejects the handle or output, if the device does not support GPM streaming-state queries, or if NVML has not been initialized.
    pub fn gpm_streaming_enabled(&self) -> Result<EnableState> {
        let mut state = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `state` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlGpmQueryIfStreamingEnabled(
                self.raw,
                &raw mut state
            ))?;
        }
        Ok(EnableState::from(sys::nvmlEnableState_t(state)))
    }

    /// Reads a sample of GPM metrics into `sample`.
    /// After two samples are gathered, [`Library::gpm_metrics`](crate::library::Library::gpm_metrics) computes metrics from them.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// * The interval between two [`Device::gpm_sample`] calls must be greater than 100 ms due to the internal sample refresh rate.
    /// * Supports device handles and MIG device handles.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or sample buffer, if the device does not support GPM sampling, or if samples are requested too quickly.
    pub fn gpm_sample(&self, sample: &mut GpmSample) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the mutable borrow of `sample` gives NVML exclusive use of the sample buffer it owns.
        unsafe { try_ffi!(sys::nvmlGpmSampleGet(self.raw, sample.as_raw())) }
    }

    /// Reads a sample of GPM metrics for the MIG GPU instance `gpu_instance_id` into `sample`.
    ///
    /// After two samples are gathered, [`Library::gpm_metrics`](crate::library::Library::gpm_metrics) computes metrics from them.
    ///
    /// For Hopper or newer fully supported devices.
    ///
    /// The interval between two [`Device::gpm_mig_sample`] calls must be greater than 100 ms due to the internal sample refresh rate.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, GPU instance id, or sample buffer, if the device does not support GPM MIG sampling, or if samples are requested too quickly.
    pub fn gpm_mig_sample(&self, gpu_instance_id: u32, sample: &mut GpmSample) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the mutable borrow of `sample` gives NVML exclusive use of the sample buffer it owns.
        unsafe {
            try_ffi!(sys::nvmlGpmMigSampleGet(
                self.raw,
                gpu_instance_id,
                sample.as_raw(),
            ))
        }
    }

    /// Requests values for a list of device fields.
    /// Allows multiple fields to be queried at once.
    /// If multiple field IDs are populated by the same driver call, NVML populates those results from one call rather than one call per field ID.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device handle or field-value buffer.
    pub fn field_values(&self, queries: &[FieldQuery]) -> Result<Vec<FieldSample>> {
        let mut values = queries
            .iter()
            .map(|query| sys::nvmlFieldValue_t {
                fieldId: query.field.0,
                scopeId: query.scope_id,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `values` holds the passed count of writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFieldValues(
                self.raw,
                to_i32(values.len(), "field values")?,
                values.as_mut_ptr(),
            ))?;
        }
        Ok(values.into_iter().map(FieldSample::from_raw).collect())
    }

    /// Clears the values of a list of device fields.
    /// Allows multiple fields to be cleared at once.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the device handle or field-value buffer.
    pub fn clear_field_values(&self, queries: &[FieldQuery]) -> Result<Vec<FieldSample>> {
        let mut values = queries
            .iter()
            .map(|query| sys::nvmlFieldValue_t {
                fieldId: query.field.0,
                scopeId: query.scope_id,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `values` holds the passed count of writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceClearFieldValues(
                self.raw,
                to_i32(values.len(), "field values")?,
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
    /// `last_seen_timestamp` represents a CPU timestamp in microseconds.
    /// Use `0` to fetch all samples maintained by the buffer.
    /// Use a timestamp returned by a previous query to get more recent samples.
    ///
    /// This wrapper queries the sample count first, then returns the samples NVML wrote.
    /// Compared with polling the current-value methods, samples provide higher-frequency data at lower polling cost.
    ///
    /// On MIG-enabled GPUs, querying the following sample types, [`SamplingType::GpuUtilization`], [`SamplingType::MemoryUtilization`], [`SamplingType::EncoderUtilization`], and [`SamplingType::DecoderUtilization`], is not currently supported.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if the device does not support the requested sample type, if NVML has not been initialized, or if NVML reports an unexpected failure.
    /// Missing sample entries are returned as an empty sample list.
    pub fn samples(&self, kind: SamplingType, last_seen_timestamp: u64) -> Result<Samples> {
        let mut value_type = sys::nvmlValueType_t::NVML_VALUE_TYPE_UNSIGNED_INT;
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and a null array with writable `value_type` and `count` asks NVML only for the sample count.
        let status = unsafe {
            sys::nvmlDeviceGetSamples(
                self.raw,
                kind.into(),
                last_seen_timestamp,
                &raw mut value_type,
                &raw mut count,
                ptr::null_mut(),
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Samples {
                value_type: value_type.into(),
                samples: Vec::new(),
            });
        }

        if status == sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND {
            return Ok(Samples {
                value_type: value_type.into(),
                samples: Vec::new(),
            });
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut samples = vec![sys::nvmlSample_t::default(); to_usize(count, "sample count")?];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `samples` holds `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSamples(
                self.raw,
                kind.into(),
                last_seen_timestamp,
                &raw mut value_type,
                &raw mut count,
                samples.as_mut_ptr(),
            ))?;
        }

        samples.truncate(to_usize(count, "sample count")?);
        Ok(Samples {
            value_type: value_type.into(),
            samples: samples
                .into_iter()
                .map(|sample| Sample::from_raw(value_type, sample))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    /// Reads a list of GPU PRM counters for Blackwell or newer devices on Linux.
    ///
    /// # Errors
    ///
    /// Returns an error if the request count cannot fit in `u32` or NVML rejects the counter batch.
    /// Per-counter NVML errors and unrecognized scalar value types are retained in the returned inner results.
    pub fn read_prm_counters(
        &self,
        requests: &[PrmCounterRequest],
    ) -> Result<Vec<Result<FieldValue>>> {
        let count = to_u32(requests.len(), "request count")?;
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
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `list` points to the `count` writable entries of `counters`.
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
