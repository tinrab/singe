use std::ptr;

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{
        EncoderSessionInfo, EncoderStats, EncoderType, FbcSessionInfo, FbcStats, Utilization,
        UtilizationCounter,
    },
    utility::to_usize,
};

impl Device {
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support utilization queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn utilization(&self) -> Result<Utilization> {
        let mut utilization = sys::nvmlUtilization_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `utilization` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetUtilizationRates(
                self.raw,
                &raw mut utilization,
            ))?;
        }
        Ok(utilization.into())
    }

    /// Returns video encoder utilization and its sampling period.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, or the device does not support the requested engine-utilization query.
    /// Also propagates NVML query failures.
    pub fn encoder_utilization(&self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetEncoderUtilization)
    }

    /// Returns video decoder utilization and its sampling period.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, or the device does not support the requested engine-utilization query.
    /// Also propagates NVML query failures.
    pub fn decoder_utilization(&self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetDecoderUtilization)
    }

    /// Returns JPEG utilization and its sampling period.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, or the device does not support the requested engine-utilization query.
    /// Also propagates NVML query failures.
    pub fn jpg_utilization(&self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetJpgUtilization)
    }

    /// Returns optical-flow utilization and its sampling period.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, or the device does not support the requested engine-utilization query.
    /// Also propagates NVML query failures.
    pub fn ofa_utilization(&self) -> Result<UtilizationCounter> {
        device_utilization_counter(self, sys::nvmlDeviceGetOfaUtilization)
    }

    /// Returns the current capacity of the device's encoder, as a percentage of maximum encoder capacity with valid values in the range 0-100.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if the device does not support the requested encoder, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_capacity(&self, encoder: EncoderType) -> Result<u32> {
        let mut capacity = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `capacity` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEncoderCapacity(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_stats(&self) -> Result<EncoderStats> {
        let mut session_count = 0;
        let mut average_fps = 0;
        let mut average_latency_us = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the three outputs are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEncoderStats(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if the active-session count changes while the wrapper is fetching sessions, if NVML reports an invalid session count, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_sessions(&self) -> Result<Vec<EncoderSessionInfo>> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and a null array with a writable `count` asks NVML only for the session count.
        let status = unsafe {
            sys::nvmlDeviceGetEncoderSessions(self.as_raw(), &raw mut count, ptr::null_mut())
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut sessions = vec![
            sys::nvmlEncoderSessionInfo_t::default();
            to_usize(count, "encoder session count")?
        ];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `sessions` holds `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEncoderSessions(
                self.as_raw(),
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }

        sessions.truncate(to_usize(count, "encoder session count")?);
        Ok(sessions.into_iter().map(Into::into).collect())
    }

    /// Returns the active frame buffer capture sessions statistics for the given device.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the stats output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fbc_stats(&self) -> Result<FbcStats> {
        let mut stats = sys::nvmlFBCStats_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `stats` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFBCStats(self.raw, &raw mut stats))?;
        }
        Ok(stats.into())
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
    /// Returns an error if the device is inaccessible, if the FBC session list changes while the wrapper is fetching it, if NVML reports an invalid session count, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fbc_sessions(&self) -> Result<Vec<FbcSessionInfo>> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and a null array with a writable `count` asks NVML only for the session count.
        let status = unsafe {
            sys::nvmlDeviceGetFBCSessions(self.as_raw(), &raw mut count, ptr::null_mut())
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut sessions =
            vec![sys::nvmlFBCSessionInfo_t::default(); to_usize(count, "FBC session count")?];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `sessions` holds `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetFBCSessions(
                self.as_raw(),
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }

        sessions.truncate(to_usize(count, "FBC session count")?);
        Ok(sessions.into_iter().map(Into::into).collect())
    }
}

fn device_utilization_counter(
    device: &Device,
    function: unsafe extern "C" fn(sys::nvmlDevice_t, *mut u32, *mut u32) -> sys::nvmlReturn_t,
) -> Result<UtilizationCounter> {
    let mut utilization = 0;
    let mut sampling_period_us = 0;
    // SAFETY: `device` holds a live NVML device handle, and `function` writes only the two local outputs.
    unsafe {
        try_ffi!(function(
            device.as_raw(),
            &raw mut utilization,
            &raw mut sampling_period_us,
        ))?;
    }
    Ok(UtilizationCounter {
        utilization,
        sampling_period_us,
    })
}
