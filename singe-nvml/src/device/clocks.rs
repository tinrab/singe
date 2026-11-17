#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use singe_nvml_sys as sys;

use crate::{
    device::{Device, field_id_for_perf_policy, field_sample_as_u64},
    error::Result,
    try_ffi,
    types::{
        AdaptiveClockInfoStatus, AutoBoostClocks, ClkMonStatus, ClockEventReasons, ClockId,
        ClockOffset, ClockRange, ClockRangeI32, ClockType, CurrentClockFreqs, DriverModelFlags,
        DynamicPstatesInfo, EnableState, PerfPolicyType, PerformanceModes, PerformanceState,
        ViolationTime, try_from_nvml_enum,
    },
    utility::{query_u32_list, struct_version, to_u32},
};

impl Device {
    /// Returns the current Auto Boosted clocks state for this device.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates to maximize performance as thermal limits allow.
    ///
    /// On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks, which NVML sets and resets.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support Auto Boosted clocks, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn auto_boosted_clocks(&self) -> Result<AutoBoostClocks> {
        let mut enabled = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        let mut default_enabled = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `enabled` and `default_enabled` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAutoBoostedClocksEnabled(
                self.raw,
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
    /// Auto Boosted clocks return to this default state when no compute processes, such as CUDA applications with active contexts, are running.
    ///
    /// For Kepler or newer non-GeForce fully supported devices and Maxwell or newer `GeForce` devices.
    /// Requires root/admin permissions.
    ///
    /// Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates to maximize performance as thermal limits allow.
    /// Disable Auto Boosted clocks when fixed clock rates are required.
    ///
    /// On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks, which NVML sets and resets.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, if the device does not support Auto Boosted clocks, if the current process lacks permission to change the default state, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_default_auto_boosted_clocks_enabled(
        &self,
        enabled: EnableState,
        flags: DriverModelFlags,
    ) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the state and flags are passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDefaultAutoBoostedClocksEnabled(
                self.raw,
                enabled.into(),
                flags.bits(),
            ))?;
        }
        Ok(())
    }

    /// Returns the current clock speeds for the device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// See [`ClockType`] for details on available clock information.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, clock type, or output, if the device cannot report the requested clock, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn clock(&self, kind: ClockType) -> Result<u32> {
        let mut clock = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `clock` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetClockInfo(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, clock type, clock ID, or output, if the device does not support this clock query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn clock_with_id(&self, kind: ClockType, clock_id: ClockId) -> Result<u32> {
        let mut clock = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `clock` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetClock(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, clock type, or output, if the device cannot report the requested maximum clock, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_clock(&self, kind: ClockType) -> Result<u32> {
        let mut clock = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `clock` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxClockInfo(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if the internal clock-frequency buffer is too small, if NVML rejects the handle or output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn current_clock_freqs(&self) -> Result<CurrentClockFreqs> {
        let mut current = sys::nvmlDeviceCurrentClockFreqs_t {
            version: struct_version::<sys::nvmlDeviceCurrentClockFreqs_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `current` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrentClockFreqs(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if the internal performance-mode buffer is too small, if NVML rejects the handle or output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn performance_modes(&self) -> Result<PerformanceModes> {
        let mut modes = sys::nvmlDevicePerfModes_t {
            version: struct_version::<sys::nvmlDevicePerfModes_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `modes` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPerformanceModes(self.raw, &raw mut modes))?;
        }
        Ok(modes.into())
    }

    /// Returns the configured application clock for the selected clock domain.
    ///
    /// # Errors
    ///
    /// Propagates clock-query errors from [`Self::clock_with_id`], including an unsupported clock domain or clock identifier and an inaccessible device.
    pub fn applications_clock(&self, kind: ClockType) -> Result<u32> {
        self.clock_with_id(kind, ClockId::AppClockTarget)
    }

    /// Returns the default application clock for the selected clock domain.
    ///
    /// # Errors
    ///
    /// Propagates clock-query errors from [`Self::clock_with_id`], including an unsupported clock domain or clock identifier and an inaccessible device.
    pub fn default_applications_clock(&self, kind: ClockType) -> Result<u32> {
        self.clock_with_id(kind, ClockId::AppClockDefault)
    }

    /// Returns the customer-defined maximum boost clock speed specified by `kind`.
    ///
    /// For Pascal or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, clock type, or output, if the device or requested clock type does not support customer boost clocks, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_customer_boost_clock(&self, kind: ClockType) -> Result<u32> {
        let mut clock = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `clock` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxCustomerBoostClock(
                self.raw,
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
    /// Returns an error if NVML rejects the handle, clock type, P-state, or outputs, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn min_max_clock_of_pstate(
        &self,
        kind: ClockType,
        pstate: PerformanceState,
    ) -> Result<ClockRange> {
        let mut min = 0;
        let mut max = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `min` and `max` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMinMaxClockOfPState(
                self.raw,
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
    /// [`Device::gpc_clock_vf_offset`], [`Device::memory_clock_vf_offset`], [`Device::gpc_clock_vf_offset_range`], and [`Device::memory_clock_vf_offset_range`] wrap deprecated NVML queries that are planned for removal.
    /// Use [`Device::clock_offsets`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the device, clock type, P-state, or outputs, if the device does not support clock-offset queries, or if NVML has not been initialized.
    pub fn clock_offsets(&self, kind: ClockType, pstate: PerformanceState) -> Result<ClockOffset> {
        let mut info = sys::nvmlClockOffset_t {
            version: struct_version::<sys::nvmlClockOffset_t>(1),
            type_: kind.into(),
            pstate: pstate.into(),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetClockOffsets(self.raw, &raw mut info))?;
        }
        Ok(info.into())
    }

    /// Sets the clock offset of a clock domain for a P-state.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Requires privileged access.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if NVML rejects the handle, clock type, P-state, or offset, if the device does not support clock-offset control, if the current process lacks permission, or if NVML has not been initialized.
    pub fn set_clock_offsets(&self, offset: ClockOffset) -> Result<()> {
        let mut info = sys::nvmlClockOffset_t {
            version: struct_version::<sys::nvmlClockOffset_t>(1),
            type_: offset.kind.into(),
            pstate: offset.pstate.into(),
            clockOffsetMHz: offset.clock_offset_mhz,
            minClockOffsetMHz: offset.min_clock_offset_mhz,
            maxClockOffsetMHz: offset.max_clock_offset_mhz,
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a local whose version matches its layout and that NVML only reads.
        unsafe { try_ffi!(sys::nvmlDeviceSetClockOffsets(self.raw, &raw mut info)) }
    }

    /// Returns the GPCCLK VF offset value.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support this deprecated offset query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gpc_clock_vf_offset(&self) -> Result<i32> {
        let mut offset = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `offset` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpcClkVfOffset(self.raw, &raw mut offset))?;
        }
        Ok(offset)
    }

    /// Returns the `MemClk` (Memory Clock) VF offset value.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support this deprecated offset query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn memory_clock_vf_offset(&self) -> Result<i32> {
        let mut offset = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `offset` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemClkVfOffset(self.raw, &raw mut offset))?;
        }
        Ok(offset)
    }

    /// Returns the supported graphics voltage-frequency clock offset range.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, or querying this clock offset range is unsupported.
    /// Also propagates NVML query failures.
    pub fn gpc_clock_vf_offset_range(&self) -> Result<ClockRangeI32> {
        device_clock_offset_range(self, sys::nvmlDeviceGetGpcClkMinMaxVfOffset)
    }

    /// Returns the supported memory voltage-frequency clock offset range.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, or querying this clock offset range is unsupported.
    /// Also propagates NVML query failures.
    pub fn memory_clock_vf_offset_range(&self) -> Result<ClockRangeI32> {
        device_clock_offset_range(self, sys::nvmlDeviceGetMemClkMinMaxVfOffset)
    }

    /// Returns the memory clocks, in MHz, that NVML accepts when locking memory clocks.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the clock list changes while the wrapper is fetching it, if NVML rejects the handle or count output, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn supported_memory_clocks(&self) -> Result<Vec<u32>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `query_u32_list` passes a writable count with either null or an array of that many entries.
        query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetSupportedMemoryClocks(self.raw, count, values)
        })
    }

    /// Returns the graphics clocks, in MHz, that NVML accepts when locking GPU clocks at `memory_clock_mhz`.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the clock list changes while the wrapper is fetching it, if NVML rejects the handle, memory clock, or output, if `memory_clock_mhz` is unsupported, if the device does not support this query, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn supported_graphics_clocks(&self, memory_clock_mhz: u32) -> Result<Vec<u32>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `query_u32_list` passes a writable count with either null or an array of that many entries.
        query_u32_list(|count, values| unsafe {
            sys::nvmlDeviceGetSupportedGraphicsClocks(self.raw, memory_clock_mhz, count, values)
        })
    }

    /// Returns the current performance state for the device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// See [`PerformanceState`] for details on allowed performance states.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support performance-state queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn performance_state(&self) -> Result<PerformanceState> {
        let mut state = sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `state` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPerformanceState(self.raw, &raw mut state))?;
        }
        Ok(state.into())
    }

    /// Returns all supported performance states (P-states) for the device.
    ///
    /// The returned vector lists the valid P-states supported by the device, without the native unknown entries that fill unused slots.
    ///
    /// The number of returned elements never exceeds 16.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed internal P-state buffer is too small, if NVML rejects the query, if the device does not support performance-state readings, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn supported_performance_states(&self) -> Result<Vec<PerformanceState>> {
        let mut states =
            [sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN; sys::NVML_MAX_GPU_PERF_PSTATES as usize];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the passed length is the length of the writable `states` array.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSupportedPerformanceStates(
                self.raw,
                states.as_mut_ptr(),
                to_u32(states.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(states
            .into_iter()
            .filter(|state| *state != sys::nvmlPstates_t::NVML_PSTATE_UNKNOWN)
            .map(PerformanceState::from)
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support clocks-event reasons, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn current_clocks_event_reasons(&self) -> Result<ClockEventReasons> {
        let mut reasons = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `reasons` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrentClocksEventReasons(
                self.raw,
                &raw mut reasons,
            ))?;
        }
        Ok(ClockEventReasons::from_bits_retain(reasons))
    }

    /// Returns the reasons currently limiting the device's clocks.
    ///
    /// # Errors
    ///
    /// Propagates NVML errors from [`Self::current_clocks_event_reasons`], including an inaccessible device or unsupported clock-event reporting.
    pub fn current_clocks_throttle_reasons(&self) -> Result<ClockEventReasons> {
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn supported_clocks_event_reasons(&self) -> Result<ClockEventReasons> {
        let mut reasons = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `reasons` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSupportedClocksEventReasons(
                self.raw,
                &raw mut reasons,
            ))?;
        }
        Ok(ClockEventReasons::from_bits_retain(reasons))
    }

    /// Returns the clock-limiting reasons supported by the device.
    ///
    /// # Errors
    ///
    /// Propagates NVML errors from [`Self::supported_clocks_event_reasons`], including an inaccessible device or unsupported clock-event reporting.
    pub fn supported_clocks_throttle_reasons(&self) -> Result<ClockEventReasons> {
        self.supported_clocks_event_reasons()
    }

    /// Returns the device's Adaptive Clock status.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support adaptive-clock status, or if NVML has not been initialized.
    pub fn adaptive_clock_info_status(&self) -> Result<AdaptiveClockInfoStatus> {
        let mut status = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `status` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAdaptiveClockInfoStatus(
                self.raw,
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support frequency-monitor status, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn clk_mon_status(&self) -> Result<ClkMonStatus> {
        let mut status = sys::nvmlClkMonStatus_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `status` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetClkMonStatus(self.raw, &raw mut status))?;
        }
        Ok(status.into())
    }

    /// Returns the duration of a performance-policy violation and its reference timestamp.
    ///
    /// # Errors
    ///
    /// Propagates field-query and per-field NVML errors, including an empty single-field response or an unknown field-value type.
    /// Also rejects a negative duration or a floating-point duration where an integer is required.
    pub fn violation_status(&self, perf_policy: PerfPolicyType) -> Result<ViolationTime> {
        let sample = self.single_field_value(field_id_for_perf_policy(perf_policy), 0)?;
        Ok(ViolationTime {
            reference_time: u64::try_from(sample.timestamp).unwrap_or_default(),
            violation_time: field_sample_as_u64(sample)?,
        })
    }

    /// Returns performance monitor samples from the associated subdevice.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if the device does not support dynamic P-state samples, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn dynamic_pstates_info(&self) -> Result<DynamicPstatesInfo> {
        let mut info = sys::nvmlGpuDynamicPstatesInfo_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDynamicPstatesInfo(
                self.raw,
                &raw mut info
            ))?;
        }
        Ok(info.into())
    }
}

fn device_clock_offset_range(
    device: &Device,
    function: unsafe extern "C" fn(sys::nvmlDevice_t, *mut i32, *mut i32) -> sys::nvmlReturn_t,
) -> Result<ClockRangeI32> {
    let mut min = 0;
    let mut max = 0;
    // SAFETY: `device` holds a live NVML device handle, and `function` writes only the two local outputs.
    unsafe {
        try_ffi!(function(device.as_raw(), &raw mut min, &raw mut max))?;
    }
    Ok(ClockRangeI32 { min, max })
}
