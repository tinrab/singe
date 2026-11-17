#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::ptr;

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{
        CurrentPending, EccCounterType, EccErrorCounts, EccSramErrorStatus,
        EccSramUniqueUncorrectedErrorCounts, EnableState, MemoryErrorType, MemoryLocation,
        PageRetirementCause, RemappedRows, RetiredPage, RowRemapperHistogram,
    },
    utility::{struct_version, to_usize},
};

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

impl Device {
    /// Returns the current and pending ECC modes for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 1.0 or higher.
    ///
    /// Changing ECC modes requires a reboot.
    /// The "pending" ECC mode refers to the target mode following the next reboot.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or outputs, if the device does not support ECC mode reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn ecc_mode(&self) -> Result<CurrentPending<EnableState>> {
        let mut current = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        let mut pending = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `current` and `pending` are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetEccMode(
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

    /// Returns the default ECC modes for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 1.0 or higher.
    ///
    /// See [`EnableState`] for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support ECC mode reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn default_ecc_mode(&self) -> Result<EnableState> {
        let mut default_mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `default_mode` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDefaultEccMode(
                self.raw,
                &raw mut default_mode
            ))?;
        }
        Ok(default_mode.into())
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
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML rejects the handle or output, if the device does not support SRAM ECC status, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn sram_ecc_error_status(&self) -> Result<EccSramErrorStatus> {
        let mut status = sys::nvmlEccSramErrorStatus_t {
            version: struct_version::<sys::nvmlEccSramErrorStatus_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `status` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSramEccErrorStatus(
                self.raw,
                &raw mut status,
            ))?;
        }
        Ok(status.into())
    }

    /// Returns unique uncorrected SRAM error locations and their occurrence counts.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the native record version is unsupported, or this SRAM error query is unavailable.
    /// Also propagates NVML query failures.
    pub fn sram_unique_uncorrected_ecc_error_counts(
        &self,
    ) -> Result<EccSramUniqueUncorrectedErrorCounts> {
        let mut counts = sys::nvmlEccSramUniqueUncorrectedErrorCounts_t {
            version: struct_version::<sys::nvmlEccSramUniqueUncorrectedErrorCounts_t>(1),
            ..Default::default()
        };

        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `counts` is a writable local whose null `entries` asks NVML only for the entry count.
        let status = unsafe {
            sys::nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(self.raw, &raw mut counts)
        };
        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        if counts.entryCount == 0 {
            return Ok(EccSramUniqueUncorrectedErrorCounts {
                entries: Vec::new(),
            });
        }

        let mut entries = vec![
            sys::nvmlEccSramUniqueUncorrectedErrorEntry_v1_t::default();
            to_usize(counts.entryCount, "SRAM error entry count")?
        ];
        counts.entries = entries.as_mut_ptr();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `counts.entries` points to `entries`, which holds `counts.entryCount` writable records.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(
                self.raw,
                &raw mut counts,
            ))?;
        }

        entries.truncate(to_usize(counts.entryCount, "SRAM error entry count")?);
        Ok(EccSramUniqueUncorrectedErrorCounts {
            entries: entries.into_iter().map(Into::into).collect(),
        })
    }

    /// Returns the total ECC error counts for the device.
    ///
    /// For Fermi or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 1.0 or higher.
    /// Requires ECC Mode to be enabled.
    ///
    /// The total error count is the sum of errors across each separate memory system, that is, the total set of errors across the entire device.
    ///
    /// See [`MemoryErrorType`] for a description of available error types.
    /// See [`EccCounterType`] for a description of available counter types.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, error type, counter type, or output, if the device does not support ECC error reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn total_ecc_errors(
        &self,
        error_type: MemoryErrorType,
        counter_type: EccCounterType,
    ) -> Result<u64> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `count` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTotalEccErrors(
                self.raw,
                error_type.into(),
                counter_type.into(),
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns ECC error counts for L1 cache, L2 cache, device memory, and registers.
    ///
    /// # Errors
    ///
    /// Propagates errors from the component [`Self::memory_error_counter`] queries, including unsupported error types, counter scopes, or memory locations.
    /// No partial record is returned if one location query fails.
    pub fn detailed_ecc_errors(
        &self,
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
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 2.0 or higher to report aggregate location-based memory error counts.
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 1.0 or higher to report all other memory error counts.
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, error type, counter type, memory location, or output, if the device does not support ECC error reporting for the requested memory location, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn memory_error_counter(
        &self,
        error_type: MemoryErrorType,
        counter_type: EccCounterType,
        location: MemoryLocation,
    ) -> Result<u64> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `count` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemoryErrorCounter(
                self.raw,
                error_type.into(),
                counter_type.into(),
                location.into(),
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Returns the list of retired pages by source, including pages that are pending retirement.
    /// The returned address information is the hardware address of the retired page.
    /// This does not match the virtual address used in CUDA, but it matches the address information in Xid 63.
    ///
    /// Each page carries the time of its retirement, which NVML reports on Pascal and newer architectures.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the retired-page list changes while the wrapper is fetching it, if NVML rejects the handle, cause, count, or outputs, if the device does not support page retirement, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn retired_pages(&self, cause: PageRetirementCause) -> Result<Vec<RetiredPage>> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and null arrays with a writable `count` ask NVML only for the page count.
        let status = unsafe {
            sys::nvmlDeviceGetRetiredPages_v2(
                self.raw,
                cause.into(),
                &raw mut count,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut addresses = vec![0u64; to_usize(count, "retired page count")?];
        let mut timestamps = vec![0u64; to_usize(count, "retired page count")?];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `addresses` and `timestamps` each hold `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRetiredPages_v2(
                self.raw,
                cause.into(),
                &raw mut count,
                addresses.as_mut_ptr(),
                timestamps.as_mut_ptr(),
            ))?;
        }

        addresses.truncate(to_usize(count, "retired page count")?);
        timestamps.truncate(to_usize(count, "retired page count")?);
        Ok(addresses
            .into_iter()
            .zip(timestamps)
            .map(|(address, timestamp)| RetiredPage {
                address,
                timestamp: Some(timestamp),
            })
            .collect())
    }

    /// Returns whether any pages are pending retirement and need a reboot to fully retire.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support page-retirement status, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn retired_pages_pending_status(&self) -> Result<EnableState> {
        let mut pending = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `pending` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRetiredPagesPendingStatus(
                self.raw,
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
    /// A pending remapping does not affect future GPU work because error containment and dynamic page blacklisting handle it.
    ///
    /// On MIG-enabled GPUs with active instances, querying the number of remapped rows is not supported.
    ///
    /// For Ampere or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or outputs, if MIG is enabled or the device does not support remapped-row reporting, or if NVML reports an unexpected failure.
    pub fn remapped_rows(&self) -> Result<RemappedRows> {
        let mut corrected = 0;
        let mut uncorrected = 0;
        let mut pending = 0;
        let mut failure_occurred = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the four outputs are writable locals.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRemappedRows(
                self.raw,
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
    /// Returns an error if NVML reports an unexpected row-remapper query failure.
    pub fn row_remapper_histogram(&self) -> Result<RowRemapperHistogram> {
        let mut histogram = sys::nvmlRowRemapperHistogramValues_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `histogram` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRowRemapperHistogram(
                self.raw,
                &raw mut histogram,
            ))?;
        }
        Ok(histogram.into())
    }

    /// Sets the ECC mode for the device.
    ///
    /// For Kepler or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 1.0 or higher.
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or mode, if the device does not support ECC mode changes, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_ecc_mode(&self, mode: EnableState) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the mode is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetEccMode(self.raw, mode.into()))?;
        }
        Ok(())
    }

    /// Clears the ECC error and other memory error counts for the device.
    ///
    /// For Kepler or newer fully supported devices.
    /// Only applicable to devices with ECC.
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 2.0 or higher to clear aggregate location-based ECC counts.
    /// Requires [`InforomObject::Ecc`](crate::types::InforomObject::Ecc) version 1.0 or higher to clear all other ECC counts.
    /// Requires root/admin permissions.
    /// Requires ECC Mode to be enabled.
    ///
    /// Sets all of the specified ECC counters to 0, including both detailed and total counts.
    ///
    /// This operation takes effect immediately.
    ///
    /// See [`EccCounterType`] for details on available counter types.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or counter type, if the device does not support clearing ECC counters, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn clear_ecc_error_counts(&self, counter_type: EccCounterType) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the counter type is passed by value.
        unsafe {
            try_ffi!(sys::nvmlDeviceClearEccErrorCounts(
                self.raw,
                counter_type.into()
            ))?;
        }
        Ok(())
    }

    /// Returns the bank remapper status of a Rubin or newer device.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the device is inaccessible, or bank-remapper reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn bank_remapper_status(&self) -> Result<BankRemapperStatus> {
        let mut status = sys::nvmlEccBankRemapperStatus_v1_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `status` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBankRemapperStatus_v1(
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
}
