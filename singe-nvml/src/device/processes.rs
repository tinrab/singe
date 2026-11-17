use std::ptr;

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{
        AccountingStats, EnableState, Pid, ProcessDetail, ProcessInfo, ProcessMode,
        ProcessUtilizationInfo, ProcessUtilizationSample,
    },
    utility::{struct_version, to_usize},
};

impl Device {
    /// Returns GPU memory and instance information for running compute processes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, process reporting is unsupported, or NVML cannot size or retrieve the process list.
    /// Also propagates native process-list query failures.
    pub fn compute_running_processes(&self) -> Result<Vec<ProcessInfo>> {
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
    /// The process list is dynamic, so it can grow between the wrapper's size query and its fetch, which returns an error.
    ///
    /// In MIG mode, a physical device handle reports aggregate information only if the caller has appropriate privileges.
    /// Per-instance information can be queried by using specific MIG device handles.
    /// Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
    /// Protected memory usage is currently not available in MIG mode or on Windows.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if the process list changes while the wrapper is fetching it, if NVML rejects the device, process mode, or output, if the query is unsupported, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn running_process_details(&self, mode: ProcessMode) -> Result<Vec<ProcessDetail>> {
        let mut list = sys::nvmlProcessDetailList_t {
            version: struct_version::<sys::nvmlProcessDetailList_t>(1),
            mode: mode.into(),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `list` is a writable local whose version matches its layout and whose null `procArray` asks NVML only for the entry count.
        let status = unsafe { sys::nvmlDeviceGetRunningProcessDetailList(self.raw, &raw mut list) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && list.numProcArrayEntries == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut entries = vec![
            sys::nvmlProcessDetail_v1_t::default();
            to_usize(list.numProcArrayEntries, "process count")?
        ];
        list.procArray = entries.as_mut_ptr();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `list.procArray` points to `entries`, which holds `numProcArrayEntries` writable records.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetRunningProcessDetailList(
                self.raw,
                &raw mut list
            ))?;
        }

        entries.truncate(to_usize(list.numProcArrayEntries, "process count")?);
        Ok(entries.into_iter().map(Into::into).collect())
    }

    /// Returns GPU memory and instance information for running graphics processes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, process reporting is unsupported, or NVML cannot size or retrieve the process list.
    /// Also propagates native process-list query failures.
    pub fn graphics_running_processes(&self) -> Result<Vec<ProcessInfo>> {
        query_process_info_list(self, sys::nvmlDeviceGetGraphicsRunningProcesses)
    }

    /// Returns GPU memory and instance information for running MPS compute processes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, process reporting is unsupported, or NVML cannot size or retrieve the process list.
    /// Also propagates native process-list query failures.
    pub fn mps_compute_running_processes(&self) -> Result<Vec<ProcessInfo>> {
        query_process_info_list(self, sys::nvmlDeviceGetMPSComputeRunningProcesses)
    }

    /// Returns the current utilization and process ID.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running.
    /// This wrapper returns utilization values as a [`Vec`] of process samples.
    /// One utilization sample structure is returned per running process that had non-zero utilization during the last sample period.
    /// It includes the CPU timestamp at which the samples were recorded.
    /// Individual utilization values are returned as unsigned integer values.
    /// If no valid sample entries are found since `last_seen_timestamp`, the returned list is empty.
    ///
    /// This wrapper queries the required capacity first, then fetches the samples.
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
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, utilization buffer, or sampling timestamp, if the device does not support process utilization sampling, if NVML has not been initialized, or if NVML reports an unexpected failure.
    /// Missing sample entries are returned as an empty list.
    pub fn process_utilization(
        &self,
        last_seen_timestamp: u64,
    ) -> Result<Vec<ProcessUtilizationSample>> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and a null array with a writable `count` asks NVML only for the sample count.
        let status = unsafe {
            sys::nvmlDeviceGetProcessUtilization(
                self.raw,
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

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut samples = vec![
            sys::nvmlProcessUtilizationSample_t::default();
            to_usize(count, "process utilization sample count")?
        ];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `samples` holds `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetProcessUtilization(
                self.raw,
                samples.as_mut_ptr(),
                &raw mut count,
                last_seen_timestamp,
            ))?;
        }

        samples.truncate(to_usize(count, "process utilization sample count")?);
        Ok(samples.into_iter().map(Into::into).collect())
    }

    /// Returns the recent utilization and process ID for all running processes.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder, jpeg decoder, OFA (Optical Flow Accelerator) for all running processes.
    /// This wrapper returns utilization values as a [`Vec`] of process utilization records.
    /// One utilization sample structure is returned per running process that had non-zero utilization during the last sample period.
    /// It includes the CPU timestamp at which the samples were recorded.
    /// Individual utilization values are returned as unsigned integer values.
    ///
    /// This wrapper queries the required capacity first, then fetches the samples.
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
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if the process list changes while the wrapper is fetching it, if NVML rejects the query, if the device does not support process utilization sampling, if NVML has not been initialized, or if NVML reports an unexpected failure.
    /// Missing sample entries are returned as an empty list.
    pub fn processes_utilization(
        &self,
        last_seen_timestamp: u64,
    ) -> Result<Vec<ProcessUtilizationInfo>> {
        let mut info = sys::nvmlProcessesUtilizationInfo_t {
            version: struct_version::<sys::nvmlProcessesUtilizationInfo_t>(1),
            lastSeenTimeStamp: last_seen_timestamp,
            processSamplesCount: 0,
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local whose version matches its layout and whose null `procUtilArray` asks NVML only for the sample count.
        let status = unsafe { sys::nvmlDeviceGetProcessesUtilizationInfo(self.raw, &raw mut info) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && info.processSamplesCount == 0 {
            return Ok(Vec::new());
        }

        if status == sys::nvmlReturn_t::NVML_ERROR_NOT_FOUND {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut entries =
            vec![
                sys::nvmlProcessUtilizationInfo_v1_t::default();
                to_usize(info.processSamplesCount, "process utilization sample count")?
            ];
        info.procUtilArray = entries.as_mut_ptr();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info.procUtilArray` points to `entries`, which holds `processSamplesCount` writable records.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetProcessesUtilizationInfo(
                self.raw,
                &raw mut info
            ))?;
        }

        entries.truncate(to_usize(
            info.processSamplesCount,
            "process utilization sample count",
        )?);
        Ok(entries.into_iter().map(Into::into).collect())
    }

    /// Returns the state of per-process accounting mode.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// See [`Device::accounting_stats`] for the reported accounting metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support accounting mode, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `mode` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAccountingMode(self.raw, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Returns the accounting stats of the process `pid`.
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
    /// Returns an error if NVML rejects the handle, process ID, or output, if no accounting stats exist for the process, if accounting is unsupported or disabled, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_stats(&self, pid: Pid) -> Result<AccountingStats> {
        let mut stats = sys::nvmlAccountingStats_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `stats` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAccountingStats(
                self.raw,
                pid.0,
                &raw mut stats
            ))?;
        }
        Ok(stats.into())
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
    /// Returns an error if the accounting PID list changes while the wrapper is fetching it, if NVML rejects the handle or count output, if accounting is unsupported or disabled, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_pids(&self) -> Result<Vec<Pid>> {
        let mut count = 0;
        let status =
            // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and a null array with a writable `count` asks NVML only for the PID count.
            unsafe { sys::nvmlDeviceGetAccountingPids(self.raw, &raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut pids = vec![0u32; to_usize(count, "accounting PID count")?];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `pids` holds `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAccountingPids(
                self.raw,
                &raw mut count,
                pids.as_mut_ptr(),
            ))?;
        }

        pids.truncate(to_usize(count, "accounting PID count")?);
        Ok(pids.into_iter().map(Pid).collect())
    }

    /// Returns the number of processes that the circular buffer with accounting pids can hold.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// This is the maximum number of processes with stored accounting information before information about the oldest processes is overwritten by newer process information.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if accounting is unsupported or disabled, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_buffer_size(&self) -> Result<u32> {
        let mut size = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `size` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetAccountingBufferSize(
                self.raw,
                &raw mut size
            ))?;
        }
        Ok(size)
    }
}

fn query_process_info_list(
    device: &Device,
    function: unsafe extern "C" fn(
        sys::nvmlDevice_t,
        *mut u32,
        *mut sys::nvmlProcessInfo_t,
    ) -> sys::nvmlReturn_t,
) -> Result<Vec<ProcessInfo>> {
    let mut count = 0;
    // SAFETY: `device` holds a live NVML device handle, and a null array with a zero count asks NVML only for the size.
    let status = unsafe { function(device.as_raw(), &raw mut count, ptr::null_mut()) };
    if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
        return Ok(Vec::new());
    }

    if status != sys::nvmlReturn_t::NVML_SUCCESS
        && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
    {
        return Err(status.into());
    }

    let mut infos = vec![sys::nvmlProcessInfo_t::default(); to_usize(count, "process count")?];
    // SAFETY: `device` holds a live NVML device handle, and `infos` holds the `count` writable entries passed as its length.
    unsafe {
        try_ffi!(function(
            device.as_raw(),
            &raw mut count,
            infos.as_mut_ptr()
        ))?;
    }

    infos.truncate(to_usize(count, "process count")?);
    Ok(infos.into_iter().map(Into::into).collect())
}
