use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;
use std::{mem, ptr};

use crate::{
    error::Result,
    try_ffi,
    types::{
        AccountingStats, EnableState, EncoderSessionInfo, EncoderStats, FbcSessionInfo, FbcStats,
        Pid, VgpuCompatibility, VgpuInstanceId, VgpuLicenseInfo, VgpuMetadata, VgpuPlacementId,
        VgpuRuntimeState, VgpuTypeId, VgpuVmId, VgpuVmIdType,
    },
    utility::{query_sized_raw, query_u32_list, struct_version},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VgpuInstance(sys::nvmlVgpuInstance_t);

impl VgpuInstance {
    pub const fn from_id(id: VgpuInstanceId) -> Self {
        Self(id.0)
    }

    pub const fn id(self) -> VgpuInstanceId {
        VgpuInstanceId(self.0)
    }

    /// Returns the vGPU type of a vGPU instance.
    ///
    /// Returns the vGPU type ID assigned to the vGPU instance.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the
    /// handle does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_type(self) -> Result<VgpuTypeId> {
        let mut vgpu_type_id = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetType(self.0, &raw mut vgpu_type_id))?;
        }
        Ok(VgpuTypeId(vgpu_type_id))
    }

    /// Query the placement ID of active vGPU instance.
    ///
    /// When vGPU heterogeneous mode is enabled, returns the placement ID for the vGPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the vGPU
    /// instance or query arguments are invalid, if the handle does not match an
    /// active vGPU instance, or if NVML reports an unexpected failure.
    pub fn placement_id(self) -> Result<VgpuPlacementId> {
        let mut placement = sys::nvmlVgpuPlacementId_t {
            version: struct_version::<sys::nvmlVgpuPlacementId_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetPlacementId(
                self.0,
                &raw mut placement
            ))?;
        }
        Ok(placement.into())
    }

    /// Queries the state of per process accounting mode on vGPU.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest driver is not running, if NVML rejects
    /// the handle or output argument, if the handle does not match an active
    /// vGPU instance, if the vGPU does not support accounting, if NVML has not
    /// been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_mode(self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetAccountingMode(
                self.0,
                &raw mut mode
            ))?;
        }
        Ok(mode.into())
    }

    /// Returns processes running on this vGPU instance that have accounting stats.
    /// Returned processes can be running or terminated.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// This wrapper queries the required process count internally.
    /// It returns an empty list when no processes are available.
    ///
    /// See [`VgpuInstance::accounting_stats`] for the per-process metrics.
    ///
    /// With a PID collision, some processes may not be accessible before the circular buffer is full.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal PID buffer is too small, if NVML
    /// rejects the handle or process-count output, if the handle does not match
    /// an active vGPU instance, if the vGPU does not support accounting or
    /// accounting is disabled, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn accounting_pids(self) -> Result<Vec<Pid>> {
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlVgpuInstanceGetAccountingPids(self.0, count, values)
        })?
        .into_iter()
        .map(Pid)
        .collect())
    }

    /// Queries process's accounting stats.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Accounting stats capture GPU utilization and other statistics across the lifetime of a process.
    /// Stats can be queried while the process is running and after it terminates.
    /// The reported running time remains 0 while the process is still running and is updated to the actual duration after termination.
    /// Accounting stats are kept in a circular buffer, newly created processes overwrite information about old processes.
    ///
    /// The returned value includes the per-process accounting metrics exposed by NVML.
    /// Use [`VgpuInstance::accounting_pids`] to list processes with available stats.
    ///
    /// * Accounting mode must be enabled.
    ///   See [`VgpuInstance::accounting_mode`].
    /// * Only compute and graphics application stats can be queried.
    ///   Monitoring application stats cannot be queried because they do not contribute to GPU utilization.
    /// * With a PID collision, only stats for the latest process to terminate are reported.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or stats output, if the
    /// handle does not match an active vGPU instance or stats are unavailable,
    /// if the vGPU does not support accounting or accounting is disabled, if
    /// NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_stats(self, pid: Pid) -> Result<AccountingStats> {
        let mut stats = sys::nvmlAccountingStats_t::default();
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetAccountingStats(
                self.0,
                pid.0,
                &raw mut stats,
            ))?;
        }
        Ok(stats.into())
    }

    /// Returns the UUID of a vGPU instance.
    ///
    /// The UUID is a globally unique identifier associated with the vGPU.
    /// It is returned as a five-part hexadecimal string that does not exceed 80
    /// characters including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the UUID buffer is too small, if NVML rejects the
    /// handle or output argument, if the handle does not match an active vGPU
    /// instance, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn uuid(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_DEVICE_UUID_BUFFER_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetUUID(
                self.0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the VM ID associated with a vGPU instance.
    ///
    /// The VM ID is returned as a string that does not exceed 80 characters
    /// including the terminating NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// The VM ID type identifies the platform-specific format of the returned string.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the VM ID buffer is too small, if NVML rejects the
    /// handle, VM ID, or VM ID type output, if the handle does not match an
    /// active vGPU instance, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn vm_id(self) -> Result<VgpuVmId> {
        let mut buffer = [0i8; sys::NVML_DEVICE_UUID_BUFFER_SIZE as usize];
        let mut vm_id_type = sys::nvmlVgpuVmIdType_t::NVML_VGPU_VM_ID_DOMAIN_ID;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetVmID(
                self.0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
                &raw mut vm_id_type,
            ))?;
        }
        Ok(VgpuVmId {
            value: string_from_c_chars(&buffer),
            kind: VgpuVmIdType::from(vm_id_type),
        })
    }

    /// Returns the NVIDIA driver version installed in the VM associated with a vGPU.
    ///
    /// The version is returned as an alphanumeric string.
    /// The version string does not exceed 80 bytes including the terminating
    /// NUL byte.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// [`VgpuInstance::vm_driver_version`] may be called at any time for a vGPU instance.
    /// The guest VM driver version is returned as `Not Available` if no NVIDIA
    /// driver is installed in the VM, or if the VM has not yet loaded and
    /// initialized the NVIDIA driver.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal driver-version buffer is too small, if
    /// NVML rejects the handle, if the handle does not match an active vGPU
    /// instance, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn vm_driver_version(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetVmDriverVersion(
                self.0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the MDEV UUID of a vGPU instance.
    ///
    /// The MDEV UUID is a globally unique identifier of the mdev device assigned
    /// to the VM.
    /// It is returned as a five-part hexadecimal string that does not exceed 80
    /// characters including the terminating NUL byte.
    /// The MDEV UUID is displayed only on KVM.
    /// This wrapper allocates the required NVML buffer internally.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the mediated-device UUID buffer is too small, if
    /// NVML rejects the handle or UUID output, if the handle does not match an
    /// active vGPU instance, if the hypervisor is not KVM, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn mdev_uuid(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_DEVICE_UUID_BUFFER_SIZE as usize];
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetMdevUUID(
                self.0,
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the framebuffer usage in bytes.
    ///
    /// Framebuffer usage is the amount of vGPU framebuffer memory that is currently in use by the VM.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the
    /// handle does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn framebuffer_usage(self) -> Result<u64> {
        let mut usage = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFbUsage(self.0, &raw mut usage))?;
        }
        Ok(usage)
    }

    pub fn license_status(self) -> Result<bool> {
        Ok(self.license_info()?.is_licensed)
    }

    /// Query the license information of the vGPU instance.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest driver is not running, if NVML rejects
    /// the handle or output argument, if the handle does not match an active
    /// vGPU instance, or if NVML reports an unexpected failure.
    pub fn license_info(self) -> Result<VgpuLicenseInfo> {
        let mut info = sys::nvmlVgpuLicenseInfo_t::default();
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetLicenseInfo_v2(
                self.0,
                &raw mut info
            ))?;
        }
        Ok(info.into())
    }

    /// Returns the frame rate limit set for the vGPU instance.
    ///
    /// Returns the value of the frame-rate limit set for the vGPU instance.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the
    /// handle does not match an active vGPU instance, if the frame-rate limiter
    /// is disabled for this vGPU type, if NVML has not been initialized, or if
    /// NVML reports an unexpected failure.
    pub fn frame_rate_limit(self) -> Result<u32> {
        let mut limit = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFrameRateLimit(
                self.0,
                &raw mut limit
            ))?;
        }
        Ok(limit)
    }

    /// Returns the current ECC mode of a vGPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the
    /// handle does not match an active vGPU instance, if the vGPU does not
    /// support ECC mode, if NVML has not been initialized, or if NVML reports an
    /// unexpected failure.
    pub fn ecc_mode(self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEccMode(self.0, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Returns the encoder capacity of a vGPU instance, as a percentage of maximum encoder capacity with valid values in the range 0-100.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or encoder query, if the
    /// handle does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn encoder_capacity(self) -> Result<u32> {
        let mut capacity = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEncoderCapacity(
                self.0,
                &raw mut capacity
            ))?;
        }
        Ok(capacity)
    }

    /// Returns the current encoder statistics of a vGPU instance.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or query arguments, if the
    /// handle does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn encoder_stats(self) -> Result<EncoderStats> {
        let mut session_count = 0;
        let mut average_fps = 0;
        let mut average_latency_us = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEncoderStats(
                self.0,
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

    /// Returns information about all active encoder sessions on a vGPU instance.
    ///
    /// This wrapper queries the required session count first, then returns the active encoder sessions as a [`Vec`].
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal session buffer is too small, if NVML
    /// rejects the handle or reports an invalid session count, if the handle
    /// does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn encoder_sessions(self) -> Result<Vec<EncoderSessionInfo>> {
        let mut count = 0;
        let status = unsafe {
            sys::nvmlVgpuInstanceGetEncoderSessions(self.0, &raw mut count, ptr::null_mut())
        };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut sessions = vec![sys::nvmlEncoderSessionInfo_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEncoderSessions(
                self.0,
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }
        sessions.truncate(count as usize);
        Ok(sessions.into_iter().map(Into::into).collect())
    }

    /// Returns the active frame buffer capture sessions statistics of a vGPU instance.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the
    /// handle does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn fbc_stats(self) -> Result<FbcStats> {
        let mut stats = sys::nvmlFBCStats_t::default();
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFBCStats(self.0, &raw mut stats))?;
        }
        Ok(stats.into())
    }

    /// Returns information about active frame buffer capture sessions on a vGPU instance.
    ///
    /// This wrapper queries the required session count first, then returns the active FBC sessions as a [`Vec`].
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// `h_resolution`, `v_resolution`, `average_fps`, and `average_latency` may be zero if no new frames were captured since the session started.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal session buffer is too small, if NVML
    /// rejects the handle or reports an invalid session count, if the handle
    /// does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn fbc_sessions(self) -> Result<Vec<FbcSessionInfo>> {
        let mut count = 0;
        let status =
            unsafe { sys::nvmlVgpuInstanceGetFBCSessions(self.0, &raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }
        if status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(status.into());
        }

        let mut sessions = vec![sys::nvmlFBCSessionInfo_t::default(); count as usize];
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFBCSessions(
                self.0,
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }
        sessions.truncate(count as usize);
        Ok(sessions.into_iter().map(Into::into).collect())
    }

    /// Returns the GPU instance ID for the given vGPU instance.
    /// Returns a valid GPU instance ID for MIG-backed vGPU instances; otherwise returns `INVALID_GPU_INSTANCE_ID`.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or query arguments, if the
    /// handle does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn gpu_instance_id(self) -> Result<u32> {
        let mut id = 0;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetGpuInstanceId(self.0, &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns the PCI ID of the given vGPU instance, that is, the PCI ID of the GPU as seen inside the VM.
    ///
    /// The vGPU PCI ID is returned as `00000000:00:00.0` if the NVIDIA driver is
    /// not installed on the vGPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest driver is not running, if the internal PCI
    /// ID buffer is too small, if NVML rejects the handle or output argument, if
    /// the handle does not match an active vGPU instance, if NVML has not been
    /// initialized, or if NVML reports an unexpected failure.
    pub fn gpu_pci_id(self) -> Result<String> {
        let mut buffer = [0i8; sys::NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE as usize];
        let mut length = buffer.len() as u32;
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetGpuPciId(
                self.0,
                buffer.as_mut_ptr(),
                &raw mut length,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the currently used runtime state size of the vGPU instance.
    ///
    /// This size represents the maximum in-memory data size used by a vGPU instance during standard operation.
    /// This measurement is exclusive of frame buffer (FB) data size assigned to the vGPU instance.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the vGPU
    /// instance or query arguments are invalid, if the handle does not match an
    /// active vGPU instance, if NVML has not been initialized, or if NVML
    /// reports an unexpected failure.
    pub fn runtime_state(self) -> Result<VgpuRuntimeState> {
        let mut state = sys::nvmlVgpuRuntimeState_t {
            version: struct_version::<sys::nvmlVgpuRuntimeState_t>(1),
            ..Default::default()
        };
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetRuntimeStateSize(
                self.0,
                &raw mut state
            ))?;
        }
        Ok(state.into())
    }

    /// Returns vGPU metadata structure for a running vGPU.
    /// The structure contains information about the vGPU and its associated VM such as the currently installed NVIDIA guest driver version, together with host driver version and an opaque data section containing internal state.
    ///
    /// [`VgpuInstance::metadata`] may be called at any time for a vGPU instance.
    /// Some fields in the returned structure are dependent on information obtained from the guest VM, which may not yet have reached a state where that information is available.
    /// The current state of these dependent fields is reflected in the info structure's [`VgpuGuestInfoState`](crate::types::VgpuGuestInfoState) field.
    ///
    /// The VMM may choose to read and save the vGPU's VM info as persistent metadata associated with the VM, and provide it to Virtual GPU Manager when creating a vGPU for subsequent instances of the VM.
    ///
    /// This wrapper allocates the metadata buffer internally and retries if NVML reports it is too small.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal metadata buffer is too small after
    /// retrying, if NVML rejects the handle or metadata arguments, if the handle
    /// does not match an active vGPU instance, or if NVML reports an unexpected
    /// failure.
    pub fn metadata(self) -> Result<VgpuMetadata> {
        let raw = query_sized_raw(|size, buffer| unsafe {
            sys::nvmlVgpuInstanceGetMetadata(self.0, buffer.cast(), size)
        })?;

        let metadata = unsafe { &*raw.as_ptr().cast::<sys::nvmlVgpuMetadata_t>() };
        let opaque_offset = mem::offset_of!(sys::nvmlVgpuMetadata_t, opaqueData);
        let opaque_end = opaque_offset.saturating_add(metadata.opaqueDataSize as usize);
        let opaque_data = raw[opaque_offset..opaque_end].to_vec();

        Ok(VgpuMetadata::from_raw(metadata, opaque_data))
    }

    pub fn compatibility_with(
        self,
        library: &crate::library::Library,
        pgpu_metadata: &crate::types::PgpuMetadata,
    ) -> Result<VgpuCompatibility> {
        let vgpu_metadata = self.metadata()?;
        library.vgpu_compatibility(&vgpu_metadata, pgpu_metadata)
    }
}
