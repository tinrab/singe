use std::{
    hash::{Hash, Hasher},
    ptr,
};

use singe_core::string_from_c_chars;
use singe_nvml_sys as sys;

use crate::{
    error::Result,
    library::Library,
    try_ffi,
    types::{
        AccountingStats, EnableState, EncoderSessionInfo, EncoderStats, FbcSessionInfo, FbcStats,
        PgpuMetadata, Pid, VgpuCompatibility, VgpuInstanceId, VgpuLicenseInfo, VgpuMetadata,
        VgpuPlacementId, VgpuRuntimeState, VgpuTypeId, VgpuVmId,
    },
    utility::{query_sized_raw, query_u32_list, struct_version, to_u32, to_usize},
    vgpu_type::VgpuType,
};

/// Identifies a vGPU instance for NVML queries and controls.
///
/// It keeps NVML initialized.
/// Equality and hashing compare the instance identifier.
#[derive(Debug, Clone)]
pub struct VgpuInstance {
    id: sys::nvmlVgpuInstance_t,
    library: Library,
}

impl PartialEq for VgpuInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for VgpuInstance {}

impl Hash for VgpuInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl VgpuInstance {
    /// Wraps an identifier and keeps `library`'s initialization alive, without checking that the instance exists.
    #[must_use]
    pub fn from_id(id: VgpuInstanceId, library: &Library) -> Self {
        Self {
            id: id.0,
            library: library.clone(),
        }
    }

    /// Returns the typed identifier used for native NVML calls.
    #[must_use]
    pub const fn id(&self) -> VgpuInstanceId {
        VgpuInstanceId(self.id)
    }

    /// Returns the NVML initialization that this instance keeps alive.
    #[must_use]
    pub const fn library(&self) -> &Library {
        &self.library
    }

    /// Returns the vGPU type of this instance.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vgpu_type(&self) -> Result<VgpuType> {
        let mut vgpu_type_id = 0;
        // SAFETY: NVML validates the instance ID, and `vgpu_type_id` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetType(self.id, &raw mut vgpu_type_id))?;
        }
        Ok(VgpuType::from_id(VgpuTypeId(vgpu_type_id), &self.library))
    }

    /// Returns the placement ID of this active instance in vGPU heterogeneous mode.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the vGPU instance or query arguments are invalid, if the handle does not match an active vGPU instance, or if NVML reports an unexpected failure.
    pub fn placement_id(&self) -> Result<VgpuPlacementId> {
        let mut placement = sys::nvmlVgpuPlacementId_t {
            version: struct_version::<sys::nvmlVgpuPlacementId_t>(1),
            ..Default::default()
        };
        // SAFETY: NVML validates the instance ID, and `placement` is a writable local struct with its version set.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetPlacementId(
                self.id,
                &raw mut placement
            ))?;
        }
        Ok(placement.into())
    }

    /// Returns whether per-process accounting is enabled on this instance.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest driver is not running, if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if the vGPU does not support accounting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: NVML validates the instance ID, and `mode` is a writable local enum.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetAccountingMode(
                self.id,
                &raw mut mode
            ))?;
        }
        Ok(mode.into())
    }

    /// Returns the running or terminated processes on this instance that have accounting stats.
    ///
    /// [`VgpuInstance::accounting_stats`] returns each one's metrics.
    /// After a PID collision, some processes may be unreachable until the circular buffer fills.
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal PID buffer is too small, if NVML rejects the handle or process-count output, if the handle does not match an active vGPU instance, if the vGPU does not support accounting or accounting is disabled, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_pids(&self) -> Result<Vec<Pid>> {
        // SAFETY: NVML validates the instance ID, and `query_u32_list` passes a writable count with a null array or one of `*count` writable entries.
        Ok(query_u32_list(|count, values| unsafe {
            sys::nvmlVgpuInstanceGetAccountingPids(self.id, count, values)
        })?
        .into_iter()
        .map(Pid)
        .collect())
    }

    /// Returns the accounting stats of process `pid` over its lifetime.
    ///
    /// Stats stay available after the process terminates, until newer processes overwrite them in a circular buffer; the running time is zero until it terminates.
    /// Accounting must be enabled (see [`VgpuInstance::accounting_mode`]), only compute and graphics processes are tracked, and after a PID collision only the latest process to terminate is reported.
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or stats output, if the handle does not match an active vGPU instance or stats are unavailable, if the vGPU does not support accounting or accounting is disabled, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn accounting_stats(&self, pid: Pid) -> Result<AccountingStats> {
        let mut stats = sys::nvmlAccountingStats_t::default();
        // SAFETY: NVML validates the instance ID, and `stats` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetAccountingStats(
                self.id,
                pid.0,
                &raw mut stats,
            ))?;
        }
        Ok(stats.into())
    }

    /// Returns the globally unique UUID of this instance, as a five-part hexadecimal string.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the UUID buffer is too small, if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn uuid(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_DEVICE_UUID_BUFFER_SIZE as usize];
        // SAFETY: NVML validates the instance ID, and `buffer` holds the `buffer.len()` writable bytes passed as its length.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetUUID(
                self.id,
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the ID of the VM running this instance, with the platform-specific kind of the ID.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the VM ID buffer is too small, if NVML rejects the handle, VM ID, or VM ID type output, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vm_id(&self) -> Result<VgpuVmId> {
        let mut buffer = [0; sys::NVML_DEVICE_UUID_BUFFER_SIZE as usize];
        let mut vm_id_type = sys::nvmlVgpuVmIdType_t::NVML_VGPU_VM_ID_DOMAIN_ID;
        // SAFETY: NVML validates the instance ID, and `buffer` holds the `buffer.len()` writable bytes passed as its length, and `vm_id_type` is a writable local enum.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetVmID(
                self.id,
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
                &raw mut vm_id_type,
            ))?;
        }
        Ok(VgpuVmId {
            value: string_from_c_chars(&buffer),
            kind: vm_id_type.into(),
        })
    }

    /// Returns the NVIDIA driver version installed in the VM running this instance.
    ///
    /// The version is `Not Available` until the VM has loaded and initialized an NVIDIA driver.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal driver-version buffer is too small, if NVML rejects the handle, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn vm_driver_version(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE as usize];
        // SAFETY: NVML validates the instance ID, and `buffer` holds the `buffer.len()` writable bytes passed as its length.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetVmDriverVersion(
                self.id,
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the UUID of the mediated device assigned to the VM, as a five-part hexadecimal string.
    ///
    /// Only KVM reports it.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the mediated-device UUID buffer is too small, if NVML rejects the handle or UUID output, if the handle does not match an active vGPU instance, if the hypervisor is not KVM, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn mdev_uuid(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_DEVICE_UUID_BUFFER_SIZE as usize];
        // SAFETY: NVML validates the instance ID, and `buffer` holds the `buffer.len()` writable bytes passed as its length.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetMdevUUID(
                self.id,
                buffer.as_mut_ptr(),
                to_u32(buffer.len(), "NVML buffer length")?,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns how many bytes of the vGPU framebuffer the VM is using.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn framebuffer_usage(&self) -> Result<u64> {
        let mut usage = 0;
        // SAFETY: NVML validates the instance ID, and `usage` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFbUsage(self.id, &raw mut usage))?;
        }
        Ok(usage)
    }

    /// Returns whether the vGPU instance has an active license.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::license_info`] fails.
    pub fn license_status(&self) -> Result<bool> {
        Ok(self.license_info()?.is_licensed)
    }

    /// Returns the license information of this instance.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest driver is not running, if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, or if NVML reports an unexpected failure.
    pub fn license_info(&self) -> Result<VgpuLicenseInfo> {
        let mut info = sys::nvmlVgpuLicenseInfo_t::default();
        // SAFETY: NVML validates the instance ID, and `info` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetLicenseInfo_v2(
                self.id,
                &raw mut info
            ))?;
        }
        Ok(info.into())
    }

    /// Returns the frame-rate limit set for this instance.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if the frame-rate limiter is disabled for this vGPU type, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn frame_rate_limit(&self) -> Result<u32> {
        let mut limit = 0;
        // SAFETY: NVML validates the instance ID, and `limit` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFrameRateLimit(
                self.id,
                &raw mut limit
            ))?;
        }
        Ok(limit)
    }

    /// Returns the current ECC mode of this instance.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if the vGPU does not support ECC mode, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn ecc_mode(&self) -> Result<EnableState> {
        let mut mode = sys::nvmlEnableState_t::NVML_FEATURE_DISABLED;
        // SAFETY: NVML validates the instance ID, and `mode` is a writable local enum.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEccMode(self.id, &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Returns the encoder capacity of this instance, as a percentage from 0 to 100 of the maximum.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or encoder query, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_capacity(&self) -> Result<u32> {
        let mut capacity = 0;
        // SAFETY: NVML validates the instance ID, and `capacity` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEncoderCapacity(
                self.id,
                &raw mut capacity
            ))?;
        }
        Ok(capacity)
    }

    /// Returns the current encoder statistics of this instance.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or query arguments, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_stats(&self) -> Result<EncoderStats> {
        let mut session_count = 0;
        let mut average_fps = 0;
        let mut average_latency_us = 0;
        // SAFETY: NVML validates the instance ID, and the three outputs are writable local integers.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEncoderStats(
                self.id,
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

    /// Returns the active encoder sessions on this instance.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal session buffer is too small, if NVML rejects the handle or reports an invalid session count, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn encoder_sessions(&self) -> Result<Vec<EncoderSessionInfo>> {
        let mut count = 0;
        // SAFETY: NVML validates the instance ID, and a null array asks it only for the count, written to the local `count`.
        let status = unsafe {
            sys::nvmlVgpuInstanceGetEncoderSessions(self.id, &raw mut count, ptr::null_mut())
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
            vec![sys::nvmlEncoderSessionInfo_t::default(); to_usize(count, "session count")?];
        // SAFETY: NVML validates the instance ID, and `sessions` holds the `count` writable entries passed as its length.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetEncoderSessions(
                self.id,
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }

        sessions.truncate(to_usize(count, "session count")?);
        Ok(sessions.into_iter().map(Into::into).collect())
    }

    /// Returns the statistics of the active frame buffer capture sessions on this instance.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fbc_stats(&self) -> Result<FbcStats> {
        let mut stats = sys::nvmlFBCStats_t::default();
        // SAFETY: NVML validates the instance ID, and `stats` is a writable local struct.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFBCStats(self.id, &raw mut stats))?;
        }
        Ok(stats.into())
    }

    /// Returns the active frame buffer capture sessions on this instance.
    ///
    /// A session's resolution, frame rate, and latency are zero until it captures a new frame.
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal session buffer is too small, if NVML rejects the handle or reports an invalid session count, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn fbc_sessions(&self) -> Result<Vec<FbcSessionInfo>> {
        let mut count = 0;
        let status =
            // SAFETY: NVML validates the instance ID, and a null array asks it only for the count, written to the local `count`.
            unsafe { sys::nvmlVgpuInstanceGetFBCSessions(self.id, &raw mut count, ptr::null_mut()) };
        if status == sys::nvmlReturn_t::NVML_SUCCESS && count == 0 {
            return Ok(Vec::new());
        }

        if status != sys::nvmlReturn_t::NVML_SUCCESS
            && status != sys::nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            return Err(status.into());
        }

        let mut sessions =
            vec![sys::nvmlFBCSessionInfo_t::default(); to_usize(count, "session count")?];
        // SAFETY: NVML validates the instance ID, and `sessions` holds the `count` writable entries passed as its length.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetFBCSessions(
                self.id,
                &raw mut count,
                sessions.as_mut_ptr(),
            ))?;
        }

        sessions.truncate(to_usize(count, "session count")?);
        Ok(sessions.into_iter().map(Into::into).collect())
    }

    /// Returns the ID of the GPU instance backing this instance.
    ///
    /// The ID is valid only for MIG-backed instances; NVML reports an invalid ID otherwise.
    ///
    /// Requires a Kepler or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or query arguments, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gpu_instance_id(&self) -> Result<u32> {
        let mut id = 0;
        // SAFETY: NVML validates the instance ID, and `id` is a writable local integer.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetGpuInstanceId(self.id, &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns the PCI ID of the GPU as seen inside the VM.
    ///
    /// The ID is `00000000:00:00.0` until the VM has an NVIDIA driver installed.
    ///
    /// # Errors
    ///
    /// Returns an error if the guest driver is not running, if the internal PCI ID buffer is too small, if NVML rejects the handle or output argument, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gpu_pci_id(&self) -> Result<String> {
        let mut buffer = [0; sys::NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE as usize];
        let mut length = to_u32(buffer.len(), "NVML buffer length")?;
        // SAFETY: NVML validates the instance ID, and `buffer` holds the `length` writable bytes passed as its capacity.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetGpuPciId(
                self.id,
                buffer.as_mut_ptr(),
                &raw mut length,
            ))?;
        }
        Ok(string_from_c_chars(&buffer))
    }

    /// Returns the runtime state size of this instance.
    ///
    /// The size is the maximum in-memory data the instance uses, excluding its framebuffer.
    ///
    /// Requires a Maxwell or newer GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the versioned request, if the vGPU instance or query arguments are invalid, if the handle does not match an active vGPU instance, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn runtime_state(&self) -> Result<VgpuRuntimeState> {
        let mut state = sys::nvmlVgpuRuntimeState_t {
            version: struct_version::<sys::nvmlVgpuRuntimeState_t>(1),
            ..Default::default()
        };
        // SAFETY: NVML validates the instance ID, and `state` is a writable local struct with its version set.
        unsafe {
            try_ffi!(sys::nvmlVgpuInstanceGetRuntimeStateSize(
                self.id,
                &raw mut state
            ))?;
        }
        Ok(state.into())
    }

    /// Returns the metadata of this running instance, including the guest and host driver versions and opaque internal state.
    ///
    /// Fields that come from the guest VM may not be available yet; [`VgpuGuestInfoState`](crate::types::VgpuGuestInfoState) in the result says whether they are.
    /// A VM manager can save the metadata and pass it to [`Library::vgpu_compatibility`] before creating a vGPU for the same VM.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal metadata buffer is too small after retrying, if NVML rejects the handle or metadata arguments, if the handle does not match an active vGPU instance, or if NVML reports an unexpected failure.
    pub fn metadata(&self) -> Result<VgpuMetadata> {
        let raw = query_sized_raw(|size, buffer| {
            // SAFETY: NVML validates the instance ID, and `query_sized_raw` passes a writable size with a null buffer or one of `*size` writable bytes.
            unsafe { sys::nvmlVgpuInstanceGetMetadata(self.id, buffer.cast(), size) }
        })?;
        VgpuMetadata::decode_raw(&raw)
    }

    /// Checks this vGPU's metadata against the supplied physical GPU metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::metadata`] or [`Library::vgpu_compatibility`] fails.
    pub fn compatibility_with(&self, pgpu_metadata: &PgpuMetadata) -> Result<VgpuCompatibility> {
        let vgpu_metadata = self.metadata()?;
        self.library
            .vgpu_compatibility(&vgpu_metadata, pgpu_metadata)
    }
}
