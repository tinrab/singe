use std::ffi::CStr;

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::{Error, Result},
    try_ffi,
    types::{
        Bar1MemoryInfo, ConfComputeGpuAttestationReport, ConfComputeGpuCertificate,
        ConfComputeMemSizeInfo, CurrentPending, DramEncryptionInfo, MemoryInfo,
    },
    utility::struct_version,
};

// NVML encodes an unlimited hard limit as `NVML_DEVICE_MEMORY_LIMIT_MAX`, which is -1 converted to `unsigned long long`.
const UNLIMITED_HARD_LIMIT: u64 = u64::MAX;

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

impl Device {
    /// Returns total, reserved, free, and used physical GPU memory in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, NVML is uninitialized, the native record version is unsupported, or NVML cannot report memory usage.
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let mut memory = sys::nvmlMemory_v2_t {
            version: struct_version::<sys::nvmlMemory_v2_t>(2),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `memory` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemoryInfo_v2(self.raw, &raw mut memory))?;
        }
        Ok(memory.into())
    }

    /// Returns total, available, and used size of BAR1 memory.
    ///
    /// BAR1 maps framebuffer memory so the CPU or third-party `PCIe` peer devices can access it directly.
    ///
    /// In MIG mode, a device handle returns aggregate information only if the caller has appropriate privileges.
    /// Per-instance information can be queried by using specific MIG device handles.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if the device does not support BAR1 memory reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn bar1_memory_info(&self) -> Result<Bar1MemoryInfo> {
        let mut memory = sys::nvmlBAR1Memory_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `memory` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBAR1MemoryInfo(self.raw, &raw mut memory))?;
        }
        Ok(memory.into())
    }

    /// Returns the device's memory bus width.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support memory-bus-width reporting, or if NVML has not been initialized.
    pub fn memory_bus_width(&self) -> Result<u32> {
        let mut width = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `width` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemoryBusWidth(self.raw, &raw mut width))?;
        }
        Ok(width)
    }

    /// Returns the current and pending DRAM Encryption modes for the device.
    ///
    /// For Blackwell or newer fully supported devices.
    /// Only applicable to devices that support DRAM Encryption.
    /// Requires [`InforomObject::Den`](crate::types::InforomObject::Den) version 1.0 or higher.
    ///
    /// Changing DRAM Encryption modes requires a reboot.
    /// The "pending" DRAM Encryption mode refers to the target mode following the next reboot.
    ///
    /// See [`EnableState`](crate::types::EnableState) for details on allowed modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML rejects the handle or outputs, if the device does not support DRAM encryption, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn dram_encryption_mode(&self) -> Result<CurrentPending<DramEncryptionInfo>> {
        let mut current = sys::nvmlDramEncryptionInfo_t {
            version: struct_version::<sys::nvmlDramEncryptionInfo_t>(1),
            ..Default::default()
        };
        let mut pending = sys::nvmlDramEncryptionInfo_t {
            version: struct_version::<sys::nvmlDramEncryptionInfo_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `current` and `pending` are writable locals whose versions match their layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetDramEncryptionMode(
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

    /// Sets the DRAM encryption mode for the device.
    ///
    /// For Kepler or newer fully supported devices.
    /// Only applicable to devices that support DRAM Encryption.
    /// Requires [`InforomObject::Den`](crate::types::InforomObject::Den) version 1.0 or higher.
    /// Requires root/admin permissions.
    ///
    /// The DRAM Encryption mode determines whether the GPU enables its DRAM Encryption support.
    ///
    /// This operation takes effect after the next reboot.
    ///
    /// See [`EnableState`](crate::types::EnableState) for details on available modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML version does not support the request layout, if the device is inaccessible, if NVML rejects the handle or requested mode, if the device does not support DRAM encryption, if the current process lacks permission, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_dram_encryption_mode(&self, dram_encryption: DramEncryptionInfo) -> Result<()> {
        let dram_encryption = sys::nvmlDramEncryptionInfo_t {
            version: struct_version::<sys::nvmlDramEncryptionInfo_t>(1),
            encryptionState: dram_encryption.encryption_state.into(),
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `dram_encryption` is a local whose version matches its layout and that NVML only reads.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetDramEncryptionMode(
                self.raw,
                &raw const dram_encryption,
            ))
        }
    }

    /// Returns Confidential Computing protected and unprotected memory sizes.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not support Confidential Computing memory-size reporting, or if NVML has not been initialized.
    pub fn conf_compute_mem_size_info(&self) -> Result<ConfComputeMemSizeInfo> {
        let mut info = sys::nvmlConfComputeMemSizeInfo_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `info` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetConfComputeMemSizeInfo(
                self.raw,
                &raw mut info
            ))?;
        }
        Ok(info.into())
    }

    /// Returns Confidential Computing protected memory usage.
    ///
    /// NVML reports no reserved amount for this query, so [`MemoryInfo::reserved`] is zero.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not support Confidential Computing protected-memory reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_protected_memory_usage(&self) -> Result<MemoryInfo> {
        let mut memory = sys::nvmlMemory_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `memory` is a writable local of the version 1 layout NVML writes.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetConfComputeProtectedMemoryUsage(
                self.raw,
                &raw mut memory,
            ))?;
        }
        Ok(MemoryInfo {
            total: memory.total,
            reserved: 0,
            free: memory.free,
            used: memory.used,
        })
    }

    /// Returns Confidential Computing GPU certificate details.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not support Confidential Computing certificate reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_gpu_certificate(&self) -> Result<ConfComputeGpuCertificate> {
        let mut certificate = sys::nvmlConfComputeGpuCertificate_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `certificate` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetConfComputeGpuCertificate(
                self.raw,
                &raw mut certificate,
            ))?;
        }
        Ok(certificate.into())
    }

    /// Returns Confidential Computing GPU attestation report.
    ///
    /// For Ampere or newer fully supported devices.
    /// Supported on Linux, Windows TCC.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the query, if the device does not support Confidential Computing attestation reports, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn conf_compute_gpu_attestation_report(&self) -> Result<ConfComputeGpuAttestationReport> {
        let mut report = sys::nvmlConfComputeGpuAttestationReport_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `report` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetConfComputeGpuAttestationReport(
                self.raw,
                &raw mut report,
            ))?;
        }
        Ok(report.into())
    }

    /// Returns the memory limits and memory used for the cgroup partition `cgroup`.
    ///
    /// For Linux only.
    /// `cgroup` is the full path to the sysfs cgroup file name.
    ///
    /// # Errors
    ///
    /// Returns an error if the installed NVML lacks this entry point, the cgroup namespace is invalid or unavailable, or memory-limit reporting is unsupported.
    /// Also propagates NVML query failures.
    pub fn memory_limits(&self, cgroup: &CStr) -> Result<MemoryLimits> {
        let mut limits = sys::nvmlGetMemoryLimits_v1_t {
            nameSpace: cgroup.as_ptr(),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `limits` is a writable local, and its `nameSpace` points into `cgroup`, which outlives the call.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMemoryLimits_v1(
                self.as_raw(),
                &raw mut limits
            ))?;
        }
        Ok(MemoryLimits {
            soft_limit: limits.softLimit,
            hard_limit: if limits.hardLimit == UNLIMITED_HARD_LIMIT {
                MemoryLimit::Unlimited
            } else {
                MemoryLimit::Bytes(limits.hardLimit)
            },
            current_used: limits.currentUsed,
        })
    }

    /// Sets the memory limits of the device for a cgroup partition.
    ///
    /// For Linux only.
    /// Requires root/admin permissions.
    /// The soft limit is the guaranteed amount; the hard limit is the maximum allocation size.
    ///
    /// # Errors
    ///
    /// Returns an error if `soft_limit` exceeds the hard limit.
    /// Also returns an error if the entry point is unavailable or NVML rejects the cgroup, limits, device, or caller permissions.
    pub fn set_memory_limits(
        &self,
        cgroup: &CStr,
        soft_limit: u64,
        hard_limit: MemoryLimit,
    ) -> Result<()> {
        let hard_limit = match hard_limit {
            MemoryLimit::Bytes(value) => value,
            MemoryLimit::Unlimited => UNLIMITED_HARD_LIMIT,
        };
        if soft_limit > hard_limit {
            return Err(Error::out_of_range("soft_limit"));
        }

        let mut limits = sys::nvmlSetMemoryLimits_v1_t {
            nameSpace: cgroup.as_ptr(),
            softLimit: soft_limit,
            hardLimit: hard_limit,
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, `limits` is a local that NVML only reads, and its `nameSpace` points into `cgroup`, which outlives the call.
        unsafe {
            try_ffi!(sys::nvmlDeviceSetMemoryLimits_v1(
                self.as_raw(),
                &raw mut limits
            ))
        }
    }

    /// Clears limits by setting the soft limit to zero and hard limit to unlimited.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::set_memory_limits`], including an unavailable entry point, invalid cgroup namespace, or insufficient permissions.
    pub fn clear_memory_limits(&self, cgroup: &CStr) -> Result<()> {
        self.set_memory_limits(cgroup, 0, MemoryLimit::Unlimited)
    }
}
