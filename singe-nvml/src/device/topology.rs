use std::{ffi::c_ulong, num::NonZero, ptr, sync::LazyLock, thread::available_parallelism};

use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{AffinityScope, P2pCapabilityIndex, P2pStatus, TopologyLevel},
    utility::{to_u32, to_usize},
};

impl Device {
    /// Returns bitmasks with the ideal CPU affinity for the device.
    /// For example, if processors 0, 1, 32, and 33 are ideal for the device and two mask words are requested, then `result[0] = 0x3` and `result[1] = 0x3`.
    /// This is equivalent to calling [`Device::cpu_affinity_within_scope`] with [`AffinityScope::Node`].
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output buffer, if the device does not support CPU affinity reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn cpu_affinity(&self) -> Result<Vec<u64>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `device_ulong_bitmask_list` passes an array of `set_size` writable words.
        device_ulong_bitmask_list(|set_size, set| unsafe {
            sys::nvmlDeviceGetCpuAffinity(self.raw, set_size, set)
        })
    }

    /// Returns bitmasks with the ideal CPU affinity within a node or socket for the device.
    /// For example, if processors 0, 1, 32, and 33 are ideal for the device and two mask words are requested, then `result[0] = 0x3` and `result[1] = 0x3`.
    ///
    /// If the requested scope is not applicable to the target topology, NVML falls back to reporting CPU affinity for the device's immediate non-I/O ancestor.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, scope, or output buffer, if the device does not support scoped CPU affinity reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn cpu_affinity_within_scope(&self, scope: AffinityScope) -> Result<Vec<u64>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `device_ulong_bitmask_list` passes an array of `set_size` writable words.
        device_ulong_bitmask_list(|set_size, set| unsafe {
            sys::nvmlDeviceGetCpuAffinityWithinScope(self.raw, set_size, set, scope.into())
        })
    }

    /// Sets the ideal affinity for the calling thread and device using the guidelines given in [`Device::cpu_affinity`].
    /// Currently supports up to 1024 processors.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, if the device does not support CPU affinity binding, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn set_cpu_affinity(&self) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the handle is the only argument.
        unsafe { try_ffi!(sys::nvmlDeviceSetCpuAffinity(self.raw)) }
    }

    /// Clears all affinity bindings of the calling thread.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn clear_cpu_affinity(&self) -> Result<()> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and the handle is the only argument.
        unsafe { try_ffi!(sys::nvmlDeviceClearCpuAffinity(self.raw)) }
    }

    /// Returns bitmasks with the ideal memory affinity within a node or socket for this device.
    /// For example, if NUMA nodes 0 and 1 are ideal within the socket and one mask word is requested, `result[0] = 0x3`.
    ///
    /// If the requested scope is not applicable to the target topology, NVML falls back to reporting memory affinity for the device's immediate non-I/O ancestor.
    ///
    /// For Kepler or newer fully supported devices.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, scope, or output buffer, if the device does not support memory affinity reporting, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn memory_affinity(&self, scope: AffinityScope) -> Result<Vec<u64>> {
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `device_ulong_bitmask_list` passes an array of `set_size` writable words.
        device_ulong_bitmask_list(|set_size, set| unsafe {
            sys::nvmlDeviceGetMemoryAffinity(self.raw, set_size, set, scope.into())
        })
    }

    /// Returns the NUMA node of the given GPU device.
    /// This only applies to platforms where the GPUs are NUMA nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, or if the current platform does not support GPU NUMA-node reporting.
    pub fn numa_node_id(&self) -> Result<u32> {
        let mut node = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `node` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetNumaNodeId(self.raw, &raw mut node))?;
        }
        Ok(node)
    }

    /// Returns the common ancestor for two devices.
    ///
    /// For all products.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects either handle or the output, if topology discovery is not supported on this device or OS, or if NVML fails during topology discovery.
    pub fn topology_common_ancestor(&self, other: &Self) -> Result<TopologyLevel> {
        let mut level = sys::nvmlGpuTopologyLevel_t::NVML_TOPOLOGY_SYSTEM;
        // SAFETY: both handles are valid while their libraries keep NVML initialized, and `level` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTopologyCommonAncestor(
                self.raw,
                other.raw,
                &raw mut level,
            ))?;
        }
        Ok(level.into())
    }

    /// Returns the set of GPUs nearest to this device at a specific interconnectivity level.
    /// Supported on Linux only.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle, topology level, count, or output buffer, if topology discovery is not supported on this device or OS, or if NVML fails during topology discovery.
    pub fn topology_nearest_gpus(&self, level: TopologyLevel) -> Result<Vec<Self>> {
        let mut count = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and a null array with a writable `count` asks NVML only for the device count.
        let status = unsafe {
            sys::nvmlDeviceGetTopologyNearestGpus(
                self.raw,
                level.into(),
                &raw mut count,
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

        let mut devices = vec![ptr::null_mut(); to_usize(count, "nearest GPU count")?];
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `devices` holds `count` writable entries.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetTopologyNearestGpus(
                self.raw,
                level.into(),
                &raw mut count,
                devices.as_mut_ptr(),
            ))?;
        }

        devices.truncate(to_usize(count, "nearest GPU count")?);
        Ok(devices
            .into_iter()
            // SAFETY: NVML wrote these handles under `self.library`'s initialization, and the vector was truncated to the count it reported.
            .map(|handle| unsafe { Self::from_raw(handle, &self.library) })
            .collect())
    }

    /// Returns the status for a P2P capability between this device and `other`.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects either device handle, the capability, or output, or if NVML reports an unexpected failure.
    pub fn p2p_status(&self, other: &Self, capability: P2pCapabilityIndex) -> Result<P2pStatus> {
        let mut status = sys::nvmlGpuP2PStatus_t::NVML_P2P_STATUS_UNKNOWN;
        // SAFETY: both handles are valid while their libraries keep NVML initialized, and `status` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetP2PStatus(
                self.raw,
                other.raw,
                capability.into(),
                &raw mut status,
            ))?;
        }
        Ok(status.into())
    }

    /// Returns whether this device and `other` are on the same physical board.
    ///
    /// For all fully supported products.
    ///
    /// # Errors
    ///
    /// Returns an error if either device is inaccessible, if NVML rejects either handle or output, if the check is unsupported, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn on_same_board(&self, other: &Self) -> Result<bool> {
        let mut on_same_board = 0;
        // SAFETY: both handles are valid while their libraries keep NVML initialized, and `on_same_board` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceOnSameBoard(
                self.raw,
                other.raw,
                &raw mut on_same_board,
            ))?;
        }
        Ok(on_same_board != 0)
    }
}

static ULONG_BITMASK_COUNT: LazyLock<usize> = LazyLock::new(|| {
    let cpu_count = available_parallelism().map_or(64, NonZero::get);
    let bits_per_ulong =
        usize::try_from(c_ulong::BITS).expect("the bit width of `c_ulong` fits `usize`");
    cpu_count.div_ceil(bits_per_ulong)
});

fn ulong_bitmask_count() -> usize {
    *ULONG_BITMASK_COUNT
}

fn device_ulong_bitmask_list(
    function: impl Fn(u32, *mut c_ulong) -> sys::nvmlReturn_t,
) -> Result<Vec<u64>> {
    let count = ulong_bitmask_count();
    let mut values = vec![0 as c_ulong; count];
    try_ffi!(function(
        to_u32(count, "NVML bitmask count")?,
        values.as_mut_ptr()
    ))?;
    Ok(values.into_iter().collect())
}
