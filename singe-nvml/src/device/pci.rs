use singe_nvml_sys as sys;

use crate::{
    device::Device,
    error::Result,
    try_ffi,
    types::{BridgeChipHierarchy, BusType, PciInfo, PciInfoExt, PcieUtilCounter},
    utility::struct_version,
};

impl Device {
    /// Returns the PCI attributes of this device.
    ///
    /// For all products.
    ///
    /// Returns the PCI information reported by NVML.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or PCI info output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn pci_info(&self) -> Result<PciInfo> {
        let mut pci = sys::nvmlPciInfo_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `pci` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPciInfo_v3(self.raw, &raw mut pci))?;
        }
        Ok(pci.into())
    }

    /// Returns the extended PCI attributes of this device.
    ///
    /// For all products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or PCI info output, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn pci_info_ext(&self) -> Result<PciInfoExt> {
        let mut pci = sys::nvmlPciInfoExt_t {
            version: struct_version::<sys::nvmlPciInfoExt_t>(1),
            ..Default::default()
        };
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `pci` is a writable local whose version matches its layout.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPciInfoExt(self.raw, &raw mut pci))?;
        }
        Ok(pci.into())
    }

    /// Returns bridge chip information for all the bridge chips on the board.
    ///
    /// For all fully supported products.
    /// Only applicable to multi-GPU products.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if bridge-chip reporting is not supported for the device, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn bridge_chip_hierarchy(&self) -> Result<BridgeChipHierarchy> {
        let mut hierarchy = sys::nvmlBridgeChipHierarchy_t::default();
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `hierarchy` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBridgeChipInfo(
                self.raw,
                &raw mut hierarchy
            ))?;
        }
        Ok(hierarchy.into())
    }

    /// Returns the current `PCIe` link generation.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if `PCIe` link information is unavailable, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn current_pcie_link_generation(&self) -> Result<u32> {
        let mut generation = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `generation` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrPcieLinkGeneration(
                self.raw,
                &raw mut generation,
            ))?;
        }
        Ok(generation)
    }

    /// Returns the current `PCIe` link width.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if `PCIe` link information is unavailable, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn current_pcie_link_width(&self) -> Result<u32> {
        let mut width = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `width` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetCurrPcieLinkWidth(
                self.raw,
                &raw mut width
            ))?;
        }
        Ok(width)
    }

    /// Returns the maximum `PCIe` link generation possible with this device and system.
    ///
    /// For example, a generation 2 `PCIe` device attached to a generation 1 `PCIe` bus reports generation 1.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if `PCIe` link information is unavailable, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_pcie_link_generation(&self) -> Result<u32> {
        let mut generation = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `generation` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxPcieLinkGeneration(
                self.raw,
                &raw mut generation,
            ))?;
        }
        Ok(generation)
    }

    /// Returns the maximum `PCIe` link generation supported by this device.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the query, if `PCIe` link information is unavailable, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn gpu_max_pcie_link_generation(&self) -> Result<u32> {
        let mut generation = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `generation` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetGpuMaxPcieLinkGeneration(
                self.raw,
                &raw mut generation,
            ))?;
        }
        Ok(generation)
    }

    /// Returns the maximum `PCIe` link width possible with this device and system.
    ///
    /// For example, a device with a 16x `PCIe` bus width attached to an 8x `PCIe` system bus reports a maximum link width of 8.
    ///
    /// For Fermi or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if `PCIe` link information is unavailable, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn max_pcie_link_width(&self) -> Result<u32> {
        let mut width = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `width` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetMaxPcieLinkWidth(self.raw, &raw mut width))?;
        }
        Ok(width)
    }

    /// Returns `PCIe` utilization information.
    /// Queries a byte counter over a 20 ms interval to report `PCIe` throughput.
    ///
    /// For Maxwell or newer fully supported devices.
    ///
    /// Not supported in virtual machines running virtual GPU (vGPU).
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle, counter, or output, if the device does not support `PCIe` utilization queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn pcie_throughput(&self, counter: PcieUtilCounter) -> Result<u32> {
        let mut throughput = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `throughput` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieThroughput(
                self.raw,
                counter.into(),
                &raw mut throughput,
            ))?;
        }
        Ok(throughput)
    }

    /// Returns the `PCIe` replay counter.
    ///
    /// For Kepler or newer fully supported devices.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support replay-counter queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn pcie_replay_counter(&self) -> Result<u32> {
        let mut counter = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `counter` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieReplayCounter(
                self.raw,
                &raw mut counter
            ))?;
        }
        Ok(counter)
    }

    /// Returns the device's `PCIe` Max Link speed in MB/s.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is inaccessible, if NVML rejects the handle or output, if the device does not support this query, or if NVML has not been initialized.
    pub fn pcie_link_max_speed(&self) -> Result<u32> {
        let mut speed = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `speed` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieLinkMaxSpeed(self.raw, &raw mut speed))?;
        }
        Ok(speed)
    }

    /// Returns the device's `PCIe` Link speed in Mbps.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML rejects the handle or output, if the device does not support `PCIe` speed queries, if NVML has not been initialized, or if NVML reports an unexpected failure.
    pub fn pcie_speed(&self) -> Result<u32> {
        let mut speed = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `speed` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetPcieSpeed(self.raw, &raw mut speed))?;
        }
        Ok(speed)
    }

    /// Returns the GPU bus type, such as `PCIe` or PCI.
    ///
    /// # Errors
    ///
    /// Returns an error if NVML has not been initialized, if NVML rejects the handle or output, or if NVML reports an unexpected failure.
    pub fn bus_type(&self) -> Result<BusType> {
        let mut bus_type = 0;
        // SAFETY: `self.raw` is valid while `self.library` keeps NVML initialized, and `bus_type` is a writable local.
        unsafe {
            try_ffi!(sys::nvmlDeviceGetBusType(self.raw, &raw mut bus_type))?;
        }
        Ok(BusType::from_raw(bus_type))
    }
}
