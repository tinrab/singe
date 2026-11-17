use std::fmt::{self, Display, Formatter};

use bitflags::{bitflags, parser};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cupti_sys as sys;

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// CUPTI device attribute named by an [`crate::activity::ActivityDeviceAttribute`] record.
    pub enum DeviceAttribute: sys::CUpti_DeviceAttribute {
        /// Number of event IDs for a device.
        ///
        /// Returns a `u32` value.
        MaxEventId = CUPTI_DEVICE_ATTR_MAX_EVENT_ID,
        /// Number of event domain IDs for a device.
        ///
        /// Returns a `u32` value.
        MaxEventDomainId = CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID,
        /// Get global memory bandwidth in Kbytes/sec.
        ///
        /// Returns a `u64` value.
        GlobalMemoryBandwidth = CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH,
        /// Get theoretical maximum number of instructions per cycle.
        ///
        /// Returns a `u32` value.
        InstructionPerCycle = CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE,
        /// Get theoretical maximum number of single precision instructions that can be executed per second.
        ///
        /// Returns a `u64` value.
        InstructionThroughputSp = CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION,
        /// Get number of frame buffers for device.
        ///
        /// Returns a `u64` value.
        MaxFrameBuffers = CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS,
        /// Get PCIE link rate in Mega bits/sec for device.
        ///
        /// Returns a `u64` value, or 0 when the bus type is not `PCIe`.
        PcieLinkRate = CUPTI_DEVICE_ATTR_PCIE_LINK_RATE,
        /// Get PCIE link width for device.
        ///
        /// Returns a `u64` value, or 0 when the bus type is not `PCIe`.
        PcieLinkWidth = CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH,
        /// Get PCIE generation for device.
        ///
        /// Returns a `u64` value, or 0 when the bus type is not `PCIe`.
        PcieGeneration = CUPTI_DEVICE_ATTR_PCIE_GEN,
        /// Get the class for the device.
        ///
        /// Returns the device class.
        DeviceClass = CUPTI_DEVICE_ATTR_DEVICE_CLASS,
        /// Get the peak single precision flop per cycle.
        ///
        /// Returns a `u64` value.
        FlopSpPerCycle = CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE,
        /// Get the peak double precision flop per cycle.
        ///
        /// Returns a `u64` value.
        FlopDpPerCycle = CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE,
        /// Get number of L2 units.
        ///
        /// Returns a `u64` value.
        MaxL2Units = CUPTI_DEVICE_ATTR_MAX_L2_UNITS,
        /// Get the maximum shared memory for the `CU_FUNC_CACHE_PREFER_SHARED` preference.
        ///
        /// Returns a `u64` value.
        MaxSharedMemoryCacheConfigPreferShared = CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED,
        /// Get the maximum shared memory for the `CU_FUNC_CACHE_PREFER_L1` preference.
        ///
        /// Returns a `u64` value.
        MaxSharedMemoryCacheConfigPreferL1 = CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1,
        /// Get the maximum shared memory for the `CU_FUNC_CACHE_PREFER_EQUAL` preference.
        ///
        /// Returns a `u64` value.
        MaxSharedMemoryCacheConfigPreferEqual = CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL,
        /// Get the peak half precision flop per cycle.
        ///
        /// Returns a `u64` value.
        FlopHpPerCycle = CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE,
        /// Check if Nvlink is connected to device.
        ///
        /// Returns 1 when at least one `NVLink` is connected to the device, or 0 otherwise.
        NvLinkPresent = CUPTI_DEVICE_ATTR_NVLINK_PRESENT,
        /// Check if Nvlink is present between GPU and CPU.
        ///
        /// Returns `NVLink` bandwidth in bytes per second, or 0 when `NVLink` is absent.
        GpuCentralProcessingUnitNvLinkBandwidth = CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW,
        /// Check if `NVSwitch` is present in the underlying topology.
        ///
        /// Returns 1 when present, or 0 otherwise.
        NvSwitchPresent = CUPTI_DEVICE_ATTR_NVSWITCH_PRESENT,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_DEVICE_ATTR_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Virtualization mode in which a CUDA device is running.
    pub enum DeviceVirtualizationMode: sys::CUpti_DeviceVirtualizationMode {
        /// No virtualization mode is associated with the device i.e.
        ///
        /// it’s a baremetal GPU.
        None = CUPTI_DEVICE_VIRTUALIZATION_MODE_NONE,
        /// The device is associated with the pass-through GPU.
        ///
        /// In this mode, an entire physical GPU is directly assigned to one virtual machine (VM).
        PassThrough = CUPTI_DEVICE_VIRTUALIZATION_MODE_PASS_THROUGH,
        /// The device is associated with the virtual GPU (vGPU).
        ///
        /// In this mode multiple virtual machines (VMs) have simultaneous, direct access to a single physical GPU.
        VirtualGpu = CUPTI_DEVICE_VIRTUALIZATION_MODE_VIRTUAL_GPU,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_DEVICE_VIRTUALIZATION_MODE_FORCE_INT,
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// Device type used in topology activity records.
    pub enum DeviceType: sys::CUpti_DevType {
        /// Invalid value decoded from CUPTI.
        Invalid = CUPTI_DEV_TYPE_INVALID,
        /// The device type is GPU.
        Gpu = CUPTI_DEV_TYPE_GPU,
        /// The device type is `NVLink` processing unit in CPU.
        Npu = CUPTI_DEV_TYPE_NPU,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_DEV_TYPE_FORCE_INT,
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    /// Capability flags for a topology link.
    pub struct LinkFlags: u32 {
        /// Is peer to peer access supported by this link.
        const PEER_ACCESS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_PEER_ACCESS.0;
        /// Is system memory access supported by this link.
        const SYSTEM_MEMORY_ACCESS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_SYSMEM_ACCESS.0;
        /// Is peer atomic access supported by this link.
        const PEER_ATOMICS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_PEER_ATOMICS.0;
        /// Is system memory atomic access supported by this link.
        const SYSTEM_MEMORY_ATOMICS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_SYSMEM_ATOMICS.0;
    }
}

impl Display for LinkFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    /// `PCIe` topology record device type.
    pub enum PcieDeviceType: sys::CUpti_PcieDeviceType {
        /// PCIE GPU record.
        Gpu = CUPTI_PCIE_DEVICE_TYPE_GPU,
        /// PCIE Bridge record.
        Bridge = CUPTI_PCIE_DEVICE_TYPE_BRIDGE,
        /// Reserved integer sentinel from CUPTI.
        ForceInt = CUPTI_PCIE_DEVICE_TYPE_FORCE_INT,
    }
}

impl_enum_display!(DeviceAttribute, {
    Self::MaxEventId => "CUPTI_DEVICE_ATTR_MAX_EVENT_ID",
    Self::MaxEventDomainId => "CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID",
    Self::GlobalMemoryBandwidth => "CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH",
    Self::InstructionPerCycle => "CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE",
    Self::InstructionThroughputSp => "CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION",
    Self::MaxFrameBuffers => "CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS",
    Self::PcieLinkRate => "CUPTI_DEVICE_ATTR_PCIE_LINK_RATE",
    Self::PcieLinkWidth => "CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH",
    Self::PcieGeneration => "CUPTI_DEVICE_ATTR_PCIE_GEN",
    Self::DeviceClass => "CUPTI_DEVICE_ATTR_DEVICE_CLASS",
    Self::FlopSpPerCycle => "CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE",
    Self::FlopDpPerCycle => "CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE",
    Self::MaxL2Units => "CUPTI_DEVICE_ATTR_MAX_L2_UNITS",
    Self::MaxSharedMemoryCacheConfigPreferShared => "CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED",
    Self::MaxSharedMemoryCacheConfigPreferL1 => "CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1",
    Self::MaxSharedMemoryCacheConfigPreferEqual => "CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL",
    Self::FlopHpPerCycle => "CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE",
    Self::NvLinkPresent => "CUPTI_DEVICE_ATTR_NVLINK_PRESENT",
    Self::GpuCentralProcessingUnitNvLinkBandwidth => "CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW",
    Self::NvSwitchPresent => "CUPTI_DEVICE_ATTR_NVSWITCH_PRESENT",
    Self::ForceInt => "CUPTI_DEVICE_ATTR_FORCE_INT",
});

impl_enum_display!(DeviceVirtualizationMode, {
    Self::None => "CUPTI_DEVICE_VIRTUALIZATION_MODE_NONE",
    Self::PassThrough => "CUPTI_DEVICE_VIRTUALIZATION_MODE_PASS_THROUGH",
    Self::VirtualGpu => "CUPTI_DEVICE_VIRTUALIZATION_MODE_VIRTUAL_GPU",
    Self::ForceInt => "CUPTI_DEVICE_VIRTUALIZATION_MODE_FORCE_INT",
});

impl_enum_display!(DeviceType, {
    Self::Invalid => "CUPTI_DEV_TYPE_INVALID",
    Self::Gpu => "CUPTI_DEV_TYPE_GPU",
    Self::Npu => "CUPTI_DEV_TYPE_NPU",
    Self::ForceInt => "CUPTI_DEV_TYPE_FORCE_INT",
});

impl_enum_display!(PcieDeviceType, {
    Self::Gpu => "CUPTI_PCIE_DEVICE_TYPE_GPU",
    Self::Bridge => "CUPTI_PCIE_DEVICE_TYPE_BRIDGE",
    Self::ForceInt => "CUPTI_PCIE_DEVICE_TYPE_FORCE_INT",
});
