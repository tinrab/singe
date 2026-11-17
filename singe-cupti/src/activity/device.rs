use std::{
    ffi::CStr,
    fmt::{self, Debug, Display, Formatter},
    slice,
};

use singe_cuda::dim::Dim3;

use super::{
    memory::{ActivityMemoryPoolProperties, ActivityUnifiedMemoryCounterFlags},
    record::ActivityRecordPayload,
    trace::ActivityObjectId,
};
use crate::{
    error::Result,
    sys,
    types::{
        ActivityEnvironmentKind, ActivityFlags, ComputeInstanceId, CudaDevice, CudaDeviceAttribute,
        DeviceAttribute, DeviceId, DeviceType, EnvironmentClocksThrottleReasons, GpuInstanceId,
        LinkFlags, NonUniformMemoryAccessId, NpuDomainId, PcieBridgeId, PcieBusId, PcieDeviceType,
        PcieDomainId, PcieHardwareDeviceId, PcieVendorId,
    },
};

/// Whether a device attribute belongs to CUPTI or the CUDA driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityDeviceAttributeKind {
    /// CUPTI-defined device attribute.
    Cupti(DeviceAttribute),
    /// CUDA driver device attribute.
    Cuda(CudaDeviceAttribute),
}

/// Native device-attribute value bits with typed numeric accessors.
///
/// The attribute selector determines which accessor is meaningful.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivityDeviceAttributeValue {
    pub(crate) bits: u64,
}

impl Debug for ActivityDeviceAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ActivityDeviceAttributeValue")
            .field(&format_args!("{:#x}", self.bits))
            .finish()
    }
}

impl ActivityDeviceAttributeValue {
    pub(crate) fn from_raw(value: sys::CUpti_ActivityDeviceAttribute__bindgen_ty_2) -> Self {
        Self {
            // SAFETY: every field of this union is a plain number, so reading the raw 64-bit pattern is always valid.
            bits: unsafe { value.vUint64 },
        }
    }

    /// Interprets all 64 value bits as an IEEE 754 `f64`.
    ///
    /// Use this interpretation only when the selected attribute reports a double-precision value.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from_bits(self.bits)
    }

    /// Returns the low 32 bits as an unsigned integer.
    ///
    /// # Panics
    ///
    /// Panics if the internal four-byte slice has an unexpected length.
    /// Its fixed range makes this condition unreachable.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        u32::from_le_bytes(
            self.bits.to_le_bytes()[..4]
                .try_into()
                .expect("four low register bytes"),
        )
    }

    /// Returns the preserved 64-bit unsigned integer representation.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.bits
    }

    /// Interprets the low 32 bits as a signed integer without numeric conversion.
    #[must_use]
    pub fn as_i32(self) -> i32 {
        self.as_u32().cast_signed()
    }

    /// Interprets all 64 bits as a signed integer without numeric conversion.
    #[must_use]
    pub fn as_i64(self) -> i64 {
        self.bits.cast_signed()
    }
}

/// Decoded activity record for a device attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityDeviceAttribute {
    /// The flags associated with the device.
    pub flags: ActivityFlags,
    /// The ID of the device that this attribute applies to.
    pub device_id: DeviceId,
    /// Attribute selector decoded as either a CUPTI attribute or a CUDA driver attribute using the native device-attribute flag.
    pub attribute: ActivityDeviceAttributeKind,
    /// Raw attribute value.
    /// Interpret it using the selected CUPTI or CUDA driver attribute and its native value type.
    pub value: ActivityDeviceAttributeValue,
}

impl ActivityDeviceAttribute {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityDeviceAttribute) -> Result<Self> {
        let flags = ActivityFlags::from_bits_truncate(value.flags.0);
        let attribute = if flags.contains(ActivityFlags::DEVICE_ATTRIBUTE_CUDEVICE) {
            // SAFETY: the device-attribute flag checked above selects this union field.
            ActivityDeviceAttributeKind::Cuda(CudaDeviceAttribute::from(unsafe {
                value.attribute.cu
            }))
        } else {
            // SAFETY: the device-attribute flag checked above selects this union field.
            ActivityDeviceAttributeKind::Cupti(DeviceAttribute::try_from(unsafe {
                value.attribute.cupti
            })?)
        };
        Ok(Self {
            flags,
            device_id: DeviceId::from(value.deviceId),
            attribute,
            value: ActivityDeviceAttributeValue::from_raw(value.value),
        })
    }
}

/// Sampled GPU and memory clocks and PCI Express link state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironmentSpeed {
    /// Streaming-multiprocessor clock frequency in MHz.
    pub sm_clock: u32,
    /// Memory clock frequency in MHz.
    pub memory_clock: u32,
    /// PCI Express link generation.
    pub pcie_link_gen: u32,
    /// PCI Express link width.
    pub pcie_link_width: u32,
    /// Reasons the device clocks are being throttled.
    pub clocks_throttle_reasons: EnvironmentClocksThrottleReasons,
}

/// Sampled GPU power consumption and the power-management threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironmentPower {
    /// Power consumed by the GPU and associated circuitry, in milliwatts.
    pub power: u32,
    /// Power-management threshold in milliwatts.
    pub power_limit: u32,
}

/// Sampled environmental value decoded according to the environment-kind tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityEnvironmentReading {
    /// Unrecognized or unsupported environmental sample kind.
    Unknown,
    /// Clock frequencies, PCI Express link state, and clock-throttling reasons.
    Speed(ActivityEnvironmentSpeed),
    /// GPU temperature sample.
    Temperature {
        /// GPU temperature in degrees Celsius.
        gpu_temperature: u32,
    },
    /// Power-consumption and power-limit sample.
    Power(ActivityEnvironmentPower),
    /// Fan-speed sample.
    Cooling {
        /// Fan speed as a percentage of its maximum.
        fan_speed: u32,
    },
}

/// Decoded activity record for CUPTI environmental data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityEnvironment {
    /// The ID of the device.
    pub device_id: DeviceId,
    /// The timestamp when this sample was retrieved, in nanoseconds.
    /// A value of 0 indicates that timestamp information could not be collected for the marker.
    pub timestamp: u64,
    /// The kind of data reported in this record.
    pub environment_kind: ActivityEnvironmentKind,
    /// Environmental reading selected by `environment_kind`.
    pub data: ActivityEnvironmentReading,
}

impl ActivityEnvironment {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityEnvironment) -> Result<Self> {
        let environment_kind = ActivityEnvironmentKind::try_from(value.environmentKind)?;
        // SAFETY: each arm reads the union field that the matched tag selects.
        let data = unsafe {
            match environment_kind {
                ActivityEnvironmentKind::Speed => {
                    let speed = value.data.speed;
                    ActivityEnvironmentReading::Speed(ActivityEnvironmentSpeed {
                        sm_clock: speed.smClock,
                        memory_clock: speed.memoryClock,
                        pcie_link_gen: speed.pcieLinkGen,
                        pcie_link_width: speed.pcieLinkWidth,
                        clocks_throttle_reasons:
                            EnvironmentClocksThrottleReasons::from_bits_truncate(
                                speed.clocksThrottleReasons.0,
                            ),
                    })
                }
                ActivityEnvironmentKind::Temperature => ActivityEnvironmentReading::Temperature {
                    gpu_temperature: value.data.temperature.gpuTemperature,
                },
                ActivityEnvironmentKind::Power => {
                    let power = value.data.power;
                    ActivityEnvironmentReading::Power(ActivityEnvironmentPower {
                        power: power.power,
                        power_limit: power.powerLimit,
                    })
                }
                ActivityEnvironmentKind::Cooling => ActivityEnvironmentReading::Cooling {
                    fan_speed: value.data.cooling.fanSpeed,
                },
                _ => ActivityEnvironmentReading::Unknown,
            }
        };
        Ok(Self {
            device_id: DeviceId::from(value.deviceId),
            timestamp: value.timestamp,
            environment_kind,
            data,
        })
    }
}

fn uuid_from_raw(value: sys::CUuuid) -> [u8; 16] {
    value.bytes.map(|byte| byte as u8)
}

/// Decoded activity record for a device.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "CUPTI reports independent device capabilities as boolean fields"
)]
pub struct ActivityDevice<'a> {
    /// The flags associated with the device.
    pub flags: ActivityFlags,
    /// The global memory bandwidth available on the device, in `kBytes`/sec.
    pub global_memory_bandwidth: u64,
    /// The amount of global memory on the device, in bytes.
    pub global_memory_size: u64,
    /// The amount of constant memory on the device, in bytes.
    pub constant_memory_size: u32,
    /// The size of the L2 cache on the device, in bytes.
    pub l2_cache_size: u32,
    /// The number of threads per warp on the device.
    pub num_threads_per_warp: u32,
    /// The core clock rate of the device, in `kHz`.
    pub core_clock_rate: u32,
    /// Number of memory copy engines on the device.
    pub num_memory_copy_engines: u32,
    /// Number of multiprocessors on the device.
    pub num_multiprocessors: u32,
    /// The maximum "instructions per cycle" possible on each device multiprocessor.
    pub max_instructions_per_cycle: u32,
    /// Maximum number of warps that can be present on a multiprocessor at any given time.
    pub max_warps_per_multiprocessor: u32,
    /// Maximum number of blocks that can be present on a multiprocessor at any given time.
    pub max_blocks_per_multiprocessor: u32,
    /// Maximum amount of shared memory available per multiprocessor, in bytes.
    pub max_shared_memory_per_multiprocessor: u32,
    /// Maximum number of 32-bit registers available per multiprocessor.
    pub max_registers_per_multiprocessor: u32,
    /// Maximum number of registers that can be allocated to a block.
    pub max_registers_per_block: u32,
    /// Maximum amount of shared memory that can be assigned to a block, in bytes.
    pub max_shared_memory_per_block: u32,
    /// Maximum number of threads allowed in a block.
    pub max_threads_per_block: u32,
    /// Maximum thread-block extent in each dimension.
    pub max_block_dim: Dim3,
    /// Maximum grid extent in each dimension.
    pub max_grid_dim: Dim3,
    /// Compute capability for the device, major number.
    pub compute_capability_major: u32,
    /// Compute capability for the device, minor number.
    pub compute_capability_minor: u32,
    /// The device ID.
    pub device_id: DeviceId,
    /// Whether error-correcting code (ECC) is enabled for the device.
    pub ecc_enabled: bool,
    /// The device UUID.
    /// This value is the globally unique immutable alphanumeric identifier of the device.
    pub uuid: [u8; 16],
    /// Device name.
    /// This borrowed view does not transfer ownership of the native name allocation.
    /// `None` represents a null native pointer.
    pub name: Option<&'a CStr>,
    /// Whether the device is visible to CUDA under the `CUDA_VISIBLE_DEVICES` environment setting.
    pub is_cuda_visible: bool,
    /// Whether Multi-Instance GPU (MIG) mode is enabled on the device.
    pub is_mig_enabled: bool,
    /// GPU Instance id for MIG enabled devices.
    /// If mig mode is disabled value is set to `UINT32_MAX`
    pub gpu_instance_id: GpuInstanceId,
    /// Compute Instance id for MIG enabled devices.
    /// If mig mode is disabled value is set to `UINT32_MAX`
    pub compute_instance_id: ComputeInstanceId,
    /// The MIG UUID.
    /// This value is the globally unique immutable alphanumeric identifier of the device.
    pub mig_uuid: [u8; 16],
    /// Whether this GPU is a non-uniform memory access (NUMA) node.
    pub is_non_uniform_memory_access_node: bool,
    /// NUMA node identifier of GPU memory; preserves the native invalid identifier when the GPU is not a NUMA node.
    pub non_uniform_memory_access_id: NonUniformMemoryAccessId,
    /// Number of TPCs on the device.
    pub num_tpcs: u32,
}

impl ActivityDevice<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityDevice6) -> Self {
        let name = if value.name.is_null() {
            None
        } else {
            // SAFETY: the pointer was checked for null, and the library keeps the NUL-terminated string alive while this value is borrowed.
            Some(unsafe { CStr::from_ptr(value.name) })
        };
        Self {
            flags: ActivityFlags::from_bits_truncate(value.flags.0),
            global_memory_bandwidth: value.globalMemoryBandwidth,
            global_memory_size: value.globalMemorySize,
            constant_memory_size: value.constantMemorySize,
            l2_cache_size: value.l2CacheSize,
            num_threads_per_warp: value.numThreadsPerWarp,
            core_clock_rate: value.coreClockRate,
            num_memory_copy_engines: value.numMemcpyEngines,
            num_multiprocessors: value.numMultiprocessors,
            max_instructions_per_cycle: value.maxIPC,
            max_warps_per_multiprocessor: value.maxWarpsPerMultiprocessor,
            max_blocks_per_multiprocessor: value.maxBlocksPerMultiprocessor,
            max_shared_memory_per_multiprocessor: value.maxSharedMemoryPerMultiprocessor,
            max_registers_per_multiprocessor: value.maxRegistersPerMultiprocessor,
            max_registers_per_block: value.maxRegistersPerBlock,
            max_shared_memory_per_block: value.maxSharedMemoryPerBlock,
            max_threads_per_block: value.maxThreadsPerBlock,
            max_block_dim: Dim3 {
                x: value.maxBlockDimX,
                y: value.maxBlockDimY,
                z: value.maxBlockDimZ,
            },
            max_grid_dim: Dim3 {
                x: value.maxGridDimX,
                y: value.maxGridDimY,
                z: value.maxGridDimZ,
            },
            compute_capability_major: value.computeCapabilityMajor,
            compute_capability_minor: value.computeCapabilityMinor,
            device_id: DeviceId::from(value.id),
            ecc_enabled: value.eccEnabled != 0,
            uuid: uuid_from_raw(value.uuid),
            name,
            is_cuda_visible: value.isCudaVisible != 0,
            is_mig_enabled: value.isMigEnabled != 0,
            gpu_instance_id: GpuInstanceId::from(value.gpuInstanceId),
            compute_instance_id: ComputeInstanceId::from(value.computeInstanceId),
            mig_uuid: uuid_from_raw(value.migUuid),
            is_non_uniform_memory_access_node: value.isNumaNode != 0,
            non_uniform_memory_access_id: NonUniformMemoryAccessId::from(value.numaId),
            num_tpcs: value.numTpcs,
        }
    }
}

/// NPU index and domain identifier used by topology records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityNpuId {
    /// Index of the NPU endpoint.
    pub index: u32,
    /// Domain containing the NPU endpoint.
    pub domain_id: NpuDomainId,
}

/// GPU or NPU identity decoded according to a topology endpoint type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityTopologyDeviceId {
    /// Unrecognized topology endpoint type.
    Unknown,
    /// GPU UUID.
    Gpu {
        /// Native 16-byte GPU UUID.
        uuid: [u8; 16],
    },
    /// NPU index and domain.
    Npu(ActivityNpuId),
}

/// Decoded `NVLink` connection information.
///
/// Borrowed strings and slices remain tied to the source record lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityNvLink<'a> {
    /// `NVLink` version.
    pub version: u32,
    /// Device type of endpoint zero.
    pub device0_type: DeviceType,
    /// Device type of endpoint one.
    pub device1_type: DeviceType,
    /// Decoded identifier of endpoint zero: a GPU UUID, an NPU identifier, or an unknown device kind.
    pub device0_id: ActivityTopologyDeviceId,
    /// Decoded identifier of endpoint one: a GPU UUID, an NPU identifier, or an unknown device kind.
    pub device1_id: ActivityTopologyDeviceId,
    /// Capabilities of the link.
    pub flags: LinkFlags,
    /// Number of physical `NVLinks` present between two devices.
    pub physical_link_count: u32,
    /// Borrowed endpoint-zero port identifiers, one per physical link.
    /// `None` indicates a null native pointer or zero links.
    pub device0_ports: Option<&'a [u32]>,
    /// Borrowed endpoint-one port identifiers, one per physical link.
    /// `None` indicates a null native pointer or zero links.
    pub device1_ports: Option<&'a [u32]>,
    /// Bandwidth of `NVLink` in kbytes/sec.
    pub bandwidth: u64,
    /// Whether an `NVSwitch` is an intermediate node in this connection.
    pub nvswitch_connected: bool,
}

impl ActivityNvLink<'_> {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityNvLink5) -> Result<Self> {
        let device0_type = DeviceType::try_from(value.typeDev0)?;
        let device1_type = DeviceType::try_from(value.typeDev1)?;
        let port_count = value.physicalNvLinkCount as usize;
        let device0_ports = if value.portDev0.is_null() || port_count == 0 {
            None
        } else {
            // SAFETY: CUPTI stores `port_count` port entries behind each non-null port pointer, valid while the record is borrowed.
            Some(unsafe { slice::from_raw_parts(value.portDev0, port_count) })
        };
        let device1_ports = if value.portDev1.is_null() || port_count == 0 {
            None
        } else {
            // SAFETY: CUPTI stores `port_count` port entries behind each non-null port pointer, valid while the record is borrowed.
            Some(unsafe { slice::from_raw_parts(value.portDev1, port_count) })
        };
        Ok(Self {
            version: value.nvlinkVersion,
            device0_type,
            device1_type,
            device0_id: nvlink_device0_id_from_raw(device0_type, value.idDev0),
            device1_id: nvlink_device1_id_from_raw(device1_type, value.idDev1),
            flags: LinkFlags::from_bits_truncate(value.flag),
            physical_link_count: value.physicalNvLinkCount,
            device0_ports,
            device1_ports,
            bandwidth: value.bandwidth,
            nvswitch_connected: value.nvswitchConnected != 0,
        })
    }
}

fn nvlink_device0_id_from_raw(
    device_type: DeviceType,
    value: sys::CUpti_ActivityNvLink5__bindgen_ty_1,
) -> ActivityTopologyDeviceId {
    // SAFETY: each arm reads the union field that the matched tag selects.
    unsafe {
        match device_type {
            DeviceType::Gpu => ActivityTopologyDeviceId::Gpu {
                uuid: uuid_from_raw(value.uuidDev),
            },
            DeviceType::Npu => ActivityTopologyDeviceId::Npu(ActivityNpuId {
                index: value.npu.index,
                domain_id: NpuDomainId::from(value.npu.domainId),
            }),
            _ => ActivityTopologyDeviceId::Unknown,
        }
    }
}

fn nvlink_device1_id_from_raw(
    device_type: DeviceType,
    value: sys::CUpti_ActivityNvLink5__bindgen_ty_2,
) -> ActivityTopologyDeviceId {
    // SAFETY: each arm reads the union field that the matched tag selects.
    unsafe {
        match device_type {
            DeviceType::Gpu => ActivityTopologyDeviceId::Gpu {
                uuid: uuid_from_raw(value.uuidDev),
            },
            DeviceType::Npu => ActivityTopologyDeviceId::Npu(ActivityNpuId {
                index: value.npu.index,
                domain_id: NpuDomainId::from(value.npu.domainId),
            }),
            _ => ActivityTopologyDeviceId::Unknown,
        }
    }
}

/// GPU identity and peer-access information for PCI Express topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcieGpuAttributes {
    /// Native 16-byte GPU UUID.
    pub uuid: [u8; 16],
    /// Native fixed-size array of CUDA device handles with peer-to-peer access capability.
    pub peer_devices: [CudaDevice; 32],
}

/// Bridge identity and downstream bus information for PCI Express topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcieBridgeAttributes {
    /// Downstream bus number used to locate devices and bridges connected below this bridge.
    pub secondary_bus: PcieBusId,
    /// Hardware device identifier of the bridge.
    pub device_id: PcieHardwareDeviceId,
    /// Vendor identifier of the bridge.
    pub vendor_id: PcieVendorId,
}

/// CUDA device handle or bridge identifier for a PCI Express topology record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityPcieDeviceId {
    /// CUDA device handle for a GPU endpoint.
    Gpu(CudaDevice),
    /// Identifier for a PCI Express bridge.
    Bridge(PcieBridgeId),
}

/// GPU or bridge properties decoded from a PCI Express device-type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivityPcieAttributes {
    /// GPU UUID and peer-access information.
    Gpu(ActivityPcieGpuAttributes),
    /// Bridge vendor, device, and downstream bus information.
    Bridge(ActivityPcieBridgeAttributes),
}

macro_rules! impl_debug_display {
    ($($name:ty),+ $(,)?) => {
        $(
            impl Display for $name {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    write!(f, "{self:?}")
                }
            }
        )+
    };
}

impl_debug_display!(
    ActivityRecordPayload<'_>,
    ActivityMemoryPoolProperties,
    ActivityObjectId,
    ActivityUnifiedMemoryCounterFlags,
    ActivityDeviceAttributeKind,
    ActivityEnvironmentReading,
    ActivityTopologyDeviceId,
    ActivityPcieDeviceId,
    ActivityPcieAttributes,
);

/// Decoded PCI Express device topology information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivityPcie {
    /// PCI Express device-type tag used to decode identity and attributes.
    pub device_type: PcieDeviceType,
    /// A unique identifier for GPU or Bridge in Topology.
    pub id: ActivityPcieDeviceId,
    /// Domain for the GPU or Bridge, required to identify which PCIE bus it belongs to in multiple NUMA systems.
    pub domain: PcieDomainId,
    /// PCIE Generation of GPU or Bridge.
    pub pcie_generationeration: u16,
    /// Link rate of the GPU or bridge in gigatransfers per second (GT/s).
    pub link_rate: u16,
    /// Link width of the GPU or bridge.
    pub link_width: u16,
    /// Upstream bus ID for the GPU or PCI bridge.
    /// Required to identify which bus it is connected to in the topology.
    pub upstream_bus: u16,
    /// GPU or bridge properties selected by `device_type`.
    pub attributes: ActivityPcieAttributes,
}

impl ActivityPcie {
    pub(crate) fn from_raw(value: &sys::CUpti_ActivityPcie) -> Result<Self> {
        let device_type = PcieDeviceType::try_from(value.type_)?;
        // SAFETY: each arm reads the union field that the matched tag selects.
        let (id, attributes) = unsafe {
            match device_type {
                PcieDeviceType::Gpu => {
                    let attributes = value.attr.gpuAttr;
                    (
                        ActivityPcieDeviceId::Gpu(CudaDevice::from(value.id.devId)),
                        ActivityPcieAttributes::Gpu(ActivityPcieGpuAttributes {
                            uuid: uuid_from_raw(attributes.uuidDev),
                            peer_devices: attributes.peerDev.map(CudaDevice::from),
                        }),
                    )
                }
                PcieDeviceType::Bridge => {
                    let attributes = value.attr.bridgeAttr;
                    (
                        ActivityPcieDeviceId::Bridge(PcieBridgeId::from(value.id.bridgeId)),
                        ActivityPcieAttributes::Bridge(ActivityPcieBridgeAttributes {
                            secondary_bus: PcieBusId::from(attributes.secondaryBus),
                            device_id: PcieHardwareDeviceId::from(attributes.deviceId),
                            vendor_id: PcieVendorId::from(attributes.vendorId),
                        }),
                    )
                }
                PcieDeviceType::ForceInt => (
                    ActivityPcieDeviceId::Bridge(PcieBridgeId::from(value.id.bridgeId)),
                    ActivityPcieAttributes::Bridge(ActivityPcieBridgeAttributes {
                        secondary_bus: PcieBusId::from(0),
                        device_id: PcieHardwareDeviceId::from(0),
                        vendor_id: PcieVendorId::from(0),
                    }),
                ),
            }
        };
        Ok(Self {
            device_type,
            id,
            domain: PcieDomainId::from(value.domain),
            pcie_generationeration: value.pcieGeneration,
            link_rate: value.linkRate,
            link_width: value.linkWidth,
            upstream_bus: value.upstreamBus,
            attributes,
        })
    }
}
