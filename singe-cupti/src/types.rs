#![allow(deprecated)]

use std::{
    fmt::{self, Display, Formatter},
    mem::size_of,
};

use bitflags::bitflags;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cupti_sys as sys;

pub(crate) type SubscriberHandle = sys::CUpti_SubscriberHandle;

macro_rules! id_newtype {
    ($name:ident, $raw:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(transparent)]
        /// CUPTI identifier wrapper.
        pub struct $name($raw);

        impl $name {
            /// Returns the raw CUPTI identifier value.
            pub const fn get(self) -> $raw {
                self.0
            }

            pub(crate) const fn as_raw(self) -> $raw {
                self.0
            }
        }

        impl From<$raw> for $name {
            fn from(value: $raw) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $raw {
            fn from(value: $name) -> Self {
                value.as_raw()
            }
        }
    };
}

id_newtype!(ContextId, u32);
id_newtype!(DeviceId, u32);
id_newtype!(GraphId, u32);
id_newtype!(ExecutableGraphId, u32);
id_newtype!(GraphNodeId, u64);
id_newtype!(StreamId, u64);
id_newtype!(ChannelId, u32);
id_newtype!(CorrelationId, u32);
id_newtype!(ExternalCorrelationId, u64);
id_newtype!(CudaEventId, u32);
id_newtype!(CudaEventSyncId, u64);
id_newtype!(ActivityMarkerId, u32);
id_newtype!(SourceLocatorId, u32);
id_newtype!(FunctionId, u32);
id_newtype!(ModuleId, u32);
id_newtype!(GridId, i64);
id_newtype!(ProcessId, u64);
id_newtype!(ThreadId, u32);
id_newtype!(CuptiDomainId, u32);
id_newtype!(UnifiedMemoryProcessorId, u32);
id_newtype!(GpuInstanceId, u32);
id_newtype!(ComputeInstanceId, u32);
id_newtype!(NonUniformMemoryAccessId, u32);
id_newtype!(NpuDomainId, u32);
id_newtype!(JitOperationCorrelationId, u64);
id_newtype!(PcieBridgeId, u32);
id_newtype!(PcieBusId, u16);
id_newtype!(PcieHardwareDeviceId, u16);
id_newtype!(PcieVendorId, u16);
id_newtype!(PcieDomainId, u32);
id_newtype!(OpenAccDeviceTypeId, u32);
id_newtype!(OpenAccDeviceNumber, u32);
id_newtype!(OpenAccAsynchronousId, u64);

impl From<u32> for StreamId {
    fn from(value: u32) -> Self {
        Self(u64::from(value))
    }
}

impl From<u16> for StreamId {
    fn from(value: u16) -> Self {
        Self(u64::from(value))
    }
}

impl From<u32> for ProcessId {
    fn from(value: u32) -> Self {
        Self(u64::from(value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// CUDA device handle used by CUPTI APIs.
pub struct CudaDevice(i32);

impl CudaDevice {
    /// Returns the raw value decoded from CUPTI.
    pub const fn get(self) -> i32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> sys::CUdevice {
        self.0
    }
}

impl From<sys::CUdevice> for CudaDevice {
    fn from(value: sys::CUdevice) -> Self {
        Self(value)
    }
}

impl From<CudaDevice> for sys::CUdevice {
    fn from(value: CudaDevice) -> Self {
        value.as_raw()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Point at which CUPTI invokes an API callback.
/// Callback payloads expose this as the callback site of the underlying CUPTI callback data.
pub enum ApiCallbackSite {
    /// The callback is at the entry of the API call.
    Enter = sys::CUpti_ApiCallbackSite::CUPTI_API_ENTER as _,
    /// The callback is at the exit of the API call.
    Exit = sys::CUpti_ApiCallbackSite::CUPTI_API_EXIT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ApiCallbackSite::CUPTI_API_CBSITE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ApiCallbackSite, ApiCallbackSite);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// Identifier for a CUPTI event.
pub struct EventId(u32);

impl EventId {
    /// Returns the raw value decoded from CUPTI.
    pub const fn get(self) -> u32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> sys::CUpti_EventID {
        self.0
    }
}

impl From<sys::CUpti_EventID> for EventId {
    fn from(value: sys::CUpti_EventID) -> Self {
        Self(value)
    }
}

impl From<EventId> for sys::CUpti_EventID {
    fn from(value: EventId) -> Self {
        value.as_raw()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// Identifier for a CUPTI event domain.
pub struct EventDomainId(u32);

impl EventDomainId {
    /// Returns the raw value decoded from CUPTI.
    pub const fn get(self) -> u32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> sys::CUpti_EventDomainID {
        self.0
    }
}

impl From<sys::CUpti_EventDomainID> for EventDomainId {
    fn from(value: sys::CUpti_EventDomainID) -> Self {
        Self(value)
    }
}

impl From<EventDomainId> for sys::CUpti_EventDomainID {
    fn from(value: EventDomainId) -> Self {
        value.as_raw()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// Identifier for a CUPTI metric.
pub struct MetricId(u32);

impl MetricId {
    /// Returns the raw value decoded from CUPTI.
    pub const fn get(self) -> u32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> sys::CUpti_MetricID {
        self.0
    }
}

impl From<sys::CUpti_MetricID> for MetricId {
    fn from(value: sys::CUpti_MetricID) -> Self {
        Self(value)
    }
}

impl From<MetricId> for sys::CUpti_MetricID {
    fn from(value: MetricId) -> Self {
        value.as_raw()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// CUDA device class returned by [`DeviceAttribute::DeviceClass`].
pub enum DeviceAttributeDeviceClass {
    /// Tesla device class.
    Tesla = sys::CUpti_DeviceAttributeDeviceClass::CUPTI_DEVICE_ATTR_DEVICE_CLASS_TESLA as _,
    /// Quadro device class.
    Quadro = sys::CUpti_DeviceAttributeDeviceClass::CUPTI_DEVICE_ATTR_DEVICE_CLASS_QUADRO as _,
    /// Geforce device class.
    Geforce = sys::CUpti_DeviceAttributeDeviceClass::CUPTI_DEVICE_ATTR_DEVICE_CLASS_GEFORCE as _,
    /// Tegra device class.
    Tegra = sys::CUpti_DeviceAttributeDeviceClass::CUPTI_DEVICE_ATTR_DEVICE_CLASS_TEGRA as _,
}

impl_enum_conversion!(
    sys::CUpti_DeviceAttributeDeviceClass,
    DeviceAttributeDeviceClass
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Device attributes readable through [`crate::event::get_device_attribute_value`].
pub enum DeviceAttribute {
    /// Number of event IDs for a device.
    ///
    /// Returns a `u32` value.
    MaxEventId = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_EVENT_ID as _,
    /// Number of event domain IDs for a device.
    ///
    /// Returns a `u32` value.
    MaxEventDomainId = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID as _,
    /// Get global memory bandwidth in Kbytes/sec.
    ///
    /// Returns a `u64` value.
    GlobalMemoryBandwidth =
        sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH as _,
    /// Get theoretical maximum number of instructions per cycle.
    ///
    /// Returns a `u32` value.
    InstructionPerCycle = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE as _,
    /// Get theoretical maximum number of single precision instructions that can be executed per second.
    ///
    /// Returns a `u64` value.
    InstructionThroughputSp =
        sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION as _,
    /// Get number of frame buffers for device.
    ///
    /// Returns a `u64` value.
    MaxFrameBuffers = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS as _,
    /// Get PCIE link rate in Mega bits/sec for device.
    ///
    /// Returns a `u64` value, or 0 when the bus type is not PCIe.
    PcieLinkRate = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_PCIE_LINK_RATE as _,
    /// Get PCIE link width for device.
    ///
    /// Returns a `u64` value, or 0 when the bus type is not PCIe.
    PcieLinkWidth = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH as _,
    /// Get PCIE generation for device.
    ///
    /// Returns a `u64` value, or 0 when the bus type is not PCIe.
    PcieGeneration = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_PCIE_GEN as _,
    /// Get the class for the device.
    ///
    /// Returns a [`DeviceAttributeDeviceClass`].
    DeviceClass = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_DEVICE_CLASS as _,
    /// Get the peak single precision flop per cycle.
    ///
    /// Returns a `u64` value.
    FlopSpPerCycle = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE as _,
    /// Get the peak double precision flop per cycle.
    ///
    /// Returns a `u64` value.
    FlopDpPerCycle = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE as _,
    /// Get number of L2 units.
    ///
    /// Returns a `u64` value.
    MaxL2Units = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_L2_UNITS as _,
    /// Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_SHARED preference.
    ///
    /// Returns a `u64` value.
    MaxSharedMemoryCacheConfigPreferShared =
        sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED
            as _,
    /// Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_L1 preference.
    ///
    /// Returns a `u64` value.
    MaxSharedMemoryCacheConfigPreferL1 =
        sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1 as _,
    /// Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_EQUAL preference.
    ///
    /// Returns a `u64` value.
    MaxSharedMemoryCacheConfigPreferEqual =
        sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL
            as _,
    /// Get the peak half precision flop per cycle.
    ///
    /// Returns a `u64` value.
    FlopHpPerCycle = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE as _,
    /// Check if Nvlink is connected to device.
    ///
    /// Returns 1 when at least one NVLink is connected to the device, or 0 otherwise.
    NvLinkPresent = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_NVLINK_PRESENT as _,
    /// Check if Nvlink is present between GPU and CPU.
    ///
    /// Returns NVLink bandwidth in bytes per second, or 0 when NVLink is absent.
    GpuCentralProcessingUnitNvLinkBandwidth =
        sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW as _,
    /// Check if NVSwitch is present in the underlying topology.
    ///
    /// Returns 1 when present, or 0 otherwise.
    NvSwitchPresent = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_NVSWITCH_PRESENT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_DeviceAttribute::CUPTI_DEVICE_ATTR_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_DeviceAttribute, DeviceAttribute);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Decoded value for a [`DeviceAttribute`] query.
pub enum DeviceAttributeValue {
    /// U32 scalar value returned by CUPTI.
    U32(u32),
    /// Device Class value decoded from CUPTI.
    DeviceClass(DeviceAttributeDeviceClass),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// CUDA driver device attribute identifier used in CUPTI records.
pub struct CudaDeviceAttribute(u32);

impl CudaDeviceAttribute {
    /// Returns the raw value decoded from CUPTI.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<sys::CUdevice_attribute> for CudaDeviceAttribute {
    fn from(value: sys::CUdevice_attribute) -> Self {
        Self(value as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Event-domain attributes readable through [`crate::event::get_event_domain_attribute_value`] or [`crate::event::get_device_event_domain_attribute_value`].
pub enum EventDomainAttribute {
    /// Event domain name.
    ///
    /// Returns a null-terminated C string.
    Name = sys::CUpti_EventDomainAttribute::CUPTI_EVENT_DOMAIN_ATTR_NAME as _,
    /// Number of instances of the domain for which event counts will be collected.
    ///
    /// Returns the number of event domain instances that can be profiled as a `u32`.
    InstanceCount = sys::CUpti_EventDomainAttribute::CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT as _,
    /// Total number of instances of the domain, including instances that cannot be profiled.
    ///
    /// Returns the total number of event domain instances as a `u32`; not all instances may be profilable.
    TotalInstanceCount =
        sys::CUpti_EventDomainAttribute::CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT as _,
    /// Collection method used for events contained in the event domain.
    ///
    /// Returns an [`EventCollectionMethod`].
    CollectionMethod =
        sys::CUpti_EventDomainAttribute::CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_EventDomainAttribute::CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_EventDomainAttribute, EventDomainAttribute);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Decoded value for an [`EventDomainAttribute`] query.
pub enum EventDomainAttributeValue {
    /// Name string returned by CUPTI.
    Name(String),
    /// U32 scalar value returned by CUPTI.
    U32(u32),
    /// Collection Method decoded from CUPTI.
    CollectionMethod(EventCollectionMethod),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Hardware or software mechanism used to collect an event.
pub enum EventCollectionMethod {
    /// Event is collected using a hardware global performance monitor.
    Pm = sys::CUpti_EventCollectionMethod::CUPTI_EVENT_COLLECTION_METHOD_PM as _,
    /// Event is collected using a hardware SM performance monitor.
    Sm = sys::CUpti_EventCollectionMethod::CUPTI_EVENT_COLLECTION_METHOD_SM as _,
    /// Event is collected using software instrumentation.
    Instrumented =
        sys::CUpti_EventCollectionMethod::CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED as _,
    /// Event is collected using NvLink throughput counter method.
    NvLinkTc = sys::CUpti_EventCollectionMethod::CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_EventCollectionMethod::CUPTI_EVENT_COLLECTION_METHOD_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_EventCollectionMethod, EventCollectionMethod);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Event-group attributes readable through [`crate::event::get_event_group_attribute_value`].
pub enum EventGroupAttribute {
    /// The domain to which the event group is bound.
    ///
    /// Event domain ID selected when the first event is added to the group.
    EventDomainId = sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID as _,
    /// \[rw\] Profile all the instances of the domain for this eventgroup.
    ///
    /// Controls load balancing across all instances of an event domain.
    ProfileAllDomainInstances =
        sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES as _,
    /// \[rw\] Reserved for user data.
    UserData = sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_USER_DATA as _,
    /// Number of events in the group.
    ///
    /// Returns a `u32` value.
    NumEvents = sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS as _,
    /// Enumerates events in the group.
    ///
    /// Returns a buffer of event IDs; use [`EventGroupAttribute::NumEvents`] for the element count.
    Events = sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_EVENTS as _,
    /// Number of instances of the domain bound to this event group that will be counted.
    ///
    /// Returns a `u32` value.
    InstanceCount = sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT as _,
    /// Returns the event group profiling scope.
    ///
    /// Sets the scope used when adding events whose profiling scope can be either device or context.
    ProfilingScope = sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_PROFILING_SCOPE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_EventGroupAttribute::CUPTI_EVENT_GROUP_ATTR_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_EventGroupAttribute, EventGroupAttribute);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Decoded value for an [`EventGroupAttribute`] query.
pub enum EventGroupAttributeValue {
    /// Event Domain ID value decoded from CUPTI.
    EventDomainId(EventDomainId),
    /// Profile All Domain Instances value decoded from CUPTI.
    ProfileAllDomainInstances(bool),
    /// Num Events value decoded from CUPTI.
    NumEvents(u32),
    /// Events value decoded from CUPTI.
    Events(Vec<EventId>),
    /// Instance Count value decoded from CUPTI.
    InstanceCount(u32),
    /// Profiling Scope value decoded from CUPTI.
    ProfilingScope(EventProfilingScope),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Writable values for supported [`EventGroupAttribute`] settings.
pub enum EventGroupAttributeSetting {
    /// Profile All Domain Instances value decoded from CUPTI.
    ProfileAllDomainInstances(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Scope at which a CUPTI event can be profiled.
pub enum EventProfilingScope {
    /// Event is collected at context scope.
    Context = sys::CUpti_EventProfilingScope::CUPTI_EVENT_PROFILING_SCOPE_CONTEXT as _,
    /// Event is collected at device scope.
    Device = sys::CUpti_EventProfilingScope::CUPTI_EVENT_PROFILING_SCOPE_DEVICE as _,
    /// Event can be collected at device or context scope.
    ///
    /// Scope is selected on the event group before events are added.
    Both = sys::CUpti_EventProfilingScope::CUPTI_EVENT_PROFILING_SCOPE_BOTH as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_EventProfilingScope::CUPTI_EVENT_PROFILING_SCOPE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_EventProfilingScope, EventProfilingScope);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Event attributes readable through [`crate::event::get_event_attribute_value`].
pub enum EventAttribute {
    /// Event name.
    ///
    /// Returns a null-terminated C string.
    Name = sys::CUpti_EventAttribute::CUPTI_EVENT_ATTR_NAME as _,
    /// Short description of event.
    ///
    /// Returns a null-terminated C string.
    ShortDescription = sys::CUpti_EventAttribute::CUPTI_EVENT_ATTR_SHORT_DESCRIPTION as _,
    /// Long description of event.
    ///
    /// Returns a null-terminated C string.
    LongDescription = sys::CUpti_EventAttribute::CUPTI_EVENT_ATTR_LONG_DESCRIPTION as _,
    /// Category of event.
    ///
    /// Returns an [`EventCategory`].
    Category = sys::CUpti_EventAttribute::CUPTI_EVENT_ATTR_CATEGORY as _,
    /// Profiling scope of the events.
    ///
    /// Returns whether the event is profiled at device scope, context scope, or both.
    ProfilingScope = sys::CUpti_EventAttribute::CUPTI_EVENT_ATTR_PROFILING_SCOPE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_EventAttribute::CUPTI_EVENT_ATTR_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_EventAttribute, EventAttribute);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// General category assigned to a CUPTI event.
pub enum EventCategory {
    /// An instruction related event.
    Instruction = sys::CUpti_EventCategory::CUPTI_EVENT_CATEGORY_INSTRUCTION as _,
    /// A memory related event.
    Memory = sys::CUpti_EventCategory::CUPTI_EVENT_CATEGORY_MEMORY as _,
    /// A cache related event.
    Cache = sys::CUpti_EventCategory::CUPTI_EVENT_CATEGORY_CACHE as _,
    /// A profile-trigger event.
    ProfileTrigger = sys::CUpti_EventCategory::CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER as _,
    /// A system event.
    System = sys::CUpti_EventCategory::CUPTI_EVENT_CATEGORY_SYSTEM as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_EventCategory::CUPTI_EVENT_CATEGORY_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_EventCategory, EventCategory);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Decoded value for an [`EventAttribute`] query.
pub enum EventAttributeValue {
    /// Name string returned by CUPTI.
    Name(String),
    /// Short Description string returned by CUPTI.
    ShortDescription(String),
    /// Long Description string returned by CUPTI.
    LongDescription(String),
    /// Category value decoded from CUPTI.
    Category(EventCategory),
    /// Profiling Scope value decoded from CUPTI.
    ProfilingScope(EventProfilingScope),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Period over which enabled event groups collect events.
pub enum EventCollectionMode {
    /// Events are collected for the whole interval while the event group is enabled.
    ///
    /// Event values are reset when the events are read. For CUDA toolkit v6.0 and older this was the default mode.
    Continuous = sys::CUpti_EventCollectionMode::CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS as _,
    /// Events are collected only while kernels execute during the enabled interval.
    ///
    /// Event values are reset when each kernel begins. Read values after each kernel launch
    /// when they need to be associated with individual launches. This mode serializes kernels
    /// on the GPU and is the default mode from CUDA toolkit 6.5 onward.
    Kernel = sys::CUpti_EventCollectionMode::CUPTI_EVENT_COLLECTION_MODE_KERNEL as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_EventCollectionMode::CUPTI_EVENT_COLLECTION_MODE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_EventCollectionMode, EventCollectionMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Flags used when reading event values from an event group.
pub enum ReadEventFlags {
    /// No flags.
    None = sys::CUpti_ReadEventFlags::CUPTI_EVENT_READ_FLAG_NONE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ReadEventFlags::CUPTI_EVENT_READ_FLAG_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ReadEventFlags, ReadEventFlags);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Group of related CUPTI callback IDs.
pub enum CallbackDomain {
    /// Invalid domain.
    Invalid = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_INVALID as _,
    /// Domain containing callback points for all driver API functions.
    DriverApi = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API as _,
    /// Domain containing callback points for all runtime API functions.
    RuntimeApi = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RUNTIME_API as _,
    /// Domain containing callback points for CUDA resource tracking.
    Resource = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RESOURCE as _,
    /// Domain containing callback points for CUDA synchronization.
    Synchronize = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_SYNCHRONIZE as _,
    /// Domain containing callback points for NVTX API functions.
    Nvtx = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_NVTX as _,
    /// Domain containing callback points for various states.
    State = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_STATE as _,
    /// Number of entries defined for this CUPTI enum.
    Size = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_SIZE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_CallbackDomain, CallbackDomain);

impl_enum_display!(CallbackDomain, {
    Self::Invalid => "CUPTI_CB_DOMAIN_INVALID",
    Self::DriverApi => "CUPTI_CB_DOMAIN_DRIVER_API",
    Self::RuntimeApi => "CUPTI_CB_DOMAIN_RUNTIME_API",
    Self::Resource => "CUPTI_CB_DOMAIN_RESOURCE",
    Self::Synchronize => "CUPTI_CB_DOMAIN_SYNCHRONIZE",
    Self::Nvtx => "CUPTI_CB_DOMAIN_NVTX",
    Self::State => "CUPTI_CB_DOMAIN_STATE",
    Self::Size => "CUPTI_CB_DOMAIN_SIZE",
    Self::ForceInt => "CUPTI_CB_DOMAIN_FORCE_INT",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Controls whether an activity flush is requested normally or forced.
pub enum ActivityFlushFlag {
    /// Requests the default activity buffer flush behavior.
    Default,
    /// Requests a forced activity buffer flush.
    Forced,
}

impl From<ActivityFlushFlag> for u32 {
    fn from(value: ActivityFlushFlag) -> Self {
        match value {
            ActivityFlushFlag::Default => 0,
            ActivityFlushFlag::Forced => {
                sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED as u32
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
/// Identifier for a CUPTI callback within a callback domain.
pub struct CallbackId(u32);

impl CallbackId {
    /// Returns the raw value decoded from CUPTI.
    pub const fn get(self) -> u32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> sys::CUpti_CallbackId {
        self.0
    }
}

impl From<sys::CUpti_CallbackId> for CallbackId {
    fn from(value: sys::CUpti_CallbackId) -> Self {
        Self(value)
    }
}

impl From<CallbackIdResource> for CallbackId {
    fn from(value: CallbackIdResource) -> Self {
        Self(value.into())
    }
}

impl From<CallbackIdSync> for CallbackId {
    fn from(value: CallbackIdSync) -> Self {
        Self(value.into())
    }
}

impl From<CallbackIdState> for CallbackId {
    fn from(value: CallbackIdState) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Callback IDs emitted for [`CallbackDomain::Resource`].
pub enum CallbackIdResource {
    /// Invalid resource callback ID.
    Invalid = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_INVALID as _,
    /// A new context has been created.
    ContextCreated = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_CONTEXT_CREATED as _,
    /// A context is about to be destroyed.
    ContextDestroyStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING as _,
    /// A new stream has been created.
    StreamCreated = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_STREAM_CREATED as _,
    /// A stream is about to be destroyed.
    StreamDestroyStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING as _,
    /// The driver has finished initializing.
    CuInitFinished = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_CU_INIT_FINISHED as _,
    /// A module has been loaded.
    ModuleLoaded = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_MODULE_LOADED as _,
    /// A module is about to be unloaded.
    ModuleUnloadStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING as _,
    /// The current module which is being profiled.
    ModuleProfiled = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_MODULE_PROFILED as _,
    /// CUDA graph has been created.
    GraphCreated = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPH_CREATED as _,
    /// CUDA graph is about to be destroyed.
    GraphDestroyStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING as _,
    /// CUDA graph is cloned.
    GraphCloned = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPH_CLONED as _,
    /// CUDA graph node is about to be created.
    GraphNodeCreateStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING as _,
    /// CUDA graph node is created.
    GraphNodeCreated = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED as _,
    /// CUDA graph node is about to be destroyed.
    GraphNodeDestroyStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING as _,
    /// Dependency on a CUDA graph node is created.
    GraphNodeDependencyCreated =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED as _,
    /// Dependency on a CUDA graph node is destroyed.
    GraphNodeDependencyDestroyStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING
            as _,
    /// An executable CUDA graph is about to be created.
    ExecutableGraphCreateStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING as _,
    /// An executable CUDA graph is created.
    ExecutableGraphCreated =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED as _,
    /// An executable CUDA graph is about to be destroyed.
    ExecutableGraphDestroyStarting =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING as _,
    /// CUDA graph node is cloned.
    GraphNodeCloned = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED as _,
    /// CUDA stream attribute is changed.
    StreamAttributeChanged =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED as _,
    /// CUDA graph node is updated.
    GraphNodeUpdated = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED as _,
    /// Params are set for the CUDA graph node in the executable graph.
    GraphNodeSetParams =
        sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS as _,
    /// Number of entries defined for this CUPTI enum.
    Size = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_SIZE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_CallbackIdResource::CUPTI_CBID_RESOURCE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_CallbackIdResource, CallbackIdResource);

impl_enum_display!(CallbackIdResource, {
    Self::Invalid => "CUPTI_CBID_RESOURCE_INVALID",
    Self::ContextCreated => "CUPTI_CBID_RESOURCE_CONTEXT_CREATED",
    Self::ContextDestroyStarting => "CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING",
    Self::StreamCreated => "CUPTI_CBID_RESOURCE_STREAM_CREATED",
    Self::StreamDestroyStarting => "CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING",
    Self::CuInitFinished => "CUPTI_CBID_RESOURCE_CU_INIT_FINISHED",
    Self::ModuleLoaded => "CUPTI_CBID_RESOURCE_MODULE_LOADED",
    Self::ModuleUnloadStarting => "CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING",
    Self::ModuleProfiled => "CUPTI_CBID_RESOURCE_MODULE_PROFILED",
    Self::GraphCreated => "CUPTI_CBID_RESOURCE_GRAPH_CREATED",
    Self::GraphDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING",
    Self::GraphCloned => "CUPTI_CBID_RESOURCE_GRAPH_CLONED",
    Self::GraphNodeCreateStarting => "CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING",
    Self::GraphNodeCreated => "CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED",
    Self::GraphNodeDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING",
    Self::GraphNodeDependencyCreated => "CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED",
    Self::GraphNodeDependencyDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING",
    Self::ExecutableGraphCreateStarting => "CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING",
    Self::ExecutableGraphCreated => "CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED",
    Self::ExecutableGraphDestroyStarting => "CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING",
    Self::GraphNodeCloned => "CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED",
    Self::StreamAttributeChanged => "CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED",
    Self::GraphNodeUpdated => "CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED",
    Self::GraphNodeSetParams => "CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS",
    Self::Size => "CUPTI_CBID_RESOURCE_SIZE",
    Self::ForceInt => "CUPTI_CBID_RESOURCE_FORCE_INT",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Callback IDs emitted for [`CallbackDomain::Synchronize`].
pub enum CallbackIdSync {
    /// Invalid synchronize callback ID.
    Invalid = sys::CUpti_CallbackIdSync::CUPTI_CBID_SYNCHRONIZE_INVALID as _,
    /// Stream synchronization has completed for the stream.
    StreamSynchronized = sys::CUpti_CallbackIdSync::CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED as _,
    /// Context synchronization has completed for the context.
    ContextSynchronized =
        sys::CUpti_CallbackIdSync::CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED as _,
    /// Number of entries defined for this CUPTI enum.
    Size = sys::CUpti_CallbackIdSync::CUPTI_CBID_SYNCHRONIZE_SIZE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_CallbackIdSync::CUPTI_CBID_SYNCHRONIZE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_CallbackIdSync, CallbackIdSync);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Callback IDs emitted for [`CallbackDomain::State`].
pub enum CallbackIdState {
    /// Invalid state callback ID.
    Invalid = sys::CUpti_CallbackIdState::CUPTI_CBID_STATE_INVALID as _,
    /// High-impact, non-recoverable error notification.
    ///
    /// CUPTI finalizes itself before issuing this callback; subscribers decide whether the
    /// application should continue without profiling or terminate.
    FatalError = sys::CUpti_CallbackIdState::CUPTI_CBID_STATE_FATAL_ERROR as _,
    /// Notification of non fatal errors - high impact, but recoverable.
    Error = sys::CUpti_CallbackIdState::CUPTI_CBID_STATE_ERROR as _,
    /// Notification of warnings - low impact, recoverable.
    Warning = sys::CUpti_CallbackIdState::CUPTI_CBID_STATE_WARNING as _,
    /// Number of entries defined for this CUPTI enum.
    Size = sys::CUpti_CallbackIdState::CUPTI_CBID_STATE_SIZE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_CallbackIdState::CUPTI_CBID_STATE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_CallbackIdState, CallbackIdState);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// CUPTI activity record kind.
///
/// Each value identifies the shape of activity record that CUPTI can emit.
pub enum ActivityKind {
    /// The activity record is invalid.
    Invalid = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID as _,
    /// A host-to-host, host-to-device, or device-to-device memory copy record.
    ///
    /// CUPTI emits this as [`crate::activity::ActivityMemoryCopy`]. Use
    /// [`ActivityKind::PeerMemoryCopy`] for peer-to-peer copies.
    MemoryCopy = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY as _,
    /// A GPU memory set record emitted as [`crate::activity::ActivityMemorySet`].
    MemorySet = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET as _,
    /// A serialized GPU kernel execution record emitted as [`crate::activity::ActivityKernel`].
    ///
    /// This kind serializes kernel execution. Prefer [`ActivityKind::ConcurrentKernel`] when
    /// kernel concurrency should be preserved.
    Kernel = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_KERNEL as _,
    /// A CUDA driver API call record emitted as [`crate::activity::ActivityApi`].
    Driver = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER as _,
    /// A CUDA runtime API call record emitted as [`crate::activity::ActivityApi`].
    Runtime = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME as _,
    /// A legacy event counter record emitted as [`crate::activity::ActivityEvent`].
    ///
    /// This kind is populated from Event API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    Event = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EVENT as _,
    /// A legacy metric value record emitted as [`crate::activity::ActivityMetric`].
    ///
    /// This kind is populated from Metric API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    Metric = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_METRIC as _,
    /// CUDA device information emitted as [`crate::activity::ActivityDevice`].
    Device = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE as _,
    /// CUDA context information emitted as [`crate::activity::ActivityContext`].
    Context = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONTEXT as _,
    /// A concurrent GPU kernel execution record emitted as [`crate::activity::ActivityKernel`].
    ConcurrentKernel = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL as _,
    /// Resource names assigned through NVTX APIs, emitted as [`crate::activity::ActivityName`].
    Name = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_NAME as _,
    /// An NVTX marker record emitted as [`crate::activity::ActivityMarker`].
    Marker = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MARKER as _,
    /// Optional NVTX marker payload emitted as [`crate::activity::ActivityMarkerData`].
    ///
    /// Enable [`ActivityKind::Marker`] as well to receive the associated marker records.
    MarkerData = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MARKER_DATA as _,
    /// Legacy source location record emitted as [`crate::activity::ActivitySourceLocator`].
    ///
    /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
    SourceLocator = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR as _,
    /// Legacy source-level global access record emitted as [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess).
    ///
    /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
    GlobalAccess = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS as _,
    /// Legacy source-level branch record emitted as [`crate::activity::ActivityBranch`].
    ///
    /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
    Branch = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_BRANCH as _,
    /// CUPTI, compiler, CUDA driver, or other overhead emitted as [`crate::activity::ActivityOverhead`].
    Overhead = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OVERHEAD as _,
    /// CUDA Dynamic Parallelism kernel record emitted as [`crate::activity::ActivityCudaDynamicParallelismKernel`].
    ///
    /// This kind is controlled through [`ActivityKind::ConcurrentKernel`] rather than enabled directly.
    CudaDynamicParallelismKernel = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CDP_KERNEL as _,
    /// CUDA Dynamic Parallelism preemption record emitted as [`crate::activity::ActivityPreemption`].
    Preemption = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PREEMPTION as _,
    /// GPU power, clock, thermal, and related environment data emitted as [`crate::activity::ActivityEnvironment`].
    Environment = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT as _,
    /// Legacy event counter instance record emitted as [`crate::activity::ActivityEventInstance`].
    ///
    /// This kind is populated from Event API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    EventInstance = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EVENT_INSTANCE as _,
    /// A peer-to-peer memory copy record emitted as [`crate::activity::ActivityPeerMemoryCopy`].
    PeerMemoryCopy = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY2 as _,
    /// Legacy metric instance record emitted as [`crate::activity::ActivityMetricInstance`].
    ///
    /// This kind is populated from Metric API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    MetricInstance = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_METRIC_INSTANCE as _,
    /// Legacy source-level instruction execution record emitted as [`crate::activity::ActivityInstructionExecution`].
    ///
    /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
    InstructionExecution = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION as _,
    /// Unified Memory counter record emitted as [`crate::activity::ActivityUnifiedMemoryCounter`].
    UnifiedMemoryCounter = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER as _,
    /// Device function record emitted as [`crate::activity::ActivityFunction`].
    Function = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_FUNCTION as _,
    /// CUDA module record emitted as [`crate::activity::ActivityModule`].
    ///
    /// This kind is populated from module callbacks rather than directly enabled or disabled.
    Module = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MODULE as _,
    /// Device attribute record emitted as [`crate::activity::ActivityDeviceAttribute`].
    ///
    /// This kind stores values collected through [`DeviceAttribute`] or CUDA device attributes.
    DeviceAttribute = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE as _,
    /// Legacy source-level shared access record emitted as [`crate::activity::ActivitySharedAccess`].
    ///
    /// This kind is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
    SharedAccess = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SHARED_ACCESS as _,
    /// Legacy PC sampling record emitted as [`crate::activity::ActivityPcSampling`].
    ///
    /// This kind serializes kernels and is unsupported starting with CUDA 13.0; use the PC Sampling API instead.
    PcSampling = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PC_SAMPLING as _,
    /// Legacy PC sampling summary record emitted as [`crate::activity::ActivityPcSamplingRecordInfo`].
    ///
    /// This kind is unsupported starting with CUDA 13.0; use the PC Sampling API instead.
    PcSamplingRecordInfo =
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO as _,
    /// Legacy SASS/source correlation record emitted as [`crate::activity::ActivityInstructionCorrelation`].
    ///
    /// CUPTI emits these records when source-level analysis or PC sampling is enabled. This kind
    /// is unsupported starting with CUDA 13.0; use the SASS Metric APIs instead.
    InstructionCorrelation =
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION as _,
    /// OpenACC data event record emitted as [`crate::activity::ActivityOpenAccData`].
    OpenAccData = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_DATA as _,
    /// OpenACC launch event record emitted as [`crate::activity::ActivityOpenAccLaunch`].
    OpenAccLaunch = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH as _,
    /// OpenACC miscellaneous event record emitted as [`crate::activity::ActivityOpenAccOther`].
    OpenAccOther = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_OTHER as _,
    /// CUDA event handle record emitted as [`crate::activity::ActivityCudaEvent`].
    CudaEvent = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CUDA_EVENT as _,
    /// CUDA stream record emitted as [`crate::activity::ActivityStream`].
    Stream = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_STREAM as _,
    /// CUDA synchronization record emitted as [`crate::activity::ActivitySynchronization`].
    Synchronization = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION as _,
    /// External API correlation record emitted as [`crate::activity::ActivityExternalCorrelation`].
    ExternalCorrelation = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION as _,
    /// NVLink topology record emitted as [`crate::activity::ActivityNvLink`].
    NvLink = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_NVLINK as _,
    /// Legacy instantaneous event record emitted as [`crate::activity::ActivityInstantaneousEvent`].
    ///
    /// This kind is populated from Event API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    InstantaneousEvent = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT as _,
    /// Legacy instantaneous event instance record emitted as [`crate::activity::ActivityInstantaneousEventInstance`].
    ///
    /// This kind is populated from Event API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    InstantaneousEventInstance =
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE as _,
    /// Legacy instantaneous metric record emitted as [`crate::activity::ActivityInstantaneousMetric`].
    ///
    /// This kind is populated from Metric API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    InstantaneousMetric = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC as _,
    /// Legacy instantaneous metric instance record emitted as [`crate::activity::ActivityInstantaneousMetricInstance`].
    ///
    /// This kind is populated from Metric API collection, is not directly enabled or disabled,
    /// and is unsupported starting with CUDA 13.0.
    InstantaneousMetricInstance =
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE as _,
    /// Legacy memory allocation record emitted as [`crate::activity::ActivityLegacyMemory`].
    Memory = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY as _,
    /// PCI topology record emitted as [`crate::activity::ActivityPcie`].
    Pcie = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PCIE as _,
    /// OpenMP parallel event record emitted as [`crate::activity::ActivityOpenMultiProcessing`].
    OpenMultiProcessing = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENMP as _,
    /// Internal CUDA driver launch record emitted as [`crate::activity::ActivityApi`].
    InternalLaunchApi = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API as _,
    /// Memory allocation record emitted as [`crate::activity::ActivityMemory`].
    Memory2 = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2 as _,
    /// Memory pool record emitted as [`crate::activity::ActivityMemoryPool`].
    MemoryPool = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY_POOL as _,
    /// CUDA graph trace record emitted as [`crate::activity::ActivityGraphTrace`].
    GraphTrace = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GRAPH_TRACE as _,
    /// JIT operation record emitted as [`crate::activity::ActivityJit`].
    Jit = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_JIT as _,
    /// Device graph trace record emitted as [`crate::activity::ActivityDeviceGraphTrace`].
    ///
    /// Enable [`ActivityKind::GraphTrace`] and device graph tracing to receive these records.
    DeviceGraphTrace = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE as _,
    /// Memory decompression record emitted as [`crate::activity::ActivityMemoryDecompress`].
    MemoryDecompress = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS as _,
    /// Confidential-computing overhead record emitted as [`crate::activity::ActivityConfidentialComputeRotation`].
    ConfidentialComputeRotation =
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION as _,
    /// CUDA graph host-node execution record emitted as [`crate::activity::ActivityGraphHostNode`].
    GraphHostNode = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE as _,
    /// Compute-engine context switch record emitted as [`crate::activity::ActivityComputeEngineContextSwitch`].
    ComputeEngineCtxSwitch =
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_COMPUTE_ENGINE_CTX_SWITCH as _,
    /// Host function launch record emitted as [`crate::activity::ActivityHostLaunch`].
    HostLaunch = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_HOST_LAUNCH as _,
    /// Green context allocation record emitted as [`crate::activity::ActivityGreenContext`].
    GreenContext = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GREEN_CONTEXT as _,
    /// Count of supported activity kinds.
    Count = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_COUNT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityKind, ActivityKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// CUPTI channel type reported by activity records.
pub enum ChannelType {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_INVALID as _,
    /// Channel is used for standard work launch and tracking.
    Compute = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_COMPUTE as _,
    /// Channel is used by an asynchronous copy engine For confidential compute configurations, work launch and completion are done using the copy engines.
    AsynchronousMemoryCopy = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY as _,
    /// Channel is used for memory decompression operations.
    Decompress = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_DECOMP as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ChannelType, ChannelType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Source and destination kind for an activity memory-copy record.
pub enum ActivityMemoryCopyKind {
    /// The memory copy kind is not known.
    Unknown = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN as _,
    /// A host to device memory copy.
    HostToDevice = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_HTOD as _,
    /// A device to host memory copy.
    DeviceToHost = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_DTOH as _,
    /// A host to device array memory copy.
    HostToArray = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_HTOA as _,
    /// A device array to host memory copy.
    ArrayToHost = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_ATOH as _,
    /// A device array to device array memory copy.
    ArrayToArray = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_ATOA as _,
    /// A device array to device memory copy.
    ArrayToDevice = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_ATOD as _,
    /// A device to device array memory copy.
    DeviceToArray = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_DTOA as _,
    /// A device to device memory copy on the same device.
    DeviceToDevice = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_DTOD as _,
    /// A host to host memory copy.
    HostToHost = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_HTOH as _,
    /// A peer to peer memory copy across different devices.
    PeerToPeer = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_PTOP as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityMemcpyKind::CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityMemcpyKind, ActivityMemoryCopyKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Memory kind reported by activity memory records.
pub enum ActivityMemoryKind {
    /// The memory kind is unknown.
    Unknown = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN as _,
    /// The memory is pageable.
    Pageable = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE as _,
    /// The memory is pinned.
    Pinned = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PINNED as _,
    /// The memory is on the device.
    Device = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE as _,
    /// The memory is an array.
    Array = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_ARRAY as _,
    /// The memory is managed.
    Managed = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED as _,
    /// The memory is device static.
    DeviceStatic = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC as _,
    /// The memory is managed static.
    ManagedStatic = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityMemoryKind, ActivityMemoryKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Operation type reported by [`crate::activity::ActivityMemory`].
pub enum ActivityMemoryOperationType {
    /// The operation is invalid.
    Invalid =
        sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID as _,
    /// Memory is allocated.
    Allocation =
        sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION
            as _,
    /// Memory is released.
    Release =
        sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityMemoryOperationType,
    ActivityMemoryOperationType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Memory pool type reported by [`crate::activity::ActivityMemoryPool`].
pub enum ActivityMemoryPoolType {
    /// The operation is invalid.
    Invalid = sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID as _,
    /// Memory pool is local to the process.
    Local = sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL as _,
    /// Memory pool is imported by the process.
    Imported = sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityMemoryPoolType, ActivityMemoryPoolType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Memory pool operation reported by [`crate::activity::ActivityMemoryPool`].
pub enum ActivityMemoryPoolOperationType {
    /// The operation is invalid.
    Invalid =
        sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_INVALID
            as _,
    /// Memory pool is created.
    Created =
        sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED
            as _,
    /// Memory pool is destroyed.
    Destroyed =
        sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED
            as _,
    /// Memory pool is trimmed.
    Trimmed =
        sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED
            as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_FORCE_INT
            as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityMemoryPoolOperationType,
    ActivityMemoryPoolOperationType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Launch mode used for device graph execution.
pub enum DeviceGraphLaunchMode {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_DeviceGraphLaunchMode::CUPTI_DEVICE_GRAPH_LAUNCH_MODE_INVALID as _,
    /// Fire And Forget value decoded from CUPTI.
    FireAndForget =
        sys::CUpti_DeviceGraphLaunchMode::CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET as _,
    /// Tail value decoded from CUPTI.
    Tail = sys::CUpti_DeviceGraphLaunchMode::CUPTI_DEVICE_GRAPH_LAUNCH_MODE_TAIL as _,
    /// Fire And Forget As Sibling value decoded from CUPTI.
    FireAndForgetAsSibling =
        sys::CUpti_DeviceGraphLaunchMode::CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET_AS_SIBLING
            as _,
}

impl_enum_conversion!(sys::CUpti_DeviceGraphLaunchMode, DeviceGraphLaunchMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Compute API family reported by activity records.
pub enum ActivityComputeApiKind {
    /// The compute API is not known.
    Unknown = sys::CUpti_ActivityComputeApiKind::CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN as _,
    /// The compute APIs are for CUDA.
    Cuda = sys::CUpti_ActivityComputeApiKind::CUPTI_ACTIVITY_COMPUTE_API_CUDA as _,
    /// The compute APIs are for CUDA running in MPS (Multi-Process Service) environment.
    CudaMultiProcessService =
        sys::CUpti_ActivityComputeApiKind::CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityComputeApiKind::CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityComputeApiKind, ActivityComputeApiKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// CUDA-in-Graphics mode associated with a CUDA context.
pub enum ContextCigMode {
    /// Regular (non-CIG) mode.
    None = sys::CUpti_ContextCigMode::CUPTI_CONTEXT_CIG_MODE_NONE as _,
    /// CIG mode.
    Cig = sys::CUpti_ContextCigMode::CUPTI_CONTEXT_CIG_MODE_CIG as _,
    /// CIG fallback mode.
    CigFallback = sys::CUpti_ContextCigMode::CUPTI_CONTEXT_CIG_MODE_CIG_FALLBACK as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ContextCigMode::CUPTI_CONTEXT_CIG_MODE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ContextCigMode, ContextCigMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Operation reported by a compute-engine context-switch record.
pub enum ComputeEngineContextSwitchOperationType {
    /// Invalid value decoded from CUPTI.
    Invalid =
        sys::CUpti_ComputeEngineCtxSwitchOperationType::CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_INVALID
            as _,
    /// The start of the CUDA context switch operation.
    Start =
        sys::CUpti_ComputeEngineCtxSwitchOperationType::CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_START
            as _,
    /// The end of the CUDA context switch operation.
    End =
        sys::CUpti_ComputeEngineCtxSwitchOperationType::CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_END
            as _,
    /// Number of entries defined for this CUPTI enum.
    Count =
        sys::CUpti_ComputeEngineCtxSwitchOperationType::CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_COUNT
            as _,
}

impl_enum_conversion!(
    sys::CUpti_ComputeEngineCtxSwitchOperationType,
    ComputeEngineContextSwitchOperationType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Synchronization operation reported by [`crate::activity::ActivitySynchronization`].
pub enum ActivitySynchronizationType {
    /// Unknown data.
    Unknown =
        sys::CUpti_ActivitySynchronizationType::CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN as _,
    /// Event synchronize API.
    EventSynchronize =
        sys::CUpti_ActivitySynchronizationType::CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE as _,
    /// Stream wait event API.
    StreamWaitEvent =
        sys::CUpti_ActivitySynchronizationType::CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT as _,
    /// Stream synchronize API.
    StreamSynchronize =
        sys::CUpti_ActivitySynchronizationType::CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE as _,
    /// Context synchronize API.
    ContextSynchronize =
        sys::CUpti_ActivitySynchronizationType::CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ActivitySynchronizationType::CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivitySynchronizationType,
    ActivitySynchronizationType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Stream flag reported by [`crate::activity::ActivityStream`].
pub enum ActivityStreamFlag {
    /// Unknown data.
    Unknown = sys::CUpti_ActivityStreamFlag::CUPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN as _,
    /// Default stream.
    Default = sys::CUpti_ActivityStreamFlag::CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT as _,
    /// Non-blocking stream.
    NonBlocking =
        sys::CUpti_ActivityStreamFlag::CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING as _,
    /// Null stream.
    Null = sys::CUpti_ActivityStreamFlag::CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL as _,
    /// Stream create Mask.
    Mask = sys::CUpti_ActivityStreamFlag::CUPTI_ACTIVITY_STREAM_CREATE_MASK as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityStreamFlag::CUPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityStreamFlag, ActivityStreamFlag);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Partitioned global cache configuration reported by kernel activity records.
pub enum ActivityPartitionedGlobalCacheConfig {
    /// Partitioned global cache config unknown.
    Unknown = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN as _,
    /// Partitioned global cache not supported.
    NotSupported = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED as _,
    /// Partitioned global cache config off.
    Off = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF as _,
    /// Partitioned global cache config on.
    On = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityPartitionedGlobalCacheConfig,
    ActivityPartitionedGlobalCacheConfig
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// CUDA kernel launch type reported by kernel activity records.
pub enum ActivityLaunchType {
    /// The kernel was launched via a regular kernel call.
    Regular = sys::CUpti_ActivityLaunchType::CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR as _,
    /// The kernel was launched via API cudaLaunchCooperativeKernel() or cuLaunchCooperativeKernel().
    CooperativeSingleDevice =
        sys::CUpti_ActivityLaunchType::CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE as _,
    /// The kernel was launched via API cudaLaunchCooperativeKernelMultiDevice() or cuLaunchCooperativeKernelMultiDevice().
    CooperativeMultiDevice =
        sys::CUpti_ActivityLaunchType::CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE as _,
    /// The kernel was launched as a CBL commandlist.
    CblCommandlist = sys::CUpti_ActivityLaunchType::CUPTI_ACTIVITY_LAUNCH_TYPE_CBL_COMMANDLIST as _,
}

impl_enum_conversion!(sys::CUpti_ActivityLaunchType, ActivityLaunchType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Shared-memory limit configuration for occupancy-related kernel metadata.
pub enum FunctionSharedMemoryLimitConfig {
    /// The shared memory limit config is default.
    Default = sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_DEFAULT as _,
    /// User has opted for a higher dynamic shared memory limit using function attribute ‘cudaFuncAttributeMaxDynamicSharedMemorySize’ for runtime API or CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES for driver API.
    OptIn = sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_OPTIN as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_FuncShmemLimitConfig,
    FunctionSharedMemoryLimitConfig
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Kind of named object referenced by activity records.
pub enum ActivityObjectKind {
    /// The object kind is not known.
    Unknown = sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_UNKNOWN as _,
    /// A process.
    Process = sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_PROCESS as _,
    /// A thread.
    Thread = sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_THREAD as _,
    /// A device.
    Device = sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_DEVICE as _,
    /// A context.
    Context = sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_CONTEXT as _,
    /// A stream.
    Stream = sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_STREAM as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityObjectKind::CUPTI_ACTIVITY_OBJECT_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityObjectKind, ActivityObjectKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Kind of CUPTI overhead reported by [`crate::activity::ActivityOverhead`].
pub enum ActivityOverheadKind {
    /// The overhead kind is not known.
    Unknown = sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_UNKNOWN as _,
    /// Compiler overhead.
    DriverCompiler = sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER as _,
    /// Activity buffer flush overhead.
    BufferFlush = sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH as _,
    /// CUPTI instrumentation overhead.
    Instrumentation =
        sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION as _,
    /// CUPTI resource creation and destruction overhead.
    Resource = sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE as _,
    /// CUDA Runtime triggered module loading overhead.
    RuntimeTriggeredModuleLoading =
        sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING
            as _,
    /// Lazy function loading overhead.
    LazyFunctionLoading =
        sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING as _,
    /// Overhead due to lack of command buffer space.
    ///
    /// See [`crate::activity::ActivityOverhead`] for the decoded overhead payload.
    CommandBufferFull =
        sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL as _,
    /// Overhead due to activity buffer request.
    ActivityBufferRequest =
        sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST as _,
    /// Overhead due to UVM activity initialization.
    UvmActivityInit =
        sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_UVM_ACTIVITY_INIT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityOverheadKind::CUPTI_ACTIVITY_OVERHEAD_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityOverheadKind, ActivityOverheadKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Virtualization mode in which a CUDA device is running.
pub enum DeviceVirtualizationMode {
    /// No virtualization mode is associated with the device i.e.
    ///
    /// it’s a baremetal GPU.
    None = sys::CUpti_DeviceVirtualizationMode::CUPTI_DEVICE_VIRTUALIZATION_MODE_NONE as _,
    /// The device is associated with the pass-through GPU.
    ///
    /// In this mode, an entire physical GPU is directly assigned to one virtual machine (VM).
    PassThrough =
        sys::CUpti_DeviceVirtualizationMode::CUPTI_DEVICE_VIRTUALIZATION_MODE_PASS_THROUGH as _,
    /// The device is associated with the virtual GPU (vGPU).
    ///
    /// In this mode multiple virtual machines (VMs) have simultaneous, direct access to a single physical GPU.
    VirtualGpu =
        sys::CUpti_DeviceVirtualizationMode::CUPTI_DEVICE_VIRTUALIZATION_MODE_VIRTUAL_GPU as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_DeviceVirtualizationMode::CUPTI_DEVICE_VIRTUALIZATION_MODE_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_DeviceVirtualizationMode,
    DeviceVirtualizationMode
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// External correlation namespace used by CUPTI correlation records.
pub enum ExternalCorrelationKind {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_INVALID as _,
    /// The external API is unknown to CUPTI.
    Unknown = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN as _,
    /// The external API is OpenACC.
    OpenAcc = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC as _,
    /// The external API is custom0.
    Custom0 = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 as _,
    /// The external API is custom1.
    Custom1 = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1 as _,
    /// The external API is custom2.
    Custom2 = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2 as _,
    /// Add new kinds before this line.
    Size = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_SIZE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ExternalCorrelationKind, ExternalCorrelationKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Activity API configuration attribute.
pub enum ActivityAttribute {
    /// Per-context device buffer size, in bytes, for concurrent kernel, memory copy,
    /// and memory set activity data.
    ///
    /// The attribute value is a `usize`. Larger buffers reduce dropped timestamps but reserve
    /// more device memory. The default is 3,200,000 bytes. This only affects future buffer
    /// allocations, so set it before CUDA initialization or context creation when possible.
    ///
    /// Device buffer count is controlled by [`ActivityAttribute::DeviceBufferPoolLimit`], and
    /// pre-allocation is controlled by [`ActivityAttribute::DeviceBufferPreAllocateValue`].
    DeviceBufferSize = sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE as _,
    /// Per-context device buffer size, in bytes, for CUDA Dynamic Parallelism activity data.
    ///
    /// The attribute value is a `usize`. Larger buffers reduce flushes but reserve more
    /// device memory. The default is 8 MiB and only future allocations are affected.
    DeviceBufferSizeCudaDynamicParallelism =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP as _,
    /// Maximum number of activity device buffers CUPTI may allocate per context.
    ///
    /// The attribute value is a `usize`. Raising the limit can help high-rate kernel, memory
    /// copy, and memory set workloads at the cost of a larger memory footprint. The default is
    /// 250 and only future allocations are affected.
    DeviceBufferPoolLimit =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT as _,
    /// Deprecated profiling semaphore pool size.
    ///
    /// The attribute value is a `usize`. CUPTI no longer uses profiling semaphore pools
    /// starting with CUDA 12.3.
    #[deprecated]
    ProfilingSemaphorePoolSize =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE as _,
    /// Deprecated maximum number of profiling semaphore pools per context.
    ///
    /// The attribute value is a `usize`. CUPTI no longer uses profiling semaphore pools
    /// starting with CUDA 12.3.
    #[deprecated]
    ProfilingSemaphorePoolLimit =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT as _,
    /// Controls whether callers provide zero-initialized activity buffers.
    ///
    /// The attribute value is a `u8`. Zero means CUPTI clears buffers before filling them;
    /// nonzero means the buffer callback must return an already-zeroed buffer. The default is 0.
    ZeroedOutActivityBuffer =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER as _,
    /// Number of activity device buffers to pre-allocate during context initialization.
    ///
    /// The attribute value is a `usize` and must be lower than
    /// [`ActivityAttribute::DeviceBufferPoolLimit`]. The default is 3.
    DeviceBufferPreAllocateValue =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE as _,
    /// Deprecated number of profiling semaphore pools to pre-allocate during context initialization.
    ///
    /// The attribute value is a `usize`. CUPTI no longer uses profiling semaphore pools
    /// starting with CUDA 12.3.
    #[deprecated]
    ProfilingSemaphorePreAllocateValue =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE
            as _,
    /// Chooses page-locked host memory for activity profiling buffers.
    ///
    /// The attribute value is a `u8`. Zero selects device memory, nonzero selects pinned host
    /// memory. Pinned host buffers are not supported on confidential-computing devices.
    MemAllocationTypeHostPinned =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED as _,
    /// Stores activity records in per-thread activity buffers.
    ///
    /// The attribute value is a `u8`. Set it before registering activity buffer callbacks and
    /// before enabling activity kinds. The default is 1.
    PerThreadActivityBuffer =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER as _,
    /// Per-context device buffer size, in bytes, for device graph activity data.
    ///
    /// The attribute value is a `usize`. Larger buffers reduce flushes but reserve more
    /// device memory. The default is 16 MiB and only future allocations are affected.
    DeviceBufferSizeDeviceGraphs =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_DEVICE_GRAPHS as _,
    /// Enables user-defined activity records.
    ///
    /// The attribute value is a `u8`. When enabled, CUPTI emits only fields selected for
    /// collection by the user. This requires CUPTI API version 13.2 or newer and must be set
    /// before enabling any activity kinds.
    UserDefinedRecords =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_USER_DEFINED_RECORDS as _,
    /// Reports or configures multiple-subscriber support.
    ///
    /// The attribute value is a `u8`: 0 disables multiple subscribers, 1 enables them, and 2
    /// means no earlier subscriber has chosen a value. Query through the
    /// activity attribute API but is selected during subscription.
    MultipleSubscriberState =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_MULTIPLE_SUBSCRIBER_STATE as _,
    /// Enables the Hardware Events System globally for activity collection.
    ///
    /// The attribute value is a `u8`. Set it before any subscriber enables activity kinds.
    EnableHes = sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_HES as _,
    /// Enables allocation source library tracking for a subscriber.
    ///
    /// The attribute value is a `u8`. It cannot be combined with the older
    /// allocation-source enable API when multiple subscribers are allowed.
    EnableAllocationSourceTracking =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_ALLOCATION_SOURCE_TRACKING as _,
    /// Enables kernel latency timestamp collection for a subscriber.
    ///
    /// The attribute value is a `u8`. The default is disabled.
    EnableKernelLatencyTimestamps =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LATENCY_TIMESTAMPS as _,
    /// Enables synchronization records even for synchronization calls that return errors.
    ///
    /// The attribute value is a `u8`. The default is disabled.
    EnableAllSyncRecords =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_ALL_SYNC_RECORDS as _,
    /// Enables device timestamp collection for CUDA event activity records.
    ///
    /// The attribute value is a `u8`. The default is disabled.
    EnableCudaEventDeviceTimestamps =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_CUDA_EVENT_DEVICE_TIMESTAMPS as _,
    /// Enables kernel launch attribute collection for kernel activity records.
    ///
    /// The attribute value is a `u8`. The default is disabled.
    EnableKernelLaunchAttributes =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LAUNCH_ATTRIBUTES as _,
    /// Enables device graph trace collection for graph activity records.
    ///
    /// The attribute value is a `u8`. The default is disabled.
    EnableDeviceGraphTrace =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_DEVICE_GRAPH_TRACE as _,
    /// Enables graph-level tracing in multi-subscriber mode.
    ///
    /// The attribute value is a `u8`. The default is disabled.
    EnableMultiSubscriberGraphLevelTrace =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_MULTI_SUBSCRIBER_GRAPH_TRACE as _,
    /// Selects the thread ID source used in activity records.
    ///
    /// The attribute value is an [`ActivityThreadIdType`]. The default is
    /// [`ActivityThreadIdType::Default`].
    ThreadIdType = sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_THREAD_ID_TYPE as _,
    /// Registers or removes the timestamp callback used for activity records.
    ///
    /// The attribute value is a CUPTI timestamp callback pointer; null unregisters the callback
    /// and restores CUPTI's default CPU timer.
    TimestampCallback = sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_TIMESTAMP_CALLBACK as _,
    /// Configures CUDA in Graphics (CIG) mode for activity collection.
    ///
    /// The attribute value is a `u8`.
    CigMode = sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_ENABLE_CIG_MODE as _,
    /// Reserved integer sentinel from CUPTI.
    DeviceBufferForceInt =
        sys::CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityAttribute, ActivityAttribute);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// State of CUPTI multiple-subscriber support.
pub enum MultipleSubscriberState {
    /// Disabled state returned by CUPTI.
    Disabled,
    /// Boolean enabled state returned by CUPTI.
    Enabled,
    /// Unset state returned by CUPTI.
    Unset,
    /// Unknown value decoded from CUPTI.
    Unknown(u8),
}

impl From<u8> for MultipleSubscriberState {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Disabled,
            1 => Self::Enabled,
            2 => Self::Unset,
            value => Self::Unknown(value),
        }
    }
}

impl From<MultipleSubscriberState> for u8 {
    fn from(value: MultipleSubscriberState) -> Self {
        match value {
            MultipleSubscriberState::Disabled => 0,
            MultipleSubscriberState::Enabled => 1,
            MultipleSubscriberState::Unset => 2,
            MultipleSubscriberState::Unknown(value) => value,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Decoded value for an [`ActivityAttribute`] query.
pub enum ActivityAttributeValue {
    /// Bytes scalar value returned by CUPTI.
    Bytes(usize),
    /// Number of entries defined for this CUPTI enum.
    Count(usize),
    /// Boolean enabled state returned by CUPTI.
    Enabled(bool),
    /// Multiple Subscriber State value decoded from CUPTI.
    MultipleSubscriberState(MultipleSubscriberState),
    /// Thread ID Type value decoded from CUPTI.
    ThreadIdType(ActivityThreadIdType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Writable value for a supported [`ActivityAttribute`].
pub enum ActivityAttributeSetting {
    /// Bytes scalar value returned by CUPTI.
    Bytes(usize),
    /// Number of entries defined for this CUPTI enum.
    Count(usize),
    /// Boolean enabled state returned by CUPTI.
    Enabled(bool),
    /// Thread ID Type value decoded from CUPTI.
    ThreadIdType(ActivityThreadIdType),
}

macro_rules! activity_field_enum {
    ($name:ident, $raw:ty, { $($variant:ident = $value:ident,)+ }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
        #[repr(u32)]
        #[non_exhaustive]
        /// CUPTI activity field identifier enum.
        pub enum $name {
            $(
                #[doc = concat!("CUPTI activity field `", stringify!($value), "`.")]
                $variant = <$raw>::$value as _,
            )+
        }

        impl_enum_conversion!($raw, $name);

        impl From<$name> for ActivityFieldId {
            fn from(value: $name) -> Self {
                let raw: u32 = value.into();
                ActivityFieldId(raw as i32)
            }
        }

        impl_enum_display!($name, {
            $(Self::$variant => stringify!($value),)+
        });
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Raw activity field identifier used in activity field selection.
pub struct ActivityFieldId(i32);

impl ActivityFieldId {
    /// Returns the raw value decoded from CUPTI.
    pub const fn get(self) -> i32 {
        self.0
    }

    pub(crate) const fn as_raw(self) -> i32 {
        self.0
    }
}

impl Display for ActivityFieldId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

activity_field_enum!(ActivityApiField, sys::CUpti_ActivityApiFieldIds, {
    Kind = API_FIELD_KIND,
    CallbackId = API_FIELD_CBID,
    Start = API_FIELD_START,
    End = API_FIELD_END,
    ProcessId = API_FIELD_PROCESS_ID,
    ThreadId = API_FIELD_THREAD_ID,
    CorrelationId = API_FIELD_CORRELATION_ID,
    ReturnValue = API_FIELD_RETURN_VALUE,
});

activity_field_enum!(ActivityDeviceField, sys::CUpti_ActivityDeviceFieldIds, {
    Kind = DEVICE_FIELD_KIND,
    Flags = DEVICE_FIELD_FLAGS,
    GlobalMemoryBandwidth = DEVICE_FIELD_GLOBAL_MEMORY_BANDWIDTH,
    GlobalMemorySize = DEVICE_FIELD_GLOBAL_MEMORY_SIZE,
    ConstantMemorySize = DEVICE_FIELD_CONSTANT_MEMORY_SIZE,
    L2CacheSize = DEVICE_FIELD_L2_CACHE_SIZE,
    NumThreadsPerWarp = DEVICE_FIELD_NUM_THREADS_PER_WARP,
    CoreClockRate = DEVICE_FIELD_CORE_CLOCK_RATE,
    NumMemoryCopyEngines = DEVICE_FIELD_NUM_MEMCPY_ENGINES,
    NumMultiprocessors = DEVICE_FIELD_NUM_MULTIPROCESSORS,
    MaxIpc = DEVICE_FIELD_MAX_IPC,
    MaxWarpsPerMultiprocessor = DEVICE_FIELD_MAX_WARPS_PER_MULTIPROCESSOR,
    MaxBlocksPerMultiprocessor = DEVICE_FIELD_MAX_BLOCKS_PER_MULTIPROCESSOR,
    MaxSharedMemoryPerMultiprocessor = DEVICE_FIELD_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
    MaxRegistersPerMultiprocessor = DEVICE_FIELD_MAX_REGISTERS_PER_MULTIPROCESSOR,
    MaxRegistersPerBlock = DEVICE_FIELD_MAX_REGISTERS_PER_BLOCK,
    MaxSharedMemoryPerBlock = DEVICE_FIELD_MAX_SHARED_MEMORY_PER_BLOCK,
    MaxThreadsPerBlock = DEVICE_FIELD_MAX_THREADS_PER_BLOCK,
    MaxBlockDimX = DEVICE_FIELD_MAX_BLOCK_DIM_X,
    MaxBlockDimY = DEVICE_FIELD_MAX_BLOCK_DIM_Y,
    MaxBlockDimZ = DEVICE_FIELD_MAX_BLOCK_DIM_Z,
    MaxGridDimX = DEVICE_FIELD_MAX_GRID_DIM_X,
    MaxGridDimY = DEVICE_FIELD_MAX_GRID_DIM_Y,
    MaxGridDimZ = DEVICE_FIELD_MAX_GRID_DIM_Z,
    ComputeCapabilityMajor = DEVICE_FIELD_COMPUTE_CAPABILITY_MAJOR,
    ComputeCapabilityMinor = DEVICE_FIELD_COMPUTE_CAPABILITY_MINOR,
    Id = DEVICE_FIELD_ID,
    EccEnabled = DEVICE_FIELD_ECC_ENABLED,
    Uuid = DEVICE_FIELD_UUID,
    Name = DEVICE_FIELD_NAME,
    IsCudaVisible = DEVICE_FIELD_IS_CUDA_VISIBLE,
    IsMigEnabled = DEVICE_FIELD_IS_MIG_ENABLED,
    GpuInstanceId = DEVICE_FIELD_GPU_INSTANCE_ID,
    ComputeInstanceId = DEVICE_FIELD_COMPUTE_INSTANCE_ID,
    MigUuid = DEVICE_FIELD_MIG_UUID,
    IsNonUniformMemoryAccessNode = DEVICE_FIELD_IS_NUMA_NODE,
    NonUniformMemoryAccessId = DEVICE_FIELD_NUMA_ID,
    NumTpcs = DEVICE_FIELD_NUM_TPCS,
    Max = DEVICE_FIELD_MAX,
});

activity_field_enum!(ActivityMemoryCopyField, sys::CUpti_ActivityMemcpyFieldIds, {
    Kind = MEMCPY_FIELD_KIND,
    CopyKind = MEMCPY_FIELD_COPY_KIND,
    SourceKind = MEMCPY_FIELD_SRC_KIND,
    DestinationKind = MEMCPY_FIELD_DST_KIND,
    Flags = MEMCPY_FIELD_FLAGS,
    Bytes = MEMCPY_FIELD_BYTES,
    Start = MEMCPY_FIELD_START,
    End = MEMCPY_FIELD_END,
    DeviceId = MEMCPY_FIELD_DEVICE_ID,
    ContextId = MEMCPY_FIELD_CONTEXT_ID,
    StreamId = MEMCPY_FIELD_STREAM_ID,
    CorrelationId = MEMCPY_FIELD_CORRELATION_ID,
    GraphNodeId = MEMCPY_FIELD_GRAPH_NODE_ID,
    GraphId = MEMCPY_FIELD_GRAPH_ID,
    ChannelId = MEMCPY_FIELD_CHANNEL_ID,
    ChannelType = MEMCPY_FIELD_CHANNEL_TYPE,
    IsDeviceLaunched = MEMCPY_FIELD_IS_DEVICE_LAUNCHED,
    CopyCount = MEMCPY_FIELD_COPY_COUNT,
});

activity_field_enum!(ActivityPeerMemoryCopyField, sys::CUpti_ActivityMemcpy2FieldIds, {
    Kind = MEMCPY2_FIELD_KIND,
    CopyKind = MEMCPY2_FIELD_COPY_KIND,
    SourceKind = MEMCPY2_FIELD_SRC_KIND,
    DestinationKind = MEMCPY2_FIELD_DST_KIND,
    Flags = MEMCPY2_FIELD_FLAGS,
    Bytes = MEMCPY2_FIELD_BYTES,
    Start = MEMCPY2_FIELD_START,
    End = MEMCPY2_FIELD_END,
    DeviceId = MEMCPY2_FIELD_DEVICE_ID,
    ContextId = MEMCPY2_FIELD_CONTEXT_ID,
    StreamId = MEMCPY2_FIELD_STREAM_ID,
    SourceDeviceId = MEMCPY2_FIELD_SRC_DEVICE_ID,
    SourceContextId = MEMCPY2_FIELD_SRC_CONTEXT_ID,
    DestinationDeviceId = MEMCPY2_FIELD_DST_DEVICE_ID,
    DestinationContextId = MEMCPY2_FIELD_DST_CONTEXT_ID,
    CorrelationId = MEMCPY2_FIELD_CORRELATION_ID,
    GraphNodeId = MEMCPY2_FIELD_GRAPH_NODE_ID,
    GraphId = MEMCPY2_FIELD_GRAPH_ID,
    ChannelId = MEMCPY2_FIELD_CHANNEL_ID,
    ChannelType = MEMCPY2_FIELD_CHANNEL_TYPE,
    Max = MEMCPY2_FIELD_MAX,
});

activity_field_enum!(ActivityMemorySetField, sys::CUpti_ActivityMemsetFieldIds, {
    Kind = MEMSET_FIELD_KIND,
    Value = MEMSET_FIELD_VALUE,
    Bytes = MEMSET_FIELD_BYTES,
    Start = MEMSET_FIELD_START,
    End = MEMSET_FIELD_END,
    DeviceId = MEMSET_FIELD_DEVICE_ID,
    ContextId = MEMSET_FIELD_CONTEXT_ID,
    StreamId = MEMSET_FIELD_STREAM_ID,
    CorrelationId = MEMSET_FIELD_CORRELATION_ID,
    Flags = MEMSET_FIELD_FLAGS,
    MemoryKind = MEMSET_FIELD_MEMORY_KIND,
    GraphNodeId = MEMSET_FIELD_GRAPH_NODE_ID,
    GraphId = MEMSET_FIELD_GRAPH_ID,
    ChannelId = MEMSET_FIELD_CHANNEL_ID,
    ChannelType = MEMSET_FIELD_CHANNEL_TYPE,
    IsDeviceLaunched = MEMSET_FIELD_IS_DEVICE_LAUNCHED,
});

activity_field_enum!(ActivityMemoryField, sys::CUpti_ActivityMemoryFieldIds, {
    Kind = MEMORY_FIELD_KIND,
    OperationType = MEMORY_FIELD_OPERATION_TYPE,
    MemoryKind = MEMORY_FIELD_MEMORY_KIND,
    CorrelationId = MEMORY_FIELD_CORRELATION_ID,
    Address = MEMORY_FIELD_ADDRESS,
    Bytes = MEMORY_FIELD_BYTES,
    Timestamp = MEMORY_FIELD_TIMESTAMP,
    ProcessId = MEMORY_FIELD_PROCESS_ID,
    DeviceId = MEMORY_FIELD_DEVICE_ID,
    ContextId = MEMORY_FIELD_CONTEXT_ID,
    StreamId = MEMORY_FIELD_STREAM_ID,
    IsAsynchronous = MEMORY_FIELD_IS_ASYNC,
    PoolType = MEMORY_FIELD_POOL_TYPE,
    PoolAddress = MEMORY_FIELD_POOL_ADDRESS,
    PoolReleaseThreshold = MEMORY_FIELD_POOL_RELEASE_THRESHOLD,
    PoolSize = MEMORY_FIELD_POOL_SIZE,
    PoolProcessId = MEMORY_FIELD_POOL_PROCESS_ID,
    PoolUtilizedSize = MEMORY_FIELD_POOL_UTILIZED_SIZE,
    Source = MEMORY_FIELD_SOURCE,
});

activity_field_enum!(
    ActivityMemoryPoolField,
    sys::CUpti_ActivityMemoryPoolFieldIds,
    {
        Kind = MEMORY_POOL_FIELD_KIND,
        OperationType = MEMORY_POOL_FIELD_OPERATION_TYPE,
        PoolType = MEMORY_POOL_FIELD_POOL_TYPE,
        CorrelationId = MEMORY_POOL_FIELD_CORRELATION_ID,
        ProcessId = MEMORY_POOL_FIELD_PROCESS_ID,
        DeviceId = MEMORY_POOL_FIELD_DEVICE_ID,
        MinBytesToKeep = MEMORY_POOL_FIELD_MIN_BYTES_TO_KEEP,
        Address = MEMORY_POOL_FIELD_ADDRESS,
        Size = MEMORY_POOL_FIELD_SIZE,
        ReleaseThreshold = MEMORY_POOL_FIELD_RELEASE_THRESHOLD,
        Timestamp = MEMORY_POOL_FIELD_TIMESTAMP,
        UtilizedSize = MEMORY_POOL_FIELD_UTILIZED_SIZE,
        IsManagedPool = MEMORY_POOL_FIELD_IS_MANAGED_POOL,
    }
);

activity_field_enum!(ActivityGraphTraceField, sys::CUpti_ActivityGraphTraceFieldIds, {
    Kind = GRAPH_TRACE_FIELD_KIND,
    CorrelationId = GRAPH_TRACE_FIELD_CORRELATION_ID,
    Start = GRAPH_TRACE_FIELD_START,
    End = GRAPH_TRACE_FIELD_END,
    DeviceId = GRAPH_TRACE_FIELD_DEVICE_ID,
    GraphId = GRAPH_TRACE_FIELD_GRAPH_ID,
    ContextId = GRAPH_TRACE_FIELD_CONTEXT_ID,
    StreamId = GRAPH_TRACE_FIELD_STREAM_ID,
    EndDeviceId = GRAPH_TRACE_FIELD_END_DEVICE_ID,
    EndContextId = GRAPH_TRACE_FIELD_END_CONTEXT_ID,
});

activity_field_enum!(
    ActivityDeviceGraphTraceField,
    sys::CUpti_ActivityDeviceGraphTraceFieldIds,
    {
        Kind = DEVICE_GRAPH_TRACE_FIELD_KIND,
        DeviceId = DEVICE_GRAPH_TRACE_FIELD_DEVICE_ID,
        Start = DEVICE_GRAPH_TRACE_FIELD_START,
        End = DEVICE_GRAPH_TRACE_FIELD_END,
        GraphId = DEVICE_GRAPH_TRACE_FIELD_GRAPH_ID,
        LauncherGraphId = DEVICE_GRAPH_TRACE_FIELD_LAUNCHER_GRAPH_ID,
        DeviceLaunchMode = DEVICE_GRAPH_TRACE_FIELD_DEVICE_LAUNCH_MODE,
        ContextId = DEVICE_GRAPH_TRACE_FIELD_CONTEXT_ID,
        StreamId = DEVICE_GRAPH_TRACE_FIELD_STREAM_ID,
    }
);

activity_field_enum!(
    ActivityGraphHostNodeField,
    sys::CUpti_ActivityGraphHostNodeFieldIds,
    {
        Kind = GRAPH_HOST_NODE_FIELD_KIND,
        StreamId = GRAPH_HOST_NODE_FIELD_STREAM_ID,
        ContextId = GRAPH_HOST_NODE_FIELD_CONTEXT_ID,
        DeviceId = GRAPH_HOST_NODE_FIELD_DEVICE_ID,
        CorrelationId = GRAPH_HOST_NODE_FIELD_CORRELATION_ID,
        GraphId = GRAPH_HOST_NODE_FIELD_GRAPH_ID,
        GraphNodeId = GRAPH_HOST_NODE_FIELD_GRAPH_NODE_ID,
        ProcessId = GRAPH_HOST_NODE_FIELD_PROCESS_ID,
        ThreadId = GRAPH_HOST_NODE_FIELD_THREAD_ID,
        Start = GRAPH_HOST_NODE_FIELD_START,
        End = GRAPH_HOST_NODE_FIELD_END,
    }
);

activity_field_enum!(ActivityHostLaunchField, sys::CUpti_ActivityHostLaunchFieldIds, {
    Kind = HOST_LAUNCH_FIELD_KIND,
    StreamId = HOST_LAUNCH_FIELD_STREAM_ID,
    ContextId = HOST_LAUNCH_FIELD_CONTEXT_ID,
    DeviceId = HOST_LAUNCH_FIELD_DEVICE_ID,
    CorrelationId = HOST_LAUNCH_FIELD_CORRELATION_ID,
    ProcessId = HOST_LAUNCH_FIELD_PROCESS_ID,
    ThreadId = HOST_LAUNCH_FIELD_THREAD_ID,
    Start = HOST_LAUNCH_FIELD_START,
    End = HOST_LAUNCH_FIELD_END,
});

activity_field_enum!(
    ActivityComputeEngineContextSwitchField,
    sys::CUpti_ActivityComputeEngineCtxSwitchFieldIds,
    {
        Kind = COMPUTE_ENGINE_CTX_SWITCH_FIELD_KIND,
        ContextId = COMPUTE_ENGINE_CTX_SWITCH_FIELD_CONTEXT_ID,
        Timestamp = COMPUTE_ENGINE_CTX_SWITCH_FIELD_TIMESTAMP,
        OperationType = COMPUTE_ENGINE_CTX_SWITCH_FIELD_OPERATION_TYPE,
    }
);

activity_field_enum!(
    ActivityGreenContextField,
    sys::CUpti_ActivityGreenContextFieldIds,
    {
        Kind = GREEN_CONTEXT_FIELD_KIND,
        ContextId = GREEN_CONTEXT_FIELD_CONTEXT_ID,
        ParentContextId = GREEN_CONTEXT_FIELD_PARENT_CONTEXT_ID,
        DeviceId = GREEN_CONTEXT_FIELD_DEVICE_ID,
        NumMultiprocessors = GREEN_CONTEXT_FIELD_NUM_MULTIPROCESSORS,
        NumTpcs = GREEN_CONTEXT_FIELD_NUM_TPCS,
        LogicalTpcMaskSize = GREEN_CONTEXT_FIELD_LOGICAL_TPC_MASK_SIZE,
        LogicalTpcMask = GREEN_CONTEXT_FIELD_LOGICAL_TPC_MASK,
    }
);

activity_field_enum!(ActivityNameField, sys::CUpti_ActivityNameFieldIds, {
    Kind = NAME_FIELD_KIND,
    ObjectKind = NAME_FIELD_OBJECT_KIND,
    ObjectId = NAME_FIELD_OBJECT_ID,
    Name = NAME_FIELD_NAME,
});

activity_field_enum!(ActivityMarkerField, sys::CUpti_ActivityMarkerFieldIds, {
    Kind = MARKER_FIELD_KIND,
    Flags = MARKER_FIELD_FLAGS,
    Timestamp = MARKER_FIELD_TIMESTAMP,
    Id = MARKER_FIELD_ID,
    ProcessId = MARKER_FIELD_PROCESS_ID,
    ThreadId = MARKER_FIELD_THREAD_ID,
    Name = MARKER_FIELD_NAME,
    Domain = MARKER_FIELD_DOMAIN,
});

activity_field_enum!(
    ActivityMarkerDataField,
    sys::CUpti_ActivityMarkerDataFieldIds,
    {
        Kind = MARKER_DATA_FIELD_KIND,
        Flags = MARKER_DATA_FIELD_FLAGS,
        Id = MARKER_DATA_FIELD_ID,
        PayloadKind = MARKER_DATA_FIELD_PAYLOAD_KIND,
        Payload = MARKER_DATA_FIELD_PAYLOAD,
        Color = MARKER_DATA_FIELD_COLOR,
        Category = MARKER_DATA_FIELD_CATEGORY,
        CuptiDomainId = MARKER_DATA_FIELD_CUPTI_DOMAIN_ID,
    }
);

activity_field_enum!(
    ActivityExternalCorrelationField,
    sys::CUpti_ActivityExternalCorrelationFieldIds,
    {
        Kind = EXTERNAL_CORRELATION_FIELD_KIND,
        ExternalKind = EXTERNAL_CORRELATION_FIELD_EXTERNAL_KIND,
        ExternalId = EXTERNAL_CORRELATION_FIELD_EXTERNAL_ID,
        CorrelationId = EXTERNAL_CORRELATION_FIELD_CORRELATION_ID,
    }
);

activity_field_enum!(ActivityCudaEventField, sys::CUpti_ActivityCudaEventFieldIds, {
    Kind = CUDA_EVENT_FIELD_KIND,
    CorrelationId = CUDA_EVENT_FIELD_CORRELATION_ID,
    ContextId = CUDA_EVENT_FIELD_CONTEXT_ID,
    StreamId = CUDA_EVENT_FIELD_STREAM_ID,
    EventId = CUDA_EVENT_FIELD_EVENT_ID,
    DeviceId = CUDA_EVENT_FIELD_DEVICE_ID,
    DeviceTimestamp = CUDA_EVENT_FIELD_DEVICE_TIMESTAMP,
    CudaEventSyncId = CUDA_EVENT_FIELD_CUDA_EVENT_SYNC_ID,
});

activity_field_enum!(ActivityOverheadField, sys::CUpti_ActivityOverheadFieldIds, {
    Kind = OVERHEAD_FIELD_KIND,
    OverheadKind = OVERHEAD_FIELD_OVERHEAD_KIND,
    ProcessId = OVERHEAD_FIELD_PROCESS_ID,
    ThreadId = OVERHEAD_FIELD_THREAD_ID,
    Start = OVERHEAD_FIELD_START,
    End = OVERHEAD_FIELD_END,
    CorrelationId = OVERHEAD_FIELD_CORRELATION_ID,
    OverheadData = OVERHEAD_FIELD_OVERHEAD_DATA,
});

activity_field_enum!(ActivityUvmCounterField, sys::CUpti_ActivityUvmCounterFieldIds, {
    Kind = UVM_COUNTER_FIELD_KIND,
    CounterKind = UVM_COUNTER_FIELD_COUNTER_KIND,
    Value = UVM_COUNTER_FIELD_VALUE,
    Start = UVM_COUNTER_FIELD_START,
    End = UVM_COUNTER_FIELD_END,
    Address = UVM_COUNTER_FIELD_ADDRESS,
    SourceId = UVM_COUNTER_FIELD_SRC_ID,
    DestinationId = UVM_COUNTER_FIELD_DST_ID,
    StreamId = UVM_COUNTER_FIELD_STREAM_ID,
    ProcessId = UVM_COUNTER_FIELD_PROCESS_ID,
    Flags = UVM_COUNTER_FIELD_FLAGS,
    Processors = UVM_COUNTER_FIELD_PROCESSORS,
    Max = UVM_COUNTER_FIELD_MAX,
});

activity_field_enum!(ActivityContextField, sys::CUpti_ActivityContextFieldIds, {
    Kind = CONTEXT_FIELD_KIND,
    ContextId = CONTEXT_FIELD_CONTEXT_ID,
    DeviceId = CONTEXT_FIELD_DEVICE_ID,
    ComputeApiKind = CONTEXT_FIELD_COMPUTE_API_KIND,
    NullStreamId = CONTEXT_FIELD_NULL_STREAM_ID,
    ParentContextId = CONTEXT_FIELD_PARENT_CONTEXT_ID,
    IsGreenContext = CONTEXT_FIELD_IS_GREEN_CONTEXT,
    NumMultiprocessors = CONTEXT_FIELD_NUM_MULTIPROCESSORS,
    CigMode = CONTEXT_FIELD_CIG_MODE,
    ProcessId = CONTEXT_FIELD_PROCESS_ID,
});

activity_field_enum!(ActivityKernelField, sys::CUpti_ActivityKernelFieldIds, {
    Kind = KERNEL_FIELD_KIND,
    CacheConfigRequested = KERNEL_FIELD_CACHE_CONFIG_REQUESTED,
    CacheConfigExecuted = KERNEL_FIELD_CACHE_CONFIG_EXECUTED,
    SharedMemoryConfig = KERNEL_FIELD_SHARED_MEMORY_CONFIG,
    RegistersPerThread = KERNEL_FIELD_REGISTERS_PER_THREAD,
    PartitionedGlobalCacheRequested = KERNEL_FIELD_PARTITIONED_GLOBAL_CACHE_REQUESTED,
    PartitionedGlobalCacheExecuted = KERNEL_FIELD_PARTITIONED_GLOBAL_CACHE_EXECUTED,
    Start = KERNEL_FIELD_START,
    End = KERNEL_FIELD_END,
    Completed = KERNEL_FIELD_COMPLETED,
    DeviceId = KERNEL_FIELD_DEVICE_ID,
    ContextId = KERNEL_FIELD_CONTEXT_ID,
    StreamId = KERNEL_FIELD_STREAM_ID,
    GridX = KERNEL_FIELD_GRID_X,
    GridY = KERNEL_FIELD_GRID_Y,
    GridZ = KERNEL_FIELD_GRID_Z,
    BlockX = KERNEL_FIELD_BLOCK_X,
    BlockY = KERNEL_FIELD_BLOCK_Y,
    BlockZ = KERNEL_FIELD_BLOCK_Z,
    StaticSharedMemory = KERNEL_FIELD_STATIC_SHARED_MEMORY,
    DynamicSharedMemory = KERNEL_FIELD_DYNAMIC_SHARED_MEMORY,
    LocalMemoryPerThread = KERNEL_FIELD_LOCAL_MEMORY_PER_THREAD,
    CorrelationId = KERNEL_FIELD_CORRELATION_ID,
    GridId = KERNEL_FIELD_GRID_ID,
    Name = KERNEL_FIELD_NAME,
    Queued = KERNEL_FIELD_QUEUED,
    Submitted = KERNEL_FIELD_SUBMITTED,
    LaunchType = KERNEL_FIELD_LAUNCH_TYPE,
    IsSharedMemoryCarveoutRequested = KERNEL_FIELD_IS_SHARED_MEMORY_CARVEOUT_REQUESTED,
    SharedMemoryCarveoutRequested = KERNEL_FIELD_SHARED_MEMORY_CARVEOUT_REQUESTED,
    SharedMemoryExecuted = KERNEL_FIELD_SHARED_MEMORY_EXECUTED,
    GraphNodeId = KERNEL_FIELD_GRAPH_NODE_ID,
    SharedMemoryLimitConfig = KERNEL_FIELD_SHMEM_LIMIT_CONFIG,
    GraphId = KERNEL_FIELD_GRAPH_ID,
    AccessPolicyWindow = KERNEL_FIELD_ACCESS_POLICY_WINDOW,
    ChannelId = KERNEL_FIELD_CHANNEL_ID,
    ChannelType = KERNEL_FIELD_CHANNEL_TYPE,
    ClusterX = KERNEL_FIELD_CLUSTER_X,
    ClusterY = KERNEL_FIELD_CLUSTER_Y,
    ClusterZ = KERNEL_FIELD_CLUSTER_Z,
    ClusterSchedulingPolicy = KERNEL_FIELD_CLUSTER_SCHEDULING_POLICY,
    LocalMemoryTotal = KERNEL_FIELD_LOCAL_MEMORY_TOTAL,
    MaxPotentialClusterSize = KERNEL_FIELD_MAX_POTENTIAL_CLUSTER_SIZE,
    MaxActiveClusters = KERNEL_FIELD_MAX_ACTIVE_CLUSTERS,
    IsDeviceLaunched = KERNEL_FIELD_IS_DEVICE_LAUNCHED,
    LaunchPriority = KERNEL_FIELD_LAUNCH_PRIORITY,
});

activity_field_enum!(
    ActivityMemoryDecompressField,
    sys::CUpti_ActivityMemDecompressFieldIds,
    {
        Kind = MEM_DECOMPRESS_FIELD_KIND,
        DeviceId = MEM_DECOMPRESS_FIELD_DEVICE_ID,
        ContextId = MEM_DECOMPRESS_FIELD_CONTEXT_ID,
        StreamId = MEM_DECOMPRESS_FIELD_STREAM_ID,
        ChannelId = MEM_DECOMPRESS_FIELD_CHANNEL_ID,
        ChannelType = MEM_DECOMPRESS_FIELD_CHANNEL_TYPE,
        CorrelationId = MEM_DECOMPRESS_FIELD_CORRELATION_ID,
        NumberOfOperations = MEM_DECOMPRESS_FIELD_NUMBER_OF_OPERATIONS,
        SourceBytes = MEM_DECOMPRESS_FIELD_SOURCE_BYTES,
        Start = MEM_DECOMPRESS_FIELD_START,
        End = MEM_DECOMPRESS_FIELD_END,
    }
);

activity_field_enum!(
    ActivityConfidentialComputeRotationField,
    sys::CUpti_ActivityConfidentialComputeRotationFieldIds,
    {
        Kind = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_KIND,
        EventType = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_EVENT_TYPE,
        DeviceId = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_DEVICE_ID,
        ContextId = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_CONTEXT_ID,
        ChannelId = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_CHANNEL_ID,
        ChannelType = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_CHANNEL_TYPE,
        Timestamp = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_TIMESTAMP,
        Max = CONFIDENTIAL_COMPUTE_ROTATION_FIELD_MAX,
    }
);

activity_field_enum!(ActivityJitField, sys::CUpti_ActivityJitFieldIds, {
    Kind = JIT_FIELD_KIND,
    EntryType = JIT_FIELD_ENTRY_TYPE,
    OperationType = JIT_FIELD_OPERATION_TYPE,
    DeviceId = JIT_FIELD_DEVICE_ID,
    Start = JIT_FIELD_START,
    End = JIT_FIELD_END,
    CorrelationId = JIT_FIELD_CORRELATION_ID,
    OperationCorrelationId = JIT_FIELD_OPERATION_CORRELATION_ID,
    CacheSize = JIT_FIELD_CACHE_SIZE,
    CachePath = JIT_FIELD_CACHE_PATH,
    ProcessId = JIT_FIELD_PROCESS_ID,
    ThreadId = JIT_FIELD_THREAD_ID,
    Max = JIT_FIELD_MAX,
});

activity_field_enum!(ActivityStreamField, sys::CUpti_ActivityStreamFieldIds, {
    Kind = STREAM_FIELD_KIND,
    ContextId = STREAM_FIELD_CONTEXT_ID,
    StreamId = STREAM_FIELD_STREAM_ID,
    Priority = STREAM_FIELD_PRIORITY,
    Flag = STREAM_FIELD_FLAG,
    CorrelationId = STREAM_FIELD_CORRELATION_ID,
});

activity_field_enum!(
    ActivitySynchronizationField,
    sys::CUpti_ActivitySynchronizationFieldIds,
    {
        Kind = SYNCHRONIZATION_FIELD_KIND,
        SynchronizationType = SYNCHRONIZATION_FIELD_TYPE,
        Start = SYNCHRONIZATION_FIELD_START,
        End = SYNCHRONIZATION_FIELD_END,
        CorrelationId = SYNCHRONIZATION_FIELD_CORRELATION_ID,
        ContextId = SYNCHRONIZATION_FIELD_CONTEXT_ID,
        StreamId = SYNCHRONIZATION_FIELD_STREAM_ID,
        CudaEventId = SYNCHRONIZATION_FIELD_CUDA_EVENT_ID,
        CudaEventSyncId = SYNCHRONIZATION_FIELD_CUDA_EVENT_SYNC_ID,
        ReturnValue = SYNCHRONIZATION_FIELD_RETURN_VALUE,
    }
);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Activity field-selection configuration passed to CUPTI activity APIs.
pub struct ActivityConfig {
    field_ids: Vec<i32>,
}

impl ActivityConfig {
    /// Creates an activity configuration from selected fields.
    pub fn from_fields<I>(field_ids: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ActivityFieldId>,
    {
        Self {
            field_ids: field_ids
                .into_iter()
                .map(|field| field.into().as_raw())
                .collect(),
        }
    }

    /// Creates an activity configuration that selects all fields.
    pub fn all_fields() -> Self {
        Self {
            field_ids: Vec::new(),
        }
    }

    pub(crate) fn as_raw_mut(&mut self) -> sys::CUpti_ActivityConfig {
        let field_ids = if self.field_ids.is_empty() {
            std::ptr::null_mut()
        } else {
            self.field_ids.as_mut_ptr()
        };
        let field_selection = sys::CUpti_ActivityFieldSelection {
            structSize: size_of::<sys::CUpti_ActivityFieldSelection>() as u64,
            numFields: self.field_ids.len() as u64,
            pFieldIds: field_ids,
        };
        sys::CUpti_ActivityConfig {
            structSize: size_of::<sys::CUpti_ActivityConfig>() as u64,
            fieldSelection: field_selection,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Sampling period used by legacy CUPTI PC-sampling activity collection.
pub enum ActivityPcSamplingPeriod {
    /// The PC sampling period is not set.
    Invalid = sys::CUpti_ActivityPCSamplingPeriod::CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID as _,
    /// Minimum sampling period available on the device.
    Min = sys::CUpti_ActivityPCSamplingPeriod::CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN as _,
    /// Sampling period in lower range.
    Low = sys::CUpti_ActivityPCSamplingPeriod::CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW as _,
    /// Medium sampling period.
    Mid = sys::CUpti_ActivityPCSamplingPeriod::CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID as _,
    /// Sampling period in higher range.
    High = sys::CUpti_ActivityPCSamplingPeriod::CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH as _,
    /// Maximum sampling period available on the device.
    Max = sys::CUpti_ActivityPCSamplingPeriod::CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ActivityPCSamplingPeriod::CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityPCSamplingPeriod,
    ActivityPcSamplingPeriod
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Kind of environment data carried by [`crate::activity::ActivityEnvironment`].
pub enum ActivityEnvironmentKind {
    /// Unknown data.
    Unknown = sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_UNKNOWN as _,
    /// The environment data is related to speed.
    Speed = sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_SPEED as _,
    /// The environment data is related to temperature.
    Temperature = sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE as _,
    /// The environment data is related to power.
    Power = sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_POWER as _,
    /// The environment data is related to cooling.
    Cooling = sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_COOLING as _,
    /// Number of entries defined for this CUPTI enum.
    Count = sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_COUNT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityEnvironmentKind::CUPTI_ACTIVITY_ENVIRONMENT_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityEnvironmentKind, ActivityEnvironmentKind);

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    /// Clock-throttle reason flags reported by CUPTI environment records.
    pub struct EnvironmentClocksThrottleReasons: u32 {
        /// No clock throttling.
        const NONE = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_NONE as _;
        /// Nothing is running on the GPU and the clocks are dropping to idle state.
        const GPU_IDLE = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_GPU_IDLE as _;
        /// The GPU clocks are limited by a user specified limit.
        const USER_DEFINED_CLOCKS = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_USER_DEFINED_CLOCKS as _;
        /// A software power scaling algorithm is reducing the clocks below requested clocks.
        const SOFTWARE_POWER_CAP = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_SW_POWER_CAP as _;
        /// Hardware slowdown to reduce the clock by a factor of two or more is engaged.
        ///
        /// This is an indicator of one of the following: 1) Temperature is too high, 2) External power brake assertion is being triggered (e.g. by the system power supply), 3) Change in power state.
        const HARDWARE_SLOWDOWN = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN as _;
        /// Some unspecified factor is reducing the clocks.
        const UNKNOWN = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_UNKNOWN as _;
        /// Throttle reason is not supported for this GPU.
        const UNSUPPORTED = sys::CUpti_EnvironmentClocksThrottleReason::CUPTI_CLOCKS_THROTTLE_REASON_UNSUPPORTED as _;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Preemption activity kind.
pub enum ActivityPreemptionKind {
    /// The preemption kind is not known.
    Unknown = sys::CUpti_ActivityPreemptionKind::CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN as _,
    /// Preemption to save CDP block.
    Save = sys::CUpti_ActivityPreemptionKind::CUPTI_ACTIVITY_PREEMPTION_KIND_SAVE as _,
    /// Preemption to restore CDP block.
    Restore = sys::CUpti_ActivityPreemptionKind::CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityPreemptionKind::CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityPreemptionKind, ActivityPreemptionKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Stall reason reported by PC-sampling activity records.
pub enum ActivityPcSamplingStallReason {
    /// Invalid reason.
    Invalid = sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID as _,
    /// No stall, instruction is selected for issue.
    None = sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE as _,
    /// Warp is blocked because next instruction is not yet available, because of instruction cache miss, or because of branching effects.
    InstructionFetch =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH as _,
    /// Instruction is waiting on an arithmetic dependency.
    ExecutionDependency =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY as _,
    /// Warp is blocked because it is waiting for a memory access to complete.
    MemoryDependency =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY as _,
    /// Texture sub-system is fully utilized or has too many outstanding requests.
    Texture = sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE as _,
    /// Warp is blocked as it is waiting at __syncthreads() or at memory barrier.
    Sync = sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC as _,
    /// Warp is blocked waiting for **constant** memory and immediate memory access to complete.
    ConstantMemoryDependency =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY as _,
    /// Compute operation cannot be performed due to the required resources not being available.
    PipeBusy =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY as _,
    /// Warp is blocked because there are too many pending memory operations.
    MemoryThrottle =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE as _,
    /// Warp was ready to issue, but some other warp issued instead.
    NotSelected =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED as _,
    /// Miscellaneous reasons.
    Other = sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER as _,
    /// Sleeping.
    Sleeping =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ActivityPCSamplingStallReason::CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityPCSamplingStallReason,
    ActivityPcSamplingStallReason
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Deprecated unified-memory counter scope.
pub enum ActivityUnifiedMemoryCounterScope {
    /// The unified memory counter scope is not known.
    Unknown = sys::CUpti_ActivityUnifiedMemoryCounterScope::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN as _,
    /// Collect unified memory counter for single process on one device.
    ProcessSingleDevice = sys::CUpti_ActivityUnifiedMemoryCounterScope::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE as _,
    /// Collect unified memory counter for single process across all devices.
    ProcessAllDevices = sys::CUpti_ActivityUnifiedMemoryCounterScope::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES as _,
    /// Number of entries defined for this CUPTI enum.
    Count = sys::CUpti_ActivityUnifiedMemoryCounterScope::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityUnifiedMemoryCounterScope::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityUnifiedMemoryCounterScope,
    ActivityUnifiedMemoryCounterScope
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Unified-memory counter kind.
pub enum ActivityUnifiedMemoryCounterKind {
    /// The unified memory counter kind is not known.
    Unknown = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN as _,
    /// Number of bytes transferred from host to device.
    BytesTransferHostToDevice = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD as _,
    /// Number of bytes transferred from device to host.
    BytesTransferDeviceToHost = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH as _,
    /// Number of CPU page faults, this is only supported on 64 bit Linux and Mac platforms.
    CentralProcessingUnitPageFaultCount = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT as _,
    /// Number of GPU page faults, this is only supported on devices with compute capability 6.0 and higher and 64 bit Linux platforms.
    GpuPageFault = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT as _,
    /// Thrashing occurs when data is frequently accessed by multiple processors and has to be constantly migrated around to achieve data locality.
    ///
    /// In this case the overhead of migration may exceed the benefits of locality. This is only supported on 64 bit Linux platforms.
    Thrashing = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING as _,
    /// Throttling is a prevention technique used by the driver to avoid further thrashing.
    ///
    /// Here, the driver doesn’t service the fault for one of the contending processors for a specific period of time, so that the other processor can run at full-speed. This is only supported on 64 bit Linux platforms.
    Throttling = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING as _,
    /// In case throttling does not help, the driver tries to pin the memory to a processor for a specific period of time.
    ///
    /// One of the contending processors will have slow access to the memory, while the other will have fast access. This is only supported on 64 bit Linux platforms.
    RemoteMap = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP as _,
    /// Number of bytes transferred from one device to another device.
    ///
    /// This is only supported on 64 bit Linux platforms.
    BytesTransferDeviceToDevice = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD as _,
    /// Number of entries defined for this CUPTI enum.
    Count = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityUnifiedMemoryCounterKind::CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityUnifiedMemoryCounterKind,
    ActivityUnifiedMemoryCounterKind
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Access type reported for unified-memory page-fault counters.
pub enum ActivityUnifiedMemoryAccessType {
    /// The unified memory access type is not known.
    Unknown = sys::CUpti_ActivityUnifiedMemoryAccessType::CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN as _,
    /// The page fault was triggered by read memory instruction.
    Read = sys::CUpti_ActivityUnifiedMemoryAccessType::CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ as _,
    /// The page fault was triggered by write memory instruction.
    Write = sys::CUpti_ActivityUnifiedMemoryAccessType::CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE as _,
    /// The page fault was triggered by atomic memory instruction.
    Atomic = sys::CUpti_ActivityUnifiedMemoryAccessType::CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC as _,
    /// The page fault was triggered by memory prefetch operation.
    Prefetch = sys::CUpti_ActivityUnifiedMemoryAccessType::CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityUnifiedMemoryAccessType,
    ActivityUnifiedMemoryAccessType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Migration cause reported for unified-memory transfer counters.
pub enum ActivityUnifiedMemoryMigrationCause {
    /// The unified memory migration cause is not known.
    Unknown = sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN as _,
    /// The unified memory migrated due to an explicit call from the user e.g.
    ///
    /// cudaMemPrefetchAsync.
    User = sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER as _,
    /// The unified memory migrated to guarantee data coherence e.g.
    ///
    /// CPU/GPU faults on Pascal+ and kernel launch on pre-Pascal GPUs.
    Coherence = sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE as _,
    /// The unified memory was speculatively migrated by the UVM driver before being accessed by the destination processor to improve performance.
    Prefetch = sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH as _,
    /// The unified memory migrated to the CPU because it was evicted to make room for another block of memory on the GPU.
    Eviction = sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION as _,
    /// The unified memory migrated to another processor because of access counter notifications.
    ///
    /// Only frequently accessed pages are migrated between CPU and GPU, or between peer GPUs.
    AccessCounters = sys::CUpti_ActivityUnifiedMemoryMigrationCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityUnifiedMemoryMigrationCause,
    ActivityUnifiedMemoryMigrationCause
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Remote-map cause reported for unified-memory counters.
pub enum ActivityUnifiedMemoryRemoteMapCause {
    /// The cause of mapping to remote memory was unknown.
    Unknown = sys::CUpti_ActivityUnifiedMemoryRemoteMapCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN as _,
    /// Mapping to remote memory was added to maintain data coherence.
    Coherence = sys::CUpti_ActivityUnifiedMemoryRemoteMapCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE as _,
    /// Mapping to remote memory was added to prevent further thrashing.
    Thrashing = sys::CUpti_ActivityUnifiedMemoryRemoteMapCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING as _,
    /// Mapping to remote memory was added to enforce the hints specified by the programmer or by performance heuristics of the UVM driver.
    Policy = sys::CUpti_ActivityUnifiedMemoryRemoteMapCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY as _,
    /// Mapping to remote memory was added because there is no more memory available on the processor and eviction was not possible.
    OutOfMemory = sys::CUpti_ActivityUnifiedMemoryRemoteMapCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY as _,
    /// Mapping to remote memory was added after the memory was evicted to make room for another block of memory on the GPU.
    Eviction = sys::CUpti_ActivityUnifiedMemoryRemoteMapCause::CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_EVICTION as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityUnifiedMemoryRemoteMapCause,
    ActivityUnifiedMemoryRemoteMapCause
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// SASS instruction class reported by instruction activity records.
pub enum ActivityInstructionClass {
    /// The instruction class is not known.
    Unknown = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN as _,
    /// Represents a 32 bit floating point operation.
    Float32 = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32 as _,
    /// Represents a 64 bit floating point operation.
    Float64 = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64 as _,
    /// Represents an integer operation.
    Integer = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER as _,
    /// Represents a bit conversion operation.
    BitConversion =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_CONVERSION as _,
    /// Represents a control flow instruction.
    ControlFlow =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW as _,
    /// Represents a global load-store instruction.
    Global = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL as _,
    /// Represents a shared load-store instruction.
    Shared = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED as _,
    /// Represents a local load-store instruction.
    Local = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL as _,
    /// Represents a generic load-store instruction.
    Generic = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC as _,
    /// Represents a surface load-store instruction.
    Surface = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE as _,
    /// Represents a constant load instruction.
    Constant = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT as _,
    /// Represents a texture load-store instruction.
    Texture = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE as _,
    /// Represents a global atomic instruction.
    GlobalAtomic =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC as _,
    /// Represents a shared atomic instruction.
    SharedAtomic =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC as _,
    /// Represents a surface atomic instruction.
    SurfaceAtomic =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC as _,
    /// Represents a inter-thread communication instruction.
    InterThreadCommunication =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION as _,
    /// Represents a barrier instruction.
    Barrier = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER as _,
    /// Represents some miscellaneous instructions which do not fit in the above classification.
    Miscellaneous =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS as _,
    /// Represents a 16 bit floating point operation.
    Float16 = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16 as _,
    /// Represents uniform instruction.
    Uniform = sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNIFORM as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ActivityInstructionClass::CUPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityInstructionClass,
    ActivityInstructionClass
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Confidential-compute key-rotation event type.
pub enum ConfidentialComputeRotationEventType {
    /// Invalid value decoded from CUPTI.
    Invalid =
        sys::CUpti_ConfidentialComputeRotationEventType::CUPTI_CONFIDENTIAL_COMPUTE_INVALID_ROTATION_EVENT as _,
    /// This channel has been blocked from accepting new CUDA work so a key rotation can be done.
    ChannelBlocked =
        sys::CUpti_ConfidentialComputeRotationEventType::CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_BLOCKED as _,
    /// This channel remains blocked and all queued CUDA work has completed.
    ///
    /// Other clients or channels may cause delays in starting the key rotation.
    ChannelDrained =
        sys::CUpti_ConfidentialComputeRotationEventType::CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_DRAINED as _,
    /// Key rotations have completed and this channel is unblocked.
    ChannelUnblocked =
        sys::CUpti_ConfidentialComputeRotationEventType::CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_UNBLOCKED as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ConfidentialComputeRotationEventType::CUPTI_CONFIDENTIAL_COMPUTE_EVENT_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ConfidentialComputeRotationEventType,
    ConfidentialComputeRotationEventType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Device type used in topology activity records.
pub enum DeviceType {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_DevType::CUPTI_DEV_TYPE_INVALID as _,
    /// The device type is GPU.
    Gpu = sys::CUpti_DevType::CUPTI_DEV_TYPE_GPU as _,
    /// The device type is NVLink processing unit in CPU.
    Npu = sys::CUpti_DevType::CUPTI_DEV_TYPE_NPU as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_DevType::CUPTI_DEV_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_DevType, DeviceType);

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    /// Capability flags for a topology link.
    pub struct LinkFlags: u32 {
        /// Is peer to peer access supported by this link.
        const PEER_ACCESS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_PEER_ACCESS as _;
        /// Is system memory access supported by this link.
        const SYSTEM_MEMORY_ACCESS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_SYSMEM_ACCESS as _;
        /// Is peer atomic access supported by this link.
        const PEER_ATOMICS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_PEER_ATOMICS as _;
        /// Is system memory atomic access supported by this link.
        const SYSTEM_MEMORY_ATOMICS = sys::CUpti_LinkFlag::CUPTI_LINK_FLAG_SYSMEM_ATOMICS as _;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// PCIe topology record device type.
pub enum PcieDeviceType {
    /// PCIE GPU record.
    Gpu = sys::CUpti_PcieDeviceType::CUPTI_PCIE_DEVICE_TYPE_GPU as _,
    /// PCIE Bridge record.
    Bridge = sys::CUpti_PcieDeviceType::CUPTI_PCIE_DEVICE_TYPE_BRIDGE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_PcieDeviceType::CUPTI_PCIE_DEVICE_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_PcieDeviceType, PcieDeviceType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// OpenACC event kind reported by OpenACC activity records.
pub enum OpenAccEventKind {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_INVALID as _,
    /// Device Init value decoded from CUPTI.
    DeviceInit = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT as _,
    /// Device Shutdown value decoded from CUPTI.
    DeviceShutdown = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN as _,
    /// Runtime Shutdown value decoded from CUPTI.
    RuntimeShutdown = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN as _,
    /// Enqueue Launch value decoded from CUPTI.
    EnqueueLaunch = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH as _,
    /// Enqueue Upload value decoded from CUPTI.
    EnqueueUpload = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD as _,
    /// Enqueue Download value decoded from CUPTI.
    EnqueueDownload = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD as _,
    /// Wait value decoded from CUPTI.
    Wait = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_WAIT as _,
    /// Implicit Wait value decoded from CUPTI.
    ImplicitWait = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT as _,
    /// Compute Construct value decoded from CUPTI.
    ComputeConstruct = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT as _,
    /// Update value decoded from CUPTI.
    Update = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_UPDATE as _,
    /// Enter Data value decoded from CUPTI.
    EnterData = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_ENTER_DATA as _,
    /// Exit Data value decoded from CUPTI.
    ExitData = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_EXIT_DATA as _,
    /// Create value decoded from CUPTI.
    Create = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_CREATE as _,
    /// Delete value decoded from CUPTI.
    Delete = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_DELETE as _,
    /// Alloc value decoded from CUPTI.
    Alloc = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_ALLOC as _,
    /// Free value decoded from CUPTI.
    Free = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_FREE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_OpenAccEventKind::CUPTI_OPENACC_EVENT_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_OpenAccEventKind, OpenAccEventKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// OpenACC construct kind reported by OpenACC activity records.
pub enum OpenAccConstructKind {
    /// Unknown value decoded from CUPTI.
    Unknown = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN as _,
    /// Parallel value decoded from CUPTI.
    Parallel = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_PARALLEL as _,
    /// Kernels value decoded from CUPTI.
    Kernels = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_KERNELS as _,
    /// Loop value decoded from CUPTI.
    Loop = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_LOOP as _,
    /// Data value decoded from CUPTI.
    Data = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_DATA as _,
    /// Enter Data value decoded from CUPTI.
    EnterData = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_ENTER_DATA as _,
    /// Exit Data value decoded from CUPTI.
    ExitData = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_EXIT_DATA as _,
    /// Host Data value decoded from CUPTI.
    HostData = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_HOST_DATA as _,
    /// Atomic value decoded from CUPTI.
    Atomic = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_ATOMIC as _,
    /// Declare value decoded from CUPTI.
    Declare = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_DECLARE as _,
    /// Init value decoded from CUPTI.
    Init = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_INIT as _,
    /// Shutdown value decoded from CUPTI.
    Shutdown = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_SHUTDOWN as _,
    /// Set value decoded from CUPTI.
    Set = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_SET as _,
    /// Update value decoded from CUPTI.
    Update = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_UPDATE as _,
    /// Routine value decoded from CUPTI.
    Routine = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_ROUTINE as _,
    /// Wait value decoded from CUPTI.
    Wait = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_WAIT as _,
    /// Runtime Api value decoded from CUPTI.
    RuntimeApi = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_RUNTIME_API as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_OpenAccConstructKind::CUPTI_OPENACC_CONSTRUCT_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_OpenAccConstructKind, OpenAccConstructKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// OpenMP event kind reported by OpenMP activity records.
pub enum OpenMultiProcessingEventKind {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_INVALID as _,
    /// Parallel value decoded from CUPTI.
    Parallel = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_PARALLEL as _,
    /// Task value decoded from CUPTI.
    Task = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_TASK as _,
    /// Thread value decoded from CUPTI.
    Thread = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_THREAD as _,
    /// Idle utilization level.
    Idle = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_IDLE as _,
    /// Wait Barrier value decoded from CUPTI.
    WaitBarrier = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER as _,
    /// Wait Taskwait value decoded from CUPTI.
    WaitTaskwait = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_OpenMpEventKind::CUPTI_OPENMP_EVENT_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_OpenMpEventKind, OpenMultiProcessingEventKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// JIT input/output entry type reported by [`crate::activity::ActivityJit`].
pub enum ActivityJitEntryType {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_ActivityJitEntryType::CUPTI_ACTIVITY_JIT_ENTRY_INVALID as _,
    /// PTX to CUBIN.
    ParallelThreadExecutionToCudaBinary =
        sys::CUpti_ActivityJitEntryType::CUPTI_ACTIVITY_JIT_ENTRY_PTX_TO_CUBIN as _,
    /// NVVM-IR to PTX.
    NvidiaVirtualMachineIntermediateRepresentationToParallelThreadExecution =
        sys::CUpti_ActivityJitEntryType::CUPTI_ACTIVITY_JIT_ENTRY_NVVM_IR_TO_PTX as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityJitEntryType::CUPTI_ACTIVITY_JIT_ENTRY_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityJitEntryType, ActivityJitEntryType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// JIT operation type reported by [`crate::activity::ActivityJit`].
pub enum ActivityJitOperationType {
    /// Invalid value decoded from CUPTI.
    Invalid = sys::CUpti_ActivityJitOperationType::CUPTI_ACTIVITY_JIT_OPERATION_INVALID as _,
    /// Loaded from the compute cache.
    CacheLoad = sys::CUpti_ActivityJitOperationType::CUPTI_ACTIVITY_JIT_OPERATION_CACHE_LOAD as _,
    /// Stored in the compute cache.
    CacheStore = sys::CUpti_ActivityJitOperationType::CUPTI_ACTIVITY_JIT_OPERATION_CACHE_STORE as _,
    /// JIT compilation.
    Compile = sys::CUpti_ActivityJitOperationType::CUPTI_ACTIVITY_JIT_OPERATION_COMPILE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_ActivityJitOperationType::CUPTI_ACTIVITY_JIT_OPERATION_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_ActivityJitOperationType,
    ActivityJitOperationType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Configuration for legacy CUPTI PC-sampling activity collection.
pub struct ActivityPcSamplingConfig {
    /// Requested PC sampling period.
    pub sampling_period: ActivityPcSamplingPeriod,
    /// Additional PC sampling period value for APIs that support it.
    pub sampling_period2: u32,
}

impl ActivityPcSamplingConfig {
    /// Creates a PC sampling configuration with the requested period.
    pub const fn new(sampling_period: ActivityPcSamplingPeriod) -> Self {
        Self {
            sampling_period,
            sampling_period2: 0,
        }
    }

    pub(crate) fn to_raw(self) -> sys::CUpti_ActivityPCSamplingConfig {
        sys::CUpti_ActivityPCSamplingConfig {
            size: size_of::<sys::CUpti_ActivityPCSamplingConfig>() as u32,
            samplingPeriod: self.sampling_period.into(),
            samplingPeriod2: self.sampling_period2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Configuration for unified-memory activity counters.
pub struct ActivityUnifiedMemoryCounterConfig {
    /// Scope over which the unified memory counter is collected.
    pub scope: ActivityUnifiedMemoryCounterScope,
    /// Unified memory counter kind to configure.
    pub kind: ActivityUnifiedMemoryCounterKind,
    /// Device identifier used when the selected scope targets one device.
    pub device_id: DeviceId,
    /// Whether the selected unified memory counter is enabled.
    pub enable: bool,
}

impl ActivityUnifiedMemoryCounterConfig {
    /// Creates a unified memory counter configuration.
    pub const fn new(kind: ActivityUnifiedMemoryCounterKind, enable: bool) -> Self {
        Self {
            scope: ActivityUnifiedMemoryCounterScope::ProcessAllDevices,
            kind,
            device_id: DeviceId(0),
            enable,
        }
    }

    pub(crate) fn to_raw(self) -> sys::CUpti_ActivityUnifiedMemoryCounterConfig {
        #[allow(deprecated)]
        sys::CUpti_ActivityUnifiedMemoryCounterConfig {
            scope: self.scope.into(),
            kind: self.kind.into(),
            deviceId: self.device_id.as_raw(),
            enable: self.enable as u32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Automatic boost state reported for a CUDA context.
pub struct ActivityAutoBoostState {
    /// Whether automatic boost is enabled.
    pub enabled: bool,
    /// Process that requested the automatic boost state.
    pub process_id: ProcessId,
}

impl From<sys::CUpti_ActivityAutoBoostState> for ActivityAutoBoostState {
    fn from(value: sys::CUpti_ActivityAutoBoostState) -> Self {
        Self {
            enabled: value.enabled != 0,
            process_id: ProcessId::from(value.pid),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Thread identifier source used by CUPTI activity records.
pub enum ActivityThreadIdType {
    /// Default type Windows uses API GetCurrentThreadId() Linux/Mac/Android/QNX use POSIX pthread API pthread_self().
    Default = sys::CUpti_ActivityThreadIdType::CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT as _,
    /// This type is based on the system API available on the underlying platform and thread-id obtained is supposed to be unique for the process lifetime.
    ///
    /// Windows uses API GetCurrentThreadId() Linux uses syscall SYS_gettid Mac uses syscall SYS_thread_selfid Android/QNX use gettid().
    System = sys::CUpti_ActivityThreadIdType::CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM as _,
    /// Add new enums before this field.
    Size = sys::CUpti_ActivityThreadIdType::CUPTI_ACTIVITY_THREAD_ID_TYPE_SIZE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_ActivityThreadIdType::CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_ActivityThreadIdType, ActivityThreadIdType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// General category assigned to a CUPTI metric.
pub enum MetricCategory {
    /// A memory related metric.
    Memory = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_MEMORY as _,
    /// An instruction related metric.
    Instruction = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_INSTRUCTION as _,
    /// A multiprocessor related metric.
    Multiprocessor = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_MULTIPROCESSOR as _,
    /// A cache related metric.
    Cache = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_CACHE as _,
    /// A texture related metric.
    Texture = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_TEXTURE as _,
    /// A Nvlink related metric.
    NvLink = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_NVLINK as _,
    /// A PCIe related metric.
    Pcie = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_PCIE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_MetricCategory::CUPTI_METRIC_CATEGORY_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_MetricCategory, MetricCategory);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Evaluation mode supported by a CUPTI metric.
pub enum MetricEvaluationMode {
    /// If this bit is set, the metric can be profiled for each instance of the domain.
    ///
    /// Metric evaluation uses event values for one event-domain instance at a time.
    PerInstance = sys::CUpti_MetricEvaluationMode::CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE as _,
    /// If this bit is set, the metric can be profiled over all instances.
    ///
    /// Metric evaluation uses event values aggregated across all instances of the domain.
    Aggregate = sys::CUpti_MetricEvaluationMode::CUPTI_METRIC_EVALUATION_MODE_AGGREGATE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_MetricEvaluationMode::CUPTI_METRIC_EVALUATION_MODE_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_MetricEvaluationMode, MetricEvaluationMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Storage kind of a CUPTI metric value.
pub enum MetricValueKind {
    /// The metric value is a 64-bit double.
    Double = sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_DOUBLE as _,
    /// The metric value is a 64-bit unsigned integer.
    Uint64 = sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_UINT64 as _,
    /// The metric value is a percentage represented by a 64-bit double.
    ///
    /// For example, 57.5% is represented by the value 57.5.
    Percent = sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_PERCENT as _,
    /// The metric value is a throughput represented by a 64-bit integer.
    ///
    /// The unit for throughput values is bytes/second.
    Throughput = sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_THROUGHPUT as _,
    /// The metric value is a 64-bit signed integer.
    Int64 = sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_INT64 as _,
    /// The metric value is a [`MetricValueUtilizationLevel`].
    UtilizationLevel = sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL as _,
    /// Metric value encoded as NVTX Extended Payload.
    NvtxExtendedPayload =
        sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_NVTX_EXTENDED_PAYLOAD as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_MetricValueKind::CUPTI_METRIC_VALUE_KIND_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_MetricValueKind, MetricValueKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Utilization level used by utilization-valued metrics.
pub enum MetricValueUtilizationLevel {
    /// Idle utilization level.
    Idle = sys::CUpti_MetricValueUtilizationLevel::CUPTI_METRIC_VALUE_UTILIZATION_IDLE as _,
    /// Low utilization level.
    Low = sys::CUpti_MetricValueUtilizationLevel::CUPTI_METRIC_VALUE_UTILIZATION_LOW as _,
    /// Mid utilization level.
    Mid = sys::CUpti_MetricValueUtilizationLevel::CUPTI_METRIC_VALUE_UTILIZATION_MID as _,
    /// High utilization level.
    High = sys::CUpti_MetricValueUtilizationLevel::CUPTI_METRIC_VALUE_UTILIZATION_HIGH as _,
    /// Max utilization level.
    Max = sys::CUpti_MetricValueUtilizationLevel::CUPTI_METRIC_VALUE_UTILIZATION_MAX as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt =
        sys::CUpti_MetricValueUtilizationLevel::CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT as _,
}

impl_enum_conversion!(
    sys::CUpti_MetricValueUtilizationLevel,
    MetricValueUtilizationLevel
);

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
/// Decoded CUPTI metric value.
pub enum MetricValue {
    /// Metric value encoded as Double.
    Double(f64),
    /// Metric value encoded as Uint64.
    Uint64(u64),
    /// Metric value encoded as Percent.
    Percent(f64),
    /// Metric value encoded as Throughput.
    Throughput(u64),
    /// Metric value encoded as Int64.
    Int64(i64),
    /// Metric value encoded as Utilization Level.
    UtilizationLevel(MetricValueUtilizationLevel),
    /// Metric value encoded as NVTX Extended Payload.
    NvtxExtendedPayload(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Metric attributes readable through [`crate::event::get_metric_attribute_value`].
pub enum MetricAttribute {
    /// Metric name.
    ///
    /// Returns a null-terminated C string.
    Name = sys::CUpti_MetricAttribute::CUPTI_METRIC_ATTR_NAME as _,
    /// Short description of metric.
    ///
    /// Returns a null-terminated C string.
    ShortDescription = sys::CUpti_MetricAttribute::CUPTI_METRIC_ATTR_SHORT_DESCRIPTION as _,
    /// Long description of metric.
    ///
    /// Returns a null-terminated C string.
    LongDescription = sys::CUpti_MetricAttribute::CUPTI_METRIC_ATTR_LONG_DESCRIPTION as _,
    /// Category of the metric.
    ///
    /// Returns a [`MetricCategory`].
    Category = sys::CUpti_MetricAttribute::CUPTI_METRIC_ATTR_CATEGORY as _,
    /// Value type of the metric.
    ///
    /// Returns a [`MetricValueKind`].
    ValueKind = sys::CUpti_MetricAttribute::CUPTI_METRIC_ATTR_VALUE_KIND as _,
    /// Metric evaluation mode.
    ///
    /// Returns a [`MetricEvaluationMode`].
    EvaluationMode = sys::CUpti_MetricAttribute::CUPTI_METRIC_ATTR_EVALUATION_MODE as _,
    /// Reserved integer sentinel from CUPTI.
    ForceInt = sys::CUpti_MetricAttribute::CUPTI_METRIC_ATTR_FORCE_INT as _,
}

impl_enum_conversion!(sys::CUpti_MetricAttribute, MetricAttribute);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
/// Decoded value for a [`MetricAttribute`] query.
pub enum MetricAttributeValue {
    /// Name string returned by CUPTI.
    Name(String),
    /// Short Description string returned by CUPTI.
    ShortDescription(String),
    /// Long Description string returned by CUPTI.
    LongDescription(String),
    /// Category value decoded from CUPTI.
    Category(MetricCategory),
    /// Value Kind value decoded from CUPTI.
    ValueKind(MetricValueKind),
    /// Evaluation Mode value decoded from CUPTI.
    EvaluationMode(MetricEvaluationMode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Device class value used by metric properties.
pub enum MetricPropertyDeviceClass {
    /// Tesla device class.
    Tesla = sys::CUpti_MetricPropertyDeviceClass::CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TESLA as _,
    /// Quadro device class.
    Quadro = sys::CUpti_MetricPropertyDeviceClass::CUPTI_METRIC_PROPERTY_DEVICE_CLASS_QUADRO as _,
    /// Geforce device class.
    Geforce = sys::CUpti_MetricPropertyDeviceClass::CUPTI_METRIC_PROPERTY_DEVICE_CLASS_GEFORCE as _,
    /// Tegra device class.
    Tegra = sys::CUpti_MetricPropertyDeviceClass::CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TEGRA as _,
}

impl_enum_conversion!(
    sys::CUpti_MetricPropertyDeviceClass,
    MetricPropertyDeviceClass
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
/// Device property required when evaluating some CUPTI metrics.
pub enum MetricPropertyId {
    /// Multiprocessor Count value decoded from CUPTI.
    MultiprocessorCount =
        sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_MULTIPROCESSOR_COUNT as _,
    /// Warps Per Multiprocessor value decoded from CUPTI.
    WarpsPerMultiprocessor =
        sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_WARPS_PER_MULTIPROCESSOR as _,
    /// Kernel GPU Time value decoded from CUPTI.
    KernelGpuTime = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_KERNEL_GPU_TIME as _,
    /// Clock Rate value decoded from CUPTI.
    ClockRate = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_CLOCK_RATE as _,
    /// Frame Buffer Count value decoded from CUPTI.
    FrameBufferCount = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_FRAME_BUFFER_COUNT as _,
    /// Global Memory Bandwidth value decoded from CUPTI.
    GlobalMemoryBandwidth =
        sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_GLOBAL_MEMORY_BANDWIDTH as _,
    /// PCIe Link Rate value decoded from CUPTI.
    PcieLinkRate = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_PCIE_LINK_RATE as _,
    /// PCIe Link Width value decoded from CUPTI.
    PcieLinkWidth = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_PCIE_LINK_WIDTH as _,
    /// PCIe Generation value decoded from CUPTI.
    PcieGeneration = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_PCIE_GEN as _,
    /// Device Class value decoded from CUPTI.
    DeviceClass = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_DEVICE_CLASS as _,
    /// Flop Sp Per Cycle value decoded from CUPTI.
    FlopSpPerCycle = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE as _,
    /// Flop Dp Per Cycle value decoded from CUPTI.
    FlopDpPerCycle = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_FLOP_DP_PER_CYCLE as _,
    /// L2 Units value decoded from CUPTI.
    L2Units = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_L2_UNITS as _,
    /// Ecc Enabled value decoded from CUPTI.
    EccEnabled = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_ECC_ENABLED as _,
    /// Flop Hp Per Cycle value decoded from CUPTI.
    FlopHpPerCycle = sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_FLOP_HP_PER_CYCLE as _,
    /// GPU Central Processing Unit NvLink Bandwidth value decoded from CUPTI.
    GpuCentralProcessingUnitNvLinkBandwidth =
        sys::CUpti_MetricPropertyID::CUPTI_METRIC_PROPERTY_GPU_CPU_NVLINK_BANDWIDTH as _,
}

impl_enum_conversion!(sys::CUpti_MetricPropertyID, MetricPropertyId);

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    /// Flags attached to CUPTI activity records.
    pub struct ActivityFlags: u32 {
        /// Indicates the activity record has no flags.
        const NONE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_NONE as _;
        /// Indicates the activity represents a device that supports concurrent kernel execution.
        ///
        /// Valid for [`ActivityKind::Device`] records.
        const DEVICE_CONCURRENT_KERNELS = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS as _;
        /// CUPTI flag `DEVICE_ATTRIBUTE_CUDEVICE`.
        const DEVICE_ATTRIBUTE_CUDEVICE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE as _;
        /// Indicates the activity represents a region start marker.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_START = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_START as _;
        /// Indicates the activity represents a region end marker.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_END = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_END as _;
        /// Indicates the activity represents an attempt to acquire a user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_ACQUIRE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE as _;
        /// Indicates the activity represents success in acquiring the user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_ACQUIRE_SUCCESS = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS as _;
        /// Indicates the activity represents failure in acquiring the user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_ACQUIRE_FAILED = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED as _;
        /// Indicates the activity represents releasing a reservation on user defined synchronization object.
        ///
        /// Valid for [`ActivityKind::Marker`] records.
        const MARKER_SYNC_RELEASE = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE as _;
        /// Mask for the number of bytes requested by each thread in [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess).
        const GLOBAL_ACCESS_KIND_SIZE_MASK = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_SIZE_MASK as _;
        /// If bit in this flag is set, the access was load, else it is a store access.
        ///
        /// Valid for [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess) records.
        const GLOBAL_ACCESS_KIND_LOAD = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD as _;
        /// If this bit in flag is set, the load access was cached else it is uncached.
        ///
        /// Valid for [`ActivityGlobalAccess`](crate::activity::ActivityGlobalAccess) records.
        const GLOBAL_ACCESS_KIND_CACHED = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED as _;
        /// Mask for the [`ActivityInstructionClass`] bits stored in instruction execution and correlation records.
        const INSTRUCTION_CLASS_MASK = sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK as _;
    }
}

impl_enum_display!(ApiCallbackSite, {
    Self::Enter => "CUPTI_API_ENTER",
    Self::Exit => "CUPTI_API_EXIT",
    Self::ForceInt => "CUPTI_API_CBSITE_FORCE_INT",
});

impl_enum_display!(DeviceAttributeDeviceClass, {
    Self::Tesla => "CUPTI_DEVICE_ATTR_DEVICE_CLASS_TESLA",
    Self::Quadro => "CUPTI_DEVICE_ATTR_DEVICE_CLASS_QUADRO",
    Self::Geforce => "CUPTI_DEVICE_ATTR_DEVICE_CLASS_GEFORCE",
    Self::Tegra => "CUPTI_DEVICE_ATTR_DEVICE_CLASS_TEGRA",
});

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

impl_enum_display!(EventDomainAttribute, {
    Self::Name => "CUPTI_EVENT_DOMAIN_ATTR_NAME",
    Self::InstanceCount => "CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT",
    Self::TotalInstanceCount => "CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT",
    Self::CollectionMethod => "CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD",
    Self::ForceInt => "CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT",
});

impl_enum_display!(EventCollectionMethod, {
    Self::Pm => "CUPTI_EVENT_COLLECTION_METHOD_PM",
    Self::Sm => "CUPTI_EVENT_COLLECTION_METHOD_SM",
    Self::Instrumented => "CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED",
    Self::NvLinkTc => "CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC",
    Self::ForceInt => "CUPTI_EVENT_COLLECTION_METHOD_FORCE_INT",
});

impl_enum_display!(EventGroupAttribute, {
    Self::EventDomainId => "CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID",
    Self::ProfileAllDomainInstances => "CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES",
    Self::UserData => "CUPTI_EVENT_GROUP_ATTR_USER_DATA",
    Self::NumEvents => "CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS",
    Self::Events => "CUPTI_EVENT_GROUP_ATTR_EVENTS",
    Self::InstanceCount => "CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT",
    Self::ProfilingScope => "CUPTI_EVENT_GROUP_ATTR_PROFILING_SCOPE",
    Self::ForceInt => "CUPTI_EVENT_GROUP_ATTR_FORCE_INT",
});

impl_enum_display!(EventProfilingScope, {
    Self::Context => "CUPTI_EVENT_PROFILING_SCOPE_CONTEXT",
    Self::Device => "CUPTI_EVENT_PROFILING_SCOPE_DEVICE",
    Self::Both => "CUPTI_EVENT_PROFILING_SCOPE_BOTH",
    Self::ForceInt => "CUPTI_EVENT_PROFILING_SCOPE_FORCE_INT",
});

impl_enum_display!(EventAttribute, {
    Self::Name => "CUPTI_EVENT_ATTR_NAME",
    Self::ShortDescription => "CUPTI_EVENT_ATTR_SHORT_DESCRIPTION",
    Self::LongDescription => "CUPTI_EVENT_ATTR_LONG_DESCRIPTION",
    Self::Category => "CUPTI_EVENT_ATTR_CATEGORY",
    Self::ProfilingScope => "CUPTI_EVENT_ATTR_PROFILING_SCOPE",
    Self::ForceInt => "CUPTI_EVENT_ATTR_FORCE_INT",
});

impl_enum_display!(EventCategory, {
    Self::Instruction => "CUPTI_EVENT_CATEGORY_INSTRUCTION",
    Self::Memory => "CUPTI_EVENT_CATEGORY_MEMORY",
    Self::Cache => "CUPTI_EVENT_CATEGORY_CACHE",
    Self::ProfileTrigger => "CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER",
    Self::System => "CUPTI_EVENT_CATEGORY_SYSTEM",
    Self::ForceInt => "CUPTI_EVENT_CATEGORY_FORCE_INT",
});

impl_enum_display!(EventCollectionMode, {
    Self::Continuous => "CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS",
    Self::Kernel => "CUPTI_EVENT_COLLECTION_MODE_KERNEL",
    Self::ForceInt => "CUPTI_EVENT_COLLECTION_MODE_FORCE_INT",
});

impl_enum_display!(ReadEventFlags, {
    Self::None => "CUPTI_EVENT_READ_FLAG_NONE",
    Self::ForceInt => "CUPTI_EVENT_READ_FLAG_FORCE_INT",
});

impl_enum_display!(CallbackIdSync, {
    Self::Invalid => "CUPTI_CBID_SYNCHRONIZE_INVALID",
    Self::StreamSynchronized => "CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED",
    Self::ContextSynchronized => "CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED",
    Self::Size => "CUPTI_CBID_SYNCHRONIZE_SIZE",
    Self::ForceInt => "CUPTI_CBID_SYNCHRONIZE_FORCE_INT",
});

impl_enum_display!(CallbackIdState, {
    Self::Invalid => "CUPTI_CBID_STATE_INVALID",
    Self::FatalError => "CUPTI_CBID_STATE_FATAL_ERROR",
    Self::Error => "CUPTI_CBID_STATE_ERROR",
    Self::Warning => "CUPTI_CBID_STATE_WARNING",
    Self::Size => "CUPTI_CBID_STATE_SIZE",
    Self::ForceInt => "CUPTI_CBID_STATE_FORCE_INT",
});

impl_enum_display!(ActivityKind, {
    Self::Invalid => "CUPTI_ACTIVITY_KIND_INVALID",
    Self::MemoryCopy => "CUPTI_ACTIVITY_KIND_MEMCPY",
    Self::MemorySet => "CUPTI_ACTIVITY_KIND_MEMSET",
    Self::Kernel => "CUPTI_ACTIVITY_KIND_KERNEL",
    Self::Driver => "CUPTI_ACTIVITY_KIND_DRIVER",
    Self::Runtime => "CUPTI_ACTIVITY_KIND_RUNTIME",
    Self::Event => "CUPTI_ACTIVITY_KIND_EVENT",
    Self::Metric => "CUPTI_ACTIVITY_KIND_METRIC",
    Self::Device => "CUPTI_ACTIVITY_KIND_DEVICE",
    Self::Context => "CUPTI_ACTIVITY_KIND_CONTEXT",
    Self::ConcurrentKernel => "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    Self::Name => "CUPTI_ACTIVITY_KIND_NAME",
    Self::Marker => "CUPTI_ACTIVITY_KIND_MARKER",
    Self::MarkerData => "CUPTI_ACTIVITY_KIND_MARKER_DATA",
    Self::SourceLocator => "CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR",
    Self::GlobalAccess => "CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS",
    Self::Branch => "CUPTI_ACTIVITY_KIND_BRANCH",
    Self::Overhead => "CUPTI_ACTIVITY_KIND_OVERHEAD",
    Self::CudaDynamicParallelismKernel => "CUPTI_ACTIVITY_KIND_CDP_KERNEL",
    Self::Preemption => "CUPTI_ACTIVITY_KIND_PREEMPTION",
    Self::Environment => "CUPTI_ACTIVITY_KIND_ENVIRONMENT",
    Self::EventInstance => "CUPTI_ACTIVITY_KIND_EVENT_INSTANCE",
    Self::PeerMemoryCopy => "CUPTI_ACTIVITY_KIND_MEMCPY2",
    Self::MetricInstance => "CUPTI_ACTIVITY_KIND_METRIC_INSTANCE",
    Self::InstructionExecution => "CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION",
    Self::UnifiedMemoryCounter => "CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER",
    Self::Function => "CUPTI_ACTIVITY_KIND_FUNCTION",
    Self::Module => "CUPTI_ACTIVITY_KIND_MODULE",
    Self::DeviceAttribute => "CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE",
    Self::SharedAccess => "CUPTI_ACTIVITY_KIND_SHARED_ACCESS",
    Self::PcSampling => "CUPTI_ACTIVITY_KIND_PC_SAMPLING",
    Self::PcSamplingRecordInfo => "CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO",
    Self::InstructionCorrelation => "CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION",
    Self::OpenAccData => "CUPTI_ACTIVITY_KIND_OPENACC_DATA",
    Self::OpenAccLaunch => "CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH",
    Self::OpenAccOther => "CUPTI_ACTIVITY_KIND_OPENACC_OTHER",
    Self::CudaEvent => "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
    Self::Stream => "CUPTI_ACTIVITY_KIND_STREAM",
    Self::Synchronization => "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    Self::ExternalCorrelation => "CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION",
    Self::NvLink => "CUPTI_ACTIVITY_KIND_NVLINK",
    Self::InstantaneousEvent => "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT",
    Self::InstantaneousEventInstance => "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE",
    Self::InstantaneousMetric => "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC",
    Self::InstantaneousMetricInstance => "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE",
    Self::Memory => "CUPTI_ACTIVITY_KIND_MEMORY",
    Self::Pcie => "CUPTI_ACTIVITY_KIND_PCIE",
    Self::OpenMultiProcessing => "CUPTI_ACTIVITY_KIND_OPENMP",
    Self::InternalLaunchApi => "CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API",
    Self::Memory2 => "CUPTI_ACTIVITY_KIND_MEMORY2",
    Self::MemoryPool => "CUPTI_ACTIVITY_KIND_MEMORY_POOL",
    Self::GraphTrace => "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    Self::Jit => "CUPTI_ACTIVITY_KIND_JIT",
    Self::DeviceGraphTrace => "CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE",
    Self::MemoryDecompress => "CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS",
    Self::ConfidentialComputeRotation => "CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION",
    Self::GraphHostNode => "CUPTI_ACTIVITY_KIND_GRAPH_HOST_NODE",
    Self::ComputeEngineCtxSwitch => "CUPTI_ACTIVITY_KIND_COMPUTE_ENGINE_CTX_SWITCH",
    Self::HostLaunch => "CUPTI_ACTIVITY_KIND_HOST_LAUNCH",
    Self::GreenContext => "CUPTI_ACTIVITY_KIND_GREEN_CONTEXT",
    Self::Count => "CUPTI_ACTIVITY_KIND_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_KIND_FORCE_INT",
});

impl_enum_display!(ChannelType, {
    Self::Invalid => "CUPTI_CHANNEL_TYPE_INVALID",
    Self::Compute => "CUPTI_CHANNEL_TYPE_COMPUTE",
    Self::AsynchronousMemoryCopy => "CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY",
    Self::Decompress => "CUPTI_CHANNEL_TYPE_DECOMP",
    Self::ForceInt => "CUPTI_CHANNEL_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityMemoryCopyKind, {
    Self::Unknown => "CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN",
    Self::HostToDevice => "CUPTI_ACTIVITY_MEMCPY_KIND_HTOD",
    Self::DeviceToHost => "CUPTI_ACTIVITY_MEMCPY_KIND_DTOH",
    Self::HostToArray => "CUPTI_ACTIVITY_MEMCPY_KIND_HTOA",
    Self::ArrayToHost => "CUPTI_ACTIVITY_MEMCPY_KIND_ATOH",
    Self::ArrayToArray => "CUPTI_ACTIVITY_MEMCPY_KIND_ATOA",
    Self::ArrayToDevice => "CUPTI_ACTIVITY_MEMCPY_KIND_ATOD",
    Self::DeviceToArray => "CUPTI_ACTIVITY_MEMCPY_KIND_DTOA",
    Self::DeviceToDevice => "CUPTI_ACTIVITY_MEMCPY_KIND_DTOD",
    Self::HostToHost => "CUPTI_ACTIVITY_MEMCPY_KIND_HTOH",
    Self::PeerToPeer => "CUPTI_ACTIVITY_MEMCPY_KIND_PTOP",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT",
});

impl_enum_display!(ActivityMemoryKind, {
    Self::Unknown => "CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN",
    Self::Pageable => "CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE",
    Self::Pinned => "CUPTI_ACTIVITY_MEMORY_KIND_PINNED",
    Self::Device => "CUPTI_ACTIVITY_MEMORY_KIND_DEVICE",
    Self::Array => "CUPTI_ACTIVITY_MEMORY_KIND_ARRAY",
    Self::Managed => "CUPTI_ACTIVITY_MEMORY_KIND_MANAGED",
    Self::DeviceStatic => "CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC",
    Self::ManagedStatic => "CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT",
});

impl_enum_display!(ActivityMemoryOperationType, {
    Self::Invalid => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID",
    Self::Allocation => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION",
    Self::Release => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityMemoryPoolType, {
    Self::Invalid => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID",
    Self::Local => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL",
    Self::Imported => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_POOL_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityMemoryPoolOperationType, {
    Self::Invalid => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_INVALID",
    Self::Created => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED",
    Self::Destroyed => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED",
    Self::Trimmed => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED",
    Self::ForceInt => "CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_FORCE_INT",
});

impl_enum_display!(DeviceGraphLaunchMode, {
    Self::Invalid => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_INVALID",
    Self::FireAndForget => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET",
    Self::Tail => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_TAIL",
    Self::FireAndForgetAsSibling => "CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET_AS_SIBLING",
});

impl_enum_display!(ActivityComputeApiKind, {
    Self::Unknown => "CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN",
    Self::Cuda => "CUPTI_ACTIVITY_COMPUTE_API_CUDA",
    Self::CudaMultiProcessService => "CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS",
    Self::ForceInt => "CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT",
});

impl_enum_display!(ContextCigMode, {
    Self::None => "CUPTI_CONTEXT_CIG_MODE_NONE",
    Self::Cig => "CUPTI_CONTEXT_CIG_MODE_CIG",
    Self::CigFallback => "CUPTI_CONTEXT_CIG_MODE_CIG_FALLBACK",
    Self::ForceInt => "CUPTI_CONTEXT_CIG_MODE_FORCE_INT",
});

impl_enum_display!(ComputeEngineContextSwitchOperationType, {
    Self::Invalid => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_INVALID",
    Self::Start => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_START",
    Self::End => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_END",
    Self::Count => "CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_COUNT",
});

impl_enum_display!(ActivitySynchronizationType, {
    Self::Unknown => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN",
    Self::EventSynchronize => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE",
    Self::StreamWaitEvent => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT",
    Self::StreamSynchronize => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE",
    Self::ContextSynchronize => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE",
    Self::ForceInt => "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityStreamFlag, {
    Self::Unknown => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN",
    Self::Default => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT",
    Self::NonBlocking => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING",
    Self::Null => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL",
    Self::Mask => "CUPTI_ACTIVITY_STREAM_CREATE_MASK",
    Self::ForceInt => "CUPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT",
});

impl_enum_display!(ActivityPartitionedGlobalCacheConfig, {
    Self::Unknown => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN",
    Self::NotSupported => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED",
    Self::Off => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF",
    Self::On => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON",
    Self::ForceInt => "CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT",
});

impl_enum_display!(ActivityLaunchType, {
    Self::Regular => "CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR",
    Self::CooperativeSingleDevice => "CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE",
    Self::CooperativeMultiDevice => "CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE",
    Self::CblCommandlist => "CUPTI_ACTIVITY_LAUNCH_TYPE_CBL_COMMANDLIST",
});

impl_enum_display!(FunctionSharedMemoryLimitConfig, {
    Self::Default => "CUPTI_FUNC_SHMEM_LIMIT_DEFAULT",
    Self::OptIn => "CUPTI_FUNC_SHMEM_LIMIT_OPTIN",
    Self::ForceInt => "CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT",
});

impl_enum_display!(ActivityObjectKind, {
    Self::Unknown => "CUPTI_ACTIVITY_OBJECT_UNKNOWN",
    Self::Process => "CUPTI_ACTIVITY_OBJECT_PROCESS",
    Self::Thread => "CUPTI_ACTIVITY_OBJECT_THREAD",
    Self::Device => "CUPTI_ACTIVITY_OBJECT_DEVICE",
    Self::Context => "CUPTI_ACTIVITY_OBJECT_CONTEXT",
    Self::Stream => "CUPTI_ACTIVITY_OBJECT_STREAM",
    Self::ForceInt => "CUPTI_ACTIVITY_OBJECT_FORCE_INT",
});

impl_enum_display!(ActivityOverheadKind, {
    Self::Unknown => "CUPTI_ACTIVITY_OVERHEAD_UNKNOWN",
    Self::DriverCompiler => "CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER",
    Self::BufferFlush => "CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH",
    Self::Instrumentation => "CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION",
    Self::Resource => "CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE",
    Self::RuntimeTriggeredModuleLoading => "CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING",
    Self::LazyFunctionLoading => "CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING",
    Self::CommandBufferFull => "CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL",
    Self::ActivityBufferRequest => "CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST",
    Self::UvmActivityInit => "CUPTI_ACTIVITY_OVERHEAD_UVM_ACTIVITY_INIT",
    Self::ForceInt => "CUPTI_ACTIVITY_OVERHEAD_FORCE_INT",
});

impl_enum_display!(DeviceVirtualizationMode, {
    Self::None => "CUPTI_DEVICE_VIRTUALIZATION_MODE_NONE",
    Self::PassThrough => "CUPTI_DEVICE_VIRTUALIZATION_MODE_PASS_THROUGH",
    Self::VirtualGpu => "CUPTI_DEVICE_VIRTUALIZATION_MODE_VIRTUAL_GPU",
    Self::ForceInt => "CUPTI_DEVICE_VIRTUALIZATION_MODE_FORCE_INT",
});

impl_enum_display!(ExternalCorrelationKind, {
    Self::Invalid => "CUPTI_EXTERNAL_CORRELATION_KIND_INVALID",
    Self::Unknown => "CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN",
    Self::OpenAcc => "CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC",
    Self::Custom0 => "CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0",
    Self::Custom1 => "CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1",
    Self::Custom2 => "CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2",
    Self::Size => "CUPTI_EXTERNAL_CORRELATION_KIND_SIZE",
    Self::ForceInt => "CUPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT",
});

impl_enum_display!(ActivityAttribute, {
    Self::DeviceBufferSize => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE",
    Self::DeviceBufferSizeCudaDynamicParallelism => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP",
    Self::DeviceBufferPoolLimit => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT",
    Self::ProfilingSemaphorePoolSize => "CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE",
    Self::ProfilingSemaphorePoolLimit => "CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT",
    Self::ZeroedOutActivityBuffer => "CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER",
    Self::DeviceBufferPreAllocateValue => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE",
    Self::ProfilingSemaphorePreAllocateValue => "CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE",
    Self::MemAllocationTypeHostPinned => "CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED",
    Self::PerThreadActivityBuffer => "CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER",
    Self::DeviceBufferSizeDeviceGraphs => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_DEVICE_GRAPHS",
    Self::UserDefinedRecords => "CUPTI_ACTIVITY_ATTR_USER_DEFINED_RECORDS",
    Self::MultipleSubscriberState => "CUPTI_ACTIVITY_ATTR_MULTIPLE_SUBSCRIBER_STATE",
    Self::EnableHes => "CUPTI_ACTIVITY_ATTR_ENABLE_HES",
    Self::EnableAllocationSourceTracking => "CUPTI_ACTIVITY_ATTR_ENABLE_ALLOCATION_SOURCE_TRACKING",
    Self::EnableKernelLatencyTimestamps => "CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LATENCY_TIMESTAMPS",
    Self::EnableAllSyncRecords => "CUPTI_ACTIVITY_ATTR_ENABLE_ALL_SYNC_RECORDS",
    Self::EnableCudaEventDeviceTimestamps => "CUPTI_ACTIVITY_ATTR_ENABLE_CUDA_EVENT_DEVICE_TIMESTAMPS",
    Self::EnableKernelLaunchAttributes => "CUPTI_ACTIVITY_ATTR_ENABLE_KERNEL_LAUNCH_ATTRIBUTES",
    Self::EnableDeviceGraphTrace => "CUPTI_ACTIVITY_ATTR_ENABLE_DEVICE_GRAPH_TRACE",
    Self::EnableMultiSubscriberGraphLevelTrace => "CUPTI_ACTIVITY_ATTR_ENABLE_MULTI_SUBSCRIBER_GRAPH_TRACE",
    Self::ThreadIdType => "CUPTI_ACTIVITY_ATTR_THREAD_ID_TYPE",
    Self::TimestampCallback => "CUPTI_ACTIVITY_ATTR_TIMESTAMP_CALLBACK",
    Self::CigMode => "CUPTI_ACTIVITY_ATTR_ENABLE_CIG_MODE",
    Self::DeviceBufferForceInt => "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT",
});

impl_enum_display!(ActivityPcSamplingPeriod, {
    Self::Invalid => "CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID",
    Self::Min => "CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN",
    Self::Low => "CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW",
    Self::Mid => "CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID",
    Self::High => "CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH",
    Self::Max => "CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX",
    Self::ForceInt => "CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_FORCE_INT",
});

impl_enum_display!(ActivityEnvironmentKind, {
    Self::Unknown => "CUPTI_ACTIVITY_ENVIRONMENT_UNKNOWN",
    Self::Speed => "CUPTI_ACTIVITY_ENVIRONMENT_SPEED",
    Self::Temperature => "CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE",
    Self::Power => "CUPTI_ACTIVITY_ENVIRONMENT_POWER",
    Self::Cooling => "CUPTI_ACTIVITY_ENVIRONMENT_COOLING",
    Self::Count => "CUPTI_ACTIVITY_ENVIRONMENT_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_ENVIRONMENT_KIND_FORCE_INT",
});

impl_enum_display!(ActivityPreemptionKind, {
    Self::Unknown => "CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN",
    Self::Save => "CUPTI_ACTIVITY_PREEMPTION_KIND_SAVE",
    Self::Restore => "CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE",
    Self::ForceInt => "CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT",
});

impl_enum_display!(ActivityPcSamplingStallReason, {
    Self::Invalid => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID",
    Self::None => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE",
    Self::InstructionFetch => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH",
    Self::ExecutionDependency => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY",
    Self::MemoryDependency => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY",
    Self::Texture => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE",
    Self::Sync => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC",
    Self::ConstantMemoryDependency => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY",
    Self::PipeBusy => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY",
    Self::MemoryThrottle => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE",
    Self::NotSelected => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED",
    Self::Other => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER",
    Self::Sleeping => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING",
    Self::ForceInt => "CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT",
});

impl_enum_display!(ActivityUnifiedMemoryCounterScope, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN",
    Self::ProcessSingleDevice => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE",
    Self::ProcessAllDevices => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES",
    Self::Count => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT",
});

impl_enum_display!(ActivityUnifiedMemoryCounterKind, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN",
    Self::BytesTransferHostToDevice => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD",
    Self::BytesTransferDeviceToHost => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH",
    Self::CentralProcessingUnitPageFaultCount => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT",
    Self::GpuPageFault => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT",
    Self::Thrashing => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING",
    Self::Throttling => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING",
    Self::RemoteMap => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP",
    Self::BytesTransferDeviceToDevice => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD",
    Self::Count => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT",
    Self::ForceInt => "CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT",
});

impl_enum_display!(ActivityUnifiedMemoryAccessType, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN",
    Self::Read => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ",
    Self::Write => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE",
    Self::Atomic => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC",
    Self::Prefetch => "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH",
});

impl_enum_display!(ActivityUnifiedMemoryMigrationCause, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN",
    Self::User => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER",
    Self::Coherence => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE",
    Self::Prefetch => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH",
    Self::Eviction => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION",
    Self::AccessCounters => "CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS",
});

impl_enum_display!(ActivityUnifiedMemoryRemoteMapCause, {
    Self::Unknown => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN",
    Self::Coherence => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE",
    Self::Thrashing => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING",
    Self::Policy => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY",
    Self::OutOfMemory => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY",
    Self::Eviction => "CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_EVICTION",
});

impl_enum_display!(ActivityInstructionClass, {
    Self::Unknown => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN",
    Self::Float32 => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32",
    Self::Float64 => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64",
    Self::Integer => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER",
    Self::BitConversion => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_CONVERSION",
    Self::ControlFlow => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW",
    Self::Global => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL",
    Self::Shared => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED",
    Self::Local => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL",
    Self::Generic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC",
    Self::Surface => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE",
    Self::Constant => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT",
    Self::Texture => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE",
    Self::GlobalAtomic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC",
    Self::SharedAtomic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC",
    Self::SurfaceAtomic => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC",
    Self::InterThreadCommunication => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION",
    Self::Barrier => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER",
    Self::Miscellaneous => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS",
    Self::Float16 => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16",
    Self::Uniform => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNIFORM",
    Self::ForceInt => "CUPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT",
});

impl_enum_display!(ConfidentialComputeRotationEventType, {
    Self::Invalid => "CUPTI_CONFIDENTIAL_COMPUTE_INVALID_ROTATION_EVENT",
    Self::ChannelBlocked => "CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_BLOCKED",
    Self::ChannelDrained => "CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_DRAINED",
    Self::ChannelUnblocked => "CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_UNBLOCKED",
    Self::ForceInt => "CUPTI_CONFIDENTIAL_COMPUTE_EVENT_TYPE_FORCE_INT",
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

impl_enum_display!(OpenAccEventKind, {
    Self::Invalid => "CUPTI_OPENACC_EVENT_KIND_INVALID",
    Self::DeviceInit => "CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT",
    Self::DeviceShutdown => "CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN",
    Self::RuntimeShutdown => "CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN",
    Self::EnqueueLaunch => "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH",
    Self::EnqueueUpload => "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD",
    Self::EnqueueDownload => "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD",
    Self::Wait => "CUPTI_OPENACC_EVENT_KIND_WAIT",
    Self::ImplicitWait => "CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT",
    Self::ComputeConstruct => "CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT",
    Self::Update => "CUPTI_OPENACC_EVENT_KIND_UPDATE",
    Self::EnterData => "CUPTI_OPENACC_EVENT_KIND_ENTER_DATA",
    Self::ExitData => "CUPTI_OPENACC_EVENT_KIND_EXIT_DATA",
    Self::Create => "CUPTI_OPENACC_EVENT_KIND_CREATE",
    Self::Delete => "CUPTI_OPENACC_EVENT_KIND_DELETE",
    Self::Alloc => "CUPTI_OPENACC_EVENT_KIND_ALLOC",
    Self::Free => "CUPTI_OPENACC_EVENT_KIND_FREE",
    Self::ForceInt => "CUPTI_OPENACC_EVENT_KIND_FORCE_INT",
});

impl_enum_display!(OpenAccConstructKind, {
    Self::Unknown => "CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN",
    Self::Parallel => "CUPTI_OPENACC_CONSTRUCT_KIND_PARALLEL",
    Self::Kernels => "CUPTI_OPENACC_CONSTRUCT_KIND_KERNELS",
    Self::Loop => "CUPTI_OPENACC_CONSTRUCT_KIND_LOOP",
    Self::Data => "CUPTI_OPENACC_CONSTRUCT_KIND_DATA",
    Self::EnterData => "CUPTI_OPENACC_CONSTRUCT_KIND_ENTER_DATA",
    Self::ExitData => "CUPTI_OPENACC_CONSTRUCT_KIND_EXIT_DATA",
    Self::HostData => "CUPTI_OPENACC_CONSTRUCT_KIND_HOST_DATA",
    Self::Atomic => "CUPTI_OPENACC_CONSTRUCT_KIND_ATOMIC",
    Self::Declare => "CUPTI_OPENACC_CONSTRUCT_KIND_DECLARE",
    Self::Init => "CUPTI_OPENACC_CONSTRUCT_KIND_INIT",
    Self::Shutdown => "CUPTI_OPENACC_CONSTRUCT_KIND_SHUTDOWN",
    Self::Set => "CUPTI_OPENACC_CONSTRUCT_KIND_SET",
    Self::Update => "CUPTI_OPENACC_CONSTRUCT_KIND_UPDATE",
    Self::Routine => "CUPTI_OPENACC_CONSTRUCT_KIND_ROUTINE",
    Self::Wait => "CUPTI_OPENACC_CONSTRUCT_KIND_WAIT",
    Self::RuntimeApi => "CUPTI_OPENACC_CONSTRUCT_KIND_RUNTIME_API",
    Self::ForceInt => "CUPTI_OPENACC_CONSTRUCT_KIND_FORCE_INT",
});

impl_enum_display!(OpenMultiProcessingEventKind, {
    Self::Invalid => "CUPTI_OPENMP_EVENT_KIND_INVALID",
    Self::Parallel => "CUPTI_OPENMP_EVENT_KIND_PARALLEL",
    Self::Task => "CUPTI_OPENMP_EVENT_KIND_TASK",
    Self::Thread => "CUPTI_OPENMP_EVENT_KIND_THREAD",
    Self::Idle => "CUPTI_OPENMP_EVENT_KIND_IDLE",
    Self::WaitBarrier => "CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER",
    Self::WaitTaskwait => "CUPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT",
    Self::ForceInt => "CUPTI_OPENMP_EVENT_KIND_FORCE_INT",
});

impl_enum_display!(ActivityJitEntryType, {
    Self::Invalid => "CUPTI_ACTIVITY_JIT_ENTRY_INVALID",
    Self::ParallelThreadExecutionToCudaBinary => "CUPTI_ACTIVITY_JIT_ENTRY_PTX_TO_CUBIN",
    Self::NvidiaVirtualMachineIntermediateRepresentationToParallelThreadExecution => "CUPTI_ACTIVITY_JIT_ENTRY_NVVM_IR_TO_PTX",
    Self::ForceInt => "CUPTI_ACTIVITY_JIT_ENTRY_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityJitOperationType, {
    Self::Invalid => "CUPTI_ACTIVITY_JIT_OPERATION_INVALID",
    Self::CacheLoad => "CUPTI_ACTIVITY_JIT_OPERATION_CACHE_LOAD",
    Self::CacheStore => "CUPTI_ACTIVITY_JIT_OPERATION_CACHE_STORE",
    Self::Compile => "CUPTI_ACTIVITY_JIT_OPERATION_COMPILE",
    Self::ForceInt => "CUPTI_ACTIVITY_JIT_OPERATION_TYPE_FORCE_INT",
});

impl_enum_display!(ActivityThreadIdType, {
    Self::Default => "CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT",
    Self::System => "CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM",
    Self::Size => "CUPTI_ACTIVITY_THREAD_ID_TYPE_SIZE",
    Self::ForceInt => "CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT",
});

impl_enum_display!(MetricCategory, {
    Self::Memory => "CUPTI_METRIC_CATEGORY_MEMORY",
    Self::Instruction => "CUPTI_METRIC_CATEGORY_INSTRUCTION",
    Self::Multiprocessor => "CUPTI_METRIC_CATEGORY_MULTIPROCESSOR",
    Self::Cache => "CUPTI_METRIC_CATEGORY_CACHE",
    Self::Texture => "CUPTI_METRIC_CATEGORY_TEXTURE",
    Self::NvLink => "CUPTI_METRIC_CATEGORY_NVLINK",
    Self::Pcie => "CUPTI_METRIC_CATEGORY_PCIE",
    Self::ForceInt => "CUPTI_METRIC_CATEGORY_FORCE_INT",
});

impl_enum_display!(MetricEvaluationMode, {
    Self::PerInstance => "CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE",
    Self::Aggregate => "CUPTI_METRIC_EVALUATION_MODE_AGGREGATE",
    Self::ForceInt => "CUPTI_METRIC_EVALUATION_MODE_FORCE_INT",
});

impl_enum_display!(MetricValueKind, {
    Self::Double => "CUPTI_METRIC_VALUE_KIND_DOUBLE",
    Self::Uint64 => "CUPTI_METRIC_VALUE_KIND_UINT64",
    Self::Percent => "CUPTI_METRIC_VALUE_KIND_PERCENT",
    Self::Throughput => "CUPTI_METRIC_VALUE_KIND_THROUGHPUT",
    Self::Int64 => "CUPTI_METRIC_VALUE_KIND_INT64",
    Self::UtilizationLevel => "CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL",
    Self::NvtxExtendedPayload => "CUPTI_METRIC_VALUE_KIND_NVTX_EXTENDED_PAYLOAD",
    Self::ForceInt => "CUPTI_METRIC_VALUE_KIND_FORCE_INT",
});

impl_enum_display!(MetricValueUtilizationLevel, {
    Self::Idle => "CUPTI_METRIC_VALUE_UTILIZATION_IDLE",
    Self::Low => "CUPTI_METRIC_VALUE_UTILIZATION_LOW",
    Self::Mid => "CUPTI_METRIC_VALUE_UTILIZATION_MID",
    Self::High => "CUPTI_METRIC_VALUE_UTILIZATION_HIGH",
    Self::Max => "CUPTI_METRIC_VALUE_UTILIZATION_MAX",
    Self::ForceInt => "CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT",
});

impl_enum_display!(MetricAttribute, {
    Self::Name => "CUPTI_METRIC_ATTR_NAME",
    Self::ShortDescription => "CUPTI_METRIC_ATTR_SHORT_DESCRIPTION",
    Self::LongDescription => "CUPTI_METRIC_ATTR_LONG_DESCRIPTION",
    Self::Category => "CUPTI_METRIC_ATTR_CATEGORY",
    Self::ValueKind => "CUPTI_METRIC_ATTR_VALUE_KIND",
    Self::EvaluationMode => "CUPTI_METRIC_ATTR_EVALUATION_MODE",
    Self::ForceInt => "CUPTI_METRIC_ATTR_FORCE_INT",
});

impl_enum_display!(MetricPropertyDeviceClass, {
    Self::Tesla => "CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TESLA",
    Self::Quadro => "CUPTI_METRIC_PROPERTY_DEVICE_CLASS_QUADRO",
    Self::Geforce => "CUPTI_METRIC_PROPERTY_DEVICE_CLASS_GEFORCE",
    Self::Tegra => "CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TEGRA",
});

impl_enum_display!(MetricPropertyId, {
    Self::MultiprocessorCount => "CUPTI_METRIC_PROPERTY_MULTIPROCESSOR_COUNT",
    Self::WarpsPerMultiprocessor => "CUPTI_METRIC_PROPERTY_WARPS_PER_MULTIPROCESSOR",
    Self::KernelGpuTime => "CUPTI_METRIC_PROPERTY_KERNEL_GPU_TIME",
    Self::ClockRate => "CUPTI_METRIC_PROPERTY_CLOCK_RATE",
    Self::FrameBufferCount => "CUPTI_METRIC_PROPERTY_FRAME_BUFFER_COUNT",
    Self::GlobalMemoryBandwidth => "CUPTI_METRIC_PROPERTY_GLOBAL_MEMORY_BANDWIDTH",
    Self::PcieLinkRate => "CUPTI_METRIC_PROPERTY_PCIE_LINK_RATE",
    Self::PcieLinkWidth => "CUPTI_METRIC_PROPERTY_PCIE_LINK_WIDTH",
    Self::PcieGeneration => "CUPTI_METRIC_PROPERTY_PCIE_GEN",
    Self::DeviceClass => "CUPTI_METRIC_PROPERTY_DEVICE_CLASS",
    Self::FlopSpPerCycle => "CUPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE",
    Self::FlopDpPerCycle => "CUPTI_METRIC_PROPERTY_FLOP_DP_PER_CYCLE",
    Self::L2Units => "CUPTI_METRIC_PROPERTY_L2_UNITS",
    Self::EccEnabled => "CUPTI_METRIC_PROPERTY_ECC_ENABLED",
    Self::FlopHpPerCycle => "CUPTI_METRIC_PROPERTY_FLOP_HP_PER_CYCLE",
    Self::GpuCentralProcessingUnitNvLinkBandwidth => "CUPTI_METRIC_PROPERTY_GPU_CPU_NVLINK_BANDWIDTH",
});
impl Display for DeviceAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::U32(value) => write!(f, "U32({value})"),
            Self::DeviceClass(value) => write!(f, "DeviceClass({value})"),
        }
    }
}

impl Display for EventDomainAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(value) => write!(f, "Name({value})"),
            Self::U32(value) => write!(f, "U32({value})"),
            Self::CollectionMethod(value) => write!(f, "CollectionMethod({value})"),
        }
    }
}

impl Display for EventGroupAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventDomainId(value) => write!(f, "EventDomainId({})", value.get()),
            Self::ProfileAllDomainInstances(value) => {
                write!(f, "ProfileAllDomainInstances({value})")
            }
            Self::NumEvents(value) => write!(f, "NumEvents({value})"),
            Self::Events(events) => {
                write!(f, "Events([")?;
                for (index, event) in events.iter().enumerate() {
                    if index != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", event.get())?;
                }
                write!(f, "])")
            }
            Self::InstanceCount(value) => write!(f, "InstanceCount({value})"),
            Self::ProfilingScope(value) => write!(f, "ProfilingScope({value})"),
        }
    }
}

impl Display for EventGroupAttributeSetting {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileAllDomainInstances(value) => {
                write!(f, "ProfileAllDomainInstances({value})")
            }
        }
    }
}

impl Display for EventAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(value) => write!(f, "Name({value})"),
            Self::ShortDescription(value) => write!(f, "ShortDescription({value})"),
            Self::LongDescription(value) => write!(f, "LongDescription({value})"),
            Self::Category(value) => write!(f, "Category({value})"),
            Self::ProfilingScope(value) => write!(f, "ProfilingScope({value})"),
        }
    }
}

impl_enum_display!(ActivityFlushFlag, {
    Self::Default => "CUPTI_ACTIVITY_FLAG_NONE",
    Self::Forced => "CUPTI_ACTIVITY_FLAG_FLUSH_FORCED",
});

impl Display for MultipleSubscriberState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "Disabled"),
            Self::Enabled => write!(f, "Enabled"),
            Self::Unset => write!(f, "Unset"),
            Self::Unknown(value) => write!(f, "Unknown({value})"),
        }
    }
}

impl Display for ActivityAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(value) => write!(f, "Bytes({value})"),
            Self::Count(value) => write!(f, "Count({value})"),
            Self::Enabled(value) => write!(f, "Enabled({value})"),
            Self::MultipleSubscriberState(value) => write!(f, "MultipleSubscriberState({value})"),
            Self::ThreadIdType(value) => write!(f, "ThreadIdType({value})"),
        }
    }
}

impl Display for ActivityAttributeSetting {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(value) => write!(f, "Bytes({value})"),
            Self::Count(value) => write!(f, "Count({value})"),
            Self::Enabled(value) => write!(f, "Enabled({value})"),
            Self::ThreadIdType(value) => write!(f, "ThreadIdType({value})"),
        }
    }
}

impl Display for MetricValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Double(value) => write!(f, "Double({value})"),
            Self::Uint64(value) => write!(f, "Uint64({value})"),
            Self::Percent(value) => write!(f, "Percent({value})"),
            Self::Throughput(value) => write!(f, "Throughput({value})"),
            Self::Int64(value) => write!(f, "Int64({value})"),
            Self::UtilizationLevel(value) => write!(f, "UtilizationLevel({value})"),
            Self::NvtxExtendedPayload(value) => {
                write!(f, "NvtxExtendedPayload({value})")
            }
        }
    }
}

impl Display for MetricAttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(value) => write!(f, "Name({value})"),
            Self::ShortDescription(value) => write!(f, "ShortDescription({value})"),
            Self::LongDescription(value) => write!(f, "LongDescription({value})"),
            Self::Category(value) => write!(f, "Category({value})"),
            Self::ValueKind(value) => write!(f, "ValueKind({value})"),
            Self::EvaluationMode(value) => write!(f, "EvaluationMode({value})"),
        }
    }
}
