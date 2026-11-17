use singe_cupti_sys as sys;

macro_rules! id_newtype {
    ($name:ident, $raw:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(transparent)]
        /// CUPTI identifier wrapper.
        pub struct $name(pub(crate) $raw);

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
    #[must_use]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// CUDA driver device attribute identifier used in CUPTI records.
pub struct CudaDeviceAttribute(u32);

impl CudaDeviceAttribute {
    /// Returns the raw value decoded from CUPTI.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<sys::CUdevice_attribute> for CudaDeviceAttribute {
    fn from(value: sys::CUdevice_attribute) -> Self {
        Self(value.0)
    }
}
