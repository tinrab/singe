use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::driver;

singe_core::native_enum! {
    /// Pointer information
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PointerAttribute: driver::CUpointer_attribute {
        /// Returns in `*data` an integer representing the locality domain ordinal of the memory allocation, or -1 if the allocation is not localized to a locality domain.
        LocalityDomainOrdinal = CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
        /// The `CUcontext` on which a pointer was allocated or registered.
        Context = CU_POINTER_ATTRIBUTE_CONTEXT,
        /// The [CUmemorytype](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1ga8a114cc994ad2e865c44ef3838eaec72) describing the physical location of a pointer.
        MemoryType = CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        /// The address at which a pointer's memory may be accessed on the device.
        DevicePointer = CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
        /// The address at which a pointer's memory may be accessed on the host.
        HostPointer = CU_POINTER_ATTRIBUTE_HOST_POINTER,
        /// A pair of tokens for use with the nv-p2p.h Linux kernel interface.
        P2pTokens = CU_POINTER_ATTRIBUTE_P2P_TOKENS,
        /// Synchronize every synchronous memory operation initiated on this region.
        SyncMemops = CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
        /// A process-wide unique ID for an allocated memory region.
        BufferId = CU_POINTER_ATTRIBUTE_BUFFER_ID,
        /// Indicates if the pointer points to managed memory.
        IsManaged = CU_POINTER_ATTRIBUTE_IS_MANAGED,
        /// A device ordinal of a device on which a pointer was allocated or registered.
        DeviceOrdinal = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        /// Reports whether the allocation supports legacy CUDA IPC memory handles.
        IsLegacyCudaIpcCapable = CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
        /// Starting address for this requested pointer.
        RangeStartAddr = CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        /// Size of the address range for this requested pointer.
        RangeSize = CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        /// 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise.
        Mapped = CU_POINTER_ATTRIBUTE_MAPPED,
        /// Bitmask of allowed [CUmemAllocationHandleType](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1ga450a23153d86fce0afe30e25d63caef9) for this allocation.
        AllowedHandleTypes = CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
        /// 1 if the memory this pointer is referencing can be used with the `GPUDirect` RDMA API.
        IsGpuDirectRdmaCapable = CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
        /// Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given.
        AccessFlags = CU_POINTER_ATTRIBUTE_ACCESS_FLAGS,
        /// Returns the mempool handle for the allocation if it was allocated from a mempool.
        ///
        /// Otherwise returns NULL.
        MempoolHandle = CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
        /// Size of the actual underlying mapping that the pointer belongs to.
        MappingSize = CU_POINTER_ATTRIBUTE_MAPPING_SIZE,
        /// The start address of the mapping that the pointer belongs to.
        MappingBaseAddr = CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR,
        /// A process-wide unique id corresponding to the physical allocation the pointer belongs to.
        MemoryBlockId = CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID,
    }
}

impl_enum_display!(PointerAttribute, {
    Self::LocalityDomainOrdinal => "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL",
    Self::Context => "CU_POINTER_ATTRIBUTE_CONTEXT",
    Self::MemoryType => "CU_POINTER_ATTRIBUTE_MEMORY_TYPE",
    Self::DevicePointer => "CU_POINTER_ATTRIBUTE_DEVICE_POINTER",
    Self::HostPointer => "CU_POINTER_ATTRIBUTE_HOST_POINTER",
    Self::P2pTokens => "CU_POINTER_ATTRIBUTE_P2P_TOKENS",
    Self::SyncMemops => "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS",
    Self::BufferId => "CU_POINTER_ATTRIBUTE_BUFFER_ID",
    Self::IsManaged => "CU_POINTER_ATTRIBUTE_IS_MANAGED",
    Self::DeviceOrdinal => "CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL",
    Self::IsLegacyCudaIpcCapable => "CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE",
    Self::RangeStartAddr => "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR",
    Self::RangeSize => "CU_POINTER_ATTRIBUTE_RANGE_SIZE",
    Self::Mapped => "CU_POINTER_ATTRIBUTE_MAPPED",
    Self::AllowedHandleTypes => "CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES",
    Self::IsGpuDirectRdmaCapable => "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE",
    Self::AccessFlags => "CU_POINTER_ATTRIBUTE_ACCESS_FLAGS",
    Self::MempoolHandle => "CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE",
    Self::MappingSize => "CU_POINTER_ATTRIBUTE_MAPPING_SIZE",
    Self::MappingBaseAddr => "CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR",
    Self::MemoryBlockId => "CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID",
});

singe_core::native_enum! {
    /// Memory types
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryType: driver::CUmemorytype {
        /// Host memory.
        Host = CU_MEMORYTYPE_HOST,
        /// Device memory.
        Device = CU_MEMORYTYPE_DEVICE,
        /// Array memory.
        Array = CU_MEMORYTYPE_ARRAY,
        /// Unified device or host memory.
        Unified = CU_MEMORYTYPE_UNIFIED,
    }
}

impl_enum_display!(MemoryType, {
    Self::Host => "CU_MEMORYTYPE_HOST",
    Self::Device => "CU_MEMORYTYPE_DEVICE",
    Self::Array => "CU_MEMORYTYPE_ARRAY",
    Self::Unified => "CU_MEMORYTYPE_UNIFIED",
});

singe_core::native_enum! {
    /// Memory advise values
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryAdvise: driver::CUmem_advise {
        /// Data will mostly be read and only occasionally be written to.
        SetReadMostly = CU_MEM_ADVISE_SET_READ_MOSTLY,
        /// Undoes the effect of [`MemoryAdvise::SetReadMostly`].
        UnsetReadMostly = CU_MEM_ADVISE_UNSET_READ_MOSTLY,
        /// Set the preferred location for the data as the specified device.
        SetPreferredLocation = CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
        /// Clear the preferred location for the data.
        UnsetPreferredLocation = CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
        /// Data will be accessed by the specified device, so prevent page faults as much as possible.
        SetAccessedBy = CU_MEM_ADVISE_SET_ACCESSED_BY,
        /// Let the Unified Memory subsystem decide on the page faulting policy for the specified device.
        UnsetAccessedBy = CU_MEM_ADVISE_UNSET_ACCESSED_BY,
    }
}

impl_enum_display!(MemoryAdvise, {
    Self::SetReadMostly => "CU_MEM_ADVISE_SET_READ_MOSTLY",
    Self::UnsetReadMostly => "CU_MEM_ADVISE_UNSET_READ_MOSTLY",
    Self::SetPreferredLocation => "CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
    Self::UnsetPreferredLocation => "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
    Self::SetAccessedBy => "CU_MEM_ADVISE_SET_ACCESSED_BY",
    Self::UnsetAccessedBy => "CU_MEM_ADVISE_UNSET_ACCESSED_BY",
});

singe_core::native_enum! {
    /// Attribute queried for a managed-memory address range.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryRangeAttribute: driver::CUmem_range_attribute {
        /// Whether the range will mostly be read and only occasionally be written to.
        ReadMostly = CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
        /// The preferred location of the range.
        PreferredLocation = CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
        /// The devices that have [`MemoryAdvise::SetAccessedBy`] set for the range.
        AccessedBy = CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
        /// The last location to which the range was prefetched.
        LastPrefetchLocation = CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    }
}

impl_enum_display!(MemoryRangeAttribute, {
    Self::ReadMostly => "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
    Self::PreferredLocation => "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",
    Self::AccessedBy => "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",
    Self::LastPrefetchLocation => "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION",
});

singe_core::native_enum! {
    /// Cache performance hint for the hit and miss portions of an access-policy window.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum AccessProperty: driver::CUaccessProperty {
        /// Normal cache persistence.
        Normal = CU_ACCESS_PROPERTY_NORMAL,
        /// Streaming access is less likely to persit from cache.
        Streaming = CU_ACCESS_PROPERTY_STREAMING,
        /// Persisting access is more likely to persist in cache.
        Persisting = CU_ACCESS_PROPERTY_PERSISTING,
    }
}

impl_enum_display!(AccessProperty, {
    Self::Normal => "CU_ACCESS_PROPERTY_NORMAL",
    Self::Streaming => "CU_ACCESS_PROPERTY_STREAMING",
    Self::Persisting => "CU_ACCESS_PROPERTY_PERSISTING",
});
