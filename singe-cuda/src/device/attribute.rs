use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::driver;

use super::Device;
use crate::{
    error::{Error, Result},
    try_ffi,
};

singe_core::native_enum! {
    /// Device properties that can be queried using the driver.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DeviceAttribute: driver::CUdevice_attribute_enum {
        /// Maximum number of threads per block.
        MaxThreadsPerBlock = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        /// Maximum block dimension X.
        MaxBlockDimX = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        /// Maximum block dimension Y.
        MaxBlockDimY = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        /// Maximum block dimension Z.
        MaxBlockDimZ = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        /// Maximum grid dimension X.
        MaxGridDimX = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        /// Maximum grid dimension Y.
        MaxGridDimY = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        /// Maximum grid dimension Z.
        MaxGridDimZ = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        /// Maximum shared memory available per block in bytes.
        MaxSharedMemoryPerBlock = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        /// Memory available on device for **constant** variables in a CUDA C kernel in bytes.
        TotalConstantMemory = CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        /// Warp size in threads.
        WarpSize = CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        /// Maximum pitch in bytes allowed by memory copies.
        MaxPitch = CU_DEVICE_ATTRIBUTE_MAX_PITCH,
        /// Maximum number of 32-bit registers available per block.
        MaxRegistersPerBlock = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        /// Typical clock frequency in kilohertz.
        ClockRate = CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        /// Alignment requirement for textures.
        TextureAlignment = CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        /// Device can possibly copy memory and execute a kernel concurrently.
        ///
        /// Deprecated.
        /// Use [`Self::AsyncEngineCount`] instead.
        GpuOverlap = CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
        /// Number of multiprocessors on device.
        MultiprocessorCount = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        /// Specifies whether there is a run time limit on kernels.
        KernelExecTimeout = CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
        /// Device is integrated with host memory.
        Integrated = CU_DEVICE_ATTRIBUTE_INTEGRATED,
        /// Device can map host memory into CUDA address space.
        CanMapHostMemory = CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        /// Compute mode, as a [`ComputeMode`] value.
        ComputeMode = CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
        /// Maximum 1D texture width.
        MaximumTexture1DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
        /// Maximum 2D texture width.
        MaximumTexture2DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
        /// Maximum 2D texture height.
        MaximumTexture2DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
        /// Maximum 3D texture width.
        MaximumTexture3DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
        /// Maximum 3D texture height.
        MaximumTexture3DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
        /// Maximum 3D texture depth.
        MaximumTexture3DDepth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
        /// Maximum 2D layered texture width.
        MaximumTexture2DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
        /// Maximum 2D layered texture height.
        MaximumTexture2DLayeredHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
        /// Maximum layers in a 2D layered texture.
        MaximumTexture2DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
        /// Alignment requirement for surfaces.
        SurfaceAlignment = CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
        /// Device can possibly execute multiple kernels concurrently.
        ConcurrentKernels = CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
        /// Device has ECC support enabled.
        EccEnabled = CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
        /// PCI bus ID of the device.
        PciBusId = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        /// PCI device ID of the device.
        PciDeviceId = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        /// Device is using TCC driver model.
        TccDriver = CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
        /// Peak memory clock frequency in kilohertz.
        MemoryClockRate = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        /// Global memory bus width in bits.
        GlobalMemoryBusWidth = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        /// Size of L2 cache in bytes.
        L2CacheSize = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        /// Maximum resident threads per multiprocessor.
        MaxThreadsPerMultiprocessor = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        /// Number of asynchronous engines.
        AsyncEngineCount = CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
        /// Device shares a unified address space with the host.
        UnifiedAddressing = CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        /// Maximum 1D layered texture width.
        MaximumTexture1DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
        /// Maximum layers in a 1D layered texture.
        MaximumTexture1DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
        /// do not use.
        CanTex2DGather = CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
        /// Maximum 2D texture width if `CUDA_ARRAY3D_TEXTURE_GATHER` is set.
        MaximumTexture2DGatherWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
        /// Maximum 2D texture height if `CUDA_ARRAY3D_TEXTURE_GATHER` is set.
        MaximumTexture2DGatherHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
        /// Alternate maximum 3D texture width.
        MaximumTexture3DWidthAlternate = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
        /// Alternate maximum 3D texture height.
        MaximumTexture3DHeightAlternate = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
        /// Alternate maximum 3D texture depth.
        MaximumTexture3DDepthAlternate = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
        /// PCI domain ID of the device.
        PciDomainId = CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
        /// Pitch alignment requirement for textures.
        TexturePitchAlignment = CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
        /// Maximum cubemap texture width/height.
        MaximumTexturecubemapWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
        /// Maximum cubemap layered texture width/height.
        MaximumTexturecubemapLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
        /// Maximum layers in a cubemap layered texture.
        MaximumTexturecubemapLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
        /// Maximum 1D surface width.
        MaximumSurface1DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
        /// Maximum 2D surface width.
        MaximumSurface2DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
        /// Maximum 2D surface height.
        MaximumSurface2DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
        /// Maximum 3D surface width.
        MaximumSurface3DWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
        /// Maximum 3D surface height.
        MaximumSurface3DHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
        /// Maximum 3D surface depth.
        MaximumSurface3DDepth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
        /// Maximum 1D layered surface width.
        MaximumSurface1DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
        /// Maximum layers in a 1D layered surface.
        MaximumSurface1DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
        /// Maximum 2D layered surface width.
        MaximumSurface2DLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
        /// Maximum 2D layered surface height.
        MaximumSurface2DLayeredHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
        /// Maximum layers in a 2D layered surface.
        MaximumSurface2DLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
        /// Maximum cubemap surface width.
        MaximumSurfacecubemapWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
        /// Maximum cubemap layered surface width.
        MaximumSurfacecubemapLayeredWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
        /// Maximum layers in a cubemap layered surface.
        MaximumSurfacecubemapLayeredLayers = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
        /// Deprecated maximum 1D linear texture width.
        ///
        /// The driver's per-format 1D linear texture width query replaces it.
        MaximumTexture1DLinearWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
        /// Maximum 2D linear texture width.
        MaximumTexture2DLinearWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
        /// Maximum 2D linear texture height.
        MaximumTexture2DLinearHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
        /// Maximum 2D linear texture pitch in bytes.
        MaximumTexture2DLinearPitch = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
        /// Maximum mipmapped 2D texture width.
        MaximumTexture2DMipmappedWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
        /// Maximum mipmapped 2D texture height.
        MaximumTexture2DMipmappedHeight = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
        /// Major compute capability version number.
        ComputeCapabilityMajor = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        /// Minor compute capability version number.
        ComputeCapabilityMinor = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        /// Maximum mipmapped 1D texture width.
        MaximumTexture1DMipmappedWidth = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
        /// Device supports stream priorities.
        StreamPrioritiesSupported = CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
        /// Device supports caching globals in L1.
        GlobalL1CacheSupported = CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
        /// Device supports caching locals in L1.
        LocalL1CacheSupported = CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
        /// Maximum shared memory available per multiprocessor in bytes.
        MaxSharedMemoryPerMultiprocessor = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        /// Maximum number of 32-bit registers available per multiprocessor.
        MaxRegistersPerMultiprocessor = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
        /// Device can allocate managed memory on this system.
        ManagedMemory = CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
        /// Device is on a multi-GPU board.
        MultiGpuBoard = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
        /// Unique id for a group of devices on the same multi-GPU board.
        MultiGpuBoardGroupId = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
        /// Link between the device and the host supports all native atomic operations.
        HostNativeAtomicSupported = CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
        /// Ratio of single precision performance (in floating-point operations per second) to double precision performance.
        SingleToDoublePrecisionPerfRatio = CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
        /// Device supports coherently accessing pageable memory without registering it.
        PageableMemoryAccess = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        /// Device can coherently access managed memory concurrently with the CPU.
        ConcurrentManagedAccess = CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
        /// Device supports compute preemption.
        ComputePreemptionSupported = CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
        /// Device can access host registered memory at the same virtual address as the CPU.
        CanUseHostPointerForRegisteredMem = CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
        /// along with v1 `MemOps` API, batched stream memory operations are supported.
        CanUseStreamMemOpsV1 = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
        /// along with v1 `MemOps` API, 64-bit batched stream memory operations are supported.
        CanUse64BitStreamMemOpsV1 = CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1,
        /// along with v1 `MemOps` API, the NOR stream wait-value comparison is supported.
        CanUseStreamWaitValueNorV1 = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1,
        /// Device supports launching cooperative kernels.
        CooperativeLaunch = CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        /// Cooperative launches across several devices are deprecated.
        CooperativeMultiDeviceLaunch = CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
        /// Maximum optin shared memory per block.
        ///
        /// That is shared memory that is available for dynamic allocation or static allocation (including architecture specific static shared memory) on this device but is not guaranteed to be portable.
        MaxSharedMemoryPerBlockOptin = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        /// The flushing stream wait-value flag and the flush-remote-writes memory operation are supported on the device.
        CanFlushRemoteWrites = CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
        /// Device supports host memory registration, as [`RegisteredHostMemory`](crate::memory::RegisteredHostMemory) uses.
        HostRegisterSupported = CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
        /// Device accesses pageable memory via the host’s page tables.
        PageableMemoryAccessUsesHostPageTables = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
        /// The host can directly access managed memory on the device without migration.
        DirectManagedMemAccessFromHost = CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
        /// Device supports the virtual memory management APIs.
        VirtualAddressManagementSupported = CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        /// Device supports exporting virtual memory allocations created for it to a POSIX file descriptor.
        HandleTypePosixFileDescriptorSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        /// Device supports exporting virtual memory allocations created for it to a Win32 NT handle.
        HandleTypeWin32HandleSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED,
        /// Device supports exporting virtual memory allocations created for it to a Win32 KMT handle.
        HandleTypeWin32KmtHandleSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED,
        /// Maximum number of blocks per multiprocessor.
        MaxBlocksPerMultiprocessor = CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
        /// Device supports compression of memory.
        GenericCompressionSupported = CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
        /// Maximum L2 persisting lines capacity setting in bytes.
        MaxPersistingL2CacheSize = CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
        /// Maximum byte size of an access policy window.
        MaxAccessPolicyWindowSize = CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
        /// Device supports requesting `GPUDirect` RDMA for virtual memory allocations.
        GpuDirectRdmaWithCudaVmmSupported = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
        /// Shared memory reserved by CUDA driver per block in bytes.
        ReservedSharedMemoryPerBlock = CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,
        /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
        SparseCudaArraySupported = CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED,
        /// Device supports [`HostRegisterFlags::READ_ONLY`](crate::memory::HostRegisterFlags::READ_ONLY) for host registrations mapped read-only to the GPU.
        ReadOnlyHostRegisterSupported = CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
        /// External timeline semaphore interop is supported on the device.
        TimelineSemaphoreInteropSupported = CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED,
        /// Device supports stream-ordered allocation and memory pools.
        MemoryPoolsSupported = CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
        /// Device supports `GPUDirect` RDMA APIs, like `nvidia_p2p_get_pages` (see <https://docs.nvidia.com/cuda/gpudirect-rdma> for more information).
        GpuDirectRdmaSupported = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
        /// The returned value is a bitmask of the driver's `GPUDirect` RDMA write-flushing options.
        GpuDirectRdmaFlushWritesOptions = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
        /// `GPUDirect` RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute.
        ///
        /// The value is one of the driver's `GPUDirect` RDMA write-ordering levels.
        GpuDirectRdmaWritesOrdering = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
        /// Handle types supported with mempool based IPC.
        MempoolSupportedHandleTypes = CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES,
        /// Indicates device supports cluster launch.
        ClusterLaunch = CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH,
        /// Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays.
        DeferredMappingCudaArraySupported = CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED,
        /// 64-bit batched stream memory operations are supported.
        CanUse64BitStreamMemOps = CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
        /// The NOR stream wait-value comparison is supported by stream memory operations.
        CanUseStreamWaitValueNor = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR,
        /// Device supports buffer sharing with `dma_buf` mechanism.
        DmaBufSupported = CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
        /// Device supports IPC Events.
        IpcEventSupported = CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED,
        /// Number of memory domains the device supports.
        MemSyncDomainCount = CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT,
        /// Device supports accessing memory using Tensor Map.
        TensorMapAccessSupported = CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
        /// Device supports exporting virtual memory allocations to a fabric handle.
        HandleTypeFabricSupported = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        /// Device supports unified function pointers.
        UnifiedFunctionPointers = CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS,
        /// NUMA configuration of a device, as the driver's NUMA configuration enum value.
        NumaConfig = CU_DEVICE_ATTRIBUTE_NUMA_CONFIG,
        /// NUMA node ID of the GPU memory.
        NumaId = CU_DEVICE_ATTRIBUTE_NUMA_ID,
        /// Device supports switch multicast and reduction operations.
        MulticastSupported = CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        /// Indicates if contexts created on this device will be shared via MPS.
        MpsEnabled = CU_DEVICE_ATTRIBUTE_MPS_ENABLED,
        /// NUMA ID of the host node closest to the device.
        ///
        /// Returns -1 when system does not support NUMA.
        HostNumaId = CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
        /// Device supports CIG with D3D12.
        D3D12CigSupported = CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED,
        /// The returned value is a bitmask of the driver's decompression algorithms.
        MemDecompressAlgorithmMask = CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,
        /// The returned valued is the maximum length in bytes of a single decompress operation that is allowed.
        MemDecompressMaximumLength = CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH,
        /// Device supports CIG with Vulkan.
        VulkanCigSupported = CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED,
        /// The combined 16-bit PCI device ID and 16-bit PCI vendor ID.
        GpuPciDeviceId = CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID,
        /// The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.
        GpuPciSubsystemId = CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID,
        /// Device supports host NUMA locations for virtual memory management allocations and mappings.
        HostNumaVirtualMemoryManagementSupported = CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        /// Device supports host NUMA locations for stream-ordered allocation and memory pools.
        HostNumaMemoryPoolsSupported = CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED,
        /// Device supports `HOST_NUMA` location IPC between nodes in a multi-node system.
        HostNumaMultinodeIpcSupported = CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED,
        /// Device supports host locations for stream-ordered allocation and memory pools.
        HostMemoryPoolsSupported = CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED,
        /// Device supports host locations for virtual memory management allocations and mappings.
        HostVirtualMemoryManagementSupported = CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        /// Device supports page-locked host memory buffer sharing with `dma_buf` mechanism.
        HostAllocDmaBufSupported = CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED,
        /// Link between the device and the host supports only some native atomic operations.
        OnlyPartialHostNativeAtomicSupported = CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED,
        /// Device supports atomic reduction operations in stream batch memory operations.
        AtomicReductionSupported = CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED,
        /// Number of locality domains.
        LocalityDomainCount = CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT,
        /// Maximum oversized shared memory per block.
        MaxOversizedSharedMemoryPerBlock = CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK,
        /// Device supports CIG streams with D3D12.
        D3D12CigStreamsSupported = CU_DEVICE_ATTRIBUTE_D3D12_CIG_STREAMS_SUPPORTED,
        /// Device supports `mmap()` of dmabuf file descriptors for CUDA device memory allocations.
        DmaBufMmapSupported = CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED,
        /// Device supports unicast logical endpoints.
        LogicalEndpointUnicastSupported = CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
        /// Device supports multicast logical endpoints.
        LogicalEndpointMulticastSupported = CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED,
        /// Device supports counted operations via logical endpoints.
        LogicalEndpointCountedOpsSupported = CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
        /// Device supports unicast logical endpoint access on the owner device.
        LogicalEndpointUnicastAccessOnOwnerDeviceSupported = CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED,
        /// Number of multiprocessors on each locality domain.
        LocalityDomainMultiprocessorCount = CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT,
        /// Handle types supported with logical endpoint IPC.
        LogicalEndpointSupportedHandleTypes = CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES,
        /// Device supports `GPUDirect` RDMA with localized memory using the default RDMA mapping link.
        GpuDirectRdmaWithLocalizedMemorySupported = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_LOCALIZED_MEMORY_SUPPORTED,
    }
}

impl Device {
    /// Returns information about the device.
    ///
    /// # Errors
    ///
    /// Returns an error if the device ordinal or attribute is invalid, CUDA is not initialized, the driver cannot query the attribute, or previous asynchronous work reports an error.
    pub fn attribute(self, attribute: DeviceAttribute) -> Result<i32> {
        let mut value = 0;
        // SAFETY: `value` is a local that `cuDeviceGetAttribute` writes through an exclusive pointer.
        unsafe {
            try_ffi!(driver::cuDeviceGetAttribute(
                &raw mut value,
                attribute.into(),
                self.id()
            ))?;
        }
        Ok(value)
    }
}

impl Device {
    /// Number of locality domains.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the corresponding [`DeviceAttribute`].
    /// Returns `Error::OutOfRange` if the native result is negative or does not fit the return type.
    pub fn locality_domain_count(self) -> Result<u32> {
        self.attribute(DeviceAttribute::LocalityDomainCount)?
            .try_into()
            .map_err(|_| Error::out_of_range("locality domain count"))
    }

    /// Number of multiprocessors on each locality domain.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the corresponding [`DeviceAttribute`].
    /// Returns `Error::OutOfRange` if the native result is negative or does not fit the return type.
    pub fn locality_domain_multiprocessor_count(self) -> Result<u32> {
        self.attribute(DeviceAttribute::LocalityDomainMultiprocessorCount)?
            .try_into()
            .map_err(|_| Error::out_of_range("locality domain multiprocessor count"))
    }

    /// Maximum oversized shared memory per block.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the corresponding [`DeviceAttribute`].
    /// Returns `Error::OutOfRange` if the native result is negative or does not fit the return type.
    pub fn maximum_oversized_shared_memory_per_block(self) -> Result<usize> {
        self.attribute(DeviceAttribute::MaxOversizedSharedMemoryPerBlock)?
            .try_into()
            .map_err(|_| Error::out_of_range("maximum oversized shared memory per block"))
    }

    /// Device supports `GPUDirect` RDMA with localized memory using the default RDMA mapping link.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying [`DeviceAttribute::GpuDirectRdmaWithLocalizedMemorySupported`].
    pub fn localized_memory_rdma_supported(self) -> Result<bool> {
        Ok(self.attribute(DeviceAttribute::GpuDirectRdmaWithLocalizedMemorySupported)? != 0)
    }
}

impl_enum_display!(DeviceAttribute, {
    DeviceAttribute::MaxThreadsPerBlock => "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    DeviceAttribute::MaxBlockDimX => "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
    DeviceAttribute::MaxBlockDimY => "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
    DeviceAttribute::MaxBlockDimZ => "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
    DeviceAttribute::MaxGridDimX => "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
    DeviceAttribute::MaxGridDimY => "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",
    DeviceAttribute::MaxGridDimZ => "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",
    DeviceAttribute::MaxSharedMemoryPerBlock => "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
    DeviceAttribute::TotalConstantMemory => "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",
    DeviceAttribute::WarpSize => "CU_DEVICE_ATTRIBUTE_WARP_SIZE",
    DeviceAttribute::MaxPitch => "CU_DEVICE_ATTRIBUTE_MAX_PITCH",
    DeviceAttribute::MaxRegistersPerBlock => "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
    DeviceAttribute::ClockRate => "CU_DEVICE_ATTRIBUTE_CLOCK_RATE",
    DeviceAttribute::TextureAlignment => "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",
    DeviceAttribute::GpuOverlap => "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP",
    DeviceAttribute::MultiprocessorCount => "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
    DeviceAttribute::KernelExecTimeout => "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",
    DeviceAttribute::Integrated => "CU_DEVICE_ATTRIBUTE_INTEGRATED",
    DeviceAttribute::CanMapHostMemory => "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",
    DeviceAttribute::ComputeMode => "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE",
    DeviceAttribute::MaximumTexture1DWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH",
    DeviceAttribute::MaximumTexture2DWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH",
    DeviceAttribute::MaximumTexture2DHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT",
    DeviceAttribute::MaximumTexture3DWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH",
    DeviceAttribute::MaximumTexture3DHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT",
    DeviceAttribute::MaximumTexture3DDepth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH",
    DeviceAttribute::MaximumTexture2DLayeredWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH",
    DeviceAttribute::MaximumTexture2DLayeredHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT",
    DeviceAttribute::MaximumTexture2DLayeredLayers => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS",
    DeviceAttribute::SurfaceAlignment => "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",
    DeviceAttribute::ConcurrentKernels => "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",
    DeviceAttribute::EccEnabled => "CU_DEVICE_ATTRIBUTE_ECC_ENABLED",
    DeviceAttribute::PciBusId => "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",
    DeviceAttribute::PciDeviceId => "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
    DeviceAttribute::TccDriver => "CU_DEVICE_ATTRIBUTE_TCC_DRIVER",
    DeviceAttribute::MemoryClockRate => "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE",
    DeviceAttribute::GlobalMemoryBusWidth => "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",
    DeviceAttribute::L2CacheSize => "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",
    DeviceAttribute::MaxThreadsPerMultiprocessor => "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
    DeviceAttribute::AsyncEngineCount => "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",
    DeviceAttribute::UnifiedAddressing => "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",
    DeviceAttribute::MaximumTexture1DLayeredWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH",
    DeviceAttribute::MaximumTexture1DLayeredLayers => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS",
    DeviceAttribute::CanTex2DGather => "CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER",
    DeviceAttribute::MaximumTexture2DGatherWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH",
    DeviceAttribute::MaximumTexture2DGatherHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT",
    DeviceAttribute::MaximumTexture3DWidthAlternate => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE",
    DeviceAttribute::MaximumTexture3DHeightAlternate => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE",
    DeviceAttribute::MaximumTexture3DDepthAlternate => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE",
    DeviceAttribute::PciDomainId => "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
    DeviceAttribute::TexturePitchAlignment => "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",
    DeviceAttribute::MaximumTexturecubemapWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH",
    DeviceAttribute::MaximumTexturecubemapLayeredWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH",
    DeviceAttribute::MaximumTexturecubemapLayeredLayers => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS",
    DeviceAttribute::MaximumSurface1DWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH",
    DeviceAttribute::MaximumSurface2DWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH",
    DeviceAttribute::MaximumSurface2DHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT",
    DeviceAttribute::MaximumSurface3DWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH",
    DeviceAttribute::MaximumSurface3DHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT",
    DeviceAttribute::MaximumSurface3DDepth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH",
    DeviceAttribute::MaximumSurface1DLayeredWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH",
    DeviceAttribute::MaximumSurface1DLayeredLayers => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS",
    DeviceAttribute::MaximumSurface2DLayeredWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH",
    DeviceAttribute::MaximumSurface2DLayeredHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT",
    DeviceAttribute::MaximumSurface2DLayeredLayers => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS",
    DeviceAttribute::MaximumSurfacecubemapWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH",
    DeviceAttribute::MaximumSurfacecubemapLayeredWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH",
    DeviceAttribute::MaximumSurfacecubemapLayeredLayers => "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS",
    DeviceAttribute::MaximumTexture1DLinearWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH",
    DeviceAttribute::MaximumTexture2DLinearWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH",
    DeviceAttribute::MaximumTexture2DLinearHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT",
    DeviceAttribute::MaximumTexture2DLinearPitch => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH",
    DeviceAttribute::MaximumTexture2DMipmappedWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH",
    DeviceAttribute::MaximumTexture2DMipmappedHeight => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT",
    DeviceAttribute::ComputeCapabilityMajor => "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
    DeviceAttribute::ComputeCapabilityMinor => "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
    DeviceAttribute::MaximumTexture1DMipmappedWidth => "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH",
    DeviceAttribute::StreamPrioritiesSupported => "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",
    DeviceAttribute::GlobalL1CacheSupported => "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",
    DeviceAttribute::LocalL1CacheSupported => "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED",
    DeviceAttribute::MaxSharedMemoryPerMultiprocessor => "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
    DeviceAttribute::MaxRegistersPerMultiprocessor => "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",
    DeviceAttribute::ManagedMemory => "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY",
    DeviceAttribute::MultiGpuBoard => "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",
    DeviceAttribute::MultiGpuBoardGroupId => "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID",
    DeviceAttribute::HostNativeAtomicSupported => "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
    DeviceAttribute::SingleToDoublePrecisionPerfRatio => "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
    DeviceAttribute::PageableMemoryAccess => "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
    DeviceAttribute::ConcurrentManagedAccess => "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
    DeviceAttribute::ComputePreemptionSupported => "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED",
    DeviceAttribute::CanUseHostPointerForRegisteredMem => "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
    DeviceAttribute::CanUseStreamMemOpsV1 => "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1",
    DeviceAttribute::CanUse64BitStreamMemOpsV1 => "CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1",
    DeviceAttribute::CanUseStreamWaitValueNorV1 => "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1",
    DeviceAttribute::CooperativeLaunch => "CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH",
    DeviceAttribute::CooperativeMultiDeviceLaunch => "CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH",
    DeviceAttribute::MaxSharedMemoryPerBlockOptin => "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
    DeviceAttribute::CanFlushRemoteWrites => "CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES",
    DeviceAttribute::HostRegisterSupported => "CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED",
    DeviceAttribute::PageableMemoryAccessUsesHostPageTables => "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
    DeviceAttribute::DirectManagedMemAccessFromHost => "CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST",
    DeviceAttribute::VirtualAddressManagementSupported => "CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED",
    DeviceAttribute::HandleTypePosixFileDescriptorSupported => "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED",
    DeviceAttribute::HandleTypeWin32HandleSupported => "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED",
    DeviceAttribute::HandleTypeWin32KmtHandleSupported => "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED",
    DeviceAttribute::MaxBlocksPerMultiprocessor => "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR",
    DeviceAttribute::GenericCompressionSupported => "CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED",
    DeviceAttribute::MaxPersistingL2CacheSize => "CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE",
    DeviceAttribute::MaxAccessPolicyWindowSize => "CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE",
    DeviceAttribute::GpuDirectRdmaWithCudaVmmSupported => "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED",
    DeviceAttribute::ReservedSharedMemoryPerBlock => "CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK",
    DeviceAttribute::SparseCudaArraySupported => "CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED",
    DeviceAttribute::ReadOnlyHostRegisterSupported => "CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED",
    DeviceAttribute::TimelineSemaphoreInteropSupported => "CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED",
    DeviceAttribute::MemoryPoolsSupported => "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED",
    DeviceAttribute::GpuDirectRdmaSupported => "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED",
    DeviceAttribute::GpuDirectRdmaFlushWritesOptions => "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS",
    DeviceAttribute::GpuDirectRdmaWritesOrdering => "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING",
    DeviceAttribute::MempoolSupportedHandleTypes => "CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES",
    DeviceAttribute::ClusterLaunch => "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH",
    DeviceAttribute::DeferredMappingCudaArraySupported => "CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED",
    DeviceAttribute::CanUse64BitStreamMemOps => "CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS",
    DeviceAttribute::CanUseStreamWaitValueNor => "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR",
    DeviceAttribute::DmaBufSupported => "CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED",
    DeviceAttribute::IpcEventSupported => "CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED",
    DeviceAttribute::MemSyncDomainCount => "CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT",
    DeviceAttribute::TensorMapAccessSupported => "CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED",
    DeviceAttribute::HandleTypeFabricSupported => "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED",
    DeviceAttribute::UnifiedFunctionPointers => "CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS",
    DeviceAttribute::NumaConfig => "CU_DEVICE_ATTRIBUTE_NUMA_CONFIG",
    DeviceAttribute::NumaId => "CU_DEVICE_ATTRIBUTE_NUMA_ID",
    DeviceAttribute::MulticastSupported => "CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED",
    DeviceAttribute::MpsEnabled => "CU_DEVICE_ATTRIBUTE_MPS_ENABLED",
    DeviceAttribute::HostNumaId => "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID",
    DeviceAttribute::D3D12CigSupported => "CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED",
    DeviceAttribute::MemDecompressAlgorithmMask => "CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK",
    DeviceAttribute::MemDecompressMaximumLength => "CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH",
    DeviceAttribute::VulkanCigSupported => "CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED",
    DeviceAttribute::GpuPciDeviceId => "CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID",
    DeviceAttribute::GpuPciSubsystemId => "CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID",
    DeviceAttribute::HostNumaVirtualMemoryManagementSupported => "CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
    DeviceAttribute::HostNumaMemoryPoolsSupported => "CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED",
    DeviceAttribute::HostNumaMultinodeIpcSupported => "CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED",
    DeviceAttribute::HostMemoryPoolsSupported => "CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED",
    DeviceAttribute::HostVirtualMemoryManagementSupported => "CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
    DeviceAttribute::HostAllocDmaBufSupported => "CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED",
    DeviceAttribute::OnlyPartialHostNativeAtomicSupported => "CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED",
    DeviceAttribute::AtomicReductionSupported => "CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED",
    DeviceAttribute::LocalityDomainCount => "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT",
    DeviceAttribute::MaxOversizedSharedMemoryPerBlock => "CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK",
    DeviceAttribute::D3D12CigStreamsSupported => "CU_DEVICE_ATTRIBUTE_D3D12_CIG_STREAMS_SUPPORTED",
    DeviceAttribute::DmaBufMmapSupported => "CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED",
    DeviceAttribute::LogicalEndpointUnicastSupported => "CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED",
    DeviceAttribute::LogicalEndpointMulticastSupported => "CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED",
    DeviceAttribute::LogicalEndpointCountedOpsSupported => "CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED",
    DeviceAttribute::LogicalEndpointUnicastAccessOnOwnerDeviceSupported => "CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED",
    DeviceAttribute::LocalityDomainMultiprocessorCount => "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT",
    DeviceAttribute::LogicalEndpointSupportedHandleTypes => "CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES",
    DeviceAttribute::GpuDirectRdmaWithLocalizedMemorySupported => "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_LOCALIZED_MEMORY_SUPPORTED",
});
