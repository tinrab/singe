use singe_cuda_sys::{driver, runtime};

use crate::{
    error::{Error, Result},
    utility::to_usize,
};

/// An opaque 16-byte UUID returned by CUDA device or fabric queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Uuid {
    /// UUID bytes in the order returned by CUDA.
    pub bytes: [u8; 16],
}

impl From<driver::CUuuid> for Uuid {
    fn from(value: driver::CUuuid) -> Self {
        Self {
            bytes: value.bytes.map(|byte| byte as u8),
        }
    }
}

impl From<Uuid> for driver::CUuuid {
    fn from(value: Uuid) -> Self {
        driver::CUuuid {
            bytes: value.bytes.map(u8::cast_signed),
        }
    }
}

/// Rust representation of CUDA device properties.
#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent capabilities and option switches can be combined freely"
)]
pub struct DeviceProperties {
    /// ASCII string identifying device.
    pub name: String,
    /// 16-byte unique identifier.
    pub uuid: Uuid,
    /// 8-byte locally unique identifier.
    /// Value is undefined on TCC and non-Windows platforms.
    pub luid: [u8; 8],
    /// LUID device node mask.
    /// Value is undefined on TCC and non-Windows platforms.
    pub luid_device_node_mask: u32,
    /// Global memory available on device in bytes.
    pub total_global_mem: usize,
    /// Shared memory available per block in bytes.
    pub shared_mem_per_block: usize,
    /// 32-bit registers available per block.
    pub regs_per_block: i32,
    /// Warp size in threads.
    pub warp_size: i32,
    /// Maximum pitch in bytes allowed by memory copies.
    pub mem_pitch: usize,
    /// Maximum number of threads per block.
    pub max_threads_per_block: i32,
    /// Maximum size of each dimension of a block.
    pub max_threads_dim: [i32; 3],
    /// Maximum size of each dimension of a grid.
    pub max_grid_size: [i32; 3],
    /// Constant memory available on device in bytes.
    pub total_const_mem: usize,
    /// Major compute capability.
    pub major: i32,
    /// Minor compute capability.
    pub minor: i32,
    /// Alignment requirement for textures.
    pub texture_alignment: usize,
    /// Pitch alignment requirement for texture references bound to pitched memory.
    pub texture_pitch_alignment: usize,
    /// Number of multiprocessors on device.
    pub multi_processor_count: i32,
    /// Device is integrated as opposed to discrete.
    pub integrated: bool,
    /// Device can map host memory into CUDA address space.
    pub can_map_host_memory: bool,
    /// Maximum 1D texture size.
    pub max_texture1d: i32,
    /// Maximum 1D mipmapped texture size.
    pub max_texture1d_mipmap: i32,
    /// Maximum 2D texture dimensions.
    pub max_texture2d: [i32; 2],
    /// Maximum 2D mipmapped texture dimensions.
    pub max_texture2d_mipmap: [i32; 2],
    /// Maximum dimensions (width, height, pitch) for 2D textures bound to linear memory.
    pub max_texture2d_linear: [i32; 3],
    /// Maximum 2D texture dimensions for texture gather operations.
    pub max_texture2d_gather: [i32; 2],
    /// Maximum 3D texture dimensions.
    pub max_texture3d: [i32; 3],
    /// Maximum alternate 3D texture dimensions.
    pub max_texture3d_alt: [i32; 3],
    /// Maximum Cubemap texture dimensions.
    pub max_texture_cubemap: i32,
    /// Maximum 1D layered texture dimensions.
    pub max_texture1d_layered: [i32; 2],
    /// Maximum 2D layered texture dimensions.
    pub max_texture2d_layered: [i32; 3],
    /// Maximum Cubemap layered texture dimensions.
    pub max_texture_cubemap_layered: [i32; 2],
    /// Maximum 1D surface size.
    pub max_surface1d: i32,
    /// Maximum 2D surface dimensions.
    pub max_surface2d: [i32; 2],
    /// Maximum 3D surface dimensions.
    pub max_surface3d: [i32; 3],
    /// Maximum 1D layered surface dimensions.
    pub max_surface1d_layered: [i32; 2],
    /// Maximum 2D layered surface dimensions.
    pub max_surface2d_layered: [i32; 3],
    /// Maximum Cubemap surface dimensions.
    pub max_surface_cubemap: i32,
    /// Maximum Cubemap layered surface dimensions.
    pub max_surface_cubemap_layered: [i32; 2],
    /// Alignment requirements for surfaces.
    pub surface_alignment: usize,
    /// Device can possibly execute multiple kernels concurrently.
    pub concurrent_kernels: bool,
    /// Device has ECC support enabled.
    pub ecc_enabled: bool,
    /// PCI bus ID of the device.
    pub pci_bus_id: i32,
    /// PCI device ID of the device.
    pub pci_device_id: i32,
    /// PCI domain ID of the device.
    pub pci_domain_id: i32,
    /// 1 if device is a Tesla device using TCC driver, 0 otherwise.
    pub tcc_driver: bool,
    /// Number of asynchronous engines.
    pub async_engine_count: i32,
    /// Device shares a unified address space with the host.
    pub unified_addressing: bool,
    /// Global memory bus width in bits.
    pub memory_bus_width: i32,
    /// Size of L2 cache in bytes.
    pub l2_cache_size: i32,
    /// Device's maximum l2 persisting lines capacity setting in bytes.
    pub persisting_l2_cache_max_size: i32,
    /// Maximum resident threads per multiprocessor.
    pub max_threads_per_multi_processor: i32,
    /// Device supports stream priorities.
    pub stream_priorities_supported: bool,
    /// Device supports caching globals in L1.
    pub global_l1_cache_supported: bool,
    /// Device supports caching locals in L1.
    pub local_l1_cache_supported: bool,
    /// Shared memory available per multiprocessor in bytes.
    pub shared_mem_per_multiprocessor: usize,
    /// 32-bit registers available per multiprocessor.
    pub regs_per_multiprocessor: i32,
    /// Device supports allocating managed memory on this system.
    pub managed_memory: bool,
    /// Device is on a multi-GPU board.
    pub is_multi_gpu_board: bool,
    /// Unique identifier for a group of devices on the same multi-GPU board.
    pub multi_gpu_board_group_id: i32,
    /// Link between the device and the host supports native atomic operations.
    pub host_native_atomic_supported: bool,
    /// Device supports coherently accessing pageable memory without registering it with [`RegisteredHostMemory`](crate::memory::RegisteredHostMemory).
    pub pageable_memory_access: bool,
    /// Device can coherently access managed memory concurrently with the CPU.
    pub concurrent_managed_access: bool,
    /// Device supports Compute Preemption.
    pub compute_preemption_supported: bool,
    /// Device can access host registered memory at the same virtual address as the CPU.
    pub can_use_host_pointer_for_registered_mem: bool,
    /// Device supports cooperative kernel launches.
    pub cooperative_launch: bool,
    /// Per device maximum shared memory per block usable by special opt-in.
    pub shared_mem_per_block_optin: usize,
    /// Device accesses pageable memory via the host's page tables.
    pub pageable_memory_access_uses_host_page_tables: bool,
    /// Host can directly access managed memory on the device without migration.
    pub direct_managed_mem_access_from_host: bool,
    /// Maximum number of resident blocks per multiprocessor.
    pub max_blocks_per_multi_processor: i32,
    /// Maximum value of the CUDA access-policy window `num_bytes` field.
    pub access_policy_max_window_size: i32,
    /// Shared memory reserved by CUDA driver per block in bytes.
    pub reserved_shared_mem_per_block: usize,
    /// Device supports host memory registration via [`RegisteredHostMemory`](crate::memory::RegisteredHostMemory).
    pub host_register_supported: bool,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
    pub sparse_cuda_array_supported: bool,
    /// Device supports [`HostRegisterFlags::READ_ONLY`](crate::memory::HostRegisterFlags::READ_ONLY) for host registrations mapped as read-only to the GPU.
    pub host_register_read_only_supported: bool,
    /// External timeline semaphore interop is supported.
    pub timeline_semaphore_interop_supported: bool,
    /// Device supports CUDA memory pools.
    pub memory_pools_supported: bool,
    /// Device supports `GPUDirect` RDMA APIs.
    pub gpu_direct_rdma_supported: bool,
    /// The returned flags may be used as subset of the supported write ordering MASTs supplied with `GPUDirect` RDMA writes.
    pub gpu_direct_rdma_flush_writes_options: u32,
    /// `GPUDirect` RDMA writes are guaranteed to be ordered with respect to other `GPUDirect` RDMA writes from the same GPU.
    pub gpu_direct_rdma_writes_ordering: i32,
    /// Handle types supported with mempool based IPC.
    pub memory_pool_supported_handle_types: u32,
    /// Indicates device supports deferred mapping CUDA arrays and mapping hints.
    pub deferred_mapping_cuda_array_supported: bool,
    /// Device supports IPC Events.
    pub ipc_event_supported: bool,
    /// Device supports Cluster Launch.
    pub cluster_launch: bool,
    /// Device supports unified function pointers.
    pub unified_function_pointers: bool,
}

impl TryFrom<runtime::cudaDeviceProp> for DeviceProperties {
    type Error = Error;

    #[expect(
        clippy::too_many_lines,
        reason = "the complete native device property record is converted field by field"
    )]
    fn try_from(value: runtime::cudaDeviceProp) -> Result<Self> {
        let end = value
            .name
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(value.name.len());
        let name_bytes: Vec<u8> = value.name[..end].iter().map(|&byte| byte as u8).collect();
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        let prop = Self {
            name,
            uuid: value.uuid.into(),
            luid: value.luid.map(|byte| byte as u8),
            luid_device_node_mask: value.luidDeviceNodeMask,
            total_global_mem: to_usize(value.totalGlobalMem, "device property size")?,
            shared_mem_per_block: to_usize(value.sharedMemPerBlock, "device property size")?,
            regs_per_block: value.regsPerBlock,
            warp_size: value.warpSize,
            mem_pitch: to_usize(value.memPitch, "device property size")?,
            max_threads_per_block: value.maxThreadsPerBlock,
            max_threads_dim: value.maxThreadsDim,
            max_grid_size: value.maxGridSize,
            total_const_mem: to_usize(value.totalConstMem, "device property size")?,
            major: value.major,
            minor: value.minor,
            texture_alignment: to_usize(value.textureAlignment, "device property size")?,
            texture_pitch_alignment: to_usize(value.texturePitchAlignment, "device property size")?,
            multi_processor_count: value.multiProcessorCount,
            integrated: value.integrated != 0,
            can_map_host_memory: value.canMapHostMemory != 0,
            max_texture1d: value.maxTexture1D,
            max_texture1d_mipmap: value.maxTexture1DMipmap,
            max_texture2d: value.maxTexture2D,
            max_texture2d_mipmap: value.maxTexture2DMipmap,
            max_texture2d_linear: value.maxTexture2DLinear,
            max_texture2d_gather: value.maxTexture2DGather,
            max_texture3d: value.maxTexture3D,
            max_texture3d_alt: value.maxTexture3DAlt,
            max_texture_cubemap: value.maxTextureCubemap,
            max_texture1d_layered: value.maxTexture1DLayered,
            max_texture2d_layered: value.maxTexture2DLayered,
            max_texture_cubemap_layered: value.maxTextureCubemapLayered,
            max_surface1d: value.maxSurface1D,
            max_surface2d: value.maxSurface2D,
            max_surface3d: value.maxSurface3D,
            max_surface1d_layered: value.maxSurface1DLayered,
            max_surface2d_layered: value.maxSurface2DLayered,
            max_surface_cubemap: value.maxSurfaceCubemap,
            max_surface_cubemap_layered: value.maxSurfaceCubemapLayered,
            surface_alignment: to_usize(value.surfaceAlignment, "device property size")?,
            concurrent_kernels: value.concurrentKernels != 0,
            ecc_enabled: value.ECCEnabled != 0,
            pci_bus_id: value.pciBusID,
            pci_device_id: value.pciDeviceID,
            pci_domain_id: value.pciDomainID,
            tcc_driver: value.tccDriver != 0,
            async_engine_count: value.asyncEngineCount,
            unified_addressing: value.unifiedAddressing != 0,
            memory_bus_width: value.memoryBusWidth,
            l2_cache_size: value.l2CacheSize,
            persisting_l2_cache_max_size: value.persistingL2CacheMaxSize,
            max_threads_per_multi_processor: value.maxThreadsPerMultiProcessor,
            stream_priorities_supported: value.streamPrioritiesSupported != 0,
            global_l1_cache_supported: value.globalL1CacheSupported != 0,
            local_l1_cache_supported: value.localL1CacheSupported != 0,
            shared_mem_per_multiprocessor: to_usize(
                value.sharedMemPerMultiprocessor,
                "device property size",
            )?,
            regs_per_multiprocessor: value.regsPerMultiprocessor,
            managed_memory: value.managedMemory != 0,
            is_multi_gpu_board: value.isMultiGpuBoard != 0,
            multi_gpu_board_group_id: value.multiGpuBoardGroupID,
            host_native_atomic_supported: value.hostNativeAtomicSupported != 0,
            pageable_memory_access: value.pageableMemoryAccess != 0,
            concurrent_managed_access: value.concurrentManagedAccess != 0,
            compute_preemption_supported: value.computePreemptionSupported != 0,
            can_use_host_pointer_for_registered_mem: value.canUseHostPointerForRegisteredMem != 0,
            cooperative_launch: value.cooperativeLaunch != 0,
            shared_mem_per_block_optin: to_usize(
                value.sharedMemPerBlockOptin,
                "device property size",
            )?,
            pageable_memory_access_uses_host_page_tables: value
                .pageableMemoryAccessUsesHostPageTables
                != 0,
            direct_managed_mem_access_from_host: value.directManagedMemAccessFromHost != 0,
            max_blocks_per_multi_processor: value.maxBlocksPerMultiProcessor,
            access_policy_max_window_size: value.accessPolicyMaxWindowSize,
            reserved_shared_mem_per_block: to_usize(
                value.reservedSharedMemPerBlock,
                "device property size",
            )?,
            host_register_supported: value.hostRegisterSupported != 0,
            sparse_cuda_array_supported: value.sparseCudaArraySupported != 0,
            host_register_read_only_supported: value.hostRegisterReadOnlySupported != 0,
            timeline_semaphore_interop_supported: value.timelineSemaphoreInteropSupported != 0,
            memory_pools_supported: value.memoryPoolsSupported != 0,
            gpu_direct_rdma_supported: value.gpuDirectRDMASupported != 0,
            gpu_direct_rdma_flush_writes_options: value.gpuDirectRDMAFlushWritesOptions,
            gpu_direct_rdma_writes_ordering: value.gpuDirectRDMAWritesOrdering,
            memory_pool_supported_handle_types: value.memoryPoolSupportedHandleTypes,
            deferred_mapping_cuda_array_supported: value.deferredMappingCudaArraySupported != 0,
            ipc_event_supported: value.ipcEventSupported != 0,
            cluster_launch: value.clusterLaunch != 0,
            unified_function_pointers: value.unifiedFunctionPointers != 0,
        };

        Ok(prop)
    }
}
