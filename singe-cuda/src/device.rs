#![allow(deprecated)]

use std::{
    ffi::CString,
    mem::{self, MaybeUninit},
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display, string_from_c_chars};
use singe_cuda_sys::driver;
use singe_cuda_sys::runtime;

use crate::{
    context::ContextFlags,
    error::{Error, Result},
    try_ffi,
    types::FunctionCache,
};

/// CUDA Limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Limit {
    /// GPU thread stack size.
    StackSize = runtime::cudaLimit::cudaLimitStackSize as _,
    /// GPU printf FIFO size.
    PrintfFifoSize = runtime::cudaLimit::cudaLimitPrintfFifoSize as _,
    /// GPU malloc heap size.
    MallocHeapSize = runtime::cudaLimit::cudaLimitMallocHeapSize as _,
    /// GPU device runtime synchronize depth.
    DevRuntimeSyncDepth = runtime::cudaLimit::cudaLimitDevRuntimeSyncDepth as _,
    /// GPU device runtime pending launch count.
    DevRuntimePendingLaunchCount = runtime::cudaLimit::cudaLimitDevRuntimePendingLaunchCount as _,
    /// A value between 0 and 128 that indicates the maximum fetch granularity of L2, in bytes.
    /// The value is a hint.
    MaxL2FetchGranularity = runtime::cudaLimit::cudaLimitMaxL2FetchGranularity as _,
    /// A size in bytes for L2 persisting lines cache size.
    PersistingL2CacheSize = runtime::cudaLimit::cudaLimitPersistingL2CacheSize as _,
}

impl_enum_conversion!(runtime::cudaLimit, Limit);

impl_enum_display!(Limit, {
    Self::StackSize => "cudaLimitStackSize",
    Self::PrintfFifoSize => "cudaLimitPrintfFifoSize",
    Self::MallocHeapSize => "cudaLimitMallocHeapSize",
    Self::DevRuntimeSyncDepth => "cudaLimitDevRuntimeSyncDepth",
    Self::DevRuntimePendingLaunchCount => "cudaLimitDevRuntimePendingLaunchCount",
    Self::MaxL2FetchGranularity => "cudaLimitMaxL2FetchGranularity",
    Self::PersistingL2CacheSize => "cudaLimitPersistingL2CacheSize",
});

/// CUDA device compute modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeMode {
    /// Default compute mode; multiple threads can use [`Device::set_current`] with this device.
    Default = runtime::cudaComputeMode::cudaComputeModeDefault as _,
    /// Compute-exclusive-thread mode; only one thread in one process can use [`Device::set_current`] with this device.
    Exclusive = runtime::cudaComputeMode::cudaComputeModeExclusive as _,
    /// Compute-prohibited mode; no threads can use [`Device::set_current`] with this device.
    Prohibited = runtime::cudaComputeMode::cudaComputeModeProhibited as _,
    /// Compute-exclusive-process mode; many threads in one process can use [`Device::set_current`] with this device.
    ExclusiveProcess = runtime::cudaComputeMode::cudaComputeModeExclusiveProcess as _,
}

impl_enum_conversion!(runtime::cudaComputeMode, ComputeMode);

bitflags::bitflags! {
    /// Flags for [`Device::enable_peer_access`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PeerAccessFlags: u32 {
        /// Default peer-access behavior.
        const DEFAULT = runtime::cudaPeerAccessDefault;
    }
}

/// Attributes queryable between two devices using [`Device::p2p_attribute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PeerToPeerAttribute {
    PerformanceRank = runtime::cudaDeviceP2PAttr::CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK as _,
    AccessSupported = runtime::cudaDeviceP2PAttr::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED as _,
    NativeAtomicSupported =
        runtime::cudaDeviceP2PAttr::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED as _,
    #[deprecated]
    CudaArrayAccessSupported =
        runtime::cudaDeviceP2PAttr::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED as _,
}

impl_enum_conversion!(runtime::cudaDeviceP2PAttr, PeerToPeerAttribute);

impl_enum_display!(PeerToPeerAttribute, {
    Self::PerformanceRank => "cudaDevP2PAttrPerformanceRank",
    Self::AccessSupported => "cudaDevP2PAttrAccessSupported",
    Self::NativeAtomicSupported => "cudaDevP2PAttrNativeAtomicSupported",
    Self::CudaArrayAccessSupported => "cudaDevP2PAttrCudaArrayAccessSupported",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamPriorityRange {
    pub least: i32,
    pub greatest: i32,
}

// TODO: use a crate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Uuid {
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
            bytes: value.bytes.map(|byte| byte as _),
        }
    }
}

/// Rust representation of CUDA device properties.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// ASCII string identifying device.
    pub name: String,
    /// 16-byte unique identifier.
    pub uuid: Uuid,
    /// 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms.
    pub luid: [u8; 8],
    /// LUID device node mask. Value is undefined on TCC and non-Windows platforms.
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
    /// Device supports coherently accessing pageable memory without calling [`DeviceMemory::register_host`](crate::memory::DeviceMemory::register_host) on it.
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
    /// Device supports host memory registration via [`DeviceMemory::register_host`](crate::memory::DeviceMemory::register_host).
    pub host_register_supported: bool,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
    pub sparse_cuda_array_supported: bool,
    /// Device supports [`HostRegisterFlags::READ_ONLY`](crate::memory::HostRegisterFlags::READ_ONLY) for host registrations mapped as read-only to the GPU.
    pub host_register_read_only_supported: bool,
    /// External timeline semaphore interop is supported.
    pub timeline_semaphore_interop_supported: bool,
    /// Device supports CUDA memory pools.
    pub memory_pools_supported: bool,
    /// Device supports GPUDirect RDMA APIs.
    pub gpu_direct_rdma_supported: bool,
    /// The returned flags may be used as subset of the supported write ordering MASTs supplied with GPUDirect RDMA writes.
    pub gpu_direct_rdma_flush_writes_options: u32,
    /// GPUDirect RDMA writes are guaranteed to be ordered with respect to other GPUDirect RDMA writes from the same GPU.
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
            total_global_mem: value.totalGlobalMem as usize,
            shared_mem_per_block: value.sharedMemPerBlock as usize,
            regs_per_block: value.regsPerBlock,
            warp_size: value.warpSize,
            mem_pitch: value.memPitch as usize,
            max_threads_per_block: value.maxThreadsPerBlock,
            max_threads_dim: value.maxThreadsDim,
            max_grid_size: value.maxGridSize,
            total_const_mem: value.totalConstMem as usize,
            major: value.major,
            minor: value.minor,
            texture_alignment: value.textureAlignment as usize,
            texture_pitch_alignment: value.texturePitchAlignment as usize,
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
            surface_alignment: value.surfaceAlignment as usize,
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
            shared_mem_per_multiprocessor: value.sharedMemPerMultiprocessor as usize,
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
            shared_mem_per_block_optin: value.sharedMemPerBlockOptin as usize,
            pageable_memory_access_uses_host_page_tables: value
                .pageableMemoryAccessUsesHostPageTables
                != 0,
            direct_managed_mem_access_from_host: value.directManagedMemAccessFromHost != 0,
            max_blocks_per_multi_processor: value.maxBlocksPerMultiProcessor,
            access_policy_max_window_size: value.accessPolicyMaxWindowSize,
            reserved_shared_mem_per_block: value.reservedSharedMemPerBlock as usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device(DeviceId);

pub type DeviceId = i32;

impl Device {
    pub const fn new(id: DeviceId) -> Self {
        Self(id)
    }

    /// Returns the number of devices with compute capability greater or equal to 2.0 that are available for execution.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the device count, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics such as [`crate::error::Status::NotInitialized`],
    /// [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`].
    pub fn count() -> Result<i32> {
        let mut count: i32 = 0;
        unsafe {
            try_ffi!(runtime::cudaGetDeviceCount(&raw mut count))?;
        }
        Ok(count)
    }

    /// Returns the current device for the calling host thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the current device, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn current() -> Result<Self> {
        let mut device_id: i32 = 0;
        unsafe {
            try_ffi!(runtime::cudaGetDevice(&raw mut device_id))?;
        }
        Ok(Self(device_id))
    }

    /// Blocks until the device has completed all preceding requested tasks.
    /// [`Device::synchronize`] returns an error if one of the preceding tasks has failed.
    /// If [`ContextFlags::SCHEDULE_BLOCKING_SYNC`] was set for this device, the
    /// host thread blocks until the device has finished its work.
    ///
    /// * Use of [`Device::synchronize`] in device code was deprecated in CUDA 11.6 and removed for compute_90+ compilation.
    ///   For compute capability &lt; 9.0, compile-time opt-in with `-D CUDA_FORCE_CDP1_IF_SUPPORTED` is required to continue using [`Device::synchronize`] in device code for now.
    ///   This is different from host-side [`Device::synchronize`], which is still supported.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails, a previous asynchronous
    /// launch reports an error, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn synchronize() -> Result<()> {
        unsafe {
            try_ffi!(runtime::cudaDeviceSynchronize())?;
        }
        Ok(())
    }

    /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
    /// Accessing these resources or passing them to subsequent API calls after
    /// reset results in undefined behavior.
    /// These resources include streams, events, arrays, mipmapped arrays, pitched allocations, texture and surface objects, external memory and semaphore objects, and graphics resources owned by the current device state.
    /// These resources also include memory allocations by [`DeviceMemory::alloc`](crate::memory::DeviceMemory::alloc), [`DeviceMemory::alloc_host`](crate::memory::DeviceMemory::alloc_host), [`DeviceMemory::alloc_managed`](crate::memory::DeviceMemory::alloc_managed) and [`sys::cudaMallocPitch`](singe_cuda_sys::runtime::cudaMallocPitch).
    /// Any subsequent call to this device reinitializes it.
    ///
    /// This call resets the device immediately.
    /// Ensure that no other host threads in the process are accessing the device when this is called.
    ///
    /// * [`Device::reset`] does not destroy memory allocated by [`DeviceMemory::alloc_async`](crate::memory::DeviceMemory::alloc_async) or [`sys::cudaMallocFromPoolAsync`](singe_cuda_sys::runtime::cudaMallocFromPoolAsync).
    ///   These memory allocations must be destroyed explicitly.
    /// * If a non-primary CUDA context is current to the thread, [`Device::reset`] will destroy only the internal CUDA runtime state for that context.
    ///
    /// # Errors
    ///
    /// Returns an error if device reset fails, a previous asynchronous launch
    /// reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn reset() -> Result<()> {
        unsafe {
            try_ffi!(runtime::cudaDeviceReset())?;
        }
        Ok(())
    }

    /// Returns the current size of limit.
    /// The following [`Limit`] values are supported.
    ///
    /// * [`Limit::StackSize`] is the stack size in bytes of each GPU thread.
    /// * [`Limit::PrintfFifoSize`] is the size in bytes of the shared FIFO used by the `printf()` device system call.
    /// * [`Limit::MallocHeapSize`] is the size in bytes of the heap used by the `malloc()` and `free()` device system calls.
    /// * [`Limit::DevRuntimeSyncDepth`] is the maximum grid depth at which a thread can issue the device runtime call [`Device::synchronize`] to wait on child grid launches to complete.
    ///   This feature is removed for devices of compute capability &gt;= 9.0, so such devices return [`crate::error::Status::UnsupportedLimit`].
    /// * [`Limit::DevRuntimePendingLaunchCount`] is the maximum number of outstanding device runtime launches.
    /// * [`Limit::MaxL2FetchGranularity`] is the L2 cache fetch granularity.
    /// * [`Limit::PersistingL2CacheSize`] is the persisting L2 cache size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `limit` is unsupported, CUDA cannot query the limit,
    /// a previous asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn limit(limit: Limit) -> Result<usize> {
        let mut value = 0;
        unsafe {
            try_ffi!(runtime::cudaDeviceGetLimit(&raw mut value, limit.into(),))?;
        }
        Ok(value as _)
    }

    /// Setting limit to value is a request by the application to update the current limit maintained by the device.
    /// The driver may modify the requested value to meet hardware requirements, such as clamping to minimum or maximum values or rounding up to the nearest element size.
    /// Use [`Device::limit`] to query the effective value.
    ///
    /// Setting each [`Limit`] has its own specific restrictions, so each is discussed here.
    ///
    /// * [`Limit::StackSize`] controls the stack size in bytes of each GPU thread.
    ///
    /// * [`Limit::PrintfFifoSize`] controls the size in bytes of the shared FIFO used by the `printf()` device system call.
    ///   Setting [`Limit::PrintfFifoSize`] must not be performed after launching any kernel that uses the `printf()` device system call; otherwise [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// * [`Limit::MallocHeapSize`] controls the size in bytes of the heap used by the `malloc()` and `free()` device system calls.
    ///   Setting [`Limit::MallocHeapSize`] must not be performed after launching any kernel that uses the `malloc()` or `free()` device system calls; otherwise [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// * [`Limit::DevRuntimeSyncDepth`] controls the maximum nesting depth of a grid at which a thread can safely call [`Device::synchronize`].
    ///   Setting this limit must be performed before any launch of a kernel that uses the device runtime and calls [`Device::synchronize`] above the default sync depth, two levels of grids.
    ///   Calls to [`Device::synchronize`] fail if this limit is violated.
    ///   This limit can be set smaller than the default or up to the maximum launch depth of 24.
    ///   Additional sync-depth levels require the runtime to reserve large amounts of device memory that can no longer be used for application allocations.
    ///   If these device-memory reservations fail, [`Device::set_limit`] returns an error, and the limit can be reset to a lower value.
    ///   This limit is only applicable to devices of compute capability &lt; 9.0.
    ///   Setting this limit on devices with other compute capabilities returns [`crate::error::Status::UnsupportedLimit`].
    ///
    /// * [`Limit::DevRuntimePendingLaunchCount`] controls the maximum number of outstanding device runtime launches that can be made from the current device.
    ///   A grid is outstanding from launch until it is known to have completed.
    ///   Device runtime launches that violate this limit fail.
    ///   If a module using the device runtime needs more pending launches than the default 2048 launches, this limit can be increased.
    ///   Sustaining additional pending launches requires the runtime to reserve larger amounts of device memory up front, which can no longer be used for allocations.
    ///   If these reservations fail, [`Device::set_limit`] returns an error, and the limit can be reset to a lower value.
    ///   This limit is only applicable to devices of compute capability 3.5 and higher.
    ///   Setting this limit on devices with compute capability less than 3.5 returns [`crate::error::Status::UnsupportedLimit`].
    ///
    /// * [`Limit::MaxL2FetchGranularity`] controls the L2 cache fetch granularity.
    ///   Values can range from 0B to 128B.
    ///   Performance hint that can be ignored or clamped depending on the platform.
    ///
    /// * [`Limit::PersistingL2CacheSize`] controls size in bytes available for persisting L2 cache.
    ///   Performance hint that can be ignored or clamped depending on the platform.
    ///
    /// # Errors
    ///
    /// Returns an error if `limit` is unsupported, `value` is invalid for that
    /// limit, CUDA cannot set the limit, a previous asynchronous launch reports
    /// an error, or CUDA reports runtime initialization diagnostics.
    pub fn set_limit(limit: Limit, value: usize) -> Result<()> {
        unsafe {
            try_ffi!(runtime::cudaDeviceSetLimit(limit.into(), value as _))?;
        }
        Ok(())
    }

    /// Records flags as the flags for the current device.
    /// If the current device has been set and that device has already been initialized, the previous flags are overwritten.
    /// If the current device has not been initialized, it is initialized with the provided flags.
    /// If no device has been made current to the calling thread, a default device is selected and initialized with the provided flags.
    ///
    /// The three least significant bits of `flags` control how the CPU thread interacts with the OS scheduler while waiting for device results.
    ///
    /// * [`ContextFlags::SCHEDULE_AUTO`]: The default value if `flags` is zero.
    ///   Uses a heuristic based on the number of active CUDA contexts in the process (`C`) and the number of logical processors in the system (`P`).
    ///   If `C > P`, CUDA yields to other OS threads when waiting for the device; otherwise, CUDA actively spins while waiting for results.
    ///   Additionally, on Tegra devices, [`ContextFlags::SCHEDULE_AUTO`] uses a heuristic based on the power profile of the platform and may choose [`ContextFlags::SCHEDULE_BLOCKING_SYNC`] for low-powered devices.
    /// * [`ContextFlags::SCHEDULE_SPIN`]: Instruct CUDA to actively spin when waiting for results from the device.
    ///   This can decrease latency when waiting for the
    ///   device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.
    /// * [`ContextFlags::SCHEDULE_YIELD`]: Instruct CUDA to yield its thread when waiting for results from the device.
    ///   This can increase latency when waiting for the
    ///   device, but can increase the performance of CPU threads performing work in parallel with the device.
    /// * [`ContextFlags::SCHEDULE_BLOCKING_SYNC`]: Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the device to finish work.
    ///
    /// This matches the deprecated CUDA runtime blocking-sync behavior now represented by [`ContextFlags::SCHEDULE_BLOCKING_SYNC`].
    /// * [`ContextFlags::MAP_HOST`]: This flag enables allocating pinned host memory that is accessible to the device.
    ///   It is implicit for the runtime but may
    ///   be absent if a context is created using the driver API.
    ///   If this flag is not set, [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer) always returns a failure code.
    /// * [`ContextFlags::LOCAL_MEMORY_RESIZE_TO_MAX`]: Instruct CUDA to not reduce local memory after resizing local memory for a kernel.
    ///   This can prevent thrashing by local memory
    ///   allocations when launching many kernels with high local memory usage at the cost of potentially increased memory usage.
    ///
    /// Deprecated: this behavior is now the default and cannot be disabled.
    /// * [`ContextFlags::SYNC_MEMORY_OPERATIONS`]: Ensures that synchronous memory operations initiated on this context always synchronize.
    ///   See further documentation
    ///   in the section titled "API Synchronization behavior" to learn more about cases when synchronous memory operations can exhibit
    ///   asynchronous behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot set the device flags, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn set_flags(flags: ContextFlags) -> Result<()> {
        unsafe { try_ffi!(runtime::cudaSetDeviceFlags(flags.bits())) }
    }

    /// Returns the flags for the current device.
    /// If there is a current device for the calling thread, the flags for the device are returned.
    /// If there is no current device, the flags for the first device are returned, which may be the default flags.
    /// Compare to the behavior of [`Device::set_flags`].
    ///
    /// Typically, the returned flags match the behavior seen if the calling
    /// thread uses a device after this call, assuming this thread or another
    /// thread does not change the flags or current device in between.
    /// If the device is not initialized, another thread can change the flags for the current device before it is initialized.
    /// Additionally, when using exclusive mode, if this thread has not requested a specific device, it may use a device other than the first device, contrary to the assumption made by this query.
    ///
    /// If a context has been created via the driver API and is current to the calling thread, the flags for that context are always returned.
    ///
    /// Returned flags may specifically include [`ContextFlags::MAP_HOST`] even though it is not accepted by [`Device::set_flags`] because it is implicit in runtime API flags.
    /// The reason for this is that the current context may have been created via the driver API in which case the flag is not implicit and may be unset.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the device flags, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn flags() -> Result<ContextFlags> {
        let mut flags_raw: u32 = 0;
        unsafe {
            try_ffi!(runtime::cudaGetDeviceFlags(&raw mut flags_raw))?;
        }
        Ok(ContextFlags::from_bits_retain(flags_raw))
    }

    /// Returns the device which has properties that best match the given prop.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot choose a matching device, the selected
    /// device cannot be represented by this wrapper, a previous asynchronous
    /// launch reports an error, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn choose(prop: &DeviceProperties) -> Result<Self> {
        // This function is tricky because cudaChooseDevice takes a *template* prop,
        // and we have a fully filled DeviceProp. We need to construct a template
        // cudaDeviceProp FFI struct based on the criteria we care about.
        // Often, only major/minor compute capability is used.
        // For simplicity here, let's assume we want an exact match on major/minor.
        // A more robust implementation would allow specifying which fields matter.

        let mut ffi_prop: runtime::cudaDeviceProp = unsafe { mem::zeroed() };
        ffi_prop.major = prop.major;
        ffi_prop.minor = prop.minor;
        // Maybe add other critical fields like managedMemory support if needed?
        ffi_prop.managedMemory = i32::from(prop.managed_memory);

        let mut device: i32 = -1;
        unsafe {
            try_ffi!(runtime::cudaChooseDevice(
                (&raw mut device).cast(),
                &raw const ffi_prop
            ))?;
        }
        if device == -1 {
            Err(Error::DeviceNotFound)
        } else {
            Ok(Self(device))
        }
    }

    /// Returns a device ordinal given a PCI bus ID string.
    ///
    /// # Errors
    ///
    /// Returns an error if `pci_bus_id` contains an interior NUL byte, CUDA
    /// cannot resolve the bus ID, a previous asynchronous launch reports an
    /// error, or CUDA reports runtime initialization diagnostics.
    pub fn by_pci_bus_id(pci_bus_id: &str) -> Result<Self> {
        let c_pci_bus_id = CString::new(pci_bus_id)?;
        let mut device: i32 = -1;
        unsafe {
            try_ffi!(runtime::cudaDeviceGetByPCIBusId(
                (&raw mut device).cast(),
                c_pci_bus_id.as_ptr(),
            ))?;
        }
        if device == -1 {
            Err(Error::DeviceNotFound)
        } else {
            Ok(Self(device))
        }
    }

    /// Sets device as the current device for the calling host thread.
    /// Valid device id's are 0 to ([`Device::count`] - 1).
    ///
    /// Device memory subsequently allocated from this host thread is physically
    /// resident on `device`.
    /// Host memory allocated or registered from this host thread has its
    /// lifetime associated with `device`.
    /// Streams and events created from this host thread are associated with
    /// `device`.
    /// Kernels launched from this host thread execute on `device`.
    ///
    /// This may be called from any host thread, for any device, at any time.
    /// This performs no synchronization with the previous or new device, and usually only takes significant time when it initializes the runtime's context state.
    /// This binds the primary context of the specified device to the calling
    /// thread; subsequent memory allocations, stream and event creations, and
    /// kernel launches are associated with that primary context.
    /// This also immediately initializes the runtime state on the primary context, and the context is current on the device immediately.
    /// It returns an error if the device is in [`ComputeMode::ExclusiveProcess`] and is occupied by another process, or if it is in [`ComputeMode::Prohibited`].
    ///
    /// It is not required to call [`sys::cudaInitDevice`](singe_cuda_sys::runtime::cudaInitDevice) before using this method.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot set the current device, the device is
    /// unavailable due to compute mode restrictions, a previous asynchronous
    /// launch reports an error, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn set_current(self) -> Result<()> {
        unsafe {
            try_ffi!(runtime::cudaSetDevice(self.0))?;
        }
        Ok(())
    }

    /// Returns this device's properties.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the device properties, the
    /// returned properties cannot be converted into the safe wrapper type, a
    /// previous asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn properties(self) -> Result<DeviceProperties> {
        unsafe {
            let mut prop = MaybeUninit::<runtime::cudaDeviceProp>::uninit();
            try_ffi!(runtime::cudaGetDeviceProperties(prop.as_mut_ptr(), self.0))?;
            prop.assume_init().try_into()
        }
    }

    /// Returns the PCI bus ID string identifying the device.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the PCI bus ID, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn pci_bus_id(self) -> Result<String> {
        const LEN: usize = 16; // Sufficient for typical PCI IDs like 0000:01:00.0
        let mut pci_bus_id_buf = [0i8; LEN];
        unsafe {
            try_ffi!(runtime::cudaDeviceGetPCIBusId(
                pci_bus_id_buf.as_mut_ptr().cast(),
                LEN as _,
                self.0,
            ))?;
            Ok(string_from_c_chars(&pci_bus_id_buf))
        }
    }

    /// On success, all allocations from the peer device are immediately accessible by the current device.
    /// They remain accessible until access is explicitly disabled using [`Device::disable_peer_access`] or either device is reset using [`Device::reset`].
    ///
    /// Access granted by this call is unidirectional; accessing memory on the current device from the peer device requires a separate symmetric call to [`Device::enable_peer_access`].
    ///
    /// There are both device-wide and system-wide limitations per system configuration, as noted in the CUDA Programming Guide under the section "Peer-to-Peer Memory Access".
    ///
    /// Returns [`crate::error::Status::InvalidDevice`] if [`Device::can_access_peer`] indicates that the current device cannot directly access memory from the peer device.
    ///
    /// Returns [`crate::error::Status::PeerAccessAlreadyEnabled`] if direct access to the peer device from the current device has already been enabled.
    ///
    /// Returns [`crate::error::Status::InvalidValue`] if flags is not 0.
    ///
    /// # Errors
    ///
    /// Returns an error if peer access cannot be enabled, `flags` is not
    /// [`PeerAccessFlags::DEFAULT`], the peer is invalid or already enabled, a
    /// previous asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn enable_peer_access(self, flags: PeerAccessFlags) -> Result<()> {
        if flags != PeerAccessFlags::DEFAULT {
            return Err(Error::InvalidValue);
        }
        unsafe { try_ffi!(runtime::cudaDeviceEnablePeerAccess(self.0, flags.bits(),)) }
    }

    /// Returns [`crate::error::Status::PeerAccessNotEnabled`] if direct access to memory on the peer device has not yet been enabled from the current device.
    ///
    /// # Errors
    ///
    /// Returns an error if peer access was not enabled, CUDA cannot disable
    /// peer access, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn disable_peer_access(self) -> Result<()> {
        unsafe { try_ffi!(runtime::cudaDeviceDisablePeerAccess(self.0)) }
    }

    /// Returns true if this device can directly access memory from `other`.
    /// If direct access from this device to `other` is possible, access may be enabled by calling [`Device::enable_peer_access`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query peer accessibility, either device
    /// is invalid, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn can_access_peer(self, other: Self) -> Result<bool> {
        let mut can_access_peer: i32 = 0;
        unsafe {
            try_ffi!(runtime::cudaDeviceCanAccessPeer(
                (&raw mut can_access_peer).cast(),
                self.0,
                other.0,
            ))?;
        }
        Ok(can_access_peer != 0)
    }

    /// Returns the value of the requested attribute of the link between devices.
    /// Supported attributes are represented by [`PeerToPeerAttribute`]:
    ///
    /// * [`PeerToPeerAttribute::PerformanceRank`]: relative performance of the link between the two devices. Lower values are better.
    /// * [`PeerToPeerAttribute::AccessSupported`]: whether peer access is enabled.
    /// * [`PeerToPeerAttribute::NativeAtomicSupported`]: whether native atomic operations over the link are supported.
    /// * [`PeerToPeerAttribute::CudaArrayAccessSupported`]: whether CUDA arrays are accessible over the link.
    ///
    /// Returns [`crate::error::Status::InvalidDevice`] if either device is invalid or if they represent the same device.
    ///
    /// Returns [`crate::error::Status::InvalidValue`] if `attrib` is not valid.
    ///
    /// # Errors
    ///
    /// Returns an error if either device is invalid, `attr` is not accepted by
    /// CUDA, CUDA cannot query the attribute, a previous asynchronous launch
    /// reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn p2p_attribute(self, attr: PeerToPeerAttribute, other: Self) -> Result<i32> {
        let mut value: i32 = 0;
        unsafe {
            try_ffi!(runtime::cudaDeviceGetP2PAttribute(
                (&raw mut value).cast(),
                attr.into(),
                self.0,
                other.0,
            ))?;
        }
        Ok(value)
    }

    /// On devices where the L1 cache and shared memory use the same hardware resources, this returns the preferred cache configuration for the current device.
    /// This setting is only a preference.
    /// The runtime uses the requested configuration if possible, but it may choose a different configuration if required to execute functions.
    ///
    /// This returns [`FunctionCache::PreferNone`] on devices where the size of the L1 cache and shared memory are fixed.
    ///
    /// The supported cache configurations are:
    ///
    /// * [`FunctionCache::PreferNone`]: no preference for shared memory or L1 (default)
    /// * [`FunctionCache::PreferShared`]: prefer larger shared memory and smaller L1 cache
    /// * [`FunctionCache::PreferL1`]: prefer larger L1 cache and smaller shared memory
    /// * [`FunctionCache::PreferEqual`]: prefer equal size L1 cache and shared memory
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the cache configuration, a
    /// previous asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn cache_config() -> Result<FunctionCache> {
        let mut config = runtime::cudaFuncCache::CU_FUNC_CACHE_PREFER_NONE;
        unsafe {
            try_ffi!(runtime::cudaDeviceGetCacheConfig(&raw mut config))?;
        }
        Ok(config.into())
    }

    /// On devices where the L1 cache and shared memory use the same hardware resources, this sets through cacheConfig the preferred cache configuration for the current device.
    /// This setting is only a preference.
    /// The runtime uses the requested configuration if possible, but it is free to choose a different configuration if required to execute the kernel.
    /// Any per-kernel cache preference set through the CUDA API takes precedence over this device-wide setting.
    /// Setting the device-wide cache configuration to [`FunctionCache::PreferNone`] causes subsequent kernel launches to prefer not changing the cache configuration unless required to launch the kernel.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.
    ///
    /// Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point.
    ///
    /// The supported cache configurations are:
    ///
    /// * [`FunctionCache::PreferNone`]: no preference for shared memory or L1 (default)
    /// * [`FunctionCache::PreferShared`]: prefer larger shared memory and smaller L1 cache
    /// * [`FunctionCache::PreferL1`]: prefer larger L1 cache and smaller shared memory
    /// * [`FunctionCache::PreferEqual`]: prefer equal size L1 cache and shared memory
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot set the cache configuration, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn set_cache_config(config: FunctionCache) -> Result<()> {
        unsafe {
            try_ffi!(runtime::cudaDeviceSetCacheConfig(config.into()))?;
        }
        Ok(())
    }

    /// Returns the least and greatest stream priority numerical values.
    /// Stream priorities follow a convention where lower numbers represent greater priorities.
    /// The range of meaningful stream priorities is given by \[greatest, least\].
    /// If a stream is created with a priority value outside this range, the priority is automatically clamped.
    /// See [`Context::create_stream_with_priority`](crate::context::Context::create_stream_with_priority) for details on creating a priority stream.
    ///
    /// Returns 0 for both values if the current context's device does not support stream priorities.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the stream-priority range, a
    /// previous asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn stream_priority_range() -> Result<StreamPriorityRange> {
        let mut least = 0;
        let mut greatest = 0;
        unsafe {
            try_ffi!(runtime::cudaDeviceGetStreamPriorityRange(
                &raw mut least,
                &raw mut greatest,
            ))?;
        }
        Ok(StreamPriorityRange { least, greatest })
    }

    pub const fn id(self) -> DeviceId {
        self.0
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        match Device::count() {
            Ok(count) => {
                println!("Found {} CUDA devices.", count);
                if count > 0 {
                    match Device::new(0).properties() {
                        Ok(props) => println!("Device 0: {}", props.name),
                        Err(e) => eprintln!("error getting properties for device 0: {:?}", e),
                    }
                }
            }
            Err(e) => eprintln!("error getting device count: {:?}", e),
        }
    }
}
