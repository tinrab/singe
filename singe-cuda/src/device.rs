#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{ffi::CString, mem};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display, string_from_c_chars};
use singe_cuda_sys::{driver, runtime};

use crate::{
    context::ContextFlags,
    driver::entry_points::{
        cuDeviceGetCliqueCount, cuDeviceGetCliqueInfo, cuDeviceGetFabricClusterUuid,
    },
    error::{Error, Result},
    try_ffi,
    types::FunctionCache,
    utility::to_usize,
};

/// CUDA Limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Limit {
    /// Per-block memory size.
    PerBlockMemorySize = runtime::cudaLimit::cudaLimitPerBlockMemorySize as _,
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
    Self::PerBlockMemorySize => "cudaLimitPerBlockMemorySize",
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

/// Attributes queryable between two devices using [`Device::p2p_attribute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PeerToPeerAttribute {
    /// Relative performance rank of the link between two devices.
    PerformanceRank = runtime::cudaDeviceP2PAttr::CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK as _,
    /// Whether peer access is supported between the devices.
    AccessSupported = runtime::cudaDeviceP2PAttr::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED as _,
    /// Whether all CUDA-valid native atomic operations are supported over the peer link.
    NativeAtomicSupported =
        runtime::cudaDeviceP2PAttr::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED as _,
    /// Deprecated selector for CUDA array peer-access support.
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

/// Inclusive stream priority limits supported by the current device.
///
/// Lower numeric values represent higher priorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamPriorityRange {
    /// Least supported priority, with the highest numeric value.
    pub least: i32,
    /// Greatest supported priority, with the lowest numeric value.
    pub greatest: i32,
}

// TODO: use a crate
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

/// A CUDA device ordinal.
///
/// The value does not own a CUDA context or validate device availability until an operation queries CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device(DeviceId);

/// Signed CUDA device ordinal used by runtime and driver APIs.
pub type DeviceId = i32;

impl Device {
    /// Wraps a device ordinal without querying CUDA, validating it, or making the device current.
    #[must_use]
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
        // SAFETY: `cudaGetDeviceCount` receives valid output locations.
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
        // SAFETY: `cudaGetDevice` receives valid output locations.
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
    /// * Use of [`Device::synchronize`] in device code was deprecated in CUDA 11.6 and removed for `compute_90`+ compilation.
    ///   For compute capability &lt; 9.0, compile-time opt-in with `-D CUDA_FORCE_CDP1_IF_SUPPORTED` is required to continue using [`Device::synchronize`] in device code for now.
    ///   This is different from host-side [`Device::synchronize`], which is still supported.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails, a previous asynchronous
    /// launch reports an error, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn synchronize() -> Result<()> {
        // SAFETY: `cudaDeviceSynchronize` takes no arguments.
        unsafe {
            try_ffi!(runtime::cudaDeviceSynchronize())?;
        }
        Ok(())
    }

    /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
    /// Accessing these resources or passing them to subsequent API calls after
    /// reset results in undefined behavior.
    /// These resources include streams, events, arrays, mipmapped arrays, pitched allocations, texture and surface objects, external memory and semaphore objects, and graphics resources owned by the current device state.
    /// These resources also include memory allocations by [`DeviceMemory::alloc`](crate::memory::DeviceMemory::alloc), [`PinnedHostMemory`](crate::memory::PinnedHostMemory), [`DeviceMemory::alloc_managed`](crate::memory::DeviceMemory::alloc_managed), and [`PitchedDeviceMemory`](crate::memory::PitchedDeviceMemory).
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
    /// # Safety
    /// All affected resources must have been released and all work completed.
    /// Exclude other host threads and any later use of pre-reset owners or handles.
    pub unsafe fn reset() -> Result<()> {
        // SAFETY: `cudaDeviceReset` takes no arguments.
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
        // SAFETY: `cudaDeviceGetLimit` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetLimit(&raw mut value, limit.into(),))?;
        }
        to_usize(value, "device limit")
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
        // SAFETY: `cudaDeviceSetLimit` receives only plain values.
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
        // SAFETY: `cudaSetDeviceFlags` receives only plain values.
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
        // SAFETY: `cudaGetDeviceFlags` receives valid output locations.
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

        // SAFETY: this plain C struct is valid when all bytes are zero.
        let mut ffi_prop: runtime::cudaDeviceProp = unsafe { mem::zeroed() };
        ffi_prop.major = prop.major;
        ffi_prop.minor = prop.minor;
        // Maybe add other critical fields like managedMemory support if needed?
        ffi_prop.managedMemory = i32::from(prop.managed_memory);

        let mut device: i32 = -1;
        // SAFETY: `cudaChooseDevice` receives valid output locations and pointers into buffers borrowed for the call.
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
        // SAFETY: `cudaDeviceGetByPCIBusId` receives valid output locations and pointers into buffers borrowed for the call.
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
        // SAFETY: `cudaSetDevice` receives only plain values.
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
        let mut prop = runtime::cudaDeviceProp::default();
        // SAFETY: `cudaGetDeviceProperties` writes only into the local `prop`.
        unsafe {
            try_ffi!(runtime::cudaGetDeviceProperties(&raw mut prop, self.0))?;
        }
        prop.try_into()
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
        // SAFETY: `cudaDeviceGetPCIBusId` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetPCIBusId(
                pci_bus_id_buf.as_mut_ptr().cast(),
                16,
                self.0,
            ))?;
            Ok(string_from_c_chars(&pci_bus_id_buf))
        }
    }

    /// Returns true if this device can directly access memory from `other`.
    /// If direct access from this device to `other` is possible, access may be enabled by calling [`Context::enable_peer_access`](crate::context::Context::enable_peer_access).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query peer accessibility, either device
    /// is invalid, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn can_access_peer(self, other: Self) -> Result<bool> {
        let mut can_access_peer: i32 = 0;
        // SAFETY: `cudaDeviceCanAccessPeer` receives valid output locations.
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
    /// * [`PeerToPeerAttribute::AccessSupported`]: whether direct peer access is supported.
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
        // SAFETY: `cudaDeviceGetP2PAttribute` receives valid output locations.
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
        // SAFETY: `cudaDeviceGetCacheConfig` receives valid output locations.
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
        // SAFETY: `cudaDeviceSetCacheConfig` receives only plain values.
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
        // SAFETY: `cudaDeviceGetStreamPriorityRange` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaDeviceGetStreamPriorityRange(
                &raw mut least,
                &raw mut greatest,
            ))?;
        }
        Ok(StreamPriorityRange { least, greatest })
    }

    /// Returns this device's ordinal.
    #[must_use]
    pub const fn id(self) -> DeviceId {
        self.0
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        match Device::count() {
            Ok(count) => {
                println!("Found {} CUDA devices.", count);
                if count > 0 {
                    match Device::new(0).properties() {
                        Ok(props) => println!("Device 0: {}", props.name),
                        Err(error) => eprintln!("error getting properties for device 0: {error:?}"),
                    }
                }
            }
            Err(error) => eprintln!("error getting device count: {error:?}"),
        }
    }
}

/// Device properties that can be queried using the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DeviceAttribute {
    /// Maximum number of threads per block.
    MaxThreadsPerBlock = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _,
    /// Maximum block dimension X.
    MaxBlockDimX = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X as _,
    /// Maximum block dimension Y.
    MaxBlockDimY = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y as _,
    /// Maximum block dimension Z.
    MaxBlockDimZ = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z as _,
    /// Maximum grid dimension X.
    MaxGridDimX = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X as _,
    /// Maximum grid dimension Y.
    MaxGridDimY = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y as _,
    /// Maximum grid dimension Z.
    MaxGridDimZ = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z as _,
    /// Maximum shared memory available per block in bytes.
    MaxSharedMemoryPerBlock = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK as _,
    /// Memory available on device for **constant** variables in a CUDA C kernel in bytes.
    TotalConstantMemory = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY as _,
    /// Warp size in threads.
    WarpSize = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_WARP_SIZE as _,
    /// Maximum pitch in bytes allowed by memory copies.
    MaxPitch = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_PITCH as _,
    /// Maximum number of 32-bit registers available per block.
    MaxRegistersPerBlock = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK as _,
    /// Typical clock frequency in kilohertz.
    ClockRate = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CLOCK_RATE as _,
    /// Alignment requirement for textures.
    TextureAlignment = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT as _,
    /// Device can possibly copy memory and execute a kernel concurrently.
    ///
    /// Deprecated. Use instead `CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`.
    GpuOverlap = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP as _,
    /// Number of multiprocessors on device.
    MultiprocessorCount = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as _,
    /// Specifies whether there is a run time limit on kernels.
    KernelExecTimeout = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT as _,
    /// Device is integrated with host memory.
    Integrated = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED as _,
    /// Device can map host memory into CUDA address space.
    CanMapHostMemory = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY as _,
    /// Compute mode (See [CUcomputemode](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1ga637aab2eadb52e1c1c048b8bad9592d1) for details).
    ComputeMode = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as _,
    /// Maximum 1D texture width.
    MaximumTexture1DWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH as _,
    /// Maximum 2D texture width.
    MaximumTexture2DWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH as _,
    /// Maximum 2D texture height.
    MaximumTexture2DHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT as _,
    /// Maximum 3D texture width.
    MaximumTexture3DWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH as _,
    /// Maximum 3D texture height.
    MaximumTexture3DHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT as _,
    /// Maximum 3D texture depth.
    MaximumTexture3DDepth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH as _,
    /// Maximum 2D layered texture width.
    MaximumTexture2DLayeredWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH as _,
    /// Maximum 2D layered texture height.
    MaximumTexture2DLayeredHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT as _,
    /// Maximum layers in a 2D layered texture.
    MaximumTexture2DLayeredLayers = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS as _,
    /// Alignment requirement for surfaces.
    SurfaceAlignment = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT as _,
    /// Device can possibly execute multiple kernels concurrently.
    ConcurrentKernels = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS as _,
    /// Device has ECC support enabled.
    EccEnabled = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ECC_ENABLED as _,
    /// PCI bus ID of the device.
    PciBusId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID as _,
    /// PCI device ID of the device.
    PciDeviceId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID as _,
    /// Device is using TCC driver model.
    TccDriver = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TCC_DRIVER as _,
    /// Peak memory clock frequency in kilohertz.
    MemoryClockRate = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE as _,
    /// Global memory bus width in bits.
    GlobalMemoryBusWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH as _,
    /// Size of L2 cache in bytes.
    L2CacheSize = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as _,
    /// Maximum resident threads per multiprocessor.
    MaxThreadsPerMultiprocessor = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR as _,
    /// Number of asynchronous engines.
    AsyncEngineCount = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT as _,
    /// Device shares a unified address space with the host.
    UnifiedAddressing = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING as _,
    /// Maximum 1D layered texture width.
    MaximumTexture1DLayeredWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH as _,
    /// Maximum layers in a 1D layered texture.
    MaximumTexture1DLayeredLayers = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS as _,
    /// do not use.
    CanTex2DGather = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER as _,
    /// Maximum 2D texture width if `CUDA_ARRAY3D_TEXTURE_GATHER` is set.
    MaximumTexture2DGatherWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH as _,
    /// Maximum 2D texture height if `CUDA_ARRAY3D_TEXTURE_GATHER` is set.
    MaximumTexture2DGatherHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT as _,
    /// Alternate maximum 3D texture width.
    MaximumTexture3DWidthAlternate = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE as _,
    /// Alternate maximum 3D texture height.
    MaximumTexture3DHeightAlternate = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE as _,
    /// Alternate maximum 3D texture depth.
    MaximumTexture3DDepthAlternate = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE as _,
    /// PCI domain ID of the device.
    PciDomainId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID as _,
    /// Pitch alignment requirement for textures.
    TexturePitchAlignment = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT as _,
    /// Maximum cubemap texture width/height.
    MaximumTexturecubemapWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH as _,
    /// Maximum cubemap layered texture width/height.
    MaximumTexturecubemapLayeredWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH as _,
    /// Maximum layers in a cubemap layered texture.
    MaximumTexturecubemapLayeredLayers = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS as _,
    /// Maximum 1D surface width.
    MaximumSurface1DWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH as _,
    /// Maximum 2D surface width.
    MaximumSurface2DWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH as _,
    /// Maximum 2D surface height.
    MaximumSurface2DHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT as _,
    /// Maximum 3D surface width.
    MaximumSurface3DWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH as _,
    /// Maximum 3D surface height.
    MaximumSurface3DHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT as _,
    /// Maximum 3D surface depth.
    MaximumSurface3DDepth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH as _,
    /// Maximum 1D layered surface width.
    MaximumSurface1DLayeredWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH as _,
    /// Maximum layers in a 1D layered surface.
    MaximumSurface1DLayeredLayers = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS as _,
    /// Maximum 2D layered surface width.
    MaximumSurface2DLayeredWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH as _,
    /// Maximum 2D layered surface height.
    MaximumSurface2DLayeredHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT as _,
    /// Maximum layers in a 2D layered surface.
    MaximumSurface2DLayeredLayers = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS as _,
    /// Maximum cubemap surface width.
    MaximumSurfacecubemapWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH as _,
    /// Maximum cubemap layered surface width.
    MaximumSurfacecubemapLayeredWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH as _,
    /// Maximum layers in a cubemap layered surface.
    MaximumSurfacecubemapLayeredLayers = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS as _,
    /// do not use.
    ///
    /// Use `cudaDeviceGetTexture1DLinearMaxWidth()` or `cuDeviceGetTexture1DLinearMaxWidth` instead.
    MaximumTexture1DLinearWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH as _,
    /// Maximum 2D linear texture width.
    MaximumTexture2DLinearWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH as _,
    /// Maximum 2D linear texture height.
    MaximumTexture2DLinearHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT as _,
    /// Maximum 2D linear texture pitch in bytes.
    MaximumTexture2DLinearPitch = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH as _,
    /// Maximum mipmapped 2D texture width.
    MaximumTexture2DMipmappedWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH as _,
    /// Maximum mipmapped 2D texture height.
    MaximumTexture2DMipmappedHeight = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT as _,
    /// Major compute capability version number.
    ComputeCapabilityMajor = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as _,
    /// Minor compute capability version number.
    ComputeCapabilityMinor = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as _,
    /// Maximum mipmapped 1D texture width.
    MaximumTexture1DMipmappedWidth = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH as _,
    /// Device supports stream priorities.
    StreamPrioritiesSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED as _,
    /// Device supports caching globals in L1.
    GlobalL1CacheSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED as _,
    /// Device supports caching locals in L1.
    LocalL1CacheSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED as _,
    /// Maximum shared memory available per multiprocessor in bytes.
    MaxSharedMemoryPerMultiprocessor = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR as _,
    /// Maximum number of 32-bit registers available per multiprocessor.
    MaxRegistersPerMultiprocessor = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR as _,
    /// Device can allocate managed memory on this system.
    ManagedMemory = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY as _,
    /// Device is on a multi-GPU board.
    MultiGpuBoard = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD as _,
    /// Unique id for a group of devices on the same multi-GPU board.
    MultiGpuBoardGroupId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID as _,
    /// Link between the device and the host supports all native atomic operations.
    HostNativeAtomicSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED as _,
    /// Ratio of single precision performance (in floating-point operations per second) to double precision performance.
    SingleToDoublePrecisionPerfRatio = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO as _,
    /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it.
    PageableMemoryAccess = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS as _,
    /// Device can coherently access managed memory concurrently with the CPU.
    ConcurrentManagedAccess = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS as _,
    /// Device supports compute preemption.
    ComputePreemptionSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED as _,
    /// Device can access host registered memory at the same virtual address as the CPU.
    CanUseHostPointerForRegisteredMem = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM as _,
    /// along with v1 `MemOps` API, cuStreamBatchMemOp and related APIs are supported.
    CanUseStreamMemOpsV1 = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 as _,
    /// along with v1 `MemOps` API, 64-bit operations are supported in cuStreamBatchMemOp and related APIs.
    CanUse64BitStreamMemOpsV1 = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 as _,
    /// along with v1 `MemOps` API, `CUstreamWaitValue_flags_enum::CU_STREAM_WAIT_VALUE_NOR` is supported.
    CanUseStreamWaitValueNorV1 = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 as _,
    /// Device supports launching cooperative kernels via `cuLaunchCooperativeKernel`.
    CooperativeLaunch = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH as _,
    /// `cuLaunchCooperativeKernelMultiDevice` is deprecated.
    CooperativeMultiDeviceLaunch = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH as _,
    /// Maximum optin shared memory per block.
    ///
    /// That is shared memory that is available for dynamic allocation or static allocation (including architecture specific static shared memory) on this device but is not guaranteed to be portable.
    MaxSharedMemoryPerBlockOptin = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN as _,
    /// The `CUstreamWaitValue_flags_enum::CU_STREAM_WAIT_VALUE_FLUSH` flag and the `CUstreamBatchMemOpType_enum::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` `MemOp` are supported on the device.
    ///
    /// See [Stream Memory Operations](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html#group__cuda__memop) for additional details.
    CanFlushRemoteWrites = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES as _,
    /// Device supports host memory registration via `::cudaHostRegister`.
    HostRegisterSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED as _,
    /// Device accesses pageable memory via the host’s page tables.
    PageableMemoryAccessUsesHostPageTables = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES as _,
    /// The host can directly access managed memory on the device without migration.
    DirectManagedMemAccessFromHost = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST as _,
    /// Use `CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`.
    VirtualAddressManagementSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED as _,
    /// Device supports exporting memory to a posix file descriptor with `cuMemExportToShareableHandle`, if requested via `cuMemCreate`.
    HandleTypePosixFileDescriptorSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED as _,
    /// Device supports exporting memory to a Win32 NT handle with `cuMemExportToShareableHandle`, if requested via `cuMemCreate`.
    HandleTypeWin32HandleSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED as _,
    /// Device supports exporting memory to a Win32 KMT handle with `cuMemExportToShareableHandle`, if requested via `cuMemCreate`.
    HandleTypeWin32KmtHandleSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED as _,
    /// Maximum number of blocks per multiprocessor.
    MaxBlocksPerMultiprocessor = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR as _,
    /// Device supports compression of memory.
    GenericCompressionSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED as _,
    /// Maximum L2 persisting lines capacity setting in bytes.
    MaxPersistingL2CacheSize = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE as _,
    /// Maximum value of [CUaccessPolicyWindow::num_bytes](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/structCUaccessPolicyWindow__v1.html#structcuaccesspolicywindow__v1_1a79ee4c983788bc68605d42aee2701eec).
    MaxAccessPolicyWindowSize = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE as _,
    /// Device supports specifying the `GPUDirect` RDMA flag with `cuMemCreate`.
    GpuDirectRdmaWithCudaVmmSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED as _,
    /// Shared memory reserved by CUDA driver per block in bytes.
    ReservedSharedMemoryPerBlock = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK as _,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
    SparseCudaArraySupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED as _,
    /// Device supports using the cuMemHostRegister flag `::CU_MEMHOSTERGISTER_READ_ONLY` to register memory that must be mapped as read-only to the GPU.
    ReadOnlyHostRegisterSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED as _,
    /// External timeline semaphore interop is supported on the device.
    TimelineSemaphoreInteropSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED as _,
    /// Device supports using the `cuMemAllocAsync` and `::cuMemPool` family of APIs.
    MemoryPoolsSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED as _,
    /// Device supports `GPUDirect` RDMA APIs, like `nvidia_p2p_get_pages` (see <https://docs.nvidia.com/cuda/gpudirect-rdma> for more information).
    GpuDirectRdmaSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED as _,
    /// The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the [CUflushGPUDirectRDMAWritesOptions](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1gaf34334d1d6892847a5d05be7ca8db3c6) enum.
    GpuDirectRdmaFlushWritesOptions = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS as _,
    /// `GPUDirect` RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute.
    ///
    /// See [CUGPUDirectRDMAWritesOrdering](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1ga8d647023012ffc318ce86814fec05376) for the numerical values returned here.
    GpuDirectRdmaWritesOrdering = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING as _,
    /// Handle types supported with mempool based IPC.
    MempoolSupportedHandleTypes = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES as _,
    /// Indicates device supports cluster launch.
    ClusterLaunch = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH as _,
    /// Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays.
    DeferredMappingCudaArraySupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED as _,
    /// 64-bit operations are supported in cuStreamBatchMemOp and related `MemOp` APIs.
    CanUse64BitStreamMemOps = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS as _,
    /// `CUstreamWaitValue_flags_enum::CU_STREAM_WAIT_VALUE_NOR` is supported by `MemOp` APIs.
    CanUseStreamWaitValueNor = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR as _,
    /// Device supports buffer sharing with `dma_buf` mechanism.
    DmaBufSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED as _,
    /// Device supports IPC Events.
    IpcEventSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED as _,
    /// Number of memory domains the device supports.
    MemSyncDomainCount = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT as _,
    /// Device supports accessing memory using Tensor Map.
    TensorMapAccessSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED as _,
    /// Device supports exporting memory to a fabric handle with `cuMemExportToShareableHandle` or requested with `cuMemCreate`.
    HandleTypeFabricSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED as _,
    /// Device supports unified function pointers.
    UnifiedFunctionPointers = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS as _,
    /// NUMA configuration of a device: value is of type [CUdeviceNumaConfig](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1gabd735e21702c410355167e04d3362eab) enum.
    NumaConfig = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_NUMA_CONFIG as _,
    /// NUMA node ID of the GPU memory.
    NumaId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_NUMA_ID as _,
    /// Device supports switch multicast and reduction operations.
    MulticastSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED as _,
    /// Indicates if contexts created on this device will be shared via MPS.
    MpsEnabled = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MPS_ENABLED as _,
    /// NUMA ID of the host node closest to the device.
    ///
    /// Returns -1 when system does not support NUMA.
    HostNumaId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID as _,
    /// Device supports CIG with D3D12.
    D3D12CigSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED as _,
    /// The returned valued shall be interpreted as a bitmask, where the individual bits are described by the [CUmemDecompressAlgorithm](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html#group__cuda__mem_1ga6c015495b909c100e19cd4ddafceee91) enum.
    MemDecompressAlgorithmMask = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK as _,
    /// The returned valued is the maximum length in bytes of a single decompress operation that is allowed.
    MemDecompressMaximumLength = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH as _,
    /// Device supports CIG with Vulkan.
    VulkanCigSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED as _,
    /// The combined 16-bit PCI device ID and 16-bit PCI vendor ID.
    GpuPciDeviceId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID as _,
    /// The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.
    GpuPciSubsystemId = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID as _,
    /// Device supports `HOST_NUMA` location with the virtual memory management APIs like `cuMemCreate`, `cuMemMap` and related APIs.
    HostNumaVirtualMemoryManagementSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED as _,
    /// Device supports `HOST_NUMA` location with the `cuMemAllocAsync` and `::cuMemPool` family of APIs.
    HostNumaMemoryPoolsSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED as _,
    /// Device supports `HOST_NUMA` location IPC between nodes in a multi-node system.
    HostNumaMultinodeIpcSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED as _,
    /// Device suports HOST location with the `cuMemAllocAsync` and `::cuMemPool` family of APIs.
    HostMemoryPoolsSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED as _,
    /// Device supports HOST location with the virtual memory management APIs like `cuMemCreate`, `cuMemMap` and related APIs.
    HostVirtualMemoryManagementSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED as _,
    /// Device supports page-locked host memory buffer sharing with `dma_buf` mechanism.
    HostAllocDmaBufSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED as _,
    /// Link between the device and the host supports only some native atomic operations.
    OnlyPartialHostNativeAtomicSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED as _,
    /// Device supports atomic reduction operations in stream batch memory operations.
    AtomicReductionSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED as _,
    /// Number of locality domains.
    LocalityDomainCount = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT as _,
    /// Maximum oversized shared memory per block.
    MaxOversizedSharedMemoryPerBlock = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK as _,
    /// Device supports CIG streams with D3D12.
    D3D12CigStreamsSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_D3D12_CIG_STREAMS_SUPPORTED as _,
    /// Device supports `mmap()` of dmabuf file descriptors for CUDA device memory allocations.
    DmaBufMmapSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED as _,
    /// Device supports unicast logical endpoints.
    LogicalEndpointUnicastSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED as _,
    /// Device supports multicast logical endpoints.
    LogicalEndpointMulticastSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED as _,
    /// Device supports counted operations via logical endpoints.
    LogicalEndpointCountedOpsSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED as _,
    /// Device supports unicast logical endpoint access on the owner device.
    LogicalEndpointUnicastAccessOnOwnerDeviceSupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED as _,
    /// Number of multiprocessors on each locality domain.
    LocalityDomainMultiprocessorCount = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT as _,
    /// Handle types supported with logical endpoint IPC.
    LogicalEndpointSupportedHandleTypes = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES as _,
    /// Device supports `GPUDirect` RDMA with localized memory using the default RDMA mapping link.
    GpuDirectRdmaWithLocalizedMemorySupported = driver::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_LOCALIZED_MEMORY_SUPPORTED as _,
}

impl_enum_conversion!(u32, driver::CUdevice_attribute_enum, DeviceAttribute);

/// Fabric clique types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CliqueType {
    /// Unicast pointer clique.
    UnicastPointer = driver::CUcliqueType_enum::CU_CLIQUE_TYPE_UNICAST_POINTER as _,
    /// Multicast pointer clique.
    MulticastPointer = driver::CUcliqueType_enum::CU_CLIQUE_TYPE_MULTICAST_POINTER as _,
    /// Unicast logical endpoint clique.
    UnicastLogicalEndpoint =
        driver::CUcliqueType_enum::CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT as _,
    /// Multicast logical endpoint clique.
    MulticastLogicalEndpoint =
        driver::CUcliqueType_enum::CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT as _,
}

impl_enum_conversion!(u32, driver::CUcliqueType_enum, CliqueType);

/// Fabric clique information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliqueInfo {
    /// Type of the fabric clique.
    pub kind: CliqueType,
    /// ID of the fabric clique.
    pub id: u32,
}

impl Device {
    /// Returns information about the device.
    ///
    /// # Errors
    ///
    /// Returns an error if the device ordinal or attribute is invalid, CUDA is not initialized, the driver cannot query the attribute, or previous asynchronous work reports an error.
    pub fn attribute(self, attribute: DeviceAttribute) -> Result<i32> {
        let mut value = 0;
        // SAFETY: `cuDeviceGetAttribute` receives valid output locations.
        unsafe {
            try_ffi!(driver::cuDeviceGetAttribute(
                &raw mut value,
                attribute.into(),
                self.id() as _
            ))?;
        }
        Ok(value)
    }

    /// Retrieves the fabric cluster UUID for the given device.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver entry point is unavailable or the native query fails, including unsupported fabric features or an invalid device.
    pub fn fabric_cluster_uuid(self) -> Result<Uuid> {
        let mut value = driver::CUuuid::default();
        // SAFETY: `cuDeviceGetFabricClusterUuid` receives valid output locations.
        unsafe {
            cuDeviceGetFabricClusterUuid(&raw mut value, self.id() as _)?;
        }
        Ok(value.into())
    }

    /// Returns the fabric clique information for the cliques that the device is part of.
    ///
    /// # Errors
    ///
    /// Returns an error if either required driver entry point is unavailable, the native count or clique-information query fails, or the reported count does not fit `usize`. No partial list is returned.
    pub fn cliques(self) -> Result<Vec<CliqueInfo>> {
        let mut count = 0;
        // SAFETY: `cuDeviceGetCliqueCount` receives valid output locations.
        unsafe {
            cuDeviceGetCliqueCount(&raw mut count, self.id() as _)?;
        }
        let mut values = vec![driver::CUcliqueInfo::default(); to_usize(count, "clique count")?];
        // SAFETY: `cuDeviceGetCliqueInfo` receives valid output locations.
        unsafe {
            cuDeviceGetCliqueInfo(values.as_mut_ptr(), &raw mut count, self.id() as _)?;
        }
        values.truncate(to_usize(count, "clique count")?);
        Ok(values
            .into_iter()
            .map(|v| CliqueInfo {
                kind: v.type_.into(),
                id: v.id,
            })
            .collect())
    }
}

impl Device {
    /// Number of locality domains.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the corresponding [`DeviceAttribute`]. Returns `Error::InvalidValue` if the native result is negative or does not fit the return type.
    pub fn locality_domain_count(self) -> Result<u32> {
        self.attribute(DeviceAttribute::LocalityDomainCount)?
            .try_into()
            .map_err(|_| Error::InvalidValue)
    }

    /// Number of multiprocessors on each locality domain.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the corresponding [`DeviceAttribute`]. Returns `Error::InvalidValue` if the native result is negative or does not fit the return type.
    pub fn locality_domain_multiprocessor_count(self) -> Result<u32> {
        self.attribute(DeviceAttribute::LocalityDomainMultiprocessorCount)?
            .try_into()
            .map_err(|_| Error::InvalidValue)
    }

    /// Maximum oversized shared memory per block.
    ///
    /// # Errors
    ///
    /// Propagates errors from querying the corresponding [`DeviceAttribute`]. Returns `Error::InvalidValue` if the native result is negative or does not fit the return type.
    pub fn maximum_oversized_shared_memory_per_block(self) -> Result<usize> {
        self.attribute(DeviceAttribute::MaxOversizedSharedMemoryPerBlock)?
            .try_into()
            .map_err(|_| Error::InvalidValue)
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

impl_enum_display!(ComputeMode, {
    ComputeMode::Default => "cudaComputeModeDefault",
    ComputeMode::Exclusive => "cudaComputeModeExclusive",
    ComputeMode::Prohibited => "cudaComputeModeProhibited",
    ComputeMode::ExclusiveProcess => "cudaComputeModeExclusiveProcess",
});

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

impl_enum_display!(CliqueType, {
    CliqueType::UnicastPointer => "CU_CLIQUE_TYPE_UNICAST_POINTER",
    CliqueType::MulticastPointer => "CU_CLIQUE_TYPE_MULTICAST_POINTER",
    CliqueType::UnicastLogicalEndpoint => "CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT",
    CliqueType::MulticastLogicalEndpoint => "CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT",
});
