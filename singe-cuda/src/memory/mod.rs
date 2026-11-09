pub mod array;
pub mod pattern;

/// Direct transfers retaining both actual CUDA contexts.
pub mod peer;

mod asynchronous;
mod device;
mod managed;
mod pinned;
mod pitched;
mod pool;
mod registered;
pub mod transfer;

use std::{
    alloc::{Layout, dealloc},
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem::{MaybeUninit, replace},
    ptr::{self, NonNull},
    sync::Arc,
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::{
    driver::{self, CUmemLocationType as MemoryLocationKind},
    runtime,
};

pub use self::pool::MemoryPoolUsage;
use crate::{
    context::Context,
    driver::entry_points::cuMemGetLocationInfo,
    error::{Error, Result},
    stream::{GraphRecordable, Stream, StreamCaptureScope},
    try_ffi,
    types::DevicePtr,
    view::{
        AllocationId, ByteBuffer, ByteBufferMut, DeviceInput, DeviceOutput, DeviceRegion,
        DeviceRegionProvider, DeviceRepr, DeviceSlice, DeviceSliceMut, HostRegion, HostSlice,
        HostSliceMut,
    },
};

/// A raw typed copy descriptor for recording in a stream capture.
///
/// This descriptor does not retain either allocation.
/// Its constructor's contract applies to every captured replay that uses the stored pointers.
#[derive(Debug, Clone, Copy)]
pub struct MemoryCopyOperation<'a, T> {
    dst: *mut T,
    src: *const T,
    count: usize,
    kind: MemoryCopyKind,
    _marker: PhantomData<&'a mut [T]>,
}

impl<T> MemoryCopyOperation<'_, T> {
    /// Creates a stream-capture-safe memcpy operation from raw pointers.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores `dst` and `src` pointer addresses in the resulting CUDA graph.
    /// The caller must ensure both pointers are valid for `count` elements whenever a captured graph using this operation is launched.
    /// If `dst` is mutable memory, it must also remain exclusive for the work ordered by those launches.
    /// Host destinations must receive initialized, valid values of `T` without duplicating Rust ownership or host references.
    pub const unsafe fn new(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
    ) -> Self {
        Self {
            dst,
            src,
            count,
            kind,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T> GraphRecordable for MemoryCopyOperation<'_, T> {
    type Output = ();

    fn record(self, scope: &StreamCaptureScope<'_>) -> Result<()> {
        // SAFETY: the caller of the unsafe `new` guaranteed `dst` and `src` stay valid for `count` elements for every launch of the captured graph.
        unsafe {
            DeviceMemory::<T>::copy_async(self.dst, self.src, self.count, self.kind, scope.stream())
        }
    }
}

/// A raw byte-pattern fill descriptor for recording in a stream capture.
///
/// `count` is measured in elements of `T`; the byte value is repeated across all `count * size_of::<T>()` bytes.
/// The descriptor does not retain the allocation.
#[derive(Debug, Clone, Copy)]
pub struct MemorySetOperation<'a, T> {
    dst: *mut T,
    value: u8,
    count: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<T> MemorySetOperation<'_, T> {
    /// Creates a stream-capture-safe memset operation from a raw pointer.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores `dst` in the resulting CUDA graph.
    /// The caller must ensure `dst` is valid for writes of `count * size_of::<T>()` bytes whenever a captured graph using this operation is launched, and that it remains exclusive for the work ordered by those launches.
    /// Any resulting host-visible values must be valid for `T`.
    pub const unsafe fn new(dst: *mut T, value: u8, count: usize) -> Self {
        Self {
            dst,
            value,
            count,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T> GraphRecordable for MemorySetOperation<'_, T> {
    type Output = ();

    fn record(self, scope: &StreamCaptureScope<'_>) -> Result<()> {
        // SAFETY: the caller of the unsafe `new` guaranteed `dst` stays valid for `count` elements for every launch of the captured graph.
        unsafe { DeviceMemory::<T>::set_async(self.dst, self.value, self.count, scope.stream()) }
    }
}

/// CUDA memory copy types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryCopyKind {
    /// Host -&gt; Host.
    HostToHost = runtime::cudaMemcpyKind::cudaMemcpyHostToHost as _,
    /// Host -&gt; Device.
    HostToDevice = runtime::cudaMemcpyKind::cudaMemcpyHostToDevice as _,
    /// Device -&gt; Host.
    DeviceToHost = runtime::cudaMemcpyKind::cudaMemcpyDeviceToHost as _,
    /// Device -&gt; Device.
    DeviceToDevice = runtime::cudaMemcpyKind::cudaMemcpyDeviceToDevice as _,
    /// Direction of the transfer is inferred from the pointer values.
    /// Requires unified virtual addressing.
    Default = runtime::cudaMemcpyKind::cudaMemcpyDefault as _,
}

impl_enum_conversion!(runtime::cudaMemcpyKind, MemoryCopyKind);

impl_enum_display!(MemoryCopyKind, {
    Self::HostToHost => "cudaMemcpyHostToHost",
    Self::HostToDevice => "cudaMemcpyHostToDevice",
    Self::DeviceToHost => "cudaMemcpyDeviceToHost",
    Self::DeviceToDevice => "cudaMemcpyDeviceToDevice",
    Self::Default => "cudaMemcpyDefault",
});

/// A borrowed CUDA array handle without an allocation owner or lifetime bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ArrayHandle(runtime::cudaArray_t);

impl ArrayHandle {
    /// Wraps a native CUDA array handle without taking ownership.
    ///
    /// # Safety
    ///
    /// `handle` must identify a live CUDA array.
    /// Keep its native owner alive for every use of this handle or a copy of it, including queued work and graph replays.
    pub const unsafe fn from_raw(handle: runtime::cudaArray_t) -> Self {
        Self(handle)
    }

    /// Returns the borrowed native array handle without retaining its owner.
    #[must_use]
    pub const fn as_raw(self) -> runtime::cudaArray_t {
        self.0
    }
}

bitflags::bitflags! {
    /// CUDA Mem Attach Flags
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MemoryAttachFlags: u32 {
        /// Memory can be accessed by any stream on any device.
        const GLOBAL = driver::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as _;
        /// Memory cannot be accessed by any stream on any device.
        const HOST = driver::CUmemAttach_flags::CU_MEM_ATTACH_HOST as _;
        /// Memory can only be accessed by a single stream on the associated device.
        const SINGLE = driver::CUmemAttach_flags::CU_MEM_ATTACH_SINGLE as _;
    }
}

impl Display for MemoryAttachFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Defines the allocation types available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAllocationType {
    /// No valid native memory allocation type is selected.
    Invalid = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_INVALID as _,
    /// This allocation type is 'pinned', i.e.
    ///
    /// cannot migrate from its current location while the application is actively using it.
    Pinned = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED as _,
    /// This allocation type is managed memory.
    Managed = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_MANAGED as _,
    /// Reserved upper-bound sentinel for native allocation types.
    Max = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_MAX as _,
}

impl_enum_conversion!(u32, driver::CUmemAllocationType, MemoryAllocationType);

/// Flags for specifying particular handle types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAllocationHandleType {
    /// Does not allow any export mechanism.
    ///
    /// >.
    None = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE as _,
    /// Allows a file descriptor to be used for exporting.
    ///
    /// Permitted only on POSIX systems. (int).
    PosixFileDescriptor =
        driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR as _,
    /// Allows a Win32 NT handle to be used for exporting.
    ///
    /// (HANDLE).
    Win32 = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32 as _,
    /// Allows a Win32 KMT handle to be used for exporting.
    ///
    /// (`D3DKMT_HANDLE`).
    Win32Kmt = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32_KMT as _,
    /// Allows a fabric handle to be used for exporting.
    ///
    /// (`CUmemFabricHandle`).
    Fabric = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC as _,
    /// Reserved upper-bound sentinel for native allocation handle types.
    Max = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_MAX as _,
}

impl_enum_conversion!(
    u32,
    driver::CUmemAllocationHandleType,
    MemoryAllocationHandleType
);

/// Specifies the memory protection flags for mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAccessFlag {
    /// Default, make the address range not accessible.
    None = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE as _,
    /// Make the address range read accessible.
    Read = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ as _,
    /// Make the address range read-write accessible.
    ReadWrite = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE as _,
    /// Reserved upper-bound sentinel for native memory access protections.
    Max = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_MAX as _,
}

impl_enum_conversion!(u32, driver::CUmemAccess_flags, MemoryAccessFlag);

/// CUDA memory pool attributes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryPoolAttribute {
    /// (value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another streams as long as a stream ordering dependency of the allocating stream on the free action exists.
    ///
    /// CUDA events and null stream interactions can create the required stream ordered dependencies. (default enabled).
    ReuseFollowEventDependencies =
        driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES as _,
    /// (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation.
    ///
    /// (default enabled).
    ReuseAllowOpportunistic =
        driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC as _,
    /// (value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by cuMemFreeAsync (default enabled).
    ReuseAllowInternalDependencies =
        driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES as _,
    /// (value type = `cuuint64_t`) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS.
    ///
    /// When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0).
    ReleaseThreshold = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD as _,
    /// (value type = `cuuint64_t`) Amount of backing memory currently allocated for the mempool.
    ReservedMemoryCurrent = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT as _,
    /// (value type = `cuuint64_t`) High watermark of backing memory allocated for the mempool since the last time it was reset.
    ///
    /// High watermark can only be reset to zero.
    ReservedMemoryHigh = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH as _,
    /// (value type = `cuuint64_t`) Amount of memory from the pool that is currently in use by the application.
    UsedMemoryCurrent = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT as _,
    /// (value type = `cuuint64_t`) High watermark of the amount of memory from the pool that was in use by the application since the last time it was reset.
    ///
    /// High watermark can only be reset to zero.
    UsedMemoryHigh = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH as _,
}

impl_enum_conversion!(u32, driver::CUmemPool_attribute, MemoryPoolAttribute);

/// Typed value supplied to or returned from a CUDA memory-pool attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MemoryPoolAttributeValue {
    /// Boolean value for a pool reuse policy.
    Bool(
        /// Whether the policy is enabled.
        bool,
    ),
    /// Unsigned byte count for a pool threshold or memory-usage statistic.
    Bytes(
        /// Number of bytes.
        u64,
    ),
}

/// Access permissions granted to a CUDA memory location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryAccessDescriptor {
    /// Memory location whose access permissions are being specified.
    pub location: MemoryLocation,
    /// Read and write permissions granted at the location.
    pub flags: MemoryAccessFlag,
}

/// Allocation, sharing, and placement properties for creating a CUDA memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryPoolProps {
    /// Type of memory allocated from the pool.
    pub alloc_type: MemoryAllocationType,
    /// Native handle type allowed when exporting pool memory.
    pub handle_type: MemoryAllocationHandleType,
    /// Backing location, or preferred location for managed allocations.
    pub location: MemoryLocation,
    /// Maximum pool size in bytes; zero uses the system-dependent limit.
    pub max_size: usize,
    /// Additional usage requirements for allocations from the pool.
    pub usage: MemoryPoolUsage,
    /// Enables `GPUDirect` RDMA access for allocations from this pool.
    pub gpu_direct_rdma_capable: bool,
}

/// Shares ownership of a CUDA memory pool and retains its context.
///
/// Clones and pool-backed allocations keep an owned pool alive.
/// Default pools are driver-owned and are not destroyed when the last wrapper is dropped.
#[derive(Debug, Clone)]
pub struct MemoryPool {
    owner: Arc<MemoryPoolOwner>,
}

#[derive(Debug)]
struct MemoryPoolOwner {
    context: Arc<Context>,
    handle: driver::CUmemoryPool,
    owned: bool,
}

// SAFETY: CUDA pool operations use shared driver handles. Binding the retained context
// establishes the calling thread's CUDA state before each operation.
unsafe impl Send for MemoryPoolOwner {}

unsafe impl Sync for MemoryPoolOwner {}

impl From<MemoryAccessDescriptor> for driver::CUmemAccessDesc {
    fn from(value: MemoryAccessDescriptor) -> Self {
        Self {
            location: value.location.into(),
            flags: value.flags.into(),
        }
    }
}

impl From<MemoryPoolProps> for driver::CUmemPoolProps {
    fn from(value: MemoryPoolProps) -> Self {
        Self {
            allocType: value.alloc_type.into(),
            handleTypes: value.handle_type.into(),
            location: value.location.into(),
            win32SecurityAttributes: ptr::null_mut(),
            maxSize: value.max_size as _,
            usage: value.usage.bits(),
            gpuDirectRDMACapable: value.gpu_direct_rdma_capable.into(),
            reserved: [0; 53],
        }
    }
}

bitflags::bitflags! {
    /// Capabilities requested for a CUDA page-locked host allocation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct HostAllocationFlags: u32 {
        /// Uses the default page-locked allocation policy.
        const DEFAULT = runtime::cudaHostAllocDefault;
        /// Makes the allocation recognized as page-locked by other CUDA contexts.
        const PORTABLE = runtime::cudaHostAllocPortable;
        /// Maps the allocation into the CUDA address space for device access.
        const MAPPED = runtime::cudaHostAllocMapped;
        /// Uses write-combined host memory, suited to CPU writes followed by device reads.
        const WRITE_COMBINED = runtime::cudaHostAllocWriteCombined;
    }
}

impl Display for HostAllocationFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Flags for CUDA host memory registration.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct HostRegisterFlags: u32 {
        /// Uses the default host registration policy.
        const DEFAULT = runtime::cudaHostRegisterDefault;
        /// Makes the registration recognized as page-locked by other CUDA contexts.
        const PORTABLE = runtime::cudaHostRegisterPortable;
        /// Maps the registered range into the CUDA address space for device access.
        const MAPPED = runtime::cudaHostRegisterMapped;
        /// Registers memory-mapped I/O storage as non-cache-coherent device memory.
        const IO_MEMORY = runtime::cudaHostRegisterIoMemory;
        /// Marks the registered range read-only to CUDA devices.
        const READ_ONLY = runtime::cudaHostRegisterReadOnly;
    }
}

impl Display for HostRegisterFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// CUDA memory types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryType {
    /// Unregistered memory.
    Unregistered = runtime::cudaMemoryType::cudaMemoryTypeUnregistered as _,
    /// Host memory.
    Host = runtime::cudaMemoryType::cudaMemoryTypeHost as _,
    /// Device memory.
    Device = runtime::cudaMemoryType::cudaMemoryTypeDevice as _,
    /// Managed memory.
    Managed = runtime::cudaMemoryType::cudaMemoryTypeManaged as _,
}

impl_enum_conversion!(runtime::cudaMemoryType, MemoryType);

impl_enum_display!(MemoryType, {
    Self::Unregistered => "cudaMemoryTypeUnregistered",
    Self::Host => "cudaMemoryTypeHost",
    Self::Device => "cudaMemoryTypeDevice",
    Self::Managed => "cudaMemoryTypeManaged",
});

/// CUDA-reported memory kind, owning device, and accessible aliases for a pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointerAttributes {
    /// CUDA classification of the memory addressed by the queried pointer.
    pub memory_type: MemoryType,
    /// Device ordinal associated with the allocation or registration.
    pub device: i32,
    /// Alias usable for device access in the current context, or null when unavailable.
    pub device_pointer: DevicePtr,
    /// Alias usable for host access, or null when the memory is not host-accessible.
    pub host_pointer: *mut (),
    /// Locality domain ordinal of the allocation; negative when not localized.
    pub locality_domain_ordinal: i32,
}

impl From<runtime::cudaPointerAttributes> for PointerAttributes {
    fn from(attr: runtime::cudaPointerAttributes) -> Self {
        Self {
            // SAFETY: every field of this union is a plain integer, so reading it is valid for any memory type.
            locality_domain_ordinal: unsafe { attr.__bindgen_anon_1.localityDomainOrdinal },
            memory_type: attr.type_.into(),
            device: attr.device,
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            device_pointer: unsafe { DevicePtr::from_raw(attr.devicePointer.cast()) },
            host_pointer: attr.hostPointer.cast(),
        }
    }
}

impl PointerAttributes {
    /// Returns the CUDA pointer attributes for a raw pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query attributes for `ptr`, `ptr` is not known to a unified-addressing context, or CUDA reports runtime initialization diagnostics.
    pub fn query<T>(ptr: *const T) -> Result<Self> {
        let mut attr_ffi = runtime::cudaPointerAttributes::default();
        // SAFETY: `cudaPointerGetAttributes` only queries `ptr` and writes into the local `attr_ffi`.
        unsafe {
            try_ffi!(runtime::cudaPointerGetAttributes(
                &raw mut attr_ffi,
                ptr.cast(),
            ))?;
        }
        Ok(attr_ffi.into())
    }

    /// Returns CUDA pointer attributes for a raw CUDA device pointer wrapper.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query attributes for `ptr`, the pointer is not known to the current CUDA context, or CUDA reports runtime initialization diagnostics.
    pub fn query_device(ptr: DevicePtr) -> Result<Self> {
        Self::query(ptr.as_ptr())
    }

    /// Returns CUDA pointer attributes for a non-null host or device pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query attributes for `ptr`, the pointer is not known to a unified-addressing context, or CUDA reports runtime initialization diagnostics.
    pub fn query_non_null<T>(ptr: NonNull<T>) -> Result<Self> {
        Self::query(ptr.as_ptr())
    }
}

/// Specifies the type and identity of a memory location.
#[derive(Debug, Default, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryLocation {
    /// Location is unspecified; no preferred location for a managed memory pool.
    #[default]
    Unspecified,
    /// Location is a device location; the payload is a device ordinal.
    Device(i32),
    /// Location is host memory.
    Host,
    /// Location is a host NUMA node; the payload is its identifier.
    Numa(i32),
    /// Location is the host NUMA node of the current thread.
    NumaCurrent,
    /// Location is not visible but the device is accessible.
    Invisible,
    /// Location is a portion of device memory, specified by the locality domain ID.
    DeviceLocalityDomain {
        /// Device ordinal containing the locality domain.
        device_id: u8,
        /// Locality domain identifier within the device.
        locality_domain_id: u8,
    },
}

impl From<driver::CUmemLocation> for MemoryLocation {
    fn from(value: driver::CUmemLocation) -> Self {
        // SAFETY: each arm reads the union field that the matched location type selects.
        unsafe {
            match value.type_ {
                MemoryLocationKind::CU_MEM_LOCATION_TYPE_DEVICE => {
                    Self::Device(value.__bindgen_anon_1.id)
                }
                MemoryLocationKind::CU_MEM_LOCATION_TYPE_HOST => Self::Host,
                MemoryLocationKind::CU_MEM_LOCATION_TYPE_HOST_NUMA => {
                    Self::Numa(value.__bindgen_anon_1.id)
                }
                MemoryLocationKind::CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT => Self::NumaCurrent,
                MemoryLocationKind::CU_MEM_LOCATION_TYPE_INVISIBLE => Self::Invisible,
                MemoryLocationKind::CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN => {
                    let location = value.__bindgen_anon_1.localized;
                    Self::DeviceLocalityDomain {
                        device_id: location.deviceId,
                        locality_domain_id: location.localityDomainId,
                    }
                }
                _ => Self::Unspecified,
            }
        }
    }
}

impl From<MemoryLocation> for driver::CUmemLocation {
    fn from(value: MemoryLocation) -> Self {
        let mut raw = Self::default();
        let (kind, id) = match value {
            MemoryLocation::Unspecified => (MemoryLocationKind::CU_MEM_LOCATION_TYPE_INVALID, 0),
            MemoryLocation::Device(id) => (MemoryLocationKind::CU_MEM_LOCATION_TYPE_DEVICE, id),
            MemoryLocation::Host => (MemoryLocationKind::CU_MEM_LOCATION_TYPE_HOST, 0),
            MemoryLocation::Numa(id) => (MemoryLocationKind::CU_MEM_LOCATION_TYPE_HOST_NUMA, id),
            MemoryLocation::NumaCurrent => (
                MemoryLocationKind::CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
                0,
            ),
            MemoryLocation::Invisible => (MemoryLocationKind::CU_MEM_LOCATION_TYPE_INVISIBLE, -2),
            MemoryLocation::DeviceLocalityDomain {
                device_id,
                locality_domain_id,
            } => {
                raw.type_ = MemoryLocationKind::CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
                raw.__bindgen_anon_1.localized =
                    driver::CUmemLocation_st__bindgen_ty_1__bindgen_ty_1 {
                        deviceId: device_id,
                        localityDomainId: locality_domain_id,
                    };
                return raw;
            }
        };
        raw.type_ = kind;
        raw.__bindgen_anon_1.id = id;
        raw
    }
}

impl Drop for MemoryPoolOwner {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        if let Err(error) = self
            .context
            .bind()
            // SAFETY: `cuMemPoolDestroy` receives live handles owned by the wrappers.
            .and_then(|()| unsafe { try_ffi!(driver::cuMemPoolDestroy(self.handle)) })
        {
            eprintln!(
                "failed to destroy cuda memory pool: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

#[derive(Debug)]
enum DeviceDeallocation {
    Immediate,
    StreamOrdered {
        stream: Stream,
        pool: Option<MemoryPool>,
    },
}

/// A queued free owns its deallocation resources without retaining its stream.
#[derive(Debug)]
pub(crate) struct DeferredFree {
    pointer: *mut u8,
    context: Arc<Context>,
    _pool: Option<MemoryPool>,
}

// SAFETY: only CUDA accesses the pointer; the retained context and pool support thread binding.
unsafe impl Send for DeferredFree {}

unsafe impl Sync for DeferredFree {}

impl DeferredFree {
    pub(crate) fn enqueue(&self, stream: runtime::cudaStream_t) -> Result<()> {
        self.context.bind()?;
        // SAFETY: `cudaFreeAsync` receives only plain values.
        unsafe { try_ffi!(runtime::cudaFreeAsync(self.pointer.cast(), stream)) }
    }
}

/// Owns a CUDA allocation independently of its typed buffers and views.
///
/// Retaining this owner keeps the allocation, context, and deallocation resources alive.
/// Raw CUDA work must still satisfy the lifetime and ordering contract of its launch.
#[derive(Debug)]
pub struct DeviceAllocation {
    id: AllocationId,
    context: Arc<Context>,
    ptr: *mut u8,
    bytes: usize,
    deallocation: DeviceDeallocation,
}

// SAFETY: device addresses are not dereferenced by the host. All retained CUDA resources
// support thread binding, and accesses remain governed by borrows or raw unsafe APIs.
unsafe impl Send for DeviceAllocation {}

unsafe impl Sync for DeviceAllocation {}

impl DeviceAllocation {
    fn from_parts(
        context: &Arc<Context>,
        ptr: *mut u8,
        bytes: usize,
        deallocation: DeviceDeallocation,
    ) -> Arc<Self> {
        Arc::new(Self {
            id: AllocationId::create(),
            context: Arc::clone(context),
            ptr,
            bytes,
            deallocation,
        })
    }

    /// Returns the identity shared by every buffer and view of this allocation.
    #[must_use]
    pub const fn id(&self) -> AllocationId {
        self.id
    }

    /// Returns the context that owns this allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the allocation's base device address.
    #[must_use]
    pub fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the full allocation size, independent of typed views.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Returns the allocation and deallocation stream for stream-ordered memory.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        match &self.deallocation {
            DeviceDeallocation::Immediate => None,
            DeviceDeallocation::StreamOrdered { stream, .. } => Some(stream),
        }
    }

    /// Returns the explicitly selected pool retained by this allocation.
    #[must_use]
    pub fn pool(&self) -> Option<&MemoryPool> {
        match &self.deallocation {
            DeviceDeallocation::Immediate => None,
            DeviceDeallocation::StreamOrdered { pool, .. } => pool.as_ref(),
        }
    }

    fn release(&mut self) -> Result<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        match &self.deallocation {
            DeviceDeallocation::Immediate => {
                self.context.bind()?;
                // SAFETY: the final owner releases the original allocation pointer.
                unsafe {
                    DeviceMemory::<u8>::free(self.ptr)?;
                }

                self.ptr = ptr::null_mut();
                Ok(())
            }
            DeviceDeallocation::StreamOrdered { stream, pool } => {
                let free = DeferredFree {
                    pointer: replace(&mut self.ptr, ptr::null_mut()),
                    context: Arc::clone(&self.context),
                    _pool: pool.clone(),
                };
                // The stream keeps failed frees and captured frees until they can be enqueued.
                stream.defer_free(free)
            }
        }
    }
}

unsafe impl DeviceRegionProvider for DeviceAllocation {
    fn region(&self) -> DeviceRegion<'_> {
        // SAFETY: this owner retains the complete allocation and its context.
        let region =
            unsafe { DeviceRegion::from_raw_parts(&self.context, self.id, self.ptr, self.bytes) }
                .expect("invalid device allocation metadata");
        self.stream().map_or(region, |stream| {
            unsafe { region.with_stream(stream) }.expect("allocation stream context mismatch")
        })
    }
}

unsafe impl<T> DeviceRegionProvider for DeviceMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let offset = self.ptr as usize - self.allocation.ptr as usize;
        self.allocation
            .region()
            .slice(offset..offset + self.byte_len())
            .expect("device buffer exceeds its allocation")
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            eprintln!(
                "failed to release cuda allocation: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

/// Represents a region of owned CUDA device memory for elements of type `T`.
#[derive(Debug)]
pub struct DeviceMemory<T> {
    allocation: Arc<DeviceAllocation>,
    /// Raw pointer to the allocated device memory.
    ptr: *mut T,
    /// Number of elements of type `T` allocated.
    length: usize,
    /// Marker for the type `T`.
    _phantom: PhantomData<T>,
}

impl<T> PartialEq for DeviceMemory<T> {
    fn eq(&self, other: &Self) -> bool {
        self.context() == other.context() && self.ptr == other.ptr && self.length == other.length
    }
}

impl<T> Eq for DeviceMemory<T> {}

/// Owned stream-ordered CUDA device memory.
///
/// Allocations are created with `cudaMallocAsync` or `cudaMallocFromPoolAsync` and are freed with `cudaFreeAsync` on the stored stream.
#[derive(Debug)]
pub struct AsyncDeviceMemory<T> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    length: usize,
    _phantom: PhantomData<T>,
}

/// Owned 2D pitched CUDA device memory.
///
/// The logical dimensions are expressed in elements of `T`; the CUDA pitch is stored as a byte stride between rows.
#[derive(Debug)]
pub struct PitchedDeviceMemory<T: DeviceRepr> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    width: usize,
    height: usize,
    pitch: usize,
    _phantom: PhantomData<T>,
}

/// Owned 3D pitched CUDA device memory.
///
/// The logical dimensions are expressed in elements of `T`; the CUDA pitch is stored as a byte stride between rows inside each slice.
#[derive(Debug)]
pub struct PitchedDeviceVolume<T: DeviceRepr> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    width: usize,
    height: usize,
    depth: usize,
    pitch: usize,
    _phantom: PhantomData<T>,
}

#[derive(Debug)]
struct HostRegistration {
    flags: HostRegisterFlags,
    context: Arc<Context>,
    id: AllocationId,
    ptr: NonNull<()>,
    bytes: usize,
}

impl HostRegistration {
    unsafe fn create(
        context: &Arc<Context>,
        ptr: NonNull<()>,
        bytes: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        context.bind()?;
        if bytes != 0 {
            // SAFETY: the typed wrapper retains the host storage for this registration's lifetime.
            unsafe {
                register_host_raw(ptr.as_ptr(), bytes, flags)?;
            }
        }
        Ok(Self {
            flags,
            context: Arc::clone(context),
            id: AllocationId::create(),
            ptr,
            bytes,
        })
    }

    fn host_region(&self) -> HostRegion<'_> {
        // SAFETY: the registration retains this range and its original flags.
        unsafe {
            HostRegion::from_raw_parts(
                &self.context,
                self.id,
                self.bytes,
                self.flags.contains(HostRegisterFlags::PORTABLE),
            )
        }
    }

    fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.context.bind()?;
        if self.bytes == 0 {
            return Ok(DevicePtr::null());
        }
        mapped_host_device_ptr(self.ptr.as_ptr())
    }

    fn mapped_device_region(&self) -> Result<DeviceRegion<'_>> {
        let pointer = self.mapped_device_ptr()?;
        // SAFETY: the registration and its typed wrapper retain the host borrow or owned vector.
        unsafe { DeviceRegion::from_raw_parts(&self.context, self.id, pointer.cast(), self.bytes) }
    }

    fn unregister(&mut self) -> Result<()> {
        if self.bytes != 0 {
            self.context.bind()?;
            // SAFETY: the registration still retains its live host range.
            unsafe { unregister_host_raw(self.ptr.as_ptr())? };
            self.bytes = 0;
        }
        Ok(())
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            eprintln!(
                "failed to unregister cuda host memory: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

/// Retains a CUDA pinned host allocation independently of its typed buffer.
///
/// Raw host and mapped device addresses require externally established validity and access ordering.
#[derive(Debug)]
pub struct PinnedAllocation {
    context: Arc<Context>,
    id: AllocationId,
    ptr: *mut u8,
    bytes: usize,
    flags: HostAllocationFlags,
}

// SAFETY: only raw addresses and immutable metadata are exposed.
// Typed borrows or unsafe caller contracts govern all host and device accesses.
unsafe impl Send for PinnedAllocation {}

unsafe impl Sync for PinnedAllocation {}

impl PinnedAllocation {
    /// Describes the full host allocation without requiring a device mapping.
    #[must_use]
    pub fn host_region(&self) -> HostRegion<'_> {
        // SAFETY: this owner retains the complete allocation and original flags.
        unsafe {
            HostRegion::from_raw_parts(
                &self.context,
                self.id,
                self.bytes,
                self.flags.contains(HostAllocationFlags::PORTABLE),
            )
        }
    }

    /// Returns the identity shared by host and mapped device views.
    #[must_use]
    pub const fn id(&self) -> AllocationId {
        self.id
    }

    /// Returns the allocation's CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the original host address, or null for an empty allocation.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the full allocation size independent of any initialized prefix.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Returns the allocation flags.
    #[must_use]
    pub const fn flags(&self) -> HostAllocationFlags {
        self.flags
    }

    /// Returns the mapped device address in the retained context.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA host-to-device mapping query errors, including memory that is not mapped for this context.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.context.bind()?;
        if self.bytes == 0 {
            // SAFETY: a null address is always a valid `DevicePtr`.
            return Ok(unsafe { DevicePtr::from_raw(ptr::null_mut()) });
        }
        mapped_host_device_ptr(self.ptr.cast())
    }

    /// Describes the full mapped allocation without asserting element initialization.
    ///
    /// # Errors
    ///
    /// Propagates mapping-query errors and rejects a nonempty null mapping or an ending address that overflows `usize`.
    pub fn mapped_device_region(&self) -> Result<DeviceRegion<'_>> {
        let pointer = self.mapped_device_ptr()?;
        // SAFETY: the owner retains the allocation and the CUDA mapping's context.
        unsafe { DeviceRegion::from_raw_parts(&self.context, self.id, pointer.cast(), self.bytes) }
    }
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Err(error) = self
            .context
            .bind()
            // SAFETY: `self.ptr` came from the matching host allocation and is freed exactly once, in `Drop`.
            .and_then(|()| unsafe { free_host_raw(self.ptr.cast()) })
        {
            eprintln!(
                "failed to free cuda pinned allocation: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

/// Owned typed CUDA page-locked host memory.
#[derive(Debug)]
pub struct PinnedHostMemory<T: DeviceRepr> {
    allocation: Arc<PinnedAllocation>,
    ptr: NonNull<T>,
    length: usize,
}

/// Mutable borrowed host memory registered with CUDA for the registration lifetime.
#[derive(Debug)]
pub struct RegisteredHostMemory<'a, T: DeviceRepr> {
    ptr: NonNull<T>,
    length: usize,
    registration: HostRegistration,
    _borrow: PhantomData<&'a mut [T]>,
}

/// Immutable borrowed host memory registered with CUDA for the registration lifetime.
#[derive(Debug)]
pub struct RegisteredHostSlice<'a, T: DeviceRepr> {
    ptr: NonNull<T>,
    length: usize,
    registration: HostRegistration,
    _borrow: PhantomData<&'a [T]>,
}

/// Owned typed host storage with an independently retained CUDA registration.
#[derive(Debug)]
pub struct RegisteredHostVec<T: DeviceRepr> {
    allocation: Arc<RegisteredAllocation>,
    ptr: NonNull<T>,
    length: usize,
}

/// Retains an owned host vector's allocation and CUDA registration independently of typed access.
///
/// Exposes raw addresses and metadata without granting initialized or exclusive access.
/// If CUDA cannot unregister the memory on destruction, backing host storage remains allocated.
#[derive(Debug)]
pub struct RegisteredAllocation {
    registration: HostRegistration,
    layout: Layout,
}

// SAFETY: only immutable metadata and raw addresses are exposed.
// Global vector allocation and deallocation are independent of the host thread.
// DeviceRepr excludes owned values and destructors, so freeing the erased storage does not access T.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "raw host pointers follow the owned or borrowed storage bounds explained above"
)]
unsafe impl Send for RegisteredAllocation {}

unsafe impl Sync for RegisteredAllocation {}

impl RegisteredAllocation {
    /// Describes the complete registration without requiring a device mapping.
    #[must_use]
    pub fn host_region(&self) -> HostRegion<'_> {
        self.registration.host_region()
    }

    /// Returns the identity shared by the registration and every mapped view.
    #[must_use]
    pub const fn id(&self) -> AllocationId {
        self.registration.id
    }

    /// Returns the context used for registration, mapping, and unregistration.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.registration.context
    }

    /// Returns the original host allocation address.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.registration.ptr.as_ptr().cast()
    }

    /// Returns the complete registered byte range, independent of initialized prefixes.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.registration.bytes
    }

    /// Returns the original allocation layout, including unused vector capacity.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns the registration flags.
    #[must_use]
    pub const fn flags(&self) -> HostRegisterFlags {
        self.registration.flags
    }

    /// Returns the device alias in the actual registration context.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA host-to-device mapping query errors, including memory that is not mapped for this context.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.registration.mapped_device_ptr()
    }

    /// Describes the complete mapped registration without asserting initialization or access rights.
    ///
    /// # Errors
    ///
    /// Propagates mapping-query errors and rejects a nonempty null mapping or an ending address that overflows `usize`.
    pub fn mapped_device_region(&self) -> Result<DeviceRegion<'_>> {
        self.registration.mapped_device_region()
    }
}

impl Drop for RegisteredAllocation {
    fn drop(&mut self) {
        if let Err(error) = self.registration.unregister() {
            // Keep backing storage allocated if CUDA could still retain its registration.
            eprintln!(
                "failed to unregister owned cuda host memory: {}",
                singe_core::ErrorChain(&error)
            );
            return;
        }
        if self.layout.size() != 0 {
            // SAFETY: the original Vec used the global allocator and exactly this layout.
            // Registration is gone, and the last retained allocation owner is being destroyed.
            unsafe { dealloc(self.as_ptr(), self.layout) };
        }
    }
}

/// Owns a CUDA managed allocation and retains its context.
///
/// Host access through the guarded callback methods waits for tracked GPU work and excludes conflicting submissions for the duration of the callback.
#[derive(Debug)]
pub struct ManagedMemory<T: DeviceRepr> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    length: usize,
    // CUDA tracks the current visibility policy for managed memory separately
    // from the pointer. Store the last policy requested through this wrapper so
    // callers can reason about stream attachment without another FFI query.
    attach_flags: MemoryAttachFlags,
    _phantom: PhantomData<T>,
}

/// A host-memory range CUDA can legally inspect.
///
/// # Safety
///
/// Implementors must guarantee the range returned by [`HostSlice::as_host_ptr`] remains valid and registered or allocated in a way CUDA accepts for the lifetime of `&self`.
/// Host metadata must report that range, the actual registration flags, and the identity shared by all mapped aliases.
pub unsafe trait CudaHostSlice<T: DeviceRepr>: HostSlice<T> {
    /// Describes the same bytes as this host slice, preserving identity across mapped aliases.
    fn host_region(&self) -> HostRegion<'_>;
}

/// A page-locked host-memory range suitable for efficient CUDA transfers.
///
/// # Safety
///
/// Implementors must guarantee the range is page-locked for the lifetime of `&self`.
pub unsafe trait PinnedHostSlice<T: DeviceRepr>: CudaHostSlice<T> {}

/// A mutable page-locked host-memory range suitable for efficient CUDA transfers.
///
/// # Safety
///
/// Implementors must uphold both [`PinnedHostSlice`] and [`HostSliceMut`] contracts.
pub unsafe trait PinnedHostSliceMut<T: DeviceRepr>:
    PinnedHostSlice<T> + HostSliceMut<T>
{
}

// SAFETY: DeviceMemory owns a CUDA allocation and does not provide CPU
// references into the allocation. Moving the owner to another thread is safe
// when `T: Send` because CUDA allocation/free operations are synchronized by
// the CUDA context APIs used by this wrapper. Shared references are safe when
// `T: Sync` because `&DeviceMemory<T>` only exposes raw device pointers and
// read-only metadata; mutation of device memory still requires CUDA operations
// whose safety contracts are enforced by the calling APIs.
unsafe impl<T: Send> Send for DeviceMemory<T> {}

unsafe impl<T: Sync> Sync for DeviceMemory<T> {}

unsafe impl<T: Send> Send for AsyncDeviceMemory<T> {}

unsafe impl<T: Sync> Sync for AsyncDeviceMemory<T> {}

// SAFETY: managed memory has the same Rust-side ownership properties as
// DeviceMemory. CUDA maintains coherence for managed allocations across host
// threads and devices; Rust references into the allocation are not exposed by
// this wrapper.
unsafe impl<T: DeviceRepr + Send> Send for ManagedMemory<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for ManagedMemory<T> {}

// SAFETY: host wrappers own or borrow their storage like `Vec<T>`, `&mut [T]` and `&[T]`,
// so they carry the same bounds on `T`. Pinned allocations and registrations can be
// released from any thread.
unsafe impl<T: DeviceRepr + Send> Send for PinnedHostMemory<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for PinnedHostMemory<T> {}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "raw host pointers follow the owned or borrowed storage bounds explained above"
)]
unsafe impl<T: DeviceRepr + Send> Send for RegisteredHostMemory<'_, T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for RegisteredHostMemory<'_, T> {}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "raw host pointers follow the owned or borrowed storage bounds explained above"
)]
unsafe impl<T: DeviceRepr + Sync> Send for RegisteredHostSlice<'_, T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for RegisteredHostSlice<'_, T> {}

unsafe impl<T: DeviceRepr + Send> Send for RegisteredHostVec<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for RegisteredHostVec<T> {}

unsafe impl<T: DeviceRepr> HostSlice<T> for PinnedHostMemory<T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> HostSliceMut<T> for PinnedHostMemory<T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

unsafe impl ByteBuffer for PinnedHostMemory<u8> {
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn byte_len(&self) -> usize {
        self.length
    }
}

unsafe impl ByteBufferMut for PinnedHostMemory<u8> {
    fn as_byte_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}

unsafe impl<T: DeviceRepr> CudaHostSlice<T> for PinnedHostMemory<T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.allocation
            .host_region()
            .slice(..self.byte_len().expect("valid host slice length"))
            .expect("host slice fits its allocation")
    }
}

unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for PinnedHostMemory<T> {}

unsafe impl<T: DeviceRepr> PinnedHostSliceMut<T> for PinnedHostMemory<T> {}

unsafe impl<T: DeviceRepr> HostSlice<T> for RegisteredHostMemory<'_, T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> HostSliceMut<T> for RegisteredHostMemory<'_, T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

unsafe impl<T: DeviceRepr> HostSlice<T> for RegisteredHostSlice<'_, T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> HostSlice<T> for RegisteredHostVec<T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> HostSliceMut<T> for RegisteredHostVec<T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

unsafe impl ByteBuffer for RegisteredHostMemory<'_, u8> {
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn byte_len(&self) -> usize {
        self.length
    }
}

unsafe impl ByteBufferMut for RegisteredHostMemory<'_, u8> {
    fn as_byte_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}

unsafe impl ByteBuffer for RegisteredHostSlice<'_, u8> {
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn byte_len(&self) -> usize {
        self.length
    }
}

unsafe impl ByteBuffer for RegisteredHostVec<u8> {
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn byte_len(&self) -> usize {
        self.len()
    }
}

unsafe impl ByteBufferMut for RegisteredHostVec<u8> {
    fn as_byte_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}

unsafe impl<T: DeviceRepr> CudaHostSlice<T> for RegisteredHostMemory<'_, T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.registration
            .host_region()
            .slice(..self.byte_len().expect("valid host slice length"))
            .expect("host slice fits its allocation")
    }
}

unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for RegisteredHostMemory<'_, T> {}

unsafe impl<T: DeviceRepr> PinnedHostSliceMut<T> for RegisteredHostMemory<'_, T> {}

unsafe impl<T: DeviceRepr> CudaHostSlice<T> for RegisteredHostSlice<'_, T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.registration
            .host_region()
            .slice(..self.byte_len().expect("valid host slice length"))
            .expect("host slice fits its allocation")
    }
}

unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for RegisteredHostSlice<'_, T> {}

unsafe impl<T: DeviceRepr> CudaHostSlice<T> for RegisteredHostVec<T> {
    fn host_region(&self) -> HostRegion<'_> {
        self.allocation
            .host_region()
            .slice(..self.byte_len().expect("valid host slice length"))
            .expect("host slice fits its allocation")
    }
}

unsafe impl<T: DeviceRepr> PinnedHostSlice<T> for RegisteredHostVec<T> {}

unsafe impl<T: DeviceRepr> PinnedHostSliceMut<T> for RegisteredHostVec<T> {}

fn mapped_host_device_ptr(ptr: *mut ()) -> Result<DevicePtr> {
    if ptr.is_null() {
        return Ok(DevicePtr::null());
    }

    let mut device_ptr = ptr::null_mut();
    // SAFETY: `cudaHostGetDevicePointer` receives valid output locations.
    unsafe {
        try_ffi!(runtime::cudaHostGetDevicePointer(
            &raw mut device_ptr,
            ptr.cast(),
            0,
        ))?;
    }
    // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
    Ok(unsafe { DevicePtr::from_raw(device_ptr.cast()) })
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for PitchedDeviceVolume<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

// SAFETY: device pointers are not dereferenced on the host; all operations bind the retained context.
unsafe impl<T: DeviceRepr + Send> Send for PitchedDeviceVolume<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for PitchedDeviceVolume<T> {}

unsafe impl<T: DeviceRepr> DeviceInput<T> for AsyncDeviceMemory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for AsyncDeviceMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for AsyncDeviceMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for AsyncDeviceMemory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr
    }
}

// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for AsyncDeviceMemory<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for AsyncDeviceMemory<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr.cast()
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for ManagedMemory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr
    }
}

// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for ManagedMemory<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for ManagedMemory<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr.cast()
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for DeviceMemory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr
    }
}

// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for DeviceMemory<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for DeviceMemory<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr.cast()
    }
}

fn checked_bytes<T>(len: usize) -> Result<usize> {
    let size = size_of::<T>();
    if size == 0 {
        return if len == 0 {
            Ok(0)
        } else {
            Err(Error::InvalidMemoryAllocationRequest)
        };
    }
    len.checked_mul(size)
        .ok_or(Error::InvalidMemoryAllocationRequest)
}

fn checked_len(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .ok_or(Error::InvalidMemoryAllocationRequest)
}

fn checked_volume(width: usize, height: usize, depth: usize) -> Result<usize> {
    checked_len(width, height)?
        .checked_mul(depth)
        .ok_or(Error::InvalidMemoryAllocationRequest)
}

fn dangling_or_non_null<T>(ptr: *mut T, len: usize) -> Result<NonNull<T>> {
    if len == 0 {
        Ok(NonNull::dangling())
    } else {
        NonNull::new(ptr).ok_or(Error::NullHandle)
    }
}

unsafe fn alloc_host_raw(size: usize) -> Result<*mut ()> {
    let mut ptr = ptr::null_mut();
    // SAFETY: `cudaMallocHost` receives valid output locations.
    unsafe {
        try_ffi!(runtime::cudaMallocHost(
            &raw mut ptr,
            size as runtime::size_t
        ))?;
    }
    Ok(ptr.cast())
}

unsafe fn alloc_pinned_raw(size: usize, flags: HostAllocationFlags) -> Result<*mut ()> {
    let mut ptr = ptr::null_mut();
    // SAFETY: `cudaHostAlloc` receives valid output locations.
    unsafe {
        try_ffi!(runtime::cudaHostAlloc(
            &raw mut ptr,
            size as _,
            flags.bits()
        ))?;
    }
    Ok(ptr.cast())
}

unsafe fn free_host_raw(ptr: *mut ()) -> Result<()> {
    // SAFETY: `cudaFreeHost` receives only plain values.
    unsafe { try_ffi!(runtime::cudaFreeHost(ptr.cast())) }
}

unsafe fn register_host_raw(ptr: *mut (), size: usize, flags: HostRegisterFlags) -> Result<()> {
    // SAFETY: `cudaHostRegister` receives only plain values.
    unsafe {
        try_ffi!(runtime::cudaHostRegister(
            ptr.cast(),
            size as _,
            flags.bits()
        ))?;
    }
    Ok(())
}

unsafe fn unregister_host_raw(ptr: *mut ()) -> Result<()> {
    // SAFETY: `cudaHostUnregister` receives only plain values.
    unsafe { try_ffi!(runtime::cudaHostUnregister(ptr.cast())) }
}

/// Retrieves memory location information for an address range.
///
/// Returns the most common location in each summary region, sampling at the specified interval.
/// A zero sampling granularity uses the system default.
///
/// # Safety
/// The range must remain allocated throughout this call and refer to managed memory or system-allocated pageable memory not registered with CUDA.
///
/// # Errors
///
/// Returns an error if summary granularity is zero or exceeds the range size, sampling granularity exceeds summary granularity, the driver entry point is unavailable, or CUDA rejects the location query.
pub unsafe fn location_info(
    ptr: DevicePtr,
    size: usize,
    summary_granularity: usize,
    sampling_granularity: usize,
) -> Result<Vec<MemoryLocation>> {
    if summary_granularity == 0
        || summary_granularity > size
        || sampling_granularity > summary_granularity
    {
        return Err(Error::InvalidValue);
    }

    let count = size.div_ceil(summary_granularity);
    let mut locations = vec![driver::CUmemLocation::default(); count];
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cuMemGetLocationInfo(
            ptr.as_ptr() as _,
            size as _,
            summary_granularity as _,
            sampling_granularity as _,
            locations.as_mut_ptr(),
        )?;
    }
    Ok(locations.into_iter().map(Into::into).collect())
}

/// Integer types that memory pool attributes are read into; every bit pattern is valid.
trait PoolAttributeValue: Copy + Default {}

impl_enum_display!(MemoryAllocationType, {
    MemoryAllocationType::Invalid => "CU_MEM_ALLOCATION_TYPE_INVALID",
    MemoryAllocationType::Pinned => "CU_MEM_ALLOCATION_TYPE_PINNED",
    MemoryAllocationType::Managed => "CU_MEM_ALLOCATION_TYPE_MANAGED",
    MemoryAllocationType::Max => "CU_MEM_ALLOCATION_TYPE_MAX",
});

impl_enum_display!(MemoryAllocationHandleType, {
    MemoryAllocationHandleType::None => "CU_MEM_HANDLE_TYPE_NONE",
    MemoryAllocationHandleType::PosixFileDescriptor => "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR",
    MemoryAllocationHandleType::Win32 => "CU_MEM_HANDLE_TYPE_WIN32",
    MemoryAllocationHandleType::Win32Kmt => "CU_MEM_HANDLE_TYPE_WIN32_KMT",
    MemoryAllocationHandleType::Fabric => "CU_MEM_HANDLE_TYPE_FABRIC",
    MemoryAllocationHandleType::Max => "CU_MEM_HANDLE_TYPE_MAX",
});

impl_enum_display!(MemoryAccessFlag, {
    MemoryAccessFlag::None => "CU_MEM_ACCESS_FLAGS_PROT_NONE",
    MemoryAccessFlag::Read => "CU_MEM_ACCESS_FLAGS_PROT_READ",
    MemoryAccessFlag::ReadWrite => "CU_MEM_ACCESS_FLAGS_PROT_READWRITE",
    MemoryAccessFlag::Max => "CU_MEM_ACCESS_FLAGS_PROT_MAX",
});

impl_enum_display!(MemoryPoolAttribute, {
    MemoryPoolAttribute::ReuseFollowEventDependencies => "CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES",
    MemoryPoolAttribute::ReuseAllowOpportunistic => "CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC",
    MemoryPoolAttribute::ReuseAllowInternalDependencies => "CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES",
    MemoryPoolAttribute::ReleaseThreshold => "CU_MEMPOOL_ATTR_RELEASE_THRESHOLD",
    MemoryPoolAttribute::ReservedMemoryCurrent => "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT",
    MemoryPoolAttribute::ReservedMemoryHigh => "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH",
    MemoryPoolAttribute::UsedMemoryCurrent => "CU_MEMPOOL_ATTR_USED_MEM_CURRENT",
    MemoryPoolAttribute::UsedMemoryHigh => "CU_MEMPOOL_ATTR_USED_MEM_HIGH",
});

#[cfg(all(test, feature = "testing"))]
mod tests;
