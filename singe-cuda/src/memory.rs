use crate::utility::{to_usize, usage_bits};

pub mod array;
pub mod pattern;

use std::{
    alloc::{Layout, dealloc},
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit, replace},
    ptr::{self, NonNull},
    result::Result as StdResult,
    slice,
    sync::Arc,
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::{
    driver::{self, CUmemLocationType as MemoryLocationKind},
    runtime,
};

use crate::{
    context::Context,
    device::DeviceAttribute,
    driver::entry_points::cuMemGetLocationInfo,
    error::{Error, Result},
    ipc::IpcMemoryHandle,
    stream::{GraphRecordable, Stream, StreamBinding, StreamCaptureScope, StreamScope},
    submission::{Access, AccessMode, Execution, Resources},
    try_ffi,
    types::{DevicePtr, MemoryAdvise},
    view::{
        AllocationId, AnyBitPatternDeviceRepr, ByteBuffer, ByteBufferMut, DeviceInput,
        DeviceOutput, DeviceOutputView, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut, HostRegion, HostSlice,
        HostSliceMut, NoUninitDeviceRepr, ZeroableDeviceRepr, cast_length,
    },
    virtual_memory::AllocationHandleTypes,
};

/// Direct transfers retaining both actual CUDA contexts.
pub mod peer;

pub mod transfer;

/// A raw typed copy descriptor for recording in a stream capture.
///
/// This descriptor does not retain either allocation. Its constructor's contract
/// applies to every captured replay that uses the stored pointers.
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
    /// Capturing this operation stores `dst` and `src` pointer addresses in the
    /// resulting CUDA graph. The caller must ensure both pointers are valid for
    /// `count` elements whenever a captured graph using this operation is
    /// launched. If `dst` is mutable memory, it must also remain exclusive for
    /// the work ordered by those launches.
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
/// `count` is measured in elements of `T`; the byte value is repeated across
/// all `count * size_of::<T>()` bytes. The descriptor does not retain the allocation.
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
    /// Capturing this operation stores `dst` in the resulting CUDA graph. The
    /// caller must ensure `dst` is valid for writes of
    /// `count * size_of::<T>()` bytes whenever a captured graph using this
    /// operation is launched, and that it remains exclusive for the work ordered
    /// by those launches.
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
    /// `handle` must identify a live CUDA array. Keep its native owner alive for
    /// every use of this handle or a copy of it, including queued work and graph replays.
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
/// Clones and pool-backed allocations keep an owned pool alive. Default pools
/// are driver-owned and are not destroyed when the last wrapper is dropped.
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
    /// Returns an error if CUDA cannot query attributes for `ptr`, `ptr` is not
    /// known to a unified-addressing context, or CUDA reports runtime
    /// initialization diagnostics.
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
    /// Returns an error if CUDA cannot query attributes for `ptr`, the pointer
    /// is not known to the current CUDA context, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn query_device(ptr: DevicePtr) -> Result<Self> {
        Self::query(ptr.as_ptr())
    }

    /// Returns CUDA pointer attributes for a non-null host or device pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query attributes for `ptr`, the pointer
    /// is not known to a unified-addressing context, or CUDA reports runtime
    /// initialization diagnostics.
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

impl MemoryPool {
    /// Creates a CUDA memory pool.
    /// `props` determines the properties of the pool such as the backing device and IPC capabilities.
    ///
    /// To create a memory pool for host memory not targeting a specific NUMA node, applications must set [`MemoryPoolProps::location`] to [`MemoryLocation::Host`].
    /// [`MemoryLocation`] is ignored for such pools.
    /// Pools created with [`MemoryLocation::Host`] are not IPC-capable and [`MemoryPoolProps::handle_type`] must be [`MemoryAllocationHandleType::None`]; any other value returns [`crate::error::Status::InvalidValue`].
    /// To create a memory pool targeting a specific host NUMA node, applications must set [`MemoryLocation`] to [`MemoryLocation::Numa`] and [`MemoryLocation`] must specify the NUMA ID of the host memory node.
    /// Specifying [`MemoryLocation::NumaCurrent`] as [`MemoryLocation`] returns [`crate::error::Status::InvalidValue`].
    /// By default, the pool's memory is accessible from the device where it is allocated.
    /// Pools created with [`MemoryLocation::Numa`] or [`MemoryLocation::Host`] are accessible from the host CPU by default.
    /// Applications can control the maximum size of the pool by specifying a non-zero value for [`MemoryPoolProps::max_size`].
    /// A value of 0 uses a system-dependent maximum pool size.
    ///
    /// Callers that intend to use [`MemoryAllocationHandleType::Fabric`] based memory sharing must ensure: (1) the `nvidia-caps-imex-channels` character device is created by the driver and is listed under `/proc/devices`; (2) at least one IMEX channel file is accessible to the process.
    ///
    /// When exporter and importer CUDA processes have been granted access to the same IMEX channel, they can securely share memory.
    ///
    /// The IMEX channel security model works per operating-system account.
    /// All processes for an account can share memory if that account has access to a valid IMEX channel.
    /// When isolation between accounts is desired, each account needs a separate IMEX channel.
    ///
    /// These channel files exist in `/dev/nvidia-caps-imex-channels/channel*` and can be created using standard OS native calls like `mknod` on Linux.
    ///
    /// To create a managed memory pool, applications must set [`MemoryPoolProps::alloc_type`] to [`MemoryAllocationType::Managed`].
    /// [`MemoryPoolProps::handle_type`] must also be [`MemoryAllocationHandleType::None`] because IPC is not supported.
    /// For managed memory pools, [`MemoryPoolProps::location`] is treated as the preferred location for all allocations created from the pool.
    /// An application can also set [`MemoryLocation::Unspecified`] to indicate no preferred location.
    /// [`MemoryPoolProps::max_size`] must be set to zero for managed memory pools.
    /// [`MemoryPoolProps::usage`] must be zero because decompression for managed memory is not supported.
    /// For managed memory pools, all devices on the system must have non-zero concurrentManagedAccess.
    /// If not, this call returns [`crate::error::Status::NotSupported`].
    ///
    /// Specifying [`MemoryAllocationHandleType::None`] creates a memory pool that does not support IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `props` describes an unsupported pool, CUDA cannot
    /// create the pool, or CUDA returns a null memory-pool handle.
    pub fn create(context: &Arc<Context>, props: MemoryPoolProps) -> Result<Self> {
        context.bind()?;
        let mut handle = ptr::null_mut();
        let props = driver::CUmemPoolProps::from(props);
        // SAFETY: `cuMemPoolCreate` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(driver::cuMemPoolCreate(&raw mut handle, &raw const props))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(MemoryPoolOwner {
                context: Arc::clone(context),
                handle,
                owned: true,
            }),
        })
    }

    /// Supported attributes are:
    ///
    /// * [`MemoryPoolAttribute::ReleaseThreshold`]: amount of reserved memory, in bytes, to keep before trying to release memory back to the OS.
    ///   When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory
    ///   back to the OS on the next call to stream, event or context synchronize.
    ///   (default 0)
    /// * [`MemoryPoolAttribute::ReuseFollowEventDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to use memory asynchronously freed in another stream as long as a stream ordering dependency of the allocating stream on
    ///   the free action exists.
    ///   CUDA events and null stream interactions can create the required stream ordered dependencies.
    ///   (default
    ///   enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowOpportunistic`]: allows reuse of already completed frees when there is no dependency between the free and allocation.
    ///   (default
    ///   enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowInternalDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to insert new stream dependencies to establish the stream ordering required to reuse a piece of memory released
    ///   by [`sys::cuMemFreeAsync`](singe_cuda_sys::driver::cuMemFreeAsync) (default enabled).
    /// * [`MemoryPoolAttribute::ReservedMemoryHigh`]: resets the high watermark that tracks the amount of backing memory allocated for the memory
    ///   pool.
    ///   It is illegal to set this attribute to a non-zero value.
    /// * [`MemoryPoolAttribute::UsedMemoryHigh`]: resets the high watermark that tracks the amount of used memory allocated for the memory
    ///   pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the value variant does not match the attribute, context binding fails, or CUDA rejects the attribute or value. Current-usage statistics are read-only, and high-water marks can only be reset to zero.
    pub fn set_attribute(
        &mut self,
        attribute: MemoryPoolAttribute,
        value: MemoryPoolAttributeValue,
    ) -> Result<()> {
        self.context().bind()?;
        // SAFETY: `cuMemPoolSetAttribute` receive valid output locations.
        unsafe {
            match (attribute, value) {
                (
                    MemoryPoolAttribute::ReuseFollowEventDependencies
                    | MemoryPoolAttribute::ReuseAllowOpportunistic
                    | MemoryPoolAttribute::ReuseAllowInternalDependencies,
                    MemoryPoolAttributeValue::Bool(value),
                ) => {
                    let mut value = u32::from(value);
                    try_ffi!(driver::cuMemPoolSetAttribute(
                        self.owner.handle,
                        attribute.into(),
                        ptr::from_mut(&mut value).cast(),
                    ))?;
                }
                (
                    MemoryPoolAttribute::ReleaseThreshold
                    | MemoryPoolAttribute::ReservedMemoryCurrent
                    | MemoryPoolAttribute::ReservedMemoryHigh
                    | MemoryPoolAttribute::UsedMemoryCurrent
                    | MemoryPoolAttribute::UsedMemoryHigh,
                    MemoryPoolAttributeValue::Bytes(value),
                ) => {
                    let mut value = value;
                    try_ffi!(driver::cuMemPoolSetAttribute(
                        self.owner.handle,
                        attribute.into(),
                        ptr::from_mut(&mut value).cast(),
                    ))?;
                }
                _ => return Err(Error::InvalidValue),
            }
        }
        Ok(())
    }

    /// Supported attributes are:
    ///
    /// * [`MemoryPoolAttribute::ReleaseThreshold`]: amount of reserved memory, in bytes, to keep before trying to release memory back to the OS.
    ///   When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory
    ///   back to the OS on the next call to stream, event or context synchronize.
    ///   (default 0)
    /// * [`MemoryPoolAttribute::ReuseFollowEventDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to use memory asynchronously freed in another stream as long as a stream ordering dependency of the allocating stream on
    ///   the free action exists.
    ///   CUDA events and null stream interactions can create the required stream ordered dependencies.
    ///   (default
    ///   enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowOpportunistic`]: allows reuse of already completed frees when there is no dependency between the free and allocation.
    ///   (default
    ///   enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowInternalDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to insert new stream dependencies to establish the stream ordering required to reuse a piece of memory released
    ///   by [`sys::cuMemFreeAsync`](singe_cuda_sys::driver::cuMemFreeAsync) (default enabled).
    /// * [`MemoryPoolAttribute::ReservedMemoryCurrent`]: backing memory currently allocated for the memory pool.
    /// * [`MemoryPoolAttribute::ReservedMemoryHigh`]: high watermark of backing memory allocated for the memory pool since the last reset.
    /// * [`MemoryPoolAttribute::UsedMemoryCurrent`]: memory from the pool that is currently in use by the application.
    /// * [`MemoryPoolAttribute::UsedMemoryHigh`]: high watermark of memory from the pool that was in use by the application.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Driver cannot report the requested pool attribute.
    pub fn attribute(&self, attribute: MemoryPoolAttribute) -> Result<MemoryPoolAttributeValue> {
        self.context().bind()?;
        // SAFETY: `cuMemPoolGetAttribute` receive valid output locations.
        unsafe {
            match attribute {
                MemoryPoolAttribute::ReuseFollowEventDependencies
                | MemoryPoolAttribute::ReuseAllowOpportunistic
                | MemoryPoolAttribute::ReuseAllowInternalDependencies => {
                    let mut value = 0u32;
                    try_ffi!(driver::cuMemPoolGetAttribute(
                        self.owner.handle,
                        attribute.into(),
                        ptr::from_mut(&mut value).cast(),
                    ))?;
                    Ok(MemoryPoolAttributeValue::Bool(value != 0))
                }
                MemoryPoolAttribute::ReleaseThreshold
                | MemoryPoolAttribute::ReservedMemoryCurrent
                | MemoryPoolAttribute::ReservedMemoryHigh
                | MemoryPoolAttribute::UsedMemoryCurrent
                | MemoryPoolAttribute::UsedMemoryHigh => {
                    let mut value = 0u64;
                    try_ffi!(driver::cuMemPoolGetAttribute(
                        self.owner.handle,
                        attribute.into(),
                        ptr::from_mut(&mut value).cast(),
                    ))?;
                    Ok(MemoryPoolAttributeValue::Bytes(value))
                }
            }
        }
    }

    /// Controls visibility of pools between devices.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Driver rejects the access descriptors.
    pub fn set_access(&mut self, access_descs: &[MemoryAccessDescriptor]) -> Result<()> {
        self.context().bind()?;
        let access_descs: Vec<_> = access_descs.iter().copied().map(Into::into).collect();
        // SAFETY: `cuMemPoolSetAccess` receives pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(driver::cuMemPoolSetAccess(
                self.owner.handle,
                access_descs.as_ptr(),
                access_descs.len() as _,
            ))?;
        }
        Ok(())
    }

    /// Returns the accessibility of the pool's memory from the specified location.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA Driver cannot report access from `location`.
    pub fn access(&self, location: MemoryLocation) -> Result<MemoryAccessFlag> {
        self.context().bind()?;
        let mut flags = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE;
        let mut location = driver::CUmemLocation_st::from(location);
        // SAFETY: `cuMemPoolGetAccess` receives valid output locations.
        unsafe {
            try_ffi!(driver::cuMemPoolGetAccess(
                &raw mut flags,
                self.owner.handle,
                &raw mut location,
            ))?;
        }
        Ok(flags.into())
    }

    /// Releases memory back to the OS until the pool contains fewer than `min_bytes_to_keep` reserved bytes, or there is no more memory that the allocator can safely release.
    /// The allocator cannot release OS allocations that back outstanding asynchronous allocations.
    /// The OS allocations may happen at different granularity from the caller's allocations.
    ///
    /// * Allocations that have not been freed count as outstanding.
    /// * Allocations that have been asynchronously freed but whose completion has not been observed on the host, for example by synchronization, can count as outstanding.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot trim the pool.
    pub fn trim_to(&mut self, min_bytes_to_keep: usize) -> Result<()> {
        self.context().bind()?;
        // SAFETY: `cuMemPoolTrimTo` receives only plain values.
        unsafe {
            try_ffi!(driver::cuMemPoolTrimTo(
                self.owner.handle,
                min_bytes_to_keep as _
            ))?;
        }
        Ok(())
    }

    /// Returns the raw CUDA memory pool handle.
    #[must_use]
    pub fn as_raw(&self) -> driver::CUmemoryPool {
        self.owner.handle
    }
}

impl MemoryPool {
    /// Returns the CUDA context retained by this allocation or pool.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.owner.context
    }

    /// Takes ownership of a CUDA pool created for this context.
    ///
    /// # Safety
    ///
    /// The handle must identify a live pool that the caller may destroy.
    /// It must not be a driver-owned default pool or have another destroying owner.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool handle is null.
    pub unsafe fn from_raw(context: &Arc<Context>, handle: driver::CUmemoryPool) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(MemoryPoolOwner {
                context: Arc::clone(context),
                handle,
                owned: true,
            }),
        })
    }

    /// Transfers an owned handle only when no allocations or other owners retain it.
    /// Driver-owned pools cannot transfer destruction responsibility.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if the pool is driver-owned or another wrapper or allocation retains it.
    pub fn into_raw(self) -> StdResult<(driver::CUmemoryPool, Arc<Context>), Self> {
        if !self.owner.owned {
            return Err(self);
        }
        match Arc::try_unwrap(self.owner) {
            Ok(mut owner) => {
                owner.owned = false;
                Ok((owner.handle, Arc::clone(&owner.context)))
            }
            Err(owner) => Err(Self { owner }),
        }
    }
}

impl MemoryPool {
    /// Borrows a driver-owned default pool for a location and allocation kind.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA default-pool lookup errors, including unsupported location or allocation-type combinations. Rejects a null returned pool handle.
    pub fn default_for(
        context: &Arc<Context>,
        location: MemoryLocation,
        allocation_type: MemoryAllocationType,
    ) -> Result<Self> {
        context.bind()?;
        let mut handle = ptr::null_mut();
        let mut location = driver::CUmemLocation::from(location);
        // SAFETY: `cuMemGetDefaultMemPool` receives valid output locations.
        unsafe {
            try_ffi!(driver::cuMemGetDefaultMemPool(
                &raw mut handle,
                &raw mut location,
                allocation_type.into()
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(MemoryPoolOwner {
                context: Arc::clone(context),
                handle,
                owned: false,
            }),
        })
    }

    /// Selects this pool for future implicit allocations of its location and kind.
    ///
    /// This changes CUDA's shared pool selection. Destroying the selected owned
    /// pool restores the driver's default pool. Existing allocations retain
    /// their original backing memory regardless of later selection changes.
    ///
    /// # Errors
    ///
    /// Propagates context-binding, pool-property query and conversion, and CUDA pool-selection errors.
    pub fn set_current(&self) -> Result<()> {
        self.context().bind()?;
        let mut location = driver::CUmemLocation::from(self.location()?);
        let allocation_type = self.allocation_type()?;
        // SAFETY: `cuMemSetMemPool` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(driver::cuMemSetMemPool(
                &raw mut location,
                allocation_type.into(),
                self.as_raw(),
            ))
        }
    }
}

impl Context {
    /// Borrows the device's default pool while retaining its CUDA context.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA default-pool lookup errors, and rejects a null returned pool handle.
    pub fn default_memory_pool(self: &Arc<Self>) -> Result<MemoryPool> {
        MemoryPool::default_for(
            self,
            MemoryLocation::Device(self.device().id()),
            MemoryAllocationType::Pinned,
        )
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
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy cuda memory pool: {error}");
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
            #[cfg(debug_assertions)]
            eprintln!("failed to release cuda allocation: {error}");
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
/// Allocations are created with `cudaMallocAsync` or
/// `cudaMallocFromPoolAsync` and are freed with `cudaFreeAsync` on the stored
/// stream.
#[derive(Debug)]
pub struct AsyncDeviceMemory<T> {
    allocation: Arc<DeviceAllocation>,
    ptr: *mut T,
    length: usize,
    _phantom: PhantomData<T>,
}

/// Owned 2D pitched CUDA device memory.
///
/// The logical dimensions are expressed in elements of `T`; the CUDA pitch is
/// stored as a byte stride between rows.
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
/// The logical dimensions are expressed in elements of `T`; the CUDA pitch is
/// stored as a byte stride between rows inside each slice.
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
            #[cfg(debug_assertions)]
            eprintln!("failed to unregister cuda host memory: {error}");
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
            #[cfg(debug_assertions)]
            eprintln!("failed to free cuda pinned allocation: {error}");
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
            #[cfg(debug_assertions)]
            eprintln!("failed to unregister owned cuda host memory: {error}");
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
/// Host access through the guarded callback methods waits for tracked GPU work
/// and excludes conflicting submissions for the duration of the callback.
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
/// Implementors must guarantee the range returned by [`HostSlice::as_host_ptr`]
/// remains valid and registered or allocated in a way CUDA accepts for the
/// lifetime of `&self`.
/// Host metadata must report that range, the actual registration flags, and the identity shared by all mapped aliases.
pub unsafe trait CudaHostSlice<T: DeviceRepr>: HostSlice<T> {
    /// Describes the same bytes as this host slice, preserving identity across mapped aliases.
    fn host_region(&self) -> HostRegion<'_>;
}

/// A page-locked host-memory range suitable for efficient CUDA transfers.
///
/// # Safety
///
/// Implementors must guarantee the range is page-locked for the lifetime of
/// `&self`.
pub unsafe trait PinnedHostSlice<T: DeviceRepr>: CudaHostSlice<T> {}

/// A mutable page-locked host-memory range suitable for efficient CUDA transfers.
///
/// # Safety
///
/// Implementors must uphold both [`PinnedHostSlice`] and [`HostSliceMut`]
/// contracts.
pub unsafe trait PinnedHostSliceMut<T: DeviceRepr>:
    PinnedHostSlice<T> + HostSliceMut<T>
{
}

/// Associated utility functions.
impl<T> DeviceMemory<T> {
    /// Allocates size bytes of linear memory on the device and returns a pointer to the allocated memory.
    /// The allocated memory is suitably aligned for any kind of variable.
    /// The memory is not cleared.
    /// [`DeviceMemory::alloc`] returns [`crate::error::Status::OutOfMemory`] on allocation failure.
    ///
    /// The device version of [`DeviceMemory::free`] cannot be used with a pointer allocated using the host API, and vice versa.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot
    /// allocate device memory, a previous asynchronous launch reports an error,
    /// or CUDA reports runtime initialization diagnostics such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`],
    /// or [`crate::error::Status::NoDevice`].
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory. The caller must use
    /// it only for `count` elements of `T` and eventually free it with a
    /// compatible CUDA free function.
    pub unsafe fn alloc(count: usize) -> Result<*mut T> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }
        let mut p = ptr::null_mut();
        // SAFETY: `cudaMalloc` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMalloc(&raw mut p, bytes as _))?;
        }
        Ok(p.cast())
    }

    /// Allocates CUDA managed memory for `count` elements.
    ///
    /// `flags` controls the initial stream attachment behavior for the managed
    /// allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot
    /// allocate managed memory, or `count` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// The returned pointer is managed CUDA memory. The caller must eventually
    /// free it with a CUDA-compatible device free and must synchronize host and
    /// device access according to CUDA managed-memory rules.
    pub unsafe fn alloc_managed(count: usize, flags: MemoryAttachFlags) -> Result<*mut T> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }
        let mut p = ptr::null_mut();
        // SAFETY: `cudaMallocManaged` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMallocManaged(
                &raw mut p,
                bytes as _,
                flags.bits(),
            ))?;
        }
        Ok(p.cast::<T>())
    }

    /// Frees the memory space pointed to by `ptr`, which must have been returned by a previous call to one of these allocation functions: [`DeviceMemory::alloc`], [`sys::cudaMallocPitch`](singe_cuda_sys::runtime::cudaMallocPitch), [`DeviceMemory::alloc_managed`], [`DeviceMemory::alloc_async`], or [`sys::cudaMallocFromPoolAsync`](singe_cuda_sys::runtime::cudaMallocFromPoolAsync).
    ///
    /// This does not perform implicit synchronization when the pointer was allocated with [`DeviceMemory::alloc_async`] or [`sys::cudaMallocFromPoolAsync`](singe_cuda_sys::runtime::cudaMallocFromPoolAsync).
    /// Callers must ensure that all accesses to this pointer have completed before invoking [`DeviceMemory::free`].
    /// For best performance and memory reuse, use [`DeviceMemory::free_async`] to free memory allocated via the stream ordered memory allocator.
    /// For all other pointers, this call may perform implicit synchronization.
    ///
    /// If [`DeviceMemory::free`] has already been called before, an error is returned.
    /// If `ptr` is null, no operation is performed.
    /// [`DeviceMemory::free`] returns an error on failure.
    ///
    /// The device version of [`DeviceMemory::free`] cannot be used with a pointer allocated using the host API, and vice versa.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot free `ptr`, `ptr` has already been
    /// freed, a previous asynchronous launch reports an error, or CUDA reports
    /// runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `ptr` must be null or a live allocation returned by a compatible CUDA
    /// device allocation function, and no work may access it after it is freed.
    /// No live allocation owner may subsequently free the pointer again.
    pub unsafe fn free(ptr: *mut T) -> Result<()> {
        // SAFETY: `cudaFree` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaFree(ptr.cast()))?;
        }
        Ok(())
    }

    /// Allocates 2D pitched device memory and returns the base pointer and row pitch in bytes.
    ///
    /// `width` is measured in elements of `T`, while the returned pitch is
    /// measured in bytes. CUDA may pad each row so that row-to-row addressing
    /// satisfies device alignment requirements.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, `T` is a
    /// non-empty zero-sized allocation, CUDA cannot allocate pitched memory, or
    /// CUDA returns a null allocation for a non-empty request.
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory. The caller must use
    /// it only as a `width` by `height` pitched allocation with the returned
    /// pitch and eventually free it with a CUDA-compatible device free.
    pub unsafe fn alloc_pitched(width: usize, height: usize) -> Result<(*mut T, usize)> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok((ptr::null_mut(), row_bytes));
        }

        let mut ptr = ptr::null_mut();
        let mut pitch = 0;
        // SAFETY: `cudaMallocPitch` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMallocPitch(
                &raw mut ptr,
                &raw mut pitch,
                row_bytes as _,
                height as _,
            ))?;
        }
        if ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok((ptr.cast(), to_usize(pitch, "allocation pitch")?))
    }

    /// Allocates 3D pitched device memory and returns CUDA's pitched pointer descriptor.
    ///
    /// `width` is measured in elements of `T`. CUDA's `cudaPitchedPtr::pitch`
    /// and `xsize` fields are measured in bytes for linear memory, and `ysize`
    /// records the logical height.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, `T` is a
    /// non-empty zero-sized allocation, CUDA cannot allocate pitched memory, or
    /// CUDA returns a null allocation for a non-empty request.
    ///
    /// # Safety
    ///
    /// The returned allocation is uninitialized device memory. The caller must
    /// treat it as a `width` by `height` by `depth` pitched allocation and free
    /// the base pointer with a CUDA-compatible device free.
    pub unsafe fn alloc_pitched_3d(
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<runtime::cudaPitchedPtr> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(runtime::cudaPitchedPtr {
                ptr: ptr::null_mut(),
                pitch: row_bytes as _,
                xsize: row_bytes as _,
                ysize: height as _,
            });
        }

        let mut pitched = runtime::cudaPitchedPtr::default();
        // SAFETY: `cudaMalloc3D` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMalloc3D(
                &raw mut pitched,
                runtime::cudaExtent {
                    width: row_bytes as _,
                    height: height as _,
                    depth: depth as _,
                },
            ))?;
        }
        if pitched.ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(pitched)
    }

    /// Copies `count` elements from `src` to `dst`.
    /// The transfer direction is specified by [`MemoryCopyKind`].
    /// [`MemoryCopyKind::Default`] is recommended when unified virtual addressing is available, in which case the transfer direction is inferred from the pointer values.
    /// Calling [`DeviceMemory::copy`] with `dst` and `src` pointers that do not match the direction of the copy results in undefined behavior.
    ///
    /// * Exhibits `synchronous` behavior for most use cases.
    /// * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered
    ///   at all.
    ///   Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and
    ///   return [`crate::error::Status::InvalidValue`].
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA rejects the
    /// pointer combination or copy kind, a previous asynchronous launch reports
    /// an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must be valid for `count` elements of `T` according to
    /// `kind`, and the source and destination regions must not overlap unless
    /// CUDA permits that transfer.
    pub unsafe fn copy(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `cudaMemcpy` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemcpy(
                dst.cast(),
                src.cast(),
                bytes as _,
                kind.into(),
            ))?;
        }
        Ok(())
    }

    /// Copies a typed 2D rectangle between pitched host or device memory regions.
    ///
    /// `width` and `height` are measured in elements and rows. `dst_pitch` and
    /// `src_pitch` are byte strides between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, either pitch
    /// is smaller than the copied row width, CUDA rejects the pointer
    /// combination or copy kind, or a previous asynchronous launch reports an
    /// error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be valid for the described rectangle according to
    /// `kind`. The source and destination regions must not overlap unless CUDA
    /// permits that transfer.
    pub unsafe fn copy_2d(
        dst: *mut T,
        dst_pitch: usize,
        src: *const T,
        src_pitch: usize,
        width: usize,
        height: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }
        if row_bytes > dst_pitch || row_bytes > src_pitch {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: `cudaMemcpy2D` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemcpy2D(
                dst.cast(),
                dst_pitch as _,
                src.cast(),
                src_pitch as _,
                row_bytes as _,
                height as _,
                kind.into(),
            ))?;
        }
        Ok(())
    }

    /// Enqueues a typed 2D rectangle copy on `stream`.
    ///
    /// `width` and `height` are measured in elements and rows. Pitches are byte
    /// strides between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, either pitch
    /// is smaller than the copied row width, CUDA cannot enqueue the copy, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for the described rectangle until
    /// `stream` reaches the enqueued copy. The pointer combination must match
    /// `kind`.
    pub unsafe fn copy_2d_async(
        dst: *mut T,
        dst_pitch: usize,
        src: *const T,
        src_pitch: usize,
        width: usize,
        height: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }
        if row_bytes > dst_pitch || row_bytes > src_pitch {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: `cudaMemcpy2DAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemcpy2DAsync(
                dst.cast(),
                dst_pitch as _,
                src.cast(),
                src_pitch as _,
                row_bytes as _,
                height as _,
                kind.into(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Copies a typed 3D extent between CUDA pitched pointer operands.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and
    /// slices. CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, either pitch is
    /// smaller than the copied row width, CUDA rejects the copy parameters, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must describe live host or device memory large enough
    /// for the requested extent according to `kind`. The source and destination
    /// regions must not overlap unless CUDA permits that transfer.
    pub unsafe fn copy_3d(
        dst: runtime::cudaPitchedPtr,
        src: runtime::cudaPitchedPtr,
        width: usize,
        height: usize,
        depth: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }
        if row_bytes > to_usize(dst.pitch, "destination pitch")?
            || row_bytes > to_usize(src.pitch, "source pitch")?
        {
            return Err(Error::InvalidMemoryAccess);
        }
        let params = runtime::cudaMemcpy3DParms {
            srcArray: ptr::null_mut(),
            srcPos: runtime::cudaPos::default(),
            srcPtr: src,
            dstArray: ptr::null_mut(),
            dstPos: runtime::cudaPos::default(),
            dstPtr: dst,
            extent: runtime::cudaExtent {
                width: row_bytes as _,
                height: height as _,
                depth: depth as _,
            },
            kind: kind.into(),
        };
        // SAFETY: `cudaMemcpy3D` receives pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemcpy3D(&raw const params))?;
        }
        Ok(())
    }

    /// Enqueues a typed 3D extent copy on `stream`.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and
    /// slices. CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, either pitch is
    /// smaller than the copied row width, CUDA cannot enqueue the copy, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for the described extent until
    /// `stream` reaches the enqueued copy. The pointer combination must match
    /// `kind`.
    pub unsafe fn copy_3d_async(
        dst: runtime::cudaPitchedPtr,
        src: runtime::cudaPitchedPtr,
        width: usize,
        height: usize,
        depth: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }
        if row_bytes > to_usize(dst.pitch, "destination pitch")?
            || row_bytes > to_usize(src.pitch, "source pitch")?
        {
            return Err(Error::InvalidMemoryAccess);
        }
        let params = runtime::cudaMemcpy3DParms {
            srcArray: ptr::null_mut(),
            srcPos: runtime::cudaPos::default(),
            srcPtr: src,
            dstArray: ptr::null_mut(),
            dstPos: runtime::cudaPos::default(),
            dstPtr: dst,
            extent: runtime::cudaExtent {
                width: row_bytes as _,
                height: height as _,
                depth: depth as _,
            },
            kind: kind.into(),
        };
        // SAFETY: `cudaMemcpy3DAsync` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemcpy3DAsync(
                &raw const params,
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Fills `count` elements at `dst` with the constant byte pattern `value`.
    ///
    /// This call is asynchronous with respect to the host unless `ptr` refers to pinned host memory.
    ///
    /// See the CUDA memset synchronization rules for when this operation blocks
    /// the host.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA rejects the
    /// pointer or size, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `count * size_of::<T>()` bytes and
    /// must refer to memory that CUDA can memset.
    pub unsafe fn set(dst: *mut T, value: u8, count: usize) -> Result<()> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `cudaMemset` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemset(dst.cast(), value.into(), bytes as _))?;
        }
        Ok(())
    }

    /// Fills a typed 2D pitched memory region with a byte value.
    ///
    /// `width` and `height` are measured in elements and rows. `pitch` is the
    /// byte stride between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, `pitch` is
    /// smaller than the row width, CUDA rejects the memory region, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes over the described pitched rectangle and
    /// must refer to memory CUDA can memset.
    pub unsafe fn set_2d(
        dst: *mut T,
        pitch: usize,
        value: u8,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }
        if row_bytes > pitch {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: `cudaMemset2D` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemset2D(
                dst.cast(),
                pitch as _,
                value.into(),
                row_bytes as _,
                height as _,
            ))?;
        }
        Ok(())
    }

    /// Fills a typed 3D pitched memory region with a byte value.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and
    /// slices. CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, `dst.pitch` is
    /// smaller than the row width, CUDA rejects the memory region, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must describe live memory large enough for the requested extent
    /// and must refer to memory CUDA can memset.
    pub unsafe fn set_3d(
        dst: runtime::cudaPitchedPtr,
        value: u8,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }
        if row_bytes > to_usize(dst.pitch, "destination pitch")? {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: `cudaMemset3D` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemset3D(
                dst,
                value.into(),
                runtime::cudaExtent {
                    width: row_bytes as _,
                    height: height as _,
                    depth: depth as _,
                },
            ))?;
        }
        Ok(())
    }

    /// Returns the total amount of memory available to the current context and the amount of memory free on the device.
    /// CUDA is not guaranteed to be able to allocate all of the memory that the OS reports as free.
    /// In a multi-tenant situation, the free-memory estimate is prone to a race condition: an allocation or free by another process or thread between estimation and reporting can make the reported free value differ from actual free memory.
    ///
    /// The integrated GPU on Tegra shares memory with CPU and other component of the `SoC`.
    /// The free and total values returned by this call exclude the SWAP memory space maintained by the OS on some platforms.
    /// The OS may move some of the memory pages into swap area as the GPU or CPU allocate or access memory.
    /// See Tegra app note on how to calculate total and free memory on Tegra.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query memory information, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn memory_info() -> Result<(usize, usize)> {
        let mut free: runtime::size_t = 0;
        let mut total: runtime::size_t = 0;
        // SAFETY: `cudaMemGetInfo` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMemGetInfo(&raw mut free, &raw mut total))?;
        }
        Ok((
            to_usize(free, "free device memory")?,
            to_usize(total, "total device memory")?,
        ))
    }

    /// Returns the attributes of `ptr`.
    /// If `ptr` was not allocated in, mapped by, or registered with a context that supports unified addressing, [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// In CUDA 11.0 and later, passing a host pointer reports [`MemoryType::Unregistered`] in [`PointerAttributes::memory_type`].
    ///
    /// * [`PointerAttributes::memory_type`] identifies the type of memory.
    ///   It can be [`MemoryType::Unregistered`] for unregistered host memory, [`MemoryType::Host`] for registered host memory, [`MemoryType::Device`] for device memory, or [`MemoryType::Managed`] for managed memory.
    ///
    /// * [`PointerAttributes::device`] is the device against which `ptr` was allocated.
    ///   If `ptr` has memory type [`MemoryType::Device`], this identifies the device on which the memory physically resides.
    ///   If `ptr` has memory type [`MemoryType::Host`], this identifies the device that was current when the allocation was made, and if that device is deinitialized then
    ///   this allocation will vanish with that device's state.
    ///
    /// * [`PointerAttributes::device_pointer`] is the device pointer alias through which the memory referred to by `ptr` may be accessed on the current device.
    ///   If the memory referred to by `ptr` cannot be accessed directly by the current device then this is null.
    ///
    /// * [`PointerAttributes::host_pointer`] is the host pointer alias through which the memory referred to by `ptr` may be accessed on the host.
    ///   If the memory referred to by `ptr` cannot be accessed directly by the host then this is null.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query attributes for `ptr`, `ptr` is not
    /// known to a unified-addressing context, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn pointer_attributes(ptr: *const T) -> Result<PointerAttributes> {
        PointerAttributes::query(ptr)
    }

    /// Enqueues a stream-ordered device allocation on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot
    /// enqueue `cudaMallocAsync`, or `count` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory whose lifetime is
    /// ordered by `stream`. The caller must ensure all accesses are stream
    /// ordered correctly and free it with [`DeviceMemory::free_async`] or a
    /// compatible CUDA free after all accesses complete.
    pub unsafe fn alloc_async(count: usize, stream: &Stream) -> Result<*mut T> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }
        let mut p = ptr::null_mut();
        // SAFETY: `cudaMallocAsync` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMallocAsync(
                &raw mut p,
                bytes as _,
                stream.as_raw()
            ))?;
        }
        Ok(p.cast::<T>())
    }

    /// Inserts a free operation into `stream`.
    /// The allocation must not be accessed after stream execution reaches the free.
    /// After this call returns, accessing the memory from any subsequent work launched on the GPU or querying its pointer attributes results in undefined behavior.
    ///
    /// During stream capture, this creates a free node and must therefore be passed the address of a graph allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the free on `stream`, `ptr` is
    /// invalid for asynchronous freeing, a previous asynchronous launch reports
    /// an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `ptr` must be null or a live stream-ordered CUDA allocation. No work may
    /// access it after `stream` reaches the enqueued free.
    pub unsafe fn free_async(ptr: *mut T, stream: &Stream) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // SAFETY: `cudaFreeAsync` receives live handles owned by the wrappers.
        unsafe { try_ffi!(runtime::cudaFreeAsync(ptr.cast(), stream.as_raw())) }
    }

    /// Enqueues a typed one-dimensional memory copy on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA cannot
    /// enqueue the copy, the pointer combination is invalid for `kind`, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for `count` elements until `stream`
    /// reaches the enqueued copy. The pointer combination must match `kind`.
    pub unsafe fn copy_async(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the borrowed stream shares the handle and context; pointer contracts are unchanged.
        unsafe {
            Self::copy_async_on(
                dst,
                src,
                count,
                kind,
                &StreamBinding::Borrowed(stream.to_borrowed()),
            )
        }
    }

    /// Enqueues a typed copy on an explicit, externally borrowed or default stream.
    ///
    /// # Safety
    /// `dst` and `src` must remain valid for `count` elements until the stream reaches
    /// the copy. The pointer combination must match `kind` and the stream's context.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized copy is requested, context binding fails, or CUDA rejects the asynchronous copy.
    pub unsafe fn copy_async_on(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
        stream: &StreamBinding,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }
        stream.context().bind()?;
        // SAFETY: `cudaMemcpyAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemcpyAsync(
                dst.cast(),
                src.cast(),
                bytes as _,
                kind.into(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Fills the first `count` bytes of the memory area pointed to by `ptr` with the constant byte `value`.
    ///
    /// [`DeviceMemory::set_async`] is asynchronous with respect to the host, so the call may return before the memset is complete.
    /// The operation can optionally be associated with a stream by passing a non-zero stream argument.
    /// If `stream` is non-zero, the operation may overlap with operations in other streams.
    ///
    /// The device version only handles device-to-device copies and cannot be given local or shared pointers.
    ///
    /// See the CUDA memset synchronization rules for when this operation blocks
    /// the host.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA cannot
    /// enqueue the memset on `stream`, a previous asynchronous launch reports an
    /// error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `count * size_of::<T>()` bytes until
    /// `stream` reaches the enqueued memset.
    pub unsafe fn set_async(dst: *mut T, value: u8, count: usize, stream: &Stream) -> Result<()> {
        // SAFETY: the borrowed binding retains this stream's context and handle; pointer contracts are unchanged.
        unsafe {
            Self::set_async_on(
                dst,
                value,
                count,
                &StreamBinding::Borrowed(stream.to_borrowed()),
            )
        }
    }

    /// Fills `count` elements with a repeated byte on an explicit, borrowed or default stream.
    /// Supports ordinary enqueue and stream capture without allocating a host source.
    ///
    /// # Errors
    /// Returns an error for byte-count overflow, context binding failure or rejected native enqueue.
    ///
    /// # Safety
    /// `dst` must remain valid and exclusive for `count * size_of::<T>()` writable bytes
    /// until completion and during every replay of a captured fill.
    pub unsafe fn set_async_on(
        dst: *mut T,
        value: u8,
        count: usize,
        stream: &StreamBinding,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }
        stream.context().bind()?;
        // SAFETY: `cudaMemsetAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemsetAsync(
                dst.cast(),
                value.into(),
                bytes as _,
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Enqueues a 2D pitched memset on `stream`.
    ///
    /// `width` and `height` are measured in elements and rows. `pitch` is the
    /// byte stride between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, `pitch` is
    /// smaller than the row width, CUDA cannot enqueue the memset, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must remain valid for writes over the described pitched rectangle
    /// until `stream` reaches the enqueued memset.
    pub unsafe fn set_2d_async(
        dst: *mut T,
        pitch: usize,
        value: u8,
        width: usize,
        height: usize,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }
        if row_bytes > pitch {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: `cudaMemset2DAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemset2DAsync(
                dst.cast(),
                pitch as _,
                value.into(),
                row_bytes as _,
                height as _,
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Enqueues a 3D pitched memset on `stream`.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and
    /// slices. CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, `dst.pitch` is
    /// smaller than the row width, CUDA cannot enqueue the memset, or a
    /// previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must remain valid for the described extent until `stream` reaches
    /// the enqueued memset.
    pub unsafe fn set_3d_async(
        dst: runtime::cudaPitchedPtr,
        value: u8,
        width: usize,
        height: usize,
        depth: usize,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }
        if row_bytes > to_usize(dst.pitch, "destination pitch")? {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: `cudaMemset3DAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemset3DAsync(
                dst,
                value.into(),
                runtime::cudaExtent {
                    width: row_bytes as _,
                    height: height as _,
                    depth: depth as _,
                },
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Prefetches memory to the specified destination location.
    /// `ptr` is the base device pointer of the memory to be prefetched, `location` specifies the destination location, `count` specifies the number of bytes to copy, and `stream` is the stream in which the operation is enqueued.
    /// The memory range must refer to managed memory allocated via [`DeviceMemory::alloc_managed`] or declared via `__managed__` variables. It may also refer to memory allocated from a managed memory pool, or to system-allocated memory on systems where [`DeviceProperties::pageable_memory_access`](crate::device::DeviceProperties::pageable_memory_access) is enabled.
    ///
    /// Using [`MemoryLocation::Device`] prefetches memory to the GPU identified by its device ID. That device, and the device associated with `stream`, must support concurrent managed access.
    /// Using [`MemoryLocation::Host`] prefetches data to host memory.
    /// Applications can request prefetching memory to a specific host NUMA node by using [`MemoryLocation::Numa`] with a valid NUMA node identifier, or to the NUMA node closest to the current thread's CPU by using [`MemoryLocation::NumaCurrent`].
    ///
    /// The start and end addresses of the memory range are rounded down and up, respectively, to CPU page-size alignment before the prefetch operation is enqueued in the stream.
    ///
    /// If no physical memory has been allocated for this region, CUDA populates and maps it on the destination device.
    /// If there is insufficient memory to prefetch the desired region, the Unified Memory driver may evict pages from other [`DeviceMemory::alloc_managed`] allocations to host memory to make room.
    /// Device memory allocated using [`DeviceMemory::alloc`] or [`sys::cudaMallocArray`](singe_cuda_sys::runtime::cudaMallocArray) is not evicted.
    ///
    /// By default, mappings to the previous location of the migrated pages are removed and mappings for the new location are only set up at the destination.
    /// The exact behavior also depends on the settings applied to this memory range via `cuMemAdvise` as described below:
    ///
    /// If read-mostly advice was set on any subset of this memory range, then that subset will create a read-only copy of the pages at the destination location.
    /// If the destination location is a host NUMA node, any pages of that subset that are already in another host NUMA node are transferred to the destination.
    ///
    /// If preferred-location advice was set on any subset of this memory range, then the pages will migrate to `location` even if it is not the preferred location of every page in the range.
    ///
    /// If accessed-by advice was set on any subset of this memory range, then mappings to those pages from all appropriate processors are updated to refer to the new location if establishing such a mapping is possible.
    /// Otherwise, those mappings are cleared.
    ///
    /// This is not required for correctness; it improves performance by allowing the application to migrate data to a suitable location before access.
    /// Memory accesses to this range are always coherent and are allowed even when the data is actively being migrated.
    ///
    /// This call is asynchronous with respect to the host and all work on other devices.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the prefetch on `stream`, the
    /// memory range or destination location is invalid, a previous asynchronous
    /// launch reports an error, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn prefetch_async(
        ptr: DevicePtr,
        count: usize,
        location: MemoryLocation,
        stream: &Stream,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        // SAFETY: `cudaMemPrefetchAsync` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemPrefetchAsync(
                ptr.as_ptr() as _,
                count as _,
                location.into(),
                0, // flags
                stream.as_raw()
            ))?;
        }
        Ok(())
    }

    /// Applies a managed-memory advice policy to a byte range.
    ///
    /// The memory range must refer to managed memory, a managed memory-pool
    /// allocation, or pageable memory on systems that support pageable memory
    /// access.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice, memory range, or location,
    /// or if a previous asynchronous launch reports an error.
    pub fn advise(
        ptr: DevicePtr,
        count: usize,
        advice: MemoryAdvise,
        location: MemoryLocation,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        // SAFETY: `cudaMemAdvise` receives pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemAdvise(
                ptr.as_ptr().cast_const().cast(),
                count as _,
                advice.into(),
                location.into(),
            ))?;
        }
        Ok(())
    }
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

impl<T: DeviceRepr> PinnedHostMemory<T> {
    /// Completes tracked device accesses and excludes submitted work while reading host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        let context = self.context();
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::Read)])?;
        Ok(read(self.as_slice()))
    }

    /// Completes tracked device accesses and excludes submitted work while modifying host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice_mut<R>(&mut self, write: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        let context = Arc::clone(self.context());
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::ReadWrite)])?;
        Ok(write(self.as_slice_mut()))
    }

    fn allocate(context: &Arc<Context>, length: usize, flags: HostAllocationFlags) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else if flags.is_empty() || flags == HostAllocationFlags::DEFAULT {
            // SAFETY: `bytes` is non-zero, and the returned allocation is owned by the new value.
            unsafe { alloc_host_raw(bytes)? }
        } else {
            // SAFETY: `bytes` is non-zero, and the returned allocation is owned by the new value.
            unsafe { alloc_pinned_raw(bytes, flags)? }
        };
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }
        let allocation = Arc::new(PinnedAllocation {
            context: Arc::clone(context),
            id: AllocationId::create(),
            ptr: ptr.cast(),
            bytes,
            flags,
        });
        if bytes != 0 && !(ptr as usize).is_multiple_of(align_of::<T>()) {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(Self {
            allocation,
            ptr: dangling_or_non_null(ptr.cast(), length)?,
            length,
        })
    }

    /// Allocates pinned host memory and copies `values` into it.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation size overflows, CUDA cannot allocate
    /// page-locked host memory, or `values` is a non-empty zero-sized element
    /// slice.
    pub fn from_slice(context: &Arc<Context>, values: &[T]) -> Result<Self> {
        Self::from_slice_with_flags(context, values, HostAllocationFlags::DEFAULT)
    }

    /// Allocates pinned host memory with `flags` and copies `values` into it.
    ///
    /// `HostAllocationFlags::DEFAULT` uses the simple `cudaMallocHost`
    /// allocation path. Other flags use `cudaHostAlloc`, allowing callers to
    /// request portable, mapped, or write-combined host memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation size overflows, CUDA cannot allocate
    /// page-locked host memory with the requested flags, or `values` is a
    /// non-empty zero-sized element slice.
    pub fn from_slice_with_flags(
        context: &Arc<Context>,
        values: &[T],
        flags: HostAllocationFlags,
    ) -> Result<Self> {
        let memory = Self::allocate(context, values.len(), flags)?;
        if !values.is_empty() {
            // SAFETY: the fresh allocation holds `values.len()` elements and cannot overlap `values`.
            unsafe {
                ptr::copy_nonoverlapping(values.as_ptr(), memory.ptr.as_ptr(), values.len());
            }
        }
        Ok(memory)
    }

    /// Copies a device range into newly allocated pinned host storage.
    ///
    /// Completes the transfer before exposing initialized host values.
    ///
    /// # Errors
    ///
    /// Propagates pinned allocation, byte-size, context-binding, readiness synchronization, stream-capture, CUDA copy, and stream synchronization errors. No initialized allocation is returned unless the transfer completes.
    pub fn from_device<B: DeviceSlice<T> + ?Sized>(source: &B) -> Result<Self>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let region = source.region();
        let mut memory =
            Self::create_uninit(region.context(), source.len(), HostAllocationFlags::DEFAULT)?;
        region.context().bind()?;
        // SAFETY: the provider contains initialized values and CUDA completes the entire destination before promotion.
        unsafe {
            transfer::copy_synchronous(
                memory.as_mut_ptr().cast::<T>(),
                source.as_device_ptr(),
                source.len(),
                MemoryCopyKind::DeviceToHost,
                region.stream(),
            )?;
            Ok(memory.assume_init())
        }
    }

    /// Takes ownership of an existing CUDA pinned host allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live host allocation returned by `cudaMallocHost` or
    /// `cudaHostAlloc` for `length` initialized elements of `T`, and no other
    /// owner may free it while the returned value is alive.
    /// A null pointer is accepted for an empty allocation.
    /// Nonempty pointers must satisfy `T`'s alignment.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, a nonempty null pointer, or a pointer not aligned for `T`.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        length: usize,
        flags: HostAllocationFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }
        if bytes != 0 && !(ptr as usize).is_multiple_of(align_of::<T>()) {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(Self {
            allocation: Arc::new(PinnedAllocation {
                context: Arc::clone(context),
                id: AllocationId::create(),
                ptr: ptr.cast(),
                bytes,
                flags,
            }),
            ptr: dangling_or_non_null(ptr, length)?,
            length,
        })
    }

    /// Transfers the raw allocation when no other owner retains it.
    ///
    /// Returns the unchanged buffer if a submission or interop user retains the allocation.
    /// A non-null returned pointer must eventually be freed with `cudaFreeHost`.
    /// Empty allocations return null rather than a typed dangling sentinel.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(
        self,
    ) -> StdResult<(*mut T, usize, HostAllocationFlags, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            length,
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                let pointer = allocation.ptr.cast();
                allocation.ptr = ptr::null_mut();
                Ok((pointer, length, allocation.flags, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                length,
            }),
        }
    }

    /// Returns the CUDA context retained by this allocation or pool.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the owner retained by asynchronous transfers and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<PinnedAllocation> {
        &self.allocation
    }

    /// Reinterprets initialized host bytes without changing allocation ownership.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    ///
    /// # Panics
    ///
    /// Panics if the exposed byte length overflows `usize`. Construction and slicing preserve a validated byte length, so this cannot occur for a valid allocation.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<PinnedHostMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let bytes = self.byte_len().expect("pinned byte length overflow");
        if size == 0
            || !bytes.is_multiple_of(size)
            || (bytes != 0 && !self.as_ptr().addr().is_multiple_of(align_of::<U>()))
        {
            return Err(self);
        }
        Ok(PinnedHostMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: bytes / size,
        })
    }

    /// Erases initialization while preserving the pinned allocation and its flags.
    #[must_use]
    pub fn into_uninit(self) -> PinnedHostMemory<MaybeUninit<T>> {
        PinnedHostMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }

    /// Allocates pinned storage without initializing its elements.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow or a nonempty zero-sized allocation. Also propagates context-binding and CUDA host allocation errors and rejects null or insufficiently aligned storage for a nonempty allocation.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
        flags: HostAllocationFlags,
    ) -> Result<PinnedHostMemory<MaybeUninit<T>>> {
        PinnedHostMemory::<MaybeUninit<T>>::allocate(context, length, flags)
    }

    /// Allocates a new pinned buffer and copies this buffer's contents into it.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate the clone or if the byte size
    /// overflows.
    pub fn try_clone(&self) -> Result<Self> {
        Self::from_slice_with_flags(self.context(), self.as_slice(), self.flags())
    }

    /// Returns this pinned allocation as an initialized host slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns this pinned allocation as a mutable initialized host slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.is_empty() {
            return &mut [];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the host pointer for this pinned allocation.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the mutable host pointer for this pinned allocation.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the number of elements in this allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this allocation contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the allocation size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a
    /// non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags requested when this allocation was created.
    #[must_use]
    pub fn flags(&self) -> HostAllocationFlags {
        self.allocation.flags()
    }

    /// Queries the flags CUDA recorded for this host allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the allocation flags or reports
    /// runtime initialization diagnostics.
    pub fn host_flags(&self) -> Result<HostAllocationFlags> {
        self.context().bind()?;
        if self.is_empty() || self.flags() == HostAllocationFlags::DEFAULT {
            return Ok(self.flags());
        }
        let mut flags = 0;
        // SAFETY: `cudaHostGetFlags` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaHostGetFlags(
                &raw mut flags,
                self.ptr.as_ptr().cast(),
            ))?;
        }
        Ok(HostAllocationFlags::from_bits_retain(flags))
    }

    /// Returns the CUDA device pointer alias for this mapped host allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation is not mapped into the current CUDA
    /// context, the device cannot map host memory, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.allocation.mapped_device_ptr()
    }

    /// Returns a device view over this allocation's mapped device alias.
    ///
    /// Host and device access through the mapped alias still require normal
    /// CUDA stream synchronization by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot provide a mapped device pointer.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        let region = self
            .allocation
            .mapped_device_region()?
            .slice(..self.byte_len()?)?;
        // SAFETY: the mapped range borrows this initialized host storage.
        unsafe { DeviceView::from_raw_parts(region, self.length) }
    }

    /// Returns a mutable device view over this allocation's mapped device alias.
    ///
    /// Host and device access through the mapped alias still require normal
    /// CUDA stream synchronization by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot provide a mapped device pointer.
    pub fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        let region = self
            .allocation
            .mapped_device_region()?
            .slice(..self.byte_len()?)?;
        // SAFETY: the mapped range exclusively borrows this initialized host storage.
        unsafe { DeviceViewMut::from_raw_parts(region, self.length) }
    }
}

impl<T: DeviceRepr> PinnedHostMemory<MaybeUninit<T>> {
    /// Exposes a completed producer's initialized prefix while retaining the original capacity.
    ///
    /// # Safety
    ///
    /// Every prefix element must be valid and all host and device producers must have completed.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<PinnedHostMemory<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks every element initialized after its producer has completed.
    ///
    /// # Safety
    ///
    /// All elements must be initialized to valid `T` values, with no writes in flight.
    #[must_use]
    pub unsafe fn assume_init(self) -> PinnedHostMemory<T> {
        let () = T::LAYOUT;
        PinnedHostMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }
}

impl<T: ZeroableDeviceRepr> PinnedHostMemory<T> {
    /// Allocates pinned host memory initialized with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow or a nonempty zero-sized allocation. Also propagates context-binding and CUDA host allocation errors and rejects null or insufficiently aligned storage for a nonempty allocation.
    pub fn zeroed(
        context: &Arc<Context>,
        length: usize,
        flags: HostAllocationFlags,
    ) -> Result<Self> {
        let memory = Self::allocate(context, length, flags)?;
        if !memory.is_empty() {
            // SAFETY: the allocation is writable and the zero pattern is valid for T.
            unsafe {
                ptr::write_bytes(memory.ptr.as_ptr(), 0, memory.length);
            }
        }
        Ok(memory)
    }
}

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

impl<'a, T: DeviceRepr> RegisteredHostMemory<'a, T> {
    /// Completes tracked device accesses and excludes submitted work while reading host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        let context = self.context();
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::Read)])?;
        Ok(read(self.as_slice()))
    }

    /// Completes tracked device accesses and excludes submitted work while modifying host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice_mut<R>(&mut self, write: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        let context = Arc::clone(self.context());
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::ReadWrite)])?;
        Ok(write(self.as_slice_mut()))
    }

    /// Registers a mutable host slice with CUDA for this wrapper's lifetime.
    ///
    /// Empty slices retain the context without registering a host range.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte length overflows, CUDA cannot
    /// register the host range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create(context: &Arc<Context>, slice: &'a mut [T]) -> Result<Self> {
        Self::create_with_flags(context, slice, HostRegisterFlags::DEFAULT)
    }

    /// Registers a mutable host slice with CUDA using `flags`.
    ///
    /// Safe mutable registration rejects [`HostRegisterFlags::READ_ONLY`] and
    /// [`HostRegisterFlags::IO_MEMORY`]. Use
    /// [`RegisteredHostMemory::create_io_memory`] for MMIO ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` are incompatible with mutable safe
    /// registration, the slice byte length overflows, CUDA cannot register the
    /// range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create_with_flags(
        context: &Arc<Context>,
        slice: &'a mut [T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        if flags.contains(HostRegisterFlags::READ_ONLY)
            || flags.contains(HostRegisterFlags::IO_MEMORY)
        {
            return Err(Error::InvalidValue);
        }
        // SAFETY: the registration borrows `slice` for its lifetime, so the pages stay valid until unregistration.
        unsafe { Self::register_slice(context, slice.as_mut_ptr(), slice.len(), flags) }
    }

    /// Registers memory-mapped I/O host memory.
    ///
    /// # Safety
    ///
    /// `slice` must represent an MMIO range satisfying CUDA's
    /// `cudaHostRegisterIoMemory` requirements.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow or a nonempty zero-sized range. Also propagates context-binding and CUDA I/O-memory registration errors.
    pub unsafe fn create_io_memory(context: &Arc<Context>, slice: &'a mut [T]) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::register_slice(
                context,
                slice.as_mut_ptr(),
                slice.len(),
                HostRegisterFlags::IO_MEMORY,
            )
        }
    }

    unsafe fn register_slice(
        context: &Arc<Context>,
        ptr: *mut T,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        let typed_ptr = dangling_or_non_null(ptr, length)?;
        let registration =
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { HostRegistration::create(context, typed_ptr.cast(), bytes, flags)? };
        Ok(Self {
            ptr: typed_ptr,
            length,
            registration,
            _borrow: PhantomData,
        })
    }

    /// Returns the registered memory as an immutable host slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered memory as a mutable host slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.is_empty() {
            return &mut [];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered host pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the mutable registered host pointer.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the number of registered elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this registration contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the registered range size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a
    /// non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags used to create this registration.
    #[must_use]
    pub const fn flags(&self) -> HostRegisterFlags {
        self.registration.flags
    }

    /// Returns the CUDA device pointer alias for this registered host range.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot map this host range into the current
    /// CUDA context or reports runtime initialization diagnostics.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.registration.mapped_device_ptr()
    }

    /// Returns the CUDA context retained for registration, mapping, and unregistration.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.registration.context
    }

    /// Borrows initialized registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors. Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the registered host storage contains valid T values and retains this shared borrow.
        unsafe { DeviceView::from_raw_parts(self.registration.mapped_device_region()?, self.len()) }
    }

    /// Exclusively borrows registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors. Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive wrapper borrow covers the complete registered host range.
        unsafe {
            DeviceViewMut::from_raw_parts(self.registration.mapped_device_region()?, self.len())
        }
    }
}

impl<'a, T: DeviceRepr> RegisteredHostSlice<'a, T> {
    /// Completes tracked device accesses and excludes submitted work while reading host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        let context = self.context();
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::Read)])?;
        Ok(read(self.as_slice()))
    }

    /// Registers an immutable host slice with CUDA for this wrapper's lifetime.
    ///
    /// Empty slices retain the context without registering a host range.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte length overflows, CUDA cannot
    /// register the host range, or `slice` is non-empty for a zero-sized `T`.
    pub fn create(context: &Arc<Context>, slice: &'a [T]) -> Result<Self> {
        Self::create_with_flags(context, slice, HostRegisterFlags::DEFAULT)
    }

    /// Registers an immutable host slice as read-only for CUDA access.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice byte length overflows, CUDA cannot
    /// register the host range as read-only, or `slice` is non-empty for a
    /// zero-sized `T`.
    pub fn create_read_only(context: &Arc<Context>, slice: &'a [T]) -> Result<Self> {
        Self::create_with_flags(context, slice, HostRegisterFlags::READ_ONLY)
    }

    /// Registers an immutable host slice with CUDA using `flags`.
    ///
    /// Immutable registration rejects [`HostRegisterFlags::IO_MEMORY`]. Use the
    /// unsafe mutable constructor for MMIO ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` contain unsupported options, the byte length
    /// overflows, CUDA cannot register the range, or `slice` is non-empty for a
    /// zero-sized `T`.
    pub fn create_with_flags(
        context: &Arc<Context>,
        slice: &'a [T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        if flags.contains(HostRegisterFlags::IO_MEMORY) {
            return Err(Error::InvalidValue);
        }
        // SAFETY: the registration borrows `slice` for its lifetime, so the pages stay valid until unregistration.
        unsafe { Self::register_slice(context, slice.as_ptr(), slice.len(), flags) }
    }

    unsafe fn register_slice(
        context: &Arc<Context>,
        ptr: *const T,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        let typed_ptr = dangling_or_non_null(ptr.cast_mut(), length)?;
        let registration =
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { HostRegistration::create(context, typed_ptr.cast(), bytes, flags)? };
        Ok(Self {
            ptr: typed_ptr,
            length,
            registration,
            _borrow: PhantomData,
        })
    }

    /// Returns the registered memory as an immutable host slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered host pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the number of registered elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this registration contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the registered range size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a
    /// non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags used to create this registration.
    #[must_use]
    pub const fn flags(&self) -> HostRegisterFlags {
        self.registration.flags
    }

    /// Returns the mapped device address in the retained registration context.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA host-to-device mapping query errors, including memory that is not mapped for this context.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.registration.mapped_device_ptr()
    }

    /// Returns the CUDA context retained for registration, mapping, and unregistration.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.registration.context
    }

    /// Borrows initialized registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors. Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the registered host storage contains valid T values and retains this shared borrow.
        unsafe { DeviceView::from_raw_parts(self.registration.mapped_device_region()?, self.len()) }
    }
}

impl<T: DeviceRepr> RegisteredHostVec<T> {
    /// Completes tracked device accesses and excludes submitted work while reading host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        let context = self.context();
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::Read)])?;
        Ok(read(self.as_slice()))
    }

    /// Completes tracked device accesses and excludes submitted work while modifying host values.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and CUDA completion errors before invoking the callback.
    pub fn with_host_slice_mut<R>(&mut self, write: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        let context = Arc::clone(self.context());
        let _guard =
            context.acquire_accesses(&[Access::host(self.host_region(), AccessMode::ReadWrite)])?;
        Ok(write(self.as_slice_mut()))
    }

    /// Copies `values` into a new `Vec<T>` and registers its storage with CUDA.
    ///
    /// Empty slices retain the context without registering a host range.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation of the `Vec` fails, the byte length
    /// overflows, CUDA cannot register the range, or `values` is non-empty for
    /// a zero-sized `T`.
    pub fn from_slice(context: &Arc<Context>, values: &[T]) -> Result<Self> {
        Self::from_slice_with_flags(context, values, HostRegisterFlags::DEFAULT)
    }

    /// Copies `values` into a new `Vec<T>` and registers it with `flags`.
    ///
    /// This constructor rejects [`HostRegisterFlags::READ_ONLY`] because the
    /// returned vector exposes mutable access, and rejects
    /// [`HostRegisterFlags::IO_MEMORY`] because a normal `Vec<T>` cannot
    /// represent an MMIO range.
    ///
    /// # Errors
    ///
    /// Returns an error if `flags` contain unsupported options, allocation of
    /// the `Vec` fails, the byte length overflows, CUDA cannot register the
    /// range, or `values` is non-empty for a zero-sized `T`.
    pub fn from_slice_with_flags(
        context: &Arc<Context>,
        values: &[T],
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        Self::from_vec_with_flags(context, values.to_vec(), flags)
    }

    /// Registers an owned vector without copying or changing its allocation.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, or an invalid allocation layout. Also propagates context-binding and CUDA host registration errors.
    pub fn from_vec(context: &Arc<Context>, values: Vec<T>) -> Result<Self> {
        Self::from_vec_with_flags(context, values, HostRegisterFlags::DEFAULT)
    }

    /// Registers an owned vector, preserving its original capacity and allocation layout.
    ///
    /// Mutable storage rejects read-only and MMIO registration flags.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, an invalid allocation layout, or unsupported read-only or I/O-memory flags. Also propagates context-binding and CUDA host registration errors.
    pub fn from_vec_with_flags(
        context: &Arc<Context>,
        mut values: Vec<T>,
        flags: HostRegisterFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        if flags.contains(HostRegisterFlags::READ_ONLY)
            || flags.contains(HostRegisterFlags::IO_MEMORY)
        {
            return Err(Error::InvalidValue);
        }
        let bytes = checked_bytes::<T>(values.len())?;
        let layout = Layout::array::<T>(values.capacity())
            .map_err(|_| Error::InvalidMemoryAllocationRequest)?;
        let ptr = NonNull::from(values.as_mut_slice()).cast();
        // SAFETY: the vector is moved into the registration, so its buffer stays valid until unregistration.
        let registration = unsafe { HostRegistration::create(context, ptr.cast(), bytes, flags)? };
        let length = values.len();
        // The shared owner unregisters and frees this original global allocation.
        let _values = ManuallyDrop::new(values);
        Ok(Self {
            allocation: Arc::new(RegisteredAllocation {
                registration,
                layout,
            }),
            ptr,
            length,
        })
    }

    /// Returns the registered vector contents as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the registered vector contents as a mutable slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: `self` owns or borrows `length` initialized elements at `ptr`, and the borrow of `self` prevents aliasing mutation.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Returns the number of registered elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this registered vector contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the registered vector size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `len * size_of::<T>()` overflows or `T` is a
    /// non-empty zero-sized element type.
    pub fn byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the flags used to register this vector.
    #[must_use]
    pub fn flags(&self) -> HostRegisterFlags {
        self.allocation.flags()
    }

    /// Returns the CUDA device pointer alias for this registered vector.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot map this vector into the current CUDA
    /// context or reports runtime initialization diagnostics.
    pub fn mapped_device_ptr(&self) -> Result<DevicePtr> {
        self.allocation.mapped_device_ptr()
    }

    /// Returns the CUDA context retained for registration, mapping, and unregistration.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Borrows initialized registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors. Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view(&self) -> Result<DeviceView<'_, T>> {
        // SAFETY: the registered host storage contains valid T values and retains this shared borrow.
        unsafe {
            DeviceView::from_raw_parts(
                self.allocation
                    .mapped_device_region()?
                    .slice(..self.byte_len()?)?,
                self.len(),
            )
        }
    }

    /// Exclusively borrows registered values through their device mapping.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors. Also rejects invalid mapped address ranges, byte-size overflow, or a view whose byte length or alignment does not match `T`.
    pub fn mapped_device_view_mut(&mut self) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive wrapper borrow covers the complete registered host range.
        unsafe {
            DeviceViewMut::from_raw_parts(
                self.allocation
                    .mapped_device_region()?
                    .slice(..self.byte_len()?)?,
                self.len(),
            )
        }
    }

    /// Returns the retained owner without granting host or device access.
    #[must_use]
    pub fn allocation(&self) -> &Arc<RegisteredAllocation> {
        &self.allocation
    }

    /// Returns the host address for this typed range.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Returns the host address under an exclusive typed borrow.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Reinterprets initialized bytes while preserving the original registration and deallocation layout.
    ///
    /// # Errors
    ///
    /// Returns the original vector if its byte length is invalid, `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<RegisteredHostVec<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let Ok(bytes) = self.byte_len() else {
            return Err(self);
        };
        if size == 0
            || !bytes.is_multiple_of(size)
            || (bytes != 0 && !self.as_ptr().addr().is_multiple_of(align_of::<U>()))
        {
            return Err(self);
        }
        Ok(RegisteredHostVec {
            allocation: self.allocation,
            ptr: if bytes == 0 {
                NonNull::dangling()
            } else {
                self.ptr.cast()
            },
            length: bytes / size,
        })
    }

    /// Erases initialization while retaining the complete allocation and registration.
    #[must_use]
    pub fn into_uninit(self) -> RegisteredHostVec<MaybeUninit<T>> {
        RegisteredHostVec {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }

    /// Allocates and registers storage for a host or device producer without initializing values.
    ///
    /// # Errors
    ///
    /// Returns an error for byte-size overflow, a nonempty zero-sized allocation, an invalid allocation layout, or unsupported read-only or I/O-memory flags. Also propagates context-binding and CUDA host registration errors.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
        flags: HostRegisterFlags,
    ) -> Result<RegisteredHostVec<MaybeUninit<T>>> {
        let () = T::LAYOUT;
        checked_bytes::<T>(length)?;
        RegisteredHostVec::from_vec_with_flags(context, vec![MaybeUninit::uninit(); length], flags)
    }
}

impl<T: DeviceRepr> RegisteredHostVec<MaybeUninit<T>> {
    /// Marks all values initialized after all host and device production completes.
    ///
    /// # Safety
    /// Every element must be valid and all producers must have completed.
    #[must_use]
    pub unsafe fn assume_init(self) -> RegisteredHostVec<T> {
        let () = T::LAYOUT;
        RegisteredHostVec {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }

    /// Exposes a completed initialized prefix while retaining the complete registration and capacity.
    ///
    /// # Safety
    /// Every prefix element must be valid and all producers must have completed.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<RegisteredHostVec<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Exclusively borrows mapped storage for a producer without asserting initialized values.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA mapping-query errors. Also rejects invalid mapped address ranges, byte-size overflow, or output metadata whose byte length or alignment does not match `T`.
    pub fn mapped_device_output(&mut self) -> Result<DeviceOutputView<'_, T>> {
        let region = self
            .allocation
            .mapped_device_region()?
            .slice(..self.byte_len()?)?;
        // SAFETY: this exclusive borrow retains the mapped registration and does not read its contents.
        unsafe { DeviceOutputView::from_raw_parts(region, self.length) }
    }
}

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

fn checked_pitched_bytes<T>(
    ptr: *mut T,
    width: usize,
    height: usize,
    depth: usize,
    pitch: usize,
) -> Result<usize> {
    let row_bytes = checked_bytes::<T>(width)?;
    checked_volume(row_bytes, height, depth)?;
    if width == 0 || height == 0 || depth == 0 {
        return Ok(0);
    }
    let bytes = checked_volume(pitch, height, depth)?;
    if ptr.is_null()
        || pitch < row_bytes
        || !(ptr as usize).is_multiple_of(align_of::<T>())
        || !pitch.is_multiple_of(align_of::<T>())
        || (ptr as usize).checked_add(bytes).is_none()
    {
        return Err(Error::InvalidMemoryAccess);
    }
    Ok(bytes)
}

// Adopt a newly allocated pointer before checking CUDA's returned pitch, so a
// rejected layout still frees the allocation in its owning context.
fn own_pitched_allocation<T>(
    context: &Arc<Context>,
    ptr: *mut T,
    width: usize,
    height: usize,
    depth: usize,
    pitch: usize,
) -> Result<Arc<DeviceAllocation>> {
    let mut allocation = DeviceAllocation {
        id: AllocationId::create(),
        context: Arc::clone(context),
        ptr: ptr.cast(),
        bytes: 0,
        deallocation: DeviceDeallocation::Immediate,
    };
    allocation.bytes = checked_pitched_bytes(ptr, width, height, depth, pitch)?;
    Ok(Arc::new(allocation))
}

impl<T: DeviceRepr> PitchedDeviceMemory<T> {
    fn allocate(context: &Arc<Context>, width: usize, height: usize) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        // SAFETY: the context is bound above, and the returned allocation is owned by the new value.
        let (ptr, pitch) = unsafe { DeviceMemory::<T>::alloc_pitched(width, height)? };
        let allocation = own_pitched_allocation(context, ptr, width, height, 1, pitch)?;
        Ok(Self {
            allocation,
            ptr,
            width,
            height,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Allocates pitched storage whose logical elements are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage. Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub fn create_uninit(
        context: &Arc<Context>,
        width: usize,
        height: usize,
    ) -> Result<PitchedDeviceMemory<MaybeUninit<T>>> {
        PitchedDeviceMemory::<MaybeUninit<T>>::allocate(context, width, height)
    }

    /// Allocates pitched storage with caller-established initialization.
    ///
    /// # Safety
    ///
    /// Initialize every logical element before reading it. CUDA row padding
    /// remains uninitialized and must not be exposed as initialized bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage. Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub unsafe fn create_unchecked(
        context: &Arc<Context>,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        Self::allocate(context, width, height)
    }

    /// Allocates pitched storage and completes zero initialization of its logical elements.
    ///
    /// # Errors
    ///
    /// Propagates pitched allocation, size-validation, access-tracking, context-binding, CUDA fill, and synchronization errors.
    pub fn zeroed(context: &Arc<Context>, width: usize, height: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height)?;
        memory.set_zeroes()?;
        // SAFETY: the completed fill initializes every logical element with a valid zero value.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a copy of contiguous row-major host values.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count. Also propagates pitched allocation, size-validation, context-binding, CUDA copy, and synchronization errors.
    pub fn from_slice(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        values: &[T],
    ) -> Result<Self> {
        Self::create_uninit(context, width, height)?.initialize_from_host(values)
    }

    /// Takes ownership of a pitched allocation in `context`.
    ///
    /// # Safety
    ///
    /// `ptr` must be the base of a live CUDA allocation compatible with `cudaFree`,
    /// with no other destruction owner. It must contain initialized, valid `T`
    /// values at every logical element. The pitch and dimensions must describe
    /// its storage, and previous accesses must be ordered before future consumers.
    /// A null pointer is accepted for an empty shape.
    ///
    /// # Errors
    ///
    /// Returns an error if row or volume size calculations overflow or a nonempty zero-sized element range is requested. For nonempty storage, also rejects a null or misaligned pointer, a pitch smaller than the row width or not aligned for `T`, or an overflowing ending address.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        width: usize,
        height: usize,
        pitch: usize,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_pitched_bytes(ptr, width, height, 1, pitch)?;
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            width,
            height,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Transfers raw ownership, or returns this buffer unchanged if another owner retains it.
    ///
    /// The returned pointer must eventually be freed in the returned context.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(self) -> StdResult<(*mut T, usize, usize, usize, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            width,
            height,
            pitch,
            _phantom: PhantomData,
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                allocation.ptr = ptr::null_mut();
                Ok((ptr, width, height, pitch, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                width,
                height,
                pitch,
                _phantom: PhantomData,
            }),
        }
    }

    /// Returns the allocation's retained CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the shared owner of the physical allocation, including row padding.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets logical row bytes while preserving pitch, capacity, and allocation identity.
    ///
    /// Row padding is excluded from the initialized representation.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the logical row byte width, or nonempty storage has a base pointer or pitch not aligned for `U`.
    ///
    /// # Panics
    ///
    /// Panics if the logical row byte width overflows `usize`. Construction validates this width, so this cannot occur for a valid pitched allocation.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<PitchedDeviceMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let row_bytes = self
            .row_byte_width()
            .expect("pitched row byte length overflow");
        if size == 0
            || !row_bytes.is_multiple_of(size)
            || (!self.is_empty()
                && (!(self.ptr as usize).is_multiple_of(align_of::<U>())
                    || !self.pitch.is_multiple_of(align_of::<U>())))
        {
            return Err(self);
        }
        Ok(PitchedDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: row_bytes / size,
            height: self.height,
            pitch: self.pitch,
            _phantom: PhantomData,
        })
    }

    /// Erases logical element initialization without changing allocation ownership.
    #[must_use]
    pub fn into_uninit(self) -> PitchedDeviceMemory<MaybeUninit<T>> {
        PitchedDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }

    /// Returns the logical row width in elements.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the logical number of rows.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the row stride in bytes.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }

    /// Returns the base device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the mutable base device pointer.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns the base pointer as a CUDA device pointer wrapper.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the logical element count.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` overflows.
    pub fn len(&self) -> Result<usize> {
        checked_len(self.width, self.height)
    }

    /// Returns the logical row width in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * size_of::<T>()` overflows or `T` is a
    /// non-empty zero-sized element type.
    pub fn row_byte_width(&self) -> Result<usize> {
        checked_bytes::<T>(self.width)
    }

    /// Returns the physical allocation span described by `pitch * height`.
    ///
    /// # Errors
    ///
    /// Returns an error if `pitch * height` overflows.
    pub fn byte_len(&self) -> Result<usize> {
        Ok(self.allocation.byte_len())
    }

    /// Returns whether any dimension is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Copies a contiguous host slice into this pitched device allocation.
    ///
    /// `src` must contain exactly `width * height` elements in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA rejects the 2D copy.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let row_bytes = self.row_byte_width()?;
        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.ptr,
                self.pitch,
                src.as_ptr(),
                row_bytes,
                self.width,
                self.height,
                MemoryCopyKind::HostToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies this pitched device allocation into a contiguous host slice.
    ///
    /// `dst` must contain exactly `width * height` elements and is filled in
    /// row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA rejects the 2D copy.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so any device bytes are valid `T` values.
        unsafe { self.copy_to_host_unchecked(dst) }
    }

    /// Copies device values whose validity is established by the producing kernel.
    ///
    /// # Safety
    ///
    /// Every copied element must be initialized and valid for `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination length differs from the logical element count or transfer size calculations overflow. Also propagates access-tracking, context-binding, CUDA copy, and default-stream synchronization errors.
    pub unsafe fn copy_to_host_unchecked(&self, dst: &mut [T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let row_bytes = self.row_byte_width()?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                dst.as_mut_ptr(),
                row_bytes,
                self.ptr,
                self.pitch,
                self.width,
                self.height,
                MemoryCopyKind::DeviceToHost,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies another pitched device allocation with the same shape into this one.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes differ, byte-size arithmetic overflows,
    /// or CUDA rejects the device-to-device 2D copy.
    pub fn copy_from_pitched(&mut self, src: &Self) -> Result<()> {
        self.context().bind()?;
        if self.context() != src.context() {
            return Err(Error::InvalidValue);
        }
        if self.width != src.width || self.height != src.height {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.ptr,
                self.pitch,
                src.ptr,
                src.pitch,
                self.width,
                self.height,
                MemoryCopyKind::DeviceToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a host-to-device 2D copy from a contiguous host slice.
    ///
    /// `src` must remain valid until `stream` reaches the copy. The
    /// [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA cannot enqueue the copy.
    pub fn copy_from_host_async<'env>(
        &'env mut self,
        src: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }
        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let row_bytes = self.row_byte_width()?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_2d_async(
                    self.ptr,
                    self.pitch,
                    src.as_ptr(),
                    row_bytes,
                    self.width,
                    self.height,
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a device-to-host 2D copy into a contiguous host slice.
    ///
    /// `dst` must remain valid until `stream` reaches the copy. The
    /// [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA cannot enqueue the copy.
    pub fn copy_to_host_async<'env>(
        &'env self,
        dst: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Read)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }
        self.context().bind()?;
        let () = T::LAYOUT;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let row_bytes = self.row_byte_width()?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_2d_async(
                    dst.as_mut_ptr(),
                    row_bytes,
                    self.ptr,
                    self.pitch,
                    self.width,
                    self.height,
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// Fills this pitched allocation with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 2D memset.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        // SAFETY: `T: ZeroableDeviceRepr`, so all-zero bytes are a valid `T`.
        unsafe { self.set_value_unchecked(0) }
    }

    /// Fills this pitched allocation with `value` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 2D memset.
    pub fn set_value(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so repeating `value` produces valid `T` values.
        unsafe { self.set_value_unchecked(value) }
    }

    /// Fills storage with a byte pattern known to be valid for `T`.
    ///
    /// # Safety
    ///
    /// The repeated byte pattern must represent a valid value of `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, transfer-size validation, CUDA fill, and default-stream synchronization errors.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { DeviceMemory::<T>::set_2d(self.ptr, self.pitch, value, self.width, self.height) }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a byte fill of this pitched allocation on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the 2D memset.
    pub fn set_value_async<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }
        self.context().bind()?;
        let () = T::LAYOUT;
        checked_len(self.row_byte_width()?, self.height)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::set_2d_async(
                    self.ptr,
                    self.pitch,
                    value,
                    self.width,
                    self.height,
                    stream.stream(),
                )
            })
        }
    }

    fn row_region(&self, row: usize) -> Result<DeviceRegion<'_>> {
        if row >= self.height {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.width == 0 {
            return self.region().slice(0..0);
        }
        let start = row
            .checked_mul(self.pitch)
            .ok_or(Error::InvalidMemoryAccess)?;
        let end = start
            .checked_add(self.row_byte_width()?)
            .ok_or(Error::InvalidMemoryAccess)?;
        self.region().slice(start..end)
    }

    /// Borrows one initialized logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row(&self, row: usize) -> Result<DeviceView<'_, T>> {
        // SAFETY: construction or promotion establishes initialized logical elements.
        unsafe { DeviceView::from_raw_parts(self.row_region(row)?, self.width) }
    }

    /// Exclusively borrows one logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row_mut(&mut self, row: usize) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive buffer borrow covers this exact row and its initialized element type.
        unsafe { DeviceViewMut::from_raw_parts(self.row_region(row)?, self.width) }
    }
}

impl<T: DeviceRepr> PitchedDeviceMemory<MaybeUninit<T>> {
    /// Copies every logical element from host values and completes before promotion.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count or transfer size calculations overflow. Also propagates context-binding, CUDA copy, and default-stream synchronization errors.
    pub fn initialize_from_host(self, values: &[T]) -> Result<PitchedDeviceMemory<T>> {
        if values.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        self.context().bind()?;
        // SAFETY: the allocation has writable storage for the complete logical rectangle.
        unsafe {
            DeviceMemory::<T>::copy_2d(
                self.ptr.cast(),
                self.pitch,
                values.as_ptr(),
                self.row_byte_width()?,
                self.width,
                self.height,
                MemoryCopyKind::HostToDevice,
            )?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()?;
        // SAFETY: the completed full copy initializes all logical elements to valid T values.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks all logical elements initialized without changing the owner or pitch.
    ///
    /// # Safety
    ///
    /// Every logical element must contain an initialized valid T value.
    /// Producers must be ordered before subsequent consumers and complete before
    /// host readback. Row padding remains uninitialized.
    #[must_use]
    pub unsafe fn assume_init(self) -> PitchedDeviceMemory<T> {
        let () = T::LAYOUT;
        PitchedDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for PitchedDeviceMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

// SAFETY: device pointers are not dereferenced on the host; all operations bind the retained context.
unsafe impl<T: DeviceRepr + Send> Send for PitchedDeviceMemory<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for PitchedDeviceMemory<T> {}

impl<T: DeviceRepr> PitchedDeviceVolume<T> {
    fn allocate(context: &Arc<Context>, width: usize, height: usize, depth: usize) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        // SAFETY: the context is bound above, and the returned allocation is owned by the new value.
        let pitched = unsafe { DeviceMemory::<T>::alloc_pitched_3d(width, height, depth)? };
        let ptr = pitched.ptr.cast();
        let pitch = to_usize(pitched.pitch, "allocation pitch")?;
        let allocation = own_pitched_allocation(context, ptr, width, height, depth, pitch)?;
        Ok(Self {
            allocation,
            ptr,
            width,
            height,
            depth,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Allocates pitched storage whose logical elements are uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage. Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub fn create_uninit(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<PitchedDeviceVolume<MaybeUninit<T>>> {
        PitchedDeviceVolume::<MaybeUninit<T>>::allocate(context, width, height, depth)
    }

    /// Allocates pitched storage with caller-established initialization.
    ///
    /// # Safety
    ///
    /// Initialize every logical element before reading it. CUDA row padding
    /// remains uninitialized and must not be exposed as initialized bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if size calculations overflow, a nonempty zero-sized element range is requested, context binding fails, or CUDA cannot allocate pitched storage. Also rejects a returned pointer or pitch that violates the requested row size, alignment, or address-range bounds.
    pub unsafe fn create_unchecked(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<Self> {
        Self::allocate(context, width, height, depth)
    }

    /// Allocates pitched storage and completes zero initialization of its logical elements.
    ///
    /// # Errors
    ///
    /// Propagates pitched allocation, size-validation, access-tracking, context-binding, CUDA fill, and synchronization errors.
    pub fn zeroed(context: &Arc<Context>, width: usize, height: usize, depth: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create_uninit(context, width, height, depth)?;
        memory.set_zeroes()?;
        // SAFETY: the completed fill initializes every logical element with a valid zero value.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates pitched storage and completes a copy of contiguous row-major host values.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count. Also propagates pitched allocation, size-validation, context-binding, CUDA copy, and synchronization errors.
    pub fn from_slice(
        context: &Arc<Context>,
        width: usize,
        height: usize,
        depth: usize,
        values: &[T],
    ) -> Result<Self> {
        Self::create_uninit(context, width, height, depth)?.initialize_from_host(values)
    }

    /// Takes ownership of a pitched allocation in `context`.
    ///
    /// # Safety
    ///
    /// `ptr` must be the base of a live CUDA allocation compatible with `cudaFree`,
    /// with no other destruction owner. It must contain initialized, valid `T`
    /// values at every logical element. The pitch and dimensions must describe
    /// its storage, and previous accesses must be ordered before future consumers.
    /// A null pointer is accepted for an empty shape.
    ///
    /// # Errors
    ///
    /// Returns an error if row or volume size calculations overflow or a nonempty zero-sized element range is requested. For nonempty storage, also rejects a null or misaligned pointer, a pitch smaller than the row width or not aligned for `T`, or an overflowing ending address.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        width: usize,
        height: usize,
        depth: usize,
        pitch: usize,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_pitched_bytes(ptr, width, height, depth, pitch)?;
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            width,
            height,
            depth,
            pitch,
            _phantom: PhantomData,
        })
    }

    /// Transfers raw ownership, or returns this buffer unchanged if another owner retains it.
    ///
    /// The returned pointer must eventually be freed in the returned context.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    #[expect(
        clippy::type_complexity,
        reason = "raw volume ownership includes its pointer, dimensions, pitch, and CUDA context"
    )]
    pub fn into_raw_parts(
        self,
    ) -> StdResult<(*mut T, usize, usize, usize, usize, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            width,
            height,
            depth,
            pitch,
            _phantom: PhantomData,
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                allocation.ptr = ptr::null_mut();
                Ok((ptr, width, height, depth, pitch, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                width,
                height,
                depth,
                pitch,
                _phantom: PhantomData,
            }),
        }
    }

    /// Returns the allocation's retained CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the shared owner of the physical allocation, including row padding.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets logical row bytes while preserving pitch, capacity, and allocation identity.
    ///
    /// Row padding is excluded from the initialized representation.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the logical row byte width, or nonempty storage has a base pointer or pitch not aligned for `U`.
    ///
    /// # Panics
    ///
    /// Panics if the logical row byte width overflows `usize`. Construction validates this width, so this cannot occur for a valid pitched allocation.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<PitchedDeviceVolume<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let size = size_of::<U>();
        let row_bytes = self
            .row_byte_width()
            .expect("pitched row byte length overflow");
        if size == 0
            || !row_bytes.is_multiple_of(size)
            || (!self.is_empty()
                && (!(self.ptr as usize).is_multiple_of(align_of::<U>())
                    || !self.pitch.is_multiple_of(align_of::<U>())))
        {
            return Err(self);
        }
        Ok(PitchedDeviceVolume {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: row_bytes / size,
            height: self.height,
            depth: self.depth,
            pitch: self.pitch,
            _phantom: PhantomData,
        })
    }

    /// Erases logical element initialization without changing allocation ownership.
    #[must_use]
    pub fn into_uninit(self) -> PitchedDeviceVolume<MaybeUninit<T>> {
        PitchedDeviceVolume {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            depth: self.depth,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }

    /// Returns the logical row width in elements.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the logical number of rows per slice.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the logical number of slices.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the row stride in bytes.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }

    /// Returns the base device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the mutable base device pointer.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns the base pointer as a CUDA device pointer wrapper.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the logical element count.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height * depth` overflows.
    pub fn len(&self) -> Result<usize> {
        checked_volume(self.width, self.height, self.depth)
    }

    /// Returns the logical row width in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * size_of::<T>()` overflows or `T` is a
    /// non-empty zero-sized element type.
    pub fn row_byte_width(&self) -> Result<usize> {
        checked_bytes::<T>(self.width)
    }

    /// Returns the physical allocation span described by `pitch * height * depth`.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte-volume calculation overflows.
    pub fn byte_len(&self) -> Result<usize> {
        Ok(self.allocation.byte_len())
    }

    /// Returns whether any dimension is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.depth == 0
    }

    /// Returns a CUDA pitched pointer descriptor for this volume.
    ///
    /// The descriptor is suitable for `cudaMemcpy3D` and `cudaMemset3D`.
    /// CUDA expects the `xsize` field in bytes for linear memory, so this
    /// method reports the typed row width converted to bytes.
    #[must_use]
    pub fn pitched_ptr(&self) -> runtime::cudaPitchedPtr {
        runtime::cudaPitchedPtr {
            ptr: self.ptr.cast(),
            pitch: self.pitch as _,
            xsize: self.row_byte_width().unwrap_or(0) as _,
            ysize: self.height as _,
        }
    }

    fn linear_host_ptr(
        ptr: *mut T,
        width: usize,
        height: usize,
    ) -> Result<runtime::cudaPitchedPtr> {
        let row_bytes = checked_bytes::<T>(width)?;
        Ok(runtime::cudaPitchedPtr {
            ptr: ptr.cast(),
            pitch: row_bytes as _,
            xsize: row_bytes as _,
            ysize: height as _,
        })
    }

    /// Copies a contiguous host slice into this 3D pitched allocation.
    ///
    /// `src` must contain exactly `width * height * depth` elements in
    /// row-major slice order.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA rejects the 3D copy.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.pitched_ptr(),
                Self::linear_host_ptr(src.as_ptr().cast_mut(), self.width, self.height)?,
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::HostToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies this 3D pitched allocation into a contiguous host slice.
    ///
    /// `dst` must contain exactly `width * height * depth` elements and is
    /// filled in row-major slice order.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA rejects the 3D copy.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so any device bytes are valid `T` values.
        unsafe { self.copy_to_host_unchecked(dst) }
    }

    /// Copies device values whose validity is established by the producing kernel.
    ///
    /// # Safety
    ///
    /// Every copied element must be initialized and valid for `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination length differs from the logical element count or transfer size calculations overflow. Also propagates access-tracking, context-binding, CUDA copy, and default-stream synchronization errors.
    pub unsafe fn copy_to_host_unchecked(&self, dst: &mut [T]) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                Self::linear_host_ptr(dst.as_mut_ptr(), self.width, self.height)?,
                self.pitched_ptr(),
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::DeviceToHost,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Copies another 3D pitched allocation with the same shape into this one.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes differ, byte-size arithmetic overflows,
    /// or CUDA rejects the device-to-device 3D copy.
    pub fn copy_from_volume(&mut self, src: &Self) -> Result<()> {
        self.context().bind()?;
        if self.context() != src.context() {
            return Err(Error::InvalidValue);
        }
        if self.width != src.width || self.height != src.height || self.depth != src.depth {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the shape and lengths were checked above, so the copy stays inside both allocations.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.pitched_ptr(),
                src.pitched_ptr(),
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::DeviceToDevice,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a host-to-device 3D copy from a contiguous host slice.
    ///
    /// `src` must remain valid until `stream` reaches the copy. The
    /// [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA cannot enqueue the copy.
    pub fn copy_from_host_async<'env>(
        &'env mut self,
        src: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }
        self.context().bind()?;
        if src.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let host = Self::linear_host_ptr(src.as_ptr().cast_mut(), self.width, self.height)?;
        checked_volume(self.row_byte_width()?, self.height, self.depth)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_3d_async(
                    self.pitched_ptr(),
                    host,
                    self.width,
                    self.height,
                    self.depth,
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a device-to-host 3D copy into a contiguous host slice.
    ///
    /// `dst` must remain valid until `stream` reaches the copy. The
    /// [`StreamScope`] lifetime enforces that requirement for safe callers.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` has the wrong length, byte-size arithmetic
    /// overflows, or CUDA cannot enqueue the copy.
    pub fn copy_to_host_async<'env>(
        &'env self,
        dst: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Read)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }
        self.context().bind()?;
        let () = T::LAYOUT;
        if dst.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let host = Self::linear_host_ptr(dst.as_mut_ptr(), self.width, self.height)?;
        checked_volume(self.row_byte_width()?, self.height, self.depth)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::copy_3d_async(
                    host,
                    self.pitched_ptr(),
                    self.width,
                    self.height,
                    self.depth,
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// Fills this 3D pitched allocation with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 3D memset.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        // SAFETY: `T: ZeroableDeviceRepr`, so all-zero bytes are a valid `T`.
        unsafe { self.set_value_unchecked(0) }
    }

    /// Fills this 3D pitched allocation with `value` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the 3D memset.
    pub fn set_value(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so repeating `value` produces valid `T` values.
        unsafe { self.set_value_unchecked(value) }
    }

    /// Fills storage with a byte pattern known to be valid for `T`.
    ///
    /// # Safety
    ///
    /// The repeated byte pattern must represent a valid value of `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, transfer-size validation, CUDA fill, and default-stream synchronization errors.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            DeviceMemory::<T>::set_3d(
                self.pitched_ptr(),
                value,
                self.width,
                self.height,
                self.depth,
            )
        }?;
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a byte fill of this 3D pitched allocation on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the 3D memset.
    pub fn set_value_async<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        if self.context() != stream.stream().context() {
            return Err(Error::InvalidValue);
        }
        self.context().bind()?;
        let () = T::LAYOUT;
        checked_volume(self.row_byte_width()?, self.height, self.depth)?;
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                DeviceMemory::<T>::set_3d_async(
                    self.pitched_ptr(),
                    value,
                    self.width,
                    self.height,
                    self.depth,
                    stream.stream(),
                )
            })
        }
    }

    fn row_region(&self, slice: usize, row: usize) -> Result<DeviceRegion<'_>> {
        if slice >= self.depth || row >= self.height {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.width == 0 {
            return self.region().slice(0..0);
        }
        let start = slice
            .checked_mul(self.height)
            .and_then(|offset| offset.checked_add(row))
            .and_then(|offset| offset.checked_mul(self.pitch))
            .ok_or(Error::InvalidMemoryAccess)?;
        let end = start
            .checked_add(self.row_byte_width()?)
            .ok_or(Error::InvalidMemoryAccess)?;
        self.region().slice(start..end)
    }

    /// Borrows one initialized logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice or row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row(&self, slice: usize, row: usize) -> Result<DeviceView<'_, T>> {
        // SAFETY: construction or promotion establishes initialized logical elements.
        unsafe { DeviceView::from_raw_parts(self.row_region(slice, row)?, self.width) }
    }

    /// Exclusively borrows one logical row, excluding CUDA's row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice or row index is out of bounds, row offset calculations overflow, or the resulting range fails byte-length or alignment validation.
    pub fn row_mut(&mut self, slice: usize, row: usize) -> Result<DeviceViewMut<'_, T>> {
        // SAFETY: the exclusive buffer borrow covers this exact row and its initialized element type.
        unsafe { DeviceViewMut::from_raw_parts(self.row_region(slice, row)?, self.width) }
    }
}

impl<T: DeviceRepr> PitchedDeviceVolume<MaybeUninit<T>> {
    /// Copies every logical element from host values and completes before promotion.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the logical element count or transfer size calculations overflow. Also propagates context-binding, CUDA copy, and default-stream synchronization errors.
    pub fn initialize_from_host(self, values: &[T]) -> Result<PitchedDeviceVolume<T>> {
        if values.len() != self.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        self.context().bind()?;
        let row_bytes = self.row_byte_width()?;
        let source = runtime::cudaPitchedPtr {
            ptr: values.as_ptr().cast_mut().cast(),
            pitch: row_bytes as _,
            xsize: row_bytes as _,
            ysize: self.height as _,
        };
        // SAFETY: both operands cover the complete logical volume, with matching element sizes.
        unsafe {
            DeviceMemory::<T>::copy_3d(
                self.pitched_ptr(),
                source,
                self.width,
                self.height,
                self.depth,
                MemoryCopyKind::HostToDevice,
            )?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()?;
        // SAFETY: the completed full copy initializes all logical elements to valid T values.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks all logical elements initialized without changing the owner or pitch.
    ///
    /// # Safety
    ///
    /// Every logical element must contain an initialized valid T value.
    /// Producers must be ordered before subsequent consumers and complete before
    /// host readback. Row padding remains uninitialized.
    #[must_use]
    pub unsafe fn assume_init(self) -> PitchedDeviceVolume<T> {
        let () = T::LAYOUT;
        PitchedDeviceVolume {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            depth: self.depth,
            pitch: self.pitch,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for PitchedDeviceVolume<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

// SAFETY: device pointers are not dereferenced on the host; all operations bind the retained context.
unsafe impl<T: DeviceRepr + Send> Send for PitchedDeviceVolume<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for PitchedDeviceVolume<T> {}

impl<T> AsyncDeviceMemory<T> {
    /// Allocates stream-ordered device memory on `stream`.
    ///
    /// The allocation is freed by enqueuing `cudaFreeAsync` on the stored
    /// stream when this owner is dropped, unless [`AsyncDeviceMemory::free_async`]
    /// is called first.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot
    /// enqueue the allocation, or `length` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid T before it is read.
    /// Consumers on other streams must wait for allocation and producer work.
    pub unsafe fn create_unchecked(length: usize, stream: &Stream) -> Result<Self> {
        let _lifecycle = stream.allocation_guard()?;
        stream.context().bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { DeviceMemory::<T>::alloc_async(length, stream)? }
        };
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                stream.context(),
                ptr.cast(),
                bytes,
                DeviceDeallocation::StreamOrdered {
                    stream: stream.clone(),
                    pool: None,
                },
            ),
            ptr,
            length,
            _phantom: PhantomData,
        })
    }

    /// Allocates stream-ordered device memory from `pool` on `stream`.
    ///
    /// The allocation is freed by enqueuing `cudaFreeAsync` on the stored
    /// stream when this owner is dropped, unless [`AsyncDeviceMemory::free_async`]
    /// is called first.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot
    /// enqueue the pool allocation, CUDA returns a null pointer for a non-empty
    /// request, or `length` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// Every element must be initialized before it is read. Consumers on other
    /// streams must wait for allocation and producer work. The returned buffer
    /// retains the pool; CUDA releases its backing resources after outstanding frees finish.
    pub unsafe fn create_in_pool_unchecked(
        length: usize,
        pool: &MemoryPool,
        stream: &Stream,
    ) -> Result<Self> {
        let _lifecycle = stream.allocation_guard()?;
        stream.context().bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let mut ptr = ptr::null_mut();
        if bytes != 0 {
            // SAFETY: `cudaMallocFromPoolAsync` receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(runtime::cudaMallocFromPoolAsync(
                    &raw mut ptr,
                    bytes as _,
                    pool.as_raw(),
                    stream.as_raw()
                ))?;
            }
            if ptr.is_null() {
                return Err(Error::NullHandle);
            }
        }
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                stream.context(),
                ptr.cast(),
                bytes,
                DeviceDeallocation::StreamOrdered {
                    stream: stream.clone(),
                    pool: Some(pool.clone()),
                },
            ),
            ptr: ptr.cast(),
            length,
            _phantom: PhantomData,
        })
    }

    /// Takes ownership of a stream-ordered CUDA allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be null for an empty allocation or point to `length` live
    /// elements allocated by `cudaMallocAsync` or `cudaMallocFromPoolAsync`.
    /// The allocation must be valid to free on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is described, or a nonempty allocation has a null pointer.
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        length: usize,
        stream: &Stream,
        pool: Option<MemoryPool>,
    ) -> Result<Self> {
        let bytes = checked_bytes::<T>(length)?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                stream.context(),
                ptr.cast(),
                bytes,
                DeviceDeallocation::StreamOrdered {
                    stream: stream.clone(),
                    pool,
                },
            ),
            ptr,
            length,
            _phantom: PhantomData,
        })
    }

    /// Enqueues the asynchronous free when this buffer holds the final allocation owner.
    ///
    /// Calling this method surfaces CUDA errors that would otherwise only be
    /// logged in debug builds from `Drop`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocationInUse`] if a submission, graph, or interop user retains the allocation.
    /// Otherwise returns an error if CUDA cannot enqueue `cudaFreeAsync` for the stored stream.
    /// A retained allocation remains live after `AllocationInUse` and is released by its final owner.
    pub fn free_async(self) -> Result<()> {
        let mut allocation =
            Arc::try_unwrap(self.allocation).map_err(|_| Error::AllocationInUse)?;
        allocation.release()
    }

    /// Returns the number of elements in this allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this allocation contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the logical allocation size in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the byte-size calculation overflows. Safe constructors reject
    /// such allocations.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        checked_bytes::<T>(self.length).expect("async device memory byte length overflow")
    }

    /// Returns the base device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the mutable base device pointer.
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Returns the base pointer as a CUDA device pointer wrapper.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the CUDA context retained by this allocation or pool.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Borrows all initialized elements as a shared device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T>
    where
        T: DeviceRepr,
    {
        DeviceView::from_buffer(self)
    }

    /// Borrows all initialized elements as an exclusive device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T>
    where
        T: DeviceRepr,
    {
        DeviceViewMut::from_buffer(self)
    }

    /// Enqueues an uninitialized allocation on `stream` using CUDA’s current pool.
    ///
    /// Access on other streams must be ordered after the allocation stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, stream or context setup fails, or CUDA cannot enqueue the allocation.
    pub fn create_uninit(
        length: usize,
        stream: &Stream,
    ) -> Result<AsyncDeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { AsyncDeviceMemory::<MaybeUninit<T>>::create_unchecked(length, stream) }
    }

    /// Enqueues an uninitialized allocation from `pool` on `stream`.
    ///
    /// The allocation retains both the pool and the stream until it can be freed.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, or stream or context setup fails. Also propagates CUDA errors for the requested pool allocation and rejects a null handle for a nonempty allocation.
    pub fn create_uninit_in_pool(
        length: usize,
        pool: &MemoryPool,
        stream: &Stream,
    ) -> Result<AsyncDeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe {
            AsyncDeviceMemory::<MaybeUninit<T>>::create_in_pool_unchecked(length, pool, stream)
        }
    }

    /// Returns the allocation owner retained by submissions, graphs, and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets initialized elements while preserving the stream and pool owner.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<AsyncDeviceMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let Ok(length) = cast_length::<U>(self.region()) else {
            return Err(self);
        };
        Ok(AsyncDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            _phantom: PhantomData,
        })
    }

    /// Erases element initialization without changing allocation identity or stream ordering.
    #[must_use]
    pub fn into_uninit(self) -> AsyncDeviceMemory<MaybeUninit<T>> {
        AsyncDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            _phantom: PhantomData,
        }
    }

    /// Returns the explicitly selected pool, if any.
    #[must_use]
    pub fn pool(&self) -> Option<&MemoryPool> {
        self.allocation.pool()
    }

    /// Returns the stream used for allocation and freeing.
    ///
    /// # Panics
    ///
    /// Panics if the allocation has no associated stream. All constructors of this stream-ordered wrapper retain a stream, so this cannot occur for a valid instance.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        self.allocation
            .stream()
            .expect("stream-ordered allocation has no stream")
    }

    /// Enqueues a byte fill of this allocation on its stored stream.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the memset.
    pub fn set_value_async(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        let mut execution = Execution::create(self.stream())?;
        let mut resources = Resources::new();
        // This allocation is freed on the same retained stream, after the fill and its event.
        // Other-stream consumers retain their own allocation owners before enqueueing.
        resources.access(self.region(), AccessMode::Write);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| {
                DeviceMemory::<T>::set_async(self.ptr, value, self.length, stream)
            })?;
        }
        execution.finish(())?;
        Ok(())
    }
}

impl<T: ZeroableDeviceRepr> AsyncDeviceMemory<T> {
    /// Allocates stream-ordered device memory and fills it with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or CUDA cannot enqueue the memset.
    pub fn zeroed(length: usize, stream: &Stream) -> Result<Self> {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let memory = unsafe { Self::create_unchecked(length, stream)? };
        // SAFETY: the allocation covers `memory.length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            DeviceMemory::<T>::set_async(memory.ptr, 0, memory.length, memory.stream())?;
        }
        Ok(memory)
    }

    /// Allocates stream-ordered device memory from `pool` and fills it with zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation from the pool fails or CUDA cannot
    /// enqueue the memset.
    pub fn zeroed_in_pool(length: usize, pool: &MemoryPool, stream: &Stream) -> Result<Self> {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let memory = unsafe { Self::create_in_pool_unchecked(length, pool, stream)? };
        // SAFETY: the allocation covers `memory.length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            DeviceMemory::<T>::set_async(memory.ptr, 0, memory.length, memory.stream())?;
        }
        Ok(memory)
    }
}

unsafe impl<T> DeviceRegionProvider for AsyncDeviceMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let offset = self.ptr as usize - self.allocation.ptr as usize;
        self.allocation
            .region()
            .slice(offset..offset + self.byte_len())
            .expect("async buffer exceeds its allocation")
    }
}

impl<T: DeviceRepr> AsyncDeviceMemory<MaybeUninit<T>> {
    /// Borrows this stream-ordered allocation as a write-only output of `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Exposes a producer-initialized prefix while retaining the full allocation and its stream.
    ///
    /// # Safety
    ///
    /// The prefix must contain valid initialized `T` values, with producing work ordered before all consumers.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<AsyncDeviceMemory<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks the entire allocation initialized without changing its allocation stream.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid T, with producer writes ordered before every consumer.
    #[must_use]
    pub unsafe fn assume_init(self) -> AsyncDeviceMemory<T> {
        let () = T::LAYOUT;
        AsyncDeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            _phantom: PhantomData,
        }
    }
}

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

impl<T> DeviceMemory<T> {
    /// Takes ownership of an existing device allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be null for an empty allocation or point to `length` live
    /// elements allocated by `cudaMallocManaged` or another CUDA allocation
    /// function compatible with `cudaFree`. `length * size_of::<T>()` must fit
    /// in `usize`.
    /// No other owner may free the pointer while the returned value is alive.
    /// The allocation must belong to `context`, and all elements must be initialized before they are read.
    ///
    /// # Panics
    ///
    /// Panics if the byte count overflows or a nonempty zero-sized allocation is described. Also panics if allocation identities are exhausted; see [`AllocationId::create`].
    pub unsafe fn from_raw_parts(context: &Arc<Context>, ptr: *mut T, length: usize) -> Self {
        Self {
            allocation: Arc::new(DeviceAllocation {
                id: AllocationId::create(),
                context: Arc::clone(context),
                ptr: ptr.cast(),
                bytes: checked_bytes::<T>(length).expect("device memory byte length overflow"),
                deallocation: DeviceDeallocation::Immediate,
            }),
            ptr,
            length,
            _phantom: PhantomData,
        }
    }

    /// Transfers ownership to the caller when no other allocation owners remain.
    ///
    /// Returns the original buffer if submissions, graphs, or foreign users
    /// still retain its allocation. On success the caller owns the pointer and
    /// must retain the returned context until the allocation is freed.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(self) -> StdResult<(*mut T, usize, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr: device_ptr,
            length,
            ..
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(allocation) => {
                let allocation = ManuallyDrop::new(allocation);
                // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
                let context = unsafe { ptr::read(&raw const allocation.context) };
                Ok((device_ptr, length, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr: device_ptr,
                length,
                _phantom: PhantomData,
            }),
        }
    }

    /// Erases initialization guarantees while preserving the allocation and its context.
    #[must_use]
    pub fn into_uninit(self) -> DeviceMemory<MaybeUninit<T>> {
        let Self {
            allocation,
            ptr,
            length,
            ..
        } = self;
        DeviceMemory {
            allocation,
            ptr: ptr.cast(),
            length,
            _phantom: PhantomData,
        }
    }

    /// Allocates uninitialized device storage while retaining its context.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, context binding fails, or CUDA cannot allocate device memory.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
    ) -> Result<DeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { DeviceMemory::<MaybeUninit<T>>::create_unchecked(context, length) }
    }

    /// Allocates typed device storage for a raw producer to initialize.
    ///
    /// Prefer [`Self::create_uninit`] when initialization is represented in the buffer's element type.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid `T` before it is read.
    /// Raw producers and consumers must use the device ABI for `T` and must not transfer Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, context binding fails, or CUDA cannot allocate device memory.
    pub unsafe fn create_unchecked(context: &Arc<Context>, length: usize) -> Result<Self> {
        context.bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let device_ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { Self::alloc(length)? }
        };
        Ok(Self {
            allocation: Arc::new(DeviceAllocation {
                id: AllocationId::create(),
                context: Arc::clone(context),
                ptr: device_ptr.cast(),
                bytes,
                deallocation: DeviceDeallocation::Immediate,
            }),
            ptr: device_ptr,
            length,
            _phantom: PhantomData,
        })
    }

    /// Allocates device memory and completes zero initialization before returning.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, access-tracking, context-binding, CUDA fill, and default-stream synchronization errors.
    pub fn zeroed(context: &Arc<Context>, length: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let mut mem = unsafe { Self::create_unchecked(context, length)? };
        mem.set_zeroes()?;
        Ok(mem)
    }

    /// Allocates device storage and completes its repeated byte fill before returning.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, access-tracking, context-binding, CUDA fill, and default-stream synchronization errors.
    pub fn filled(context: &Arc<Context>, length: usize, value: u8) -> Result<Self>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let mut memory = unsafe { Self::create_unchecked(context, length)? };
        memory.set_value(value)?;
        Ok(memory)
    }

    /// Allocates device memory and completes the upload before returning.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn from_slice(context: &Arc<Context>, v: &[T]) -> Result<Self>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: DeviceRepr`, so copying the initialized host values yields valid device values.
        unsafe { Self::from_slice_unchecked(context, v) }
    }

    /// Uploads values with an externally established device ABI.
    ///
    /// This supports vendor ABI descriptors whose padding or embedded device addresses are not modeled by `DeviceRepr`.
    ///
    /// # Safety
    ///
    /// `T` must match the device ABI and must not transfer Rust ownership or host references.
    /// Embedded addresses must remain valid for every operation that consumes the uploaded descriptors.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn from_slice_unchecked(context: &Arc<Context>, v: &[T]) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        let mem = unsafe { Self::create_unchecked(context, v.len())? };
        if !v.is_empty() {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                transfer::copy_synchronous(
                    mem.ptr,
                    v.as_ptr(),
                    v.len(),
                    MemoryCopyKind::HostToDevice,
                    None,
                )?;
            }
        }
        Ok(mem)
    }

    /// # Safety
    ///
    /// The caller must ensure `v` remains valid and unmodified until `stream`
    /// has completed the transfer.
    /// The uploaded values must match the device ABI without transferring Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate device memory or enqueue the
    /// host-to-device copy.
    pub unsafe fn from_slice_async(v: &[T], stream: &Stream) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        let mut mem = unsafe { Self::create_unchecked(stream.context(), v.len())? };
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            mem.copy_from_host_async_unchecked(v, stream)?;
        }
        Ok(mem)
    }

    /// Returns the number of elements in the logical allocation range.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the logical allocation range contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the size of the logical allocation range in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the element count multiplied by the element size overflows `usize`. Allocation construction validates this size, so this condition cannot occur for a valid allocation.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        checked_bytes::<T>(self.length).expect("device memory byte length overflow")
    }

    /// Returns the allocation’s CUDA-accessible element pointer without synchronizing pending work.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the allocation’s CUDA-accessible mutable element pointer.
    ///
    /// The pointer does not establish execution ordering or permit conflicting accesses.
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Returns the allocation’s address as an untyped CUDA device pointer.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the context that owns this allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the shared owner retained by asynchronous submissions and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Preserves allocation ownership while exposing initialized representation bytes.
    #[must_use]
    pub fn into_bytes(self) -> DeviceMemory<u8>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let length = self.byte_len();
        DeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            _phantom: PhantomData,
        }
    }

    /// Reinterprets initialized storage without reallocating it.
    ///
    /// Returns the original buffer if alignment or byte length is incompatible.
    /// The original allocation owner and deallocation metadata are preserved.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<DeviceMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let Ok(length) = cast_length::<U>(self.region()) else {
            return Err(self);
        };
        Ok(DeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            _phantom: PhantomData,
        })
    }

    fn bind(&self) -> Result<()> {
        self.context().bind()
    }

    fn bind_stream(&self, stream: &Stream) -> Result<()> {
        if self.context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.bind()
    }

    /// Copies host values and waits until the device range contains the uploaded values.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow. Also propagates access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_from_host(&mut self, host_slice: &[T]) -> Result<()>
    where
        T: DeviceRepr,
    {
        transfer::copy_from_host(self, host_slice)
    }

    /// Copies from page-locked host memory into this device allocation.
    ///
    /// The source length must match this allocation's length. This is the
    /// pinned-memory-specific counterpart to [`DeviceMemory::copy_from_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows,
    /// or CUDA rejects the host-to-device copy.
    pub fn copy_from_pinned<B>(&mut self, src: &B) -> Result<()>
    where
        B: PinnedHostSlice<T> + ?Sized,
        T: DeviceRepr,
    {
        let context = Arc::clone(self.context());
        self.bind()?;
        let () = T::LAYOUT;
        if HostSlice::len(src) != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.length == 0 {
            return Ok(());
        }
        checked_bytes::<T>(self.length)?;
        let _access = context.acquire_accesses(&[
            Access::new(self.region(), AccessMode::Write),
            Access::host(src.host_region(), AccessMode::Read),
        ])?;
        // SAFETY: the guards acquired above cover both regions, and the lengths were checked to match.
        unsafe {
            transfer::copy_synchronous(
                self.ptr,
                src.as_host_ptr(),
                self.length,
                MemoryCopyKind::HostToDevice,
                None,
            )
        }
    }

    /// Enqueues a copy from page-locked host memory into this device allocation.
    ///
    /// The [`StreamScope`] ties the source and destination lifetimes to the
    /// queued copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows,
    /// or CUDA cannot enqueue the host-to-device copy.
    pub fn copy_from_pinned_async<'env, B>(
        &'env mut self,
        src: &'env B,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        B: PinnedHostSlice<T> + ?Sized,
        T: DeviceRepr,
    {
        self.bind_stream(stream.stream())?;
        let () = T::LAYOUT;
        if HostSlice::len(src) != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        checked_bytes::<T>(self.length)?;
        if self.length == 0 {
            return Ok(());
        }
        let accesses = [
            Access::new(self.region(), AccessMode::Write),
            Access::host(src.host_region(), AccessMode::Read),
        ];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    self.ptr,
                    src.as_host_ptr(),
                    self.length,
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a host-to-device copy of the entire allocation in a stream scope.
    ///
    /// The scope retains both borrows until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the element counts differ, the byte count overflows, or the scope’s stream belongs to another context. Also propagates context-binding, scope submission, and CUDA copy errors.
    pub fn copy_from_host_async<'env>(
        &'env mut self,
        host_slice: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        checked_bytes::<T>(self.len())?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    self.ptr,
                    host_slice.as_ptr(),
                    self.len(),
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until
    /// `stream` has completed the transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the host and device element counts differ, the stream belongs to another context, or the byte count overflows. Also propagates context-binding and CUDA asynchronous copy errors.
    pub unsafe fn copy_from_host_async_unchecked(
        &mut self,
        host_slice: &[T],
        stream: &Stream,
    ) -> Result<()> {
        self.bind_stream(stream)?;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::copy_async(
                self.as_mut_ptr(),
                host_slice.as_ptr(),
                self.len(),
                MemoryCopyKind::HostToDevice,
                stream,
            )
        }
    }

    /// Returns a capture operation that copies from host memory into this device allocation.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores the host and device pointer addresses in
    /// the resulting CUDA graph. The caller must ensure `self` and `host_slice`
    /// remain valid whenever a captured graph using this operation is launched.
    /// The destination allocation must remain exclusive for the work ordered by
    /// those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if the source and destination element counts differ.
    pub unsafe fn copy_from_host_operation<'a>(
        &'a mut self,
        host_slice: &'a [T],
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe {
            MemoryCopyOperation::new(
                self.as_mut_ptr(),
                host_slice.as_ptr(),
                self.len(),
                MemoryCopyKind::HostToDevice,
            )
        })
    }

    /// Copies all device elements into an equally sized host slice and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow. Also propagates access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host(&self, host_slice: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        transfer::copy_to_host(self, host_slice)
    }

    /// Copies device values to host storage under a kernel-specific validity contract.
    ///
    /// # Safety
    ///
    /// Every source element must be initialized and contain a valid value of `T`.
    /// `T` must be transferable without duplicating ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if host and device element counts differ or the byte count overflows. Also propagates context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn copy_to_host_unchecked(&self, host_slice: &mut [T]) -> Result<()> {
        self.bind()?;
        if host_slice.len() != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller establishes source value validity.
        unsafe {
            transfer::copy_synchronous(
                host_slice.as_mut_ptr(),
                self.ptr,
                self.length,
                MemoryCopyKind::DeviceToHost,
                None,
            )
        }
    }

    /// Copies this device allocation into page-locked host memory.
    ///
    /// The destination length must match this allocation's length. This is the
    /// pinned-memory-specific counterpart to [`DeviceMemory::copy_to_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows,
    /// or CUDA rejects the device-to-host copy.
    pub fn copy_to_pinned<B>(&self, dst: &mut B) -> Result<()>
    where
        B: PinnedHostSliceMut<T> + ?Sized,
        T: AnyBitPatternDeviceRepr,
    {
        let context = Arc::clone(self.context());
        self.bind()?;
        let () = T::LAYOUT;
        if HostSlice::len(dst) != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.length == 0 {
            return Ok(());
        }
        checked_bytes::<T>(self.length)?;
        let _access = context.acquire_accesses(&[
            Access::new(self.region(), AccessMode::Read),
            Access::host(dst.host_region(), AccessMode::Write),
        ])?;
        // SAFETY: the guards acquired above cover both regions, and the lengths were checked to match.
        unsafe {
            transfer::copy_synchronous(
                dst.as_host_mut_ptr(),
                self.ptr,
                self.length,
                MemoryCopyKind::DeviceToHost,
                None,
            )
        }
    }

    /// Enqueues a copy from this device allocation into page-locked host memory.
    ///
    /// The [`StreamScope`] ties the source and destination lifetimes to the
    /// queued copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows,
    /// or CUDA cannot enqueue the device-to-host copy.
    pub fn copy_to_pinned_async<'env, B>(
        &'env self,
        dst: &'env mut B,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        B: PinnedHostSliceMut<T> + ?Sized,
        T: AnyBitPatternDeviceRepr,
    {
        self.bind_stream(stream.stream())?;
        let () = T::LAYOUT;
        if HostSlice::len(dst) != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        checked_bytes::<T>(self.length)?;
        if self.length == 0 {
            return Ok(());
        }
        let accesses = [
            Access::new(self.region(), AccessMode::Read),
            Access::host(dst.host_region(), AccessMode::Write),
        ];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    dst.as_host_mut_ptr(),
                    self.ptr,
                    self.length,
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a device-to-host copy of the entire allocation in a stream scope.
    ///
    /// The scope retains the source and exclusive destination borrows until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the element counts differ, the byte count overflows, or the scope’s stream belongs to another context. Also propagates context-binding, scope submission, and CUDA copy errors.
    pub fn copy_to_host_async<'env>(
        &'env self,
        host_slice: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        checked_bytes::<T>(self.len())?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }
        let accesses = [Access::new(self.region(), AccessMode::Read)];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    host_slice.as_mut_ptr(),
                    self.ptr,
                    self.len(),
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until
    /// `stream` has completed the transfer.
    /// Every source element must be initialized and valid for `T`, without Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if the host and device element counts differ, the stream belongs to another context, or the byte count overflows. Also propagates context-binding and CUDA asynchronous copy errors.
    pub unsafe fn copy_to_host_async_unchecked(
        &self,
        host_slice: &mut [T],
        stream: &Stream,
    ) -> Result<()> {
        self.bind_stream(stream)?;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::copy_async(
                host_slice.as_mut_ptr(),
                self.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToHost,
                stream,
            )
        }
    }

    /// Returns a capture operation that copies this allocation into host memory.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores the device and host pointer addresses in
    /// the resulting CUDA graph. The caller must ensure `self` and `host_slice`
    /// remain valid whenever a captured graph using this operation is launched.
    /// The host destination must remain exclusive for the work ordered by those
    /// launches.
    ///
    /// # Errors
    ///
    /// Returns an error if the source and destination element counts differ.
    pub unsafe fn copy_to_host_operation<'a>(
        &'a self,
        host_slice: &'a mut [T],
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe {
            MemoryCopyOperation::new(
                host_slice.as_mut_ptr(),
                self.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToHost,
            )
        })
    }

    /// Copies all device elements into a new host vector and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates byte-size overflow, access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host_vec(&self) -> Result<Vec<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        transfer::copy_to_host_vec(self)
    }

    /// Copies initialized device values into a host vector under a kernel-specific validity contract.
    ///
    /// # Safety
    ///
    /// Every source element must contain a valid value of `T`.
    /// `T` must be transferable without duplicating ownership or host references.
    ///
    /// # Errors
    ///
    /// Propagates byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn copy_to_host_vec_unchecked(&self) -> Result<Vec<T>> {
        self.bind()?;
        // SAFETY: the caller establishes source value validity.
        unsafe { transfer::copy_to_host_vec_unchecked(self.ptr, self.length, None) }
    }

    /// Copies a disjoint device range into this allocation and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if contexts or element counts differ, the ranges overlap within one allocation, or the byte count overflows. Also propagates access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_from_device<S>(&mut self, source: &S) -> Result<()>
    where
        T: DeviceRepr,
        S: DeviceSlice<T> + ?Sized,
    {
        transfer::copy_device(self, source)
    }

    /// Enqueues a device-to-device copy of the entire allocation in a stream scope.
    ///
    /// The scope retains both borrows until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if source and destination contexts or element counts differ, the byte count overflows, or the scope’s stream belongs to another context. Also propagates context-binding, scope submission, and CUDA copy errors.
    pub fn copy_from_device_async<'env>(
        &'env mut self,
        src: &'env Self,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        if self.context() != src.context() {
            return Err(Error::MemoryContextMismatch);
        }
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        checked_bytes::<T>(self.len())?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }
        let accesses = [
            Access::new(self.region(), AccessMode::Write),
            Access::new(src.region(), AccessMode::Read),
        ];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    self.ptr,
                    src.as_ptr(),
                    self.len(),
                    MemoryCopyKind::DeviceToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` and `src` both remain valid until
    /// `stream` has completed the transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if source and destination contexts or element counts differ, the stream belongs to another context, or the byte count overflows. Also propagates context-binding and CUDA asynchronous copy errors.
    pub unsafe fn copy_from_device_async_unchecked(
        &mut self,
        src: &Self,
        stream: &Stream,
    ) -> Result<()> {
        if self.context() != src.context() {
            return Err(Error::MemoryContextMismatch);
        }
        self.bind_stream(stream)?;
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::copy_async(
                self.as_mut_ptr(),
                src.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToDevice,
                stream,
            )
        }
    }

    /// Returns a capture operation that copies from another device allocation into this allocation.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores both device pointer addresses in the
    /// resulting CUDA graph. The caller must ensure `self` and `src` remain
    /// valid whenever a captured graph using this operation is launched. The
    /// destination allocation must remain exclusive for the work ordered by
    /// those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if the source and destination element counts differ.
    pub unsafe fn copy_from_device_operation<'a>(
        &'a mut self,
        src: &'a Self,
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe {
            MemoryCopyOperation::new(
                self.as_mut_ptr(),
                src.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToDevice,
            )
        })
    }

    /// Fills every element with zero bytes and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.bind()?;
        let () = T::LAYOUT;
        if self.length == 0 {
            return Ok(());
        }
        // SAFETY: the allocation covers `length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            Self::set(self.ptr, 0, self.length)?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Fills every byte of the allocation with `value` and waits for completion.
    ///
    /// The byte pattern repeats across each element; `value` is not converted to `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub fn set_value(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so repeating `value` produces valid `T` values.
        unsafe { self.set_value_unchecked(value) }
    }

    /// Fills storage with a byte pattern known to be valid for `T`.
    ///
    /// # Safety
    ///
    /// The repeated byte pattern must represent a valid value of `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.bind()?;
        if self.length == 0 {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::set(self.ptr, value, self.length)?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a repeated-byte fill of the entire allocation in a stream scope.
    ///
    /// The scope retains the exclusive allocation borrow until the fill completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows or the scope’s stream belongs to another context. Also propagates context-binding, scope submission, and CUDA fill errors.
    pub fn set_value_async<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;

        checked_bytes::<T>(self.len())?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }
        let accesses = [Access::new(self.region(), AccessMode::Write)];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::set_async(self.ptr, value, self.len(), stream.stream())
            })
        }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` remains valid until `stream` has
    /// completed the memset.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the memset on `stream`.
    pub unsafe fn set_value_async_unchecked(&mut self, value: u8, stream: &Stream) -> Result<()> {
        self.bind_stream(stream)?;
        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::set_async(self.as_mut_ptr(), value, self.len(), stream) }
    }

    /// Returns a capture operation that fills this device allocation with `value`.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores this allocation's pointer address in the
    /// resulting CUDA graph. The caller must ensure `self` remains valid and
    /// exclusive whenever a captured graph using this operation is launched.
    pub unsafe fn set_value_operation(&mut self, value: u8) -> MemorySetOperation<'_, T> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { MemorySetOperation::new(self.as_mut_ptr(), value, self.len()) }
    }

    /// Takes a pointer to the base of an existing device memory allocation created with [`DeviceMemory::alloc`] and exports it for use in another process.
    /// This is a lightweight operation and may be called multiple times on an allocation without adverse effects.
    ///
    /// If a region of memory is freed with [`DeviceMemory::free`] and a subsequent call to [`DeviceMemory::alloc`] returns memory with the same device address, [`DeviceMemory::ipc_handle`] returns a unique handle for the new memory.
    ///
    /// IPC is restricted to devices with unified-addressing support on Linux and Windows.
    /// IPC on Windows is supported for compatibility but is not recommended because of its performance cost.
    /// Check device IPC support through the device properties exposed by this crate, for example [`DeviceProperties::ipc_event_supported`](crate::device::DeviceProperties::ipc_event_supported).
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation is empty, CUDA cannot export an IPC
    /// handle for the allocation, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn ipc_handle(&self) -> Result<IpcMemoryHandle> {
        self.bind()?;
        if self.is_empty() {
            // Cannot get handle for null pointer / zero size? Check docs.
            return Err(Error::InvalidMemoryAccess);
        }
        let mut handle = MaybeUninit::uninit();
        // SAFETY: `cudaIpcGetMemHandle` writes only the local handle for this live allocation.
        unsafe {
            try_ffi!(runtime::cudaIpcGetMemHandle(
                handle.as_mut_ptr(),
                self.as_ptr().cast_mut().cast(),
            ))?;
            Ok(IpcMemoryHandle::from_raw(handle.assume_init()))
        }
    }

    /// Allocates independent device storage and copies all elements into it.
    ///
    /// Returns after the device-to-device copy completes.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    pub fn try_clone(&self) -> Result<Self> {
        // SAFETY: the memory is fully initialized below before it is returned.
        let new_mem = unsafe { Self::create_unchecked(self.context(), self.length)? };
        if self.length == 0 || size_of::<T>() == 0 {
            return Ok(new_mem);
        }

        // SAFETY: the guards acquired above cover both regions, and the lengths were checked to match.
        unsafe {
            transfer::copy_synchronous(
                new_mem.as_mut_ptr(),
                self.as_ptr(),
                self.length,
                MemoryCopyKind::DeviceToDevice,
                None,
            )?;
        }

        Ok(new_mem)
    }
}

impl<T> DeviceMemory<MaybeUninit<T>> {
    /// Exposes an initialized prefix while retaining the complete allocation.
    ///
    /// # Safety
    ///
    /// The first `length` elements must contain valid initialized values of
    /// `T`. Producing work must be ordered before every consumer and host read.
    /// The remainder stays allocated but is no longer exposed as typed elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<DeviceMemory<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Converts storage initialized by a device producer into initialized device memory.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid, initialized `T` before any consumer reads it.
    /// Pending initialization must be ordered before every subsequent consumer and host readback.
    #[must_use]
    pub unsafe fn assume_init(self) -> DeviceMemory<T> {
        let Self {
            allocation,
            ptr,
            length,
            ..
        } = self;
        DeviceMemory {
            allocation,
            ptr: ptr.cast(),
            length,
            _phantom: PhantomData,
        }
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

impl<T: DeviceRepr> ManagedMemory<T> {
    /// Allocates CUDA managed memory under a caller-established initialization contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot
    /// allocate managed memory, or a non-empty zero-sized allocation is
    /// requested.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid T before host or device reads.
    pub unsafe fn create_unchecked(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        context.bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { DeviceMemory::<T>::alloc_managed(length, attach_flags)? }
        };
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            length,
            attach_flags,
            _phantom: PhantomData,
        })
    }

    /// Allocates CUDA managed memory and completes zero initialization.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, access-tracking, context-binding, CUDA fill, and synchronization errors.
    pub fn zeroed(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        // SAFETY: the memory is fully initialized below before it is returned.
        let mut memory = unsafe { Self::create_unchecked(context, length, attach_flags)? };
        memory.set_zeroes()?;
        Ok(memory)
    }

    /// Takes ownership of an existing managed allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be null for an empty allocation or identify live managed storage
    /// for `length` elements in `context`, allocated with `cudaMallocManaged`.
    /// The attachment flags must describe that allocation's policy.
    /// No other owner may free it, and each element must be initialized before it is read.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length overflows or a nonempty allocation has a null pointer.
    pub unsafe fn from_raw_parts(
        context: &Arc<Context>,
        ptr: *mut T,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<Self> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            allocation: DeviceAllocation::from_parts(
                context,
                ptr.cast(),
                bytes,
                DeviceDeallocation::Immediate,
            ),
            ptr,
            length,
            attach_flags,
            _phantom: PhantomData,
        })
    }

    /// Transfers the allocation when no submission, graph, or foreign owner retains it.
    ///
    /// Returns the unchanged buffer if the allocation is shared.
    /// On success the caller must free the pointer and keep its returned context alive until then.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(
        self,
    ) -> StdResult<(*mut T, usize, MemoryAttachFlags, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr,
            length,
            attach_flags,
            ..
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                allocation.ptr = ptr::null_mut();
                Ok((ptr, length, attach_flags, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
                length,
                attach_flags,
                _phantom: PhantomData,
            }),
        }
    }

    /// Returns the CUDA context retained by this allocation or pool.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the allocation owner retained by submissions, graphs, and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Reinterprets initialized elements while preserving managed allocation ownership and attachment flags.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<ManagedMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let Ok(length) = cast_length::<U>(self.region()) else {
            return Err(self);
        };
        Ok(ManagedMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            attach_flags: self.attach_flags,
            _phantom: PhantomData,
        })
    }

    /// Erases initialization while retaining the managed allocation and its attachment flags.
    #[must_use]
    pub fn into_uninit(self) -> ManagedMemory<MaybeUninit<T>> {
        ManagedMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            attach_flags: self.attach_flags,
            _phantom: PhantomData,
        }
    }

    fn bind_stream(&self, stream: &Stream) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.context().bind()
    }

    /// Allocates CUDA managed memory without initializing its elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, the attachment policy is unsupported, context binding fails, or CUDA cannot allocate managed memory.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Result<ManagedMemory<MaybeUninit<T>>> {
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { ManagedMemory::<MaybeUninit<T>>::create_unchecked(context, length, attach_flags) }
    }

    /// Returns the number of elements in the logical allocation range.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the logical allocation range contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the size of the logical allocation range in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the element count multiplied by the element size overflows `usize`. Allocation construction validates this size, so this condition cannot occur for a valid allocation.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.checked_byte_len()
            .expect("managed memory byte length overflow")
    }

    fn checked_byte_len(&self) -> Result<usize> {
        checked_bytes::<T>(self.length)
    }

    /// Returns the attachment policy recorded when the managed allocation was created.
    #[must_use]
    pub const fn attach_flags(&self) -> MemoryAttachFlags {
        self.attach_flags
    }

    /// Returns the allocation’s CUDA-accessible element pointer without synchronizing pending work.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the allocation’s CUDA-accessible mutable element pointer.
    ///
    /// The pointer does not establish execution ordering or permit conflicting accesses.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Borrows all initialized elements as a shared device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView::from_buffer(self)
    }

    /// Borrows all initialized elements as an exclusive device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::from_buffer(self)
    }

    /// Completes submitted accesses and excludes tracked GPU work while reading managed values.
    /// Requires [concurrent managed access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#unified-memory-on-windows-wsl-and-tegra) because unrelated streams may remain active.
    ///
    /// # Errors
    ///
    /// Returns `Error::ManagedHostAccessUnsupported` if the device lacks concurrent managed access. Also propagates device-attribute query, access-tracking, and CUDA completion errors before invoking the callback.
    pub fn with_host_slice<R>(&self, read: impl FnOnce(&[T]) -> R) -> Result<R> {
        if self
            .context()
            .device()
            .attribute(DeviceAttribute::ConcurrentManagedAccess)?
            == 0
        {
            return Err(Error::ManagedHostAccessUnsupported);
        }
        self.context().with_host_access(&[self.region()], || {
            // SAFETY: the shared borrow and tracked access guard exclude conflicting access.
            Ok(read(unsafe { self.as_host_slice() }))
        })
    }

    /// Completes submitted accesses and excludes tracked GPU work while modifying managed values.
    /// Requires concurrent managed access support because unrelated streams may remain active.
    ///
    /// # Errors
    ///
    /// Returns `Error::ManagedHostAccessUnsupported` if the device lacks concurrent managed access. Also propagates device-attribute query, access-tracking, and CUDA completion errors before invoking the callback.
    pub fn with_host_slice_mut<R>(&mut self, write: impl FnOnce(&mut [T]) -> R) -> Result<R> {
        if self
            .context()
            .device()
            .attribute(DeviceAttribute::ConcurrentManagedAccess)?
            == 0
        {
            return Err(Error::ManagedHostAccessUnsupported);
        }
        let context = Arc::clone(self.context());
        let _guard = context.acquire_host_access(&[self.region()])?;
        // SAFETY: the mutable borrow and tracked access guard exclude device access.
        Ok(write(unsafe { self.as_host_slice_mut() }))
    }

    /// Returns a host slice over this managed allocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure no GPU work or other CPU reference can
    /// concurrently mutate the same memory for the returned lifetime, and that
    /// the allocation is accessible from the host at the point of access.
    #[must_use]
    pub unsafe fn as_host_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { slice::from_raw_parts(self.ptr, self.length) }
    }

    /// Returns a mutable host slice over this managed allocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure no GPU work or other CPU reference can
    /// concurrently access the same memory for the returned lifetime, and that
    /// the allocation is accessible from the host at the point of access.
    pub unsafe fn as_host_slice_mut(&mut self) -> &mut [T] {
        if self.is_empty() {
            return &mut [];
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.length) }
    }

    /// Fills every element with zero bytes and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.context().bind()?;
        let () = T::LAYOUT;
        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the allocation covers `length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            DeviceMemory::<T>::set(self.ptr, 0, self.length)?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a residency prefetch of the entire managed range to `location`.
    ///
    /// The stream orders the request; completion is asynchronous.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another context, byte-size calculation overflows, context binding fails, or CUDA rejects the prefetch request or destination.
    pub fn prefetch_to(&self, location: MemoryLocation, stream: &Stream) -> Result<()> {
        self.bind_stream(stream)?;
        DeviceMemory::<T>::prefetch_async(
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            unsafe { DevicePtr::from_raw(self.ptr.cast::<()>()) },
            self.checked_byte_len()?,
            location,
            stream,
        )
    }

    /// Applies a managed-memory advice policy to this allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice or location.
    pub fn advise(&self, advice: MemoryAdvise, location: MemoryLocation) -> Result<()> {
        self.context().bind()?;
        DeviceMemory::<T>::advise(
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            unsafe { DevicePtr::from_raw(self.ptr.cast::<()>()) },
            self.checked_byte_len()?,
            advice,
            location,
        )
    }

    /// Advises CUDA that this managed allocation is mostly read.
    ///
    /// CUDA may create read-only copies of pages on processors that read this
    /// range. Writes remain coherent but can require additional migration or
    /// invalidation work.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice for this allocation.
    pub fn set_read_mostly(&self) -> Result<()> {
        self.advise(MemoryAdvise::SetReadMostly, Self::ignored_advice_location())
    }

    /// Removes explicit read-mostly advice from this managed allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice for this allocation.
    pub fn unset_read_mostly(&self) -> Result<()> {
        self.advise(
            MemoryAdvise::UnsetReadMostly,
            Self::ignored_advice_location(),
        )
    }

    /// Advises CUDA to prefer placing this managed allocation at `location`.
    ///
    /// This guides migration policy but does not immediately migrate pages.
    /// Use [`ManagedMemory::prefetch_to`] when an explicit migration is desired.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the location or advice for this
    /// allocation.
    pub fn set_preferred_location(&self, location: MemoryLocation) -> Result<()> {
        self.advise(MemoryAdvise::SetPreferredLocation, location)
    }

    /// Clears the preferred-location advice for this managed allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice for this allocation.
    pub fn unset_preferred_location(&self) -> Result<()> {
        self.advise(
            MemoryAdvise::UnsetPreferredLocation,
            Self::ignored_advice_location(),
        )
    }

    /// Advises CUDA that `location` will access this managed allocation.
    ///
    /// This can establish or maintain mappings for the target processor without
    /// migrating the data there.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the location or advice for this
    /// allocation.
    pub fn set_accessed_by(&self, location: MemoryLocation) -> Result<()> {
        self.advise(MemoryAdvise::SetAccessedBy, location)
    }

    /// Clears accessed-by advice for `location`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the location or advice for this
    /// allocation.
    pub fn unset_accessed_by(&self, location: MemoryLocation) -> Result<()> {
        self.advise(MemoryAdvise::UnsetAccessedBy, location)
    }

    const fn ignored_advice_location() -> MemoryLocation {
        MemoryLocation::Host
    }

    /// Attaches this managed allocation to `stream` with the requested attach flags.
    ///
    /// This enqueues `cudaStreamAttachMemAsync` and updates the stored attach
    /// flags after CUDA accepts the request.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot attach the allocation to `stream`, the
    /// flags are invalid for the allocation, or a previous asynchronous launch
    /// reports an error.
    pub fn attach_to_stream(&mut self, stream: &Stream, flags: MemoryAttachFlags) -> Result<()> {
        self.bind_stream(stream)?;
        // SAFETY: `cudaStreamAttachMemAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaStreamAttachMemAsync(
                stream.as_raw(),
                self.ptr.cast(),
                self.checked_byte_len()? as _,
                flags.bits(),
            ))?;
        }
        self.attach_flags = flags;
        Ok(())
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for ManagedMemory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for ManagedMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for ManagedMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for ManagedMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let offset = self.ptr as usize - self.allocation.ptr as usize;
        self.allocation
            .region()
            .slice(offset..offset + self.byte_len())
            .expect("managed buffer exceeds its allocation")
    }
}

impl<T: DeviceRepr> ManagedMemory<MaybeUninit<T>> {
    /// Borrows this managed allocation as a write-only output of `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Exposes an initialized prefix while retaining the full managed allocation.
    ///
    /// # Safety
    ///
    /// The prefix must contain valid initialized `T` values and all producing work must have completed.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<ManagedMemory<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks every element initialized after its producer has completed.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid T and all producer writes must have completed.
    #[must_use]
    pub unsafe fn assume_init(self) -> ManagedMemory<T> {
        let () = T::LAYOUT;
        ManagedMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
            attach_flags: self.attach_flags,
            _phantom: PhantomData,
        }
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

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::{error::Status, testing};

    #[test]
    fn test_it_works() -> Result<()> {
        // SAFETY: the test buffers outlive the synchronous copies in this block.
        unsafe {
            let host_in = [1, 2, 3];

            let device_ptr = DeviceMemory::alloc(3)?;
            let attributes = PointerAttributes::query_non_null(NonNull::new(device_ptr).unwrap())?;
            assert_eq!(attributes.memory_type, MemoryType::Device);

            DeviceMemory::copy(
                device_ptr,
                host_in.as_ptr(),
                3,
                MemoryCopyKind::HostToDevice,
            )?;
            let mut host_out = [0, 0, 0];
            DeviceMemory::copy(
                host_out.as_mut_ptr(),
                device_ptr,
                3,
                MemoryCopyKind::DeviceToHost,
            )?;
            assert_eq!(host_out, host_in);

            DeviceMemory::free(device_ptr)?;
        }
        Ok(())
    }

    #[test]
    fn test_scoped_async_copy_round_trip() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let host_in = [4_i32, 5, 6];
        // SAFETY: the buffer is fully written by the operation before it is read.
        let mut device = unsafe { DeviceMemory::create_unchecked(&context, host_in.len()) }?;
        let mut host_out = [0_i32; 3];

        stream.sync_scope(|scope| device.copy_from_host_async(&host_in, scope))?;
        stream.sync_scope(|scope| device.copy_to_host_async(&mut host_out, scope))?;

        assert_eq!(host_out, host_in);

        Ok(())
    }

    #[test]
    fn test_async_device_memory_frees_on_stream() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let mut memory = AsyncDeviceMemory::<u32>::zeroed(8, &stream)?;
        assert_eq!(memory.len(), 8);
        assert_eq!(memory.byte_len(), size_of::<[u32; 8]>());
        memory.set_value_async(0)?;
        memory.free_async()?;
        stream.synchronize()?;

        Ok(())
    }

    #[test]
    fn test_async_device_memory_allocates_from_pool() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;
        let props = MemoryPoolProps {
            alloc_type: MemoryAllocationType::Pinned,
            handle_type: MemoryAllocationHandleType::None,
            location: MemoryLocation::Device(context.device().id()),
            max_size: 0,
            usage: MemoryPoolUsage::empty(),
            gpu_direct_rdma_capable: false,
        };
        let pool = match MemoryPool::create(&context, props) {
            Ok(pool) => pool,
            Err(Error::Cuda {
                code: Status::NotSupported,
                ..
            }) => return Ok(()),
            Err(error) => return Err(error),
        };

        let memory = AsyncDeviceMemory::<u32>::zeroed_in_pool(8, &pool, &stream)?;
        assert_eq!(memory.len(), 8);
        assert_eq!(memory.stream().as_raw(), stream.as_raw());
        memory.free_async()?;
        stream.synchronize()?;

        Ok(())
    }

    #[test]
    fn test_pinned_host_memory_from_slice_and_zeroes() -> Result<()> {
        let (_lock, _ctx) = testing::bootstrap()?;

        let host = PinnedHostMemory::from_slice(&_ctx, &[1_i32, 2, 3])?;
        assert_eq!(host.as_slice(), &[1, 2, 3]);
        assert_eq!(host.len(), 3);
        assert_eq!(host.byte_len()?, 3 * size_of::<i32>());

        let zeroes = PinnedHostMemory::<u32>::zeroed(&_ctx, 4, HostAllocationFlags::DEFAULT)?;
        assert_eq!(zeroes.as_slice(), &[0, 0, 0, 0]);

        Ok(())
    }

    #[test]
    fn test_pinned_host_try_clone_and_raw_round_trip() -> Result<()> {
        let (_lock, _ctx) = testing::bootstrap()?;

        let host = PinnedHostMemory::from_slice_with_flags(
            &_ctx,
            &[1_u32, 2, 3],
            HostAllocationFlags::PORTABLE,
        )?;
        let cloned = host.try_clone()?;
        assert_eq!(cloned.as_slice(), host.as_slice());
        assert_eq!(cloned.flags(), host.flags());

        let (ptr, len, flags, allocation_context) = host.into_raw_parts().unwrap();
        // SAFETY: the pointer and length come from the allocation that the test just made.
        let rebuilt =
            unsafe { PinnedHostMemory::from_raw_parts(&allocation_context, ptr, len, flags) }?;
        assert_eq!(rebuilt.as_slice(), &[1, 2, 3]);
        assert_eq!(rebuilt.flags(), HostAllocationFlags::PORTABLE);

        Ok(())
    }

    #[test]
    fn test_mapped_pinned_host_memory_reports_device_alias() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        if !context.device().properties()?.can_map_host_memory {
            return Ok(());
        }

        let mut host = PinnedHostMemory::from_slice_with_flags(
            &context,
            &[5_u32, 6, 7, 8],
            HostAllocationFlags::MAPPED,
        )?;
        let flags = match host.host_flags() {
            Ok(flags) => flags,
            Err(Error::Cuda {
                code: Status::NotSupported | Status::InvalidValue,
                ..
            }) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert!(flags.contains(HostAllocationFlags::MAPPED));

        let mapped = match host.mapped_device_ptr() {
            Ok(mapped) => mapped,
            Err(Error::Cuda {
                code: Status::NotSupported | Status::InvalidValue,
                ..
            }) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert!(!mapped.is_null());
        assert_eq!(host.mapped_device_view()?.len(), host.len());
        assert_eq!(host.mapped_device_view_mut()?.len(), host.len());

        Ok(())
    }

    #[test]
    fn test_pinned_host_copy_round_trip() -> Result<()> {
        let (_lock, _ctx) = testing::bootstrap()?;

        let host_in = PinnedHostMemory::from_slice(&_ctx, &[7_i32, 8, 9])?;
        // SAFETY: the buffer is fully written by the operation before it is read.
        let mut device = unsafe { DeviceMemory::create_unchecked(&_ctx, host_in.len()) }?;
        device.copy_from_pinned(&host_in)?;

        let mut host_out =
            PinnedHostMemory::<i32>::zeroed(&_ctx, host_in.len(), HostAllocationFlags::DEFAULT)?;
        device.copy_to_pinned(&mut host_out)?;

        assert_eq!(host_out.as_slice(), host_in.as_slice());

        Ok(())
    }

    #[test]
    fn test_pinned_host_scoped_async_copy_round_trip() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let host_in = PinnedHostMemory::from_slice(&context, &[10_i32, 11, 12])?;
        // SAFETY: the buffer is fully written by the operation before it is read.
        let mut device = unsafe { DeviceMemory::create_unchecked(&context, host_in.len()) }?;
        let mut host_out =
            PinnedHostMemory::<i32>::zeroed(&context, host_in.len(), HostAllocationFlags::DEFAULT)?;

        stream.sync_scope(|scope| device.copy_from_pinned_async(&host_in, scope))?;
        stream.sync_scope(|scope| device.copy_to_pinned_async(&mut host_out, scope))?;

        assert_eq!(host_out.as_slice(), host_in.as_slice());

        Ok(())
    }

    #[test]
    fn test_pitched_device_memory_copy_round_trip() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let host_in = vec![1_u32, 2, 3, 4, 5, 6];
        let mut pitched = PitchedDeviceMemory::from_slice(&context, 3, 2, &host_in)?;
        assert_eq!(pitched.width(), 3);
        assert_eq!(pitched.height(), 2);
        assert!(pitched.pitch() >= 3 * size_of::<u32>());
        pitched.copy_from_host(&host_in)?;

        let mut copied = PitchedDeviceMemory::zeroed(&context, 3, 2)?;
        copied.copy_from_pitched(&pitched)?;

        let mut host_out = vec![0_u32; host_in.len()];
        copied.copy_to_host(&mut host_out)?;
        assert_eq!(host_out, host_in);

        Ok(())
    }

    #[test]
    fn test_pitched_device_memory_scoped_async_copy_round_trip() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let host_in = vec![10_u32, 11, 12, 13, 14, 15];
        let mut pitched = PitchedDeviceMemory::zeroed(&context, 2, 3)?;
        let mut host_out = vec![0_u32; host_in.len()];

        stream.sync_scope(|scope| pitched.copy_from_host_async(&host_in, scope))?;
        stream.sync_scope(|scope| pitched.copy_to_host_async(&mut host_out, scope))?;

        assert_eq!(host_out, host_in);

        Ok(())
    }

    #[test]
    fn test_pitched_device_memory_allows_empty_shapes() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let mut pitched = PitchedDeviceMemory::<u32>::zeroed(&context, 0, 8)?;
        assert!(pitched.is_empty());
        assert_eq!(pitched.row_byte_width()?, 0);
        assert_eq!(pitched.byte_len()?, 0);
        pitched.copy_from_host(&[])?;
        pitched.copy_to_host(&mut [])?;

        Ok(())
    }

    #[test]
    fn test_pitched_device_volume_copy_round_trip() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let host_in: Vec<u32> = (0..24).collect();
        let mut volume = PitchedDeviceVolume::from_slice(&context, 4, 3, 2, &host_in)?;
        assert_eq!(volume.width(), 4);
        assert_eq!(volume.height(), 3);
        assert_eq!(volume.depth(), 2);
        assert!(volume.pitch() >= 4 * size_of::<u32>());
        volume.copy_from_host(&host_in)?;

        let mut copied = PitchedDeviceVolume::zeroed(&context, 4, 3, 2)?;
        copied.copy_from_volume(&volume)?;

        let mut host_out = vec![0_u32; host_in.len()];
        copied.copy_to_host(&mut host_out)?;
        assert_eq!(host_out, host_in);

        Ok(())
    }

    #[test]
    fn test_pitched_device_volume_scoped_async_copy_round_trip() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let host_in: Vec<u32> = (100..112).collect();
        let mut volume = PitchedDeviceVolume::zeroed(&context, 3, 2, 2)?;
        let mut host_out = vec![0_u32; host_in.len()];

        stream.sync_scope(|scope| volume.copy_from_host_async(&host_in, scope))?;
        stream.sync_scope(|scope| volume.copy_to_host_async(&mut host_out, scope))?;

        assert_eq!(host_out, host_in);

        Ok(())
    }

    #[test]
    fn test_pitched_device_volume_allows_empty_shapes() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let mut volume = PitchedDeviceVolume::<u32>::zeroed(&context, 4, 0, 2)?;
        assert!(volume.is_empty());
        assert_eq!(volume.row_byte_width()?, 4 * size_of::<u32>());
        assert_eq!(volume.byte_len()?, 0);
        volume.copy_from_host(&[])?;
        volume.copy_to_host(&mut [])?;

        Ok(())
    }

    #[test]
    fn test_registered_host_empty_slices_are_noops() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let mut values: Vec<u32> = Vec::new();
        let registered = RegisteredHostMemory::create(&context, &mut values)?;
        assert!(registered.is_empty());
        assert_eq!(registered.byte_len()?, 0);

        let readonly = RegisteredHostSlice::create_read_only(&context, &values)?;
        assert!(readonly.is_empty());
        assert_eq!(readonly.byte_len()?, 0);
        assert!(readonly.flags().contains(HostRegisterFlags::READ_ONLY));

        let registered_vec = RegisteredHostVec::<u32>::from_slice(&context, &[])?;
        assert!(registered_vec.is_empty());
        assert_eq!(registered_vec.byte_len()?, 0);
        assert!(matches!(
            RegisteredHostVec::from_slice_with_flags(
                &context,
                &[1_u32],
                HostRegisterFlags::READ_ONLY
            ),
            Err(Error::InvalidValue)
        ));

        Ok(())
    }

    #[test]
    fn test_registered_host_memory_tracks_borrow() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        if !context.device().properties()?.host_register_supported {
            return Ok(());
        }

        let mut values = vec![1_u32, 2, 3, 4];
        {
            let mut registered = match RegisteredHostMemory::create(&context, &mut values) {
                Ok(registered) => registered,
                Err(Error::Cuda {
                    code: Status::NotSupported,
                    ..
                }) => return Ok(()),
                Err(error) => return Err(error),
            };
            assert_eq!(registered.as_slice(), &[1, 2, 3, 4]);
            registered.as_slice_mut()[1] = 7;
            assert_eq!(registered.byte_len()?, 4 * size_of::<u32>());
        }
        assert_eq!(values, vec![1, 7, 3, 4]);

        let read_only = match RegisteredHostSlice::create_read_only(&context, &values) {
            Ok(read_only) => read_only,
            Err(Error::Cuda {
                code: Status::NotSupported,
                ..
            }) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(read_only.as_slice(), values.as_slice());
        assert!(read_only.flags().contains(HostRegisterFlags::READ_ONLY));

        let registered_vec = match RegisteredHostVec::from_slice(&context, &values) {
            Ok(registered_vec) => registered_vec,
            Err(Error::Cuda {
                code: Status::NotSupported,
                ..
            }) => return Ok(()),
            Err(error) => return Err(error),
        };
        assert_eq!(registered_vec.as_slice(), values.as_slice());

        Ok(())
    }

    #[test]
    fn test_managed_memory_advise_uses_typed_wrapper() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let properties = context.device().properties()?;
        if !properties.managed_memory {
            return Ok(());
        }

        let managed = ManagedMemory::<u32>::zeroed(&context, 4, MemoryAttachFlags::GLOBAL)?;
        let host = MemoryLocation::Host;

        for result in [
            managed.set_read_mostly(),
            managed.unset_read_mostly(),
            managed.set_preferred_location(host),
            managed.unset_preferred_location(),
        ] {
            match result {
                Ok(()) => {}
                Err(Error::Cuda {
                    code: Status::NotSupported,
                    ..
                }) => return Ok(()),
                Err(error) => return Err(error),
            }
        }

        if properties.concurrent_managed_access {
            let device = MemoryLocation::Device(context.device().id());
            managed.set_accessed_by(device)?;
            managed.unset_accessed_by(device)?;
        }

        Ok(())
    }

    #[test]
    fn test_pinned_host_rejects_non_empty_zst() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        #[derive(Clone, Copy)]
        struct Zst;

        unsafe impl DeviceRepr for Zst {}

        unsafe impl ZeroableDeviceRepr for Zst {}

        assert!(matches!(
            PinnedHostMemory::<Zst>::from_slice(&context, &[Zst]),
            Err(Error::InvalidMemoryAllocationRequest)
        ));
        assert!(matches!(
            PinnedHostMemory::<Zst>::zeroed(&context, 1, HostAllocationFlags::DEFAULT),
            Err(Error::InvalidMemoryAllocationRequest)
        ));
        Ok(())
    }

    #[test]
    fn test_managed_memory_tracks_metadata_and_views() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        if !context.device().properties()?.managed_memory {
            return Ok(());
        }
        let mut managed = ManagedMemory::<u32>::zeroed(&context, 4, MemoryAttachFlags::HOST)?;
        // SAFETY: the test synchronizes the device before touching managed memory from the host.
        unsafe {
            managed.as_host_slice_mut().copy_from_slice(&[1, 2, 3, 4]);
        }
        assert_eq!(managed.len(), 4);
        assert_eq!(managed.byte_len(), 4 * size_of::<u32>());
        assert_eq!(managed.attach_flags(), MemoryAttachFlags::HOST);
        assert_eq!(managed.view().len(), 4);
        assert_eq!(managed.view_mut().len(), 4);
        // SAFETY: the test synchronizes the device before touching managed memory from the host.
        unsafe {
            assert_eq!(managed.as_host_slice(), &[1, 2, 3, 4]);
            managed.as_host_slice_mut()[2] = 9;
            assert_eq!(managed.as_host_slice()[2], 9);
        }
        let address = managed.as_ptr();
        let (ptr, length, flags, allocation_context) = managed.into_raw_parts().unwrap();
        assert_eq!(ptr.cast_const(), address);
        assert_eq!(length, 4);
        assert_eq!(flags, MemoryAttachFlags::HOST);
        // SAFETY: reconstruct the unique owner using its original allocation and context.
        let _managed =
            unsafe { ManagedMemory::from_raw_parts(&allocation_context, ptr, length, flags) }?;
        Ok(())
    }
}

/// Retrieves memory location information for an address range.
///
/// Returns the most common location in each summary region, sampling at the
/// specified interval. A zero sampling granularity uses the system default.
///
/// # Safety
/// The range must remain allocated throughout this call and refer to managed
/// memory or system-allocated pageable memory not registered with CUDA.
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

impl<T: DeviceRepr> ManagedMemory<T> {
    /// Gets residency information for this memory range.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid summary or sampling granularity or byte-size overflow. Also propagates context-binding, driver entry-point lookup, and CUDA location-query errors.
    pub fn location_info(
        &self,
        summary_granularity: usize,
        sampling_granularity: usize,
    ) -> Result<Vec<MemoryLocation>> {
        self.context().bind()?;
        // SAFETY: the pointer is the live managed allocation owned by `self`.
        unsafe {
            location_info(
                DevicePtr::from_raw(self.ptr.cast()),
                self.checked_byte_len()?,
                summary_granularity,
                sampling_granularity,
            )
        }
    }
}

/// Integer types that memory pool attributes are read into; every bit pattern is valid.
trait PoolAttributeValue: Copy + Default {}

impl PoolAttributeValue for i32 {}

impl PoolAttributeValue for u32 {}

impl PoolAttributeValue for u64 {}

impl MemoryPool {
    /// Returns the locality domain ID of a localized pool, or `None` for a full-device pool.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors. Also rejects an unexpected negative locality-domain identifier other than the full-device sentinel.
    pub fn locality_domain_id(&self) -> Result<Option<u32>> {
        let location_type = self.location_type()?;
        if location_type != driver::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN {
            return Ok(None);
        }
        let value: i32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_LOCALITY_DOMAIN_ID)?;
        // CUDA normalizes a localized pool to a full-device pool on single-domain devices.
        if value == -1 {
            return Ok(None);
        }
        let value = u32::try_from(value).map_err(|_| Error::out_of_range("locality domain id"))?;
        Ok(Some(value))
    }

    /// Returns the allocation type of the memory pool.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors. Also rejects an unknown allocation-type value.
    pub fn allocation_type(&self) -> Result<MemoryAllocationType> {
        let value: u32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_ALLOCATION_TYPE)?;
        MemoryAllocationType::try_from(value).map_err(|_| Error::InvalidValue)
    }

    /// Returns the location of the memory pool.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors. Also rejects unknown location types and identifiers that do not fit the localized device representation.
    pub fn location(&self) -> Result<MemoryLocation> {
        let kind = self.location_type()?;
        let id: i32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_LOCATION_ID)?;
        if let Some(locality_domain_id) = self.locality_domain_id()? {
            return Ok(MemoryLocation::DeviceLocalityDomain {
                device_id: id.try_into().map_err(|_| Error::InvalidValue)?,
                locality_domain_id: locality_domain_id
                    .try_into()
                    .map_err(|_| Error::InvalidValue)?,
            });
        }
        Ok(driver::CUmemLocation {
            type_: kind,
            __bindgen_anon_1: driver::CUmemLocation_st__bindgen_ty_1 { id },
        }
        .into())
    }

    /// Maximum size of the pool in bytes, including alignment requirements.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors.
    pub fn maximum_size(&self) -> Result<u64> {
        self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_MAX_POOL_SIZE)
    }

    /// Indicates whether hardware decompression is enabled for allocations from this pool.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors.
    pub fn hardware_decompression_enabled(&self) -> Result<bool> {
        let value: i32 = self
            .read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED)?;
        Ok(value != 0)
    }

    /// Available export handle types for the memory pool.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors.
    pub fn export_handle_types(&self) -> Result<AllocationHandleTypes> {
        let value: u32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES)?;
        Ok(AllocationHandleTypes::from_bits_retain(value))
    }

    fn location_type(&self) -> Result<driver::CUmemLocationType> {
        let value: u32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_LOCATION_TYPE)?;
        driver::CUmemLocationType::try_from(value).map_err(|_| Error::InvalidValue)
    }

    /// Reads an integer-valued pool attribute; enum attributes are read as `u32`.
    fn read_attribute<T: PoolAttributeValue>(
        &self,
        attribute: driver::CUmemPool_attribute,
    ) -> Result<T> {
        self.context().bind()?;
        let mut value = T::default();
        // SAFETY: `T` is an integer type whose size matches the attribute's documented
        // representation, and `value` is a writable local.
        unsafe {
            try_ffi!(driver::cuMemPoolGetAttribute(
                self.as_raw(),
                attribute,
                (&raw mut value).cast()
            ))?;
        }
        Ok(value)
    }
}

bitflags::bitflags! {
    /// Usage restrictions for allocations from a memory pool.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MemoryPoolUsage: u16 {
        /// Allocations from this pool will be used for hardware decompression.
        const HARDWARE_DECOMPRESSION = usage_bits(driver::CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS);
    }
}

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
