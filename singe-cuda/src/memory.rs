use std::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ptr, slice,
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::{driver, runtime};

use crate::{
    error::{Error, Result},
    ipc::IpcMemoryHandle,
    stream::{GraphRecordable, Stream, StreamCaptureScope, StreamScope},
    try_ffi,
    types::DevicePtr,
    view::{
        DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut, ZeroableDeviceRepr,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct MemoryCopyOperation<'a, T> {
    dst: *mut T,
    src: *const T,
    count: usize,
    kind: MemoryCopyKind,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> MemoryCopyOperation<'a, T> {
    /// Creates a stream-capture-safe memcpy operation from raw pointers.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores `dst` and `src` pointer addresses in the
    /// resulting CUDA graph. The caller must ensure both pointers are valid for
    /// `count` elements whenever a captured graph using this operation is
    /// launched. If `dst` is mutable memory, it must also remain exclusive for
    /// the work ordered by those launches.
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
        unsafe {
            DeviceMemory::<T>::copy_async(self.dst, self.src, self.count, self.kind, scope.stream())
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemorySetOperation<'a, T> {
    dst: *mut T,
    value: u8,
    count: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> MemorySetOperation<'a, T> {
    /// Creates a stream-capture-safe memset operation from a raw pointer.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores `dst` in the resulting CUDA graph. The
    /// caller must ensure `dst` is valid for writes of
    /// `count * size_of::<T>()` bytes whenever a captured graph using this
    /// operation is launched, and that it remains exclusive for the work ordered
    /// by those launches.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ArrayHandle(runtime::cudaArray_t);

impl ArrayHandle {
    pub const unsafe fn from_raw(handle: runtime::cudaArray_t) -> Self {
        Self(handle)
    }

    pub const fn as_raw(self) -> runtime::cudaArray_t {
        self.0
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MemoryAttachFlags: u32 {
        const GLOBAL = driver::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as _;
        const HOST = driver::CUmemAttach_flags::CU_MEM_ATTACH_HOST as _;
        const SINGLE = driver::CUmemAttach_flags::CU_MEM_ATTACH_SINGLE as _;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAllocationType {
    Invalid = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_INVALID as _,
    Pinned = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED as _,
    Managed = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_MANAGED as _,
    Max = driver::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_MAX as _,
}

impl_enum_conversion!(u32, driver::CUmemAllocationType, MemoryAllocationType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAllocationHandleType {
    None = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE as _,
    PosixFileDescriptor =
        driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR as _,
    Win32 = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32 as _,
    Win32Kmt = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32_KMT as _,
    Fabric = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC as _,
    Max = driver::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_MAX as _,
}

impl_enum_conversion!(
    u32,
    driver::CUmemAllocationHandleType,
    MemoryAllocationHandleType
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAccessFlag {
    None = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE as _,
    Read = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ as _,
    ReadWrite = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE as _,
    Max = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_MAX as _,
}

impl_enum_conversion!(u32, driver::CUmemAccess_flags, MemoryAccessFlag);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryPoolAttribute {
    ReuseFollowEventDependencies =
        driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES as _,
    ReuseAllowOpportunistic =
        driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC as _,
    ReuseAllowInternalDependencies =
        driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES as _,
    ReleaseThreshold = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD as _,
    ReservedMemoryCurrent = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT as _,
    ReservedMemoryHigh = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH as _,
    UsedMemoryCurrent = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT as _,
    UsedMemoryHigh = driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH as _,
}

impl_enum_conversion!(u32, driver::CUmemPool_attribute, MemoryPoolAttribute);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MemoryPoolAttributeValue {
    Bool(bool),
    Bytes(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryAccessDescriptor {
    pub location: MemoryLocation,
    pub flags: MemoryAccessFlag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryPoolProps {
    pub alloc_type: MemoryAllocationType,
    pub handle_type: MemoryAllocationHandleType,
    pub location: MemoryLocation,
    pub max_size: usize,
    pub usage: u16,
}

#[derive(Debug)]
pub struct MemoryPool {
    handle: driver::CUmemoryPool,
}

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
            usage: value.usage,
            reserved: [0; 54],
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct HostAllocationFlags: u32 {
        const DEFAULT = runtime::cudaHostAllocDefault;
        const PORTABLE = runtime::cudaHostAllocPortable;
        const MAPPED = runtime::cudaHostAllocMapped;
        const WRITE_COMBINED = runtime::cudaHostAllocWriteCombined;
    }
}

bitflags::bitflags! {
    /// Flags for [`DeviceMemory::register_host`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct HostRegisterFlags: u32 {
        const DEFAULT = runtime::cudaHostRegisterDefault;
        const PORTABLE = runtime::cudaHostRegisterPortable;
        const MAPPED = runtime::cudaHostRegisterMapped;
        const IO_MEMORY = runtime::cudaHostRegisterIoMemory;
        const READ_ONLY = runtime::cudaHostRegisterReadOnly;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointerAttributes {
    pub memory_type: MemoryType,
    pub device: i32,
    pub device_pointer: DevicePtr,
    pub host_pointer: *mut (),
}

impl From<runtime::cudaPointerAttributes> for PointerAttributes {
    fn from(attr: runtime::cudaPointerAttributes) -> Self {
        Self {
            memory_type: attr.type_.into(),
            device: attr.device,
            device_pointer: unsafe { DevicePtr::from_raw(attr.devicePointer.cast()) },
            host_pointer: attr.hostPointer.cast(),
        }
    }
}

#[repr(u32)]
#[derive(
    Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, TryFromPrimitive, IntoPrimitive,
)]
#[non_exhaustive]
pub enum MemoryLocationKind {
    Invalid = driver::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_INVALID as _,
    Device = driver::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE as _,
    Host = driver::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_HOST as _,
    Numa = driver::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_HOST_NUMA as _,
    NumaCurrent = driver::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT as _,
    Max = driver::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_MAX as _,
}

impl_enum_conversion!(driver::CUmemLocationType_enum, MemoryLocationKind);

impl_enum_display!(MemoryLocationKind, {
    Self::Invalid => "CU_MEM_LOCATION_TYPE_INVALID",
    Self::Device => "CU_MEM_LOCATION_TYPE_DEVICE",
    Self::Host => "CU_MEM_LOCATION_TYPE_HOST",
    Self::Numa => "CU_MEM_LOCATION_TYPE_HOST_NUMA",
    Self::NumaCurrent => "CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT",
    Self::Max => "CU_MEM_LOCATION_TYPE_MAX",
});

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct MemoryLocation {
    pub kind: MemoryLocationKind,
    pub id: i32,
}

impl From<driver::CUmemLocation_st> for MemoryLocation {
    fn from(s: driver::CUmemLocation_st) -> Self {
        Self {
            kind: s.type_.into(),
            id: unsafe { s.__bindgen_anon_1.id },
        }
    }
}

impl From<MemoryLocation> for driver::CUmemLocation_st {
    fn from(m: MemoryLocation) -> Self {
        Self {
            type_: m.kind.into(),
            __bindgen_anon_1: driver::CUmemLocation_st__bindgen_ty_1 { id: m.id as _ },
        }
    }
}

impl Default for MemoryLocation {
    fn default() -> Self {
        driver::CUmemLocation_st::default().into()
    }
}

impl MemoryPool {
    /// Creates a CUDA memory pool.
    /// `props` determines the properties of the pool such as the backing device and IPC capabilities.
    ///
    /// To create a memory pool for host memory not targeting a specific NUMA node, applications must set [`MemoryPoolProps::location`] to [`MemoryLocationKind::Host`].
    /// [`MemoryLocation::id`] is ignored for such pools.
    /// Pools created with [`MemoryLocationKind::Host`] are not IPC-capable and [`MemoryPoolProps::handle_type`] must be [`MemoryAllocationHandleType::None`]; any other value returns [`crate::error::Status::InvalidValue`].
    /// To create a memory pool targeting a specific host NUMA node, applications must set [`MemoryLocation::kind`] to [`MemoryLocationKind::Numa`] and [`MemoryLocation::id`] must specify the NUMA ID of the host memory node.
    /// Specifying [`MemoryLocationKind::NumaCurrent`] as [`MemoryLocation::kind`] returns [`crate::error::Status::InvalidValue`].
    /// By default, the pool's memory is accessible from the device where it is allocated.
    /// Pools created with [`MemoryLocationKind::Numa`] or [`MemoryLocationKind::Host`] are accessible from the host CPU by default.
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
    /// An application can also set [`MemoryLocationKind::Invalid`] to indicate no preferred location.
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
    pub fn create(props: MemoryPoolProps) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let props = driver::CUmemPoolProps::from(props);
        unsafe {
            try_ffi!(driver::cuMemPoolCreate(&raw mut handle, &raw const props))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle })
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
    pub fn set_attribute(
        &mut self,
        attribute: MemoryPoolAttribute,
        value: MemoryPoolAttributeValue,
    ) -> Result<()> {
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
                        self.handle,
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
                        self.handle,
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
        unsafe {
            match attribute {
                MemoryPoolAttribute::ReuseFollowEventDependencies
                | MemoryPoolAttribute::ReuseAllowOpportunistic
                | MemoryPoolAttribute::ReuseAllowInternalDependencies => {
                    let mut value = 0u32;
                    try_ffi!(driver::cuMemPoolGetAttribute(
                        self.handle,
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
                        self.handle,
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
        let access_descs: Vec<_> = access_descs.iter().copied().map(Into::into).collect();
        unsafe {
            try_ffi!(driver::cuMemPoolSetAccess(
                self.handle,
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
        let mut flags = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE;
        let mut location = driver::CUmemLocation_st::from(location);
        unsafe {
            try_ffi!(driver::cuMemPoolGetAccess(
                &raw mut flags,
                self.handle,
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
        unsafe {
            try_ffi!(driver::cuMemPoolTrimTo(self.handle, min_bytes_to_keep as _))?;
        }
        Ok(())
    }

    pub const fn as_raw(&self) -> driver::CUmemoryPool {
        self.handle
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(driver::cuMemPoolDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda memory pool: {err}");
            }
        }
    }
}

/// Represents a region of owned CUDA device memory for elements of type `T`.
#[derive(Debug)]
pub struct DeviceMemory<T> {
    /// Raw pointer to the allocated device memory.
    ptr: *mut T,
    /// Number of elements of type `T` allocated.
    length: usize,
    /// Marker for the type `T`.
    _phantom: PhantomData<T>,
}

impl<T> PartialEq for DeviceMemory<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.length == other.length
    }
}

impl<T> Eq for DeviceMemory<T> {}

#[derive(Debug)]
pub struct ManagedMemory<T: DeviceRepr> {
    ptr: *mut T,
    length: usize,
    // CUDA tracks the current visibility policy for managed memory separately
    // from the pointer. Store the last policy requested through this wrapper so
    // callers can reason about stream attachment without another FFI query.
    attach_flags: MemoryAttachFlags,
    _phantom: PhantomData<T>,
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
        let Some(bytes) = count.checked_mul(size_of::<T>()) else {
            return Err(Error::InvalidMemoryAllocationRequest);
        };
        let mut p = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaMalloc(&raw mut p, bytes as _))?;
        }
        Ok(p.cast())
    }

    pub unsafe fn alloc_managed(count: usize, flags: MemoryAttachFlags) -> Result<*mut T> {
        let Some(bytes) = count.checked_mul(size_of::<T>()) else {
            return Err(Error::InvalidMemoryAllocationRequest);
        };
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }
        let mut p = ptr::null_mut();
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
    pub unsafe fn free(ptr: *mut T) -> Result<()> {
        unsafe {
            try_ffi!(runtime::cudaFree(ptr.cast()))?;
        }
        Ok(())
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
        let Some(bytes) = count.checked_mul(size_of::<T>()) else {
            return Err(Error::InvalidMemoryAllocationRequest);
        };
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

    /// Fills the first `count` bytes of the memory area pointed to by `ptr` with the constant byte `value`.
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
        let Some(bytes) = count.checked_mul(size_of::<T>()) else {
            return Err(Error::InvalidMemoryAllocationRequest);
        };
        unsafe {
            try_ffi!(runtime::cudaMemset(dst.cast(), value.into(), bytes as _))?;
        }
        Ok(())
    }

    pub unsafe fn alloc_host(size: usize) -> Result<*mut ()> {
        let mut ptr = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaMallocHost(
                &raw mut ptr,
                size as runtime::size_t
            ))?;
        }
        Ok(ptr.cast())
    }

    /// Frees host memory returned by [`DeviceMemory::alloc_host`] or [`DeviceMemory::alloc_pinned`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot free the host allocation, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `ptr` must be null or a live host allocation returned by a compatible
    /// CUDA host allocation function.
    pub unsafe fn free_host(ptr: *mut ()) -> Result<()> {
        unsafe { try_ffi!(runtime::cudaFreeHost(ptr.cast())) }
    }

    /// Allocates size bytes of host memory that is page-locked and accessible to the device.
    /// The driver tracks the allocated virtual memory ranges and automatically accelerates calls such as [`DeviceMemory::copy`].
    /// Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory obtained with functions such as `malloc()`.
    /// Allocating excessive amounts of pinned memory may degrade system performance, since it reduces the amount of memory available to the system for paging.
    /// As a result, use this sparingly to allocate staging areas for data exchange between host and device.
    ///
    /// `flags` selects allocation options:
    ///
    /// * [`HostAllocationFlags::DEFAULT`]: equivalent to [`DeviceMemory::alloc_host`].
    /// * [`HostAllocationFlags::PORTABLE`]: the memory returned by this call is considered pinned memory by all CUDA contexts, not just the one that performed
    ///   the allocation.
    /// * [`HostAllocationFlags::MAPPED`]: maps the allocation into the CUDA address space.
    ///   The device pointer to the memory may be obtained by calling [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer).
    /// * [`HostAllocationFlags::WRITE_COMBINED`]: allocates the memory as write-combined (WC).
    ///   WC memory can be transferred across the PCI Express bus more quickly on some
    ///   system configurations, but cannot be read efficiently by most CPUs.
    ///   WC memory is a good option for buffers written
    ///   by the CPU and read by the device via mapped pinned memory or host-&gt;device transfers.
    ///
    /// All of these flags are orthogonal to one another: a developer may allocate memory that is portable, mapped and/or write-combined with no restrictions.
    ///
    /// For [`HostAllocationFlags::MAPPED`] to have any effect, the CUDA context must support [`ContextFlags::MAP_HOST`](crate::context::ContextFlags::MAP_HOST), which can be checked via [`Device::flags`](crate::device::Device::flags).
    /// [`ContextFlags::MAP_HOST`](crate::context::ContextFlags::MAP_HOST) is implicitly set for contexts created via the runtime API.
    ///
    /// [`HostAllocationFlags::MAPPED`] may be specified on CUDA contexts for devices that do not support mapped pinned memory.
    /// The failure is deferred to [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer) because the memory may be mapped into other CUDA contexts via [`HostAllocationFlags::PORTABLE`].
    ///
    /// Memory allocated by this method must be freed with [`DeviceMemory::free_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate pinned host memory, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized host memory. The caller must ensure
    /// it is accessed within `size` bytes and freed with [`DeviceMemory::free_host`].
    pub unsafe fn alloc_pinned(size: usize, flags: HostAllocationFlags) -> Result<*mut ()> {
        let mut ptr = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaHostAlloc(
                &raw mut ptr,
                size as _,
                flags.bits()
            ))?;
        }
        Ok(ptr.cast())
    }

    /// Page-locks the memory range specified by `ptr` and `size`, and maps it for the devices selected by `flags`.
    /// This memory range also is added to the same tracking mechanism as [`DeviceMemory::alloc_pinned`] to automatically accelerate calls to functions such as [`DeviceMemory::copy`].
    /// Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory that has not been registered.
    /// Page-locking excessive amounts of memory may degrade system performance, since it reduces the amount of memory available to the system for paging.
    /// As a result, use this sparingly to register staging areas for data exchange between host and device.
    ///
    /// On systems where [`DeviceProperties::pageable_memory_access_uses_host_page_tables`](crate::device::DeviceProperties::pageable_memory_access_uses_host_page_tables) is enabled, [`DeviceMemory::register_host`] does not page-lock the memory range specified by `ptr` and instead only populates unpopulated pages.
    ///
    /// [`DeviceMemory::register_host`] is supported only on I/O coherent devices where [`DeviceProperties::host_register_supported`](crate::device::DeviceProperties::host_register_supported) is enabled.
    ///
    /// `flags` selects registration options:
    ///
    /// * [`HostRegisterFlags::DEFAULT`]: on a system with unified virtual addressing, the memory is both mapped and portable.
    ///   On a system with no unified virtual addressing, the memory is neither mapped nor portable.
    ///
    /// * [`HostRegisterFlags::PORTABLE`]: the memory returned by this call is considered pinned memory by all CUDA contexts, not just the one that performed
    ///   the allocation.
    ///
    /// * [`HostRegisterFlags::MAPPED`]: maps the allocation into the CUDA address space.
    ///   The device pointer to the memory may be obtained by calling [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer).
    ///
    /// * [`HostRegisterFlags::IO_MEMORY`]: the passed memory pointer is treated as pointing to some memory-mapped I/O space, for example belonging to a third-party PCIe device,
    ///   and it is marked as non-cache-coherent and contiguous.
    ///
    /// * [`HostRegisterFlags::READ_ONLY`]: the passed memory pointer is treated as pointing to memory that is considered read-only by the device.
    ///   On platforms without
    ///   [`DeviceProperties::pageable_memory_access_uses_host_page_tables`](crate::device::DeviceProperties::pageable_memory_access_uses_host_page_tables), this flag is required to register memory mapped to the CPU as read-only.
    ///   Query support with [`DeviceProperties::host_register_read_only_supported`](crate::device::DeviceProperties::host_register_read_only_supported).
    ///   Using this flag with a current context associated with a device that does not have this attribute set makes [`DeviceMemory::register_host`] return [`crate::error::Status::NotSupported`].
    ///
    /// All of these flags are orthogonal to one another: a developer may page-lock memory that is portable or mapped with no restrictions.
    ///
    /// The CUDA context must have been created with [`ContextFlags::MAP_HOST`](crate::context::ContextFlags::MAP_HOST) for [`HostRegisterFlags::MAPPED`] to have any effect.
    ///
    /// [`HostRegisterFlags::MAPPED`] may be specified on CUDA contexts for devices that do not support mapped pinned memory.
    /// The failure is deferred to [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer) because the memory may be mapped into other CUDA contexts via [`HostRegisterFlags::PORTABLE`].
    ///
    /// On devices where [`DeviceProperties::can_use_host_pointer_for_registered_mem`](crate::device::DeviceProperties::can_use_host_pointer_for_registered_mem) is enabled, the memory can also be accessed from the device using the original host pointer.
    /// The device pointer returned by [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer) may or may not match the original host pointer and depends on the devices visible to the application.
    /// If all devices visible to the application have a non-zero value for the device attribute, the device pointer returned by [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer) matches the original pointer.
    /// If any device visible to the application has a zero value for the device attribute, the device pointer returned by [`sys::cudaHostGetDevicePointer`](singe_cuda_sys::runtime::cudaHostGetDevicePointer) does not match the original host pointer, but is suitable for use on all devices provided Unified Virtual Addressing is enabled.
    /// In such systems, it is valid to access the memory using either pointer on devices that have a non-zero value for the device attribute.
    /// Such devices must access the memory through only one of the two pointers, not both.
    ///
    /// The memory page-locked by this method must be unregistered with [`DeviceMemory::unregister_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot register the host range, the pointer,
    /// size, or flags are invalid, a previous asynchronous launch reports an
    /// error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `ptr..ptr + size` must be a valid host memory range and must remain valid
    /// until it is unregistered.
    pub unsafe fn register_host(ptr: *mut (), size: usize, flags: HostRegisterFlags) -> Result<()> {
        unsafe {
            try_ffi!(runtime::cudaHostRegister(
                ptr.cast(),
                size as _,
                flags.bits()
            ))?;
        }
        Ok(())
    }

    /// Unmaps the memory range whose base address is specified by `ptr`, and makes it pageable again.
    ///
    /// The base address must be the same one specified to [`DeviceMemory::register_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot unregister the host range, `ptr` is not
    /// the base address of a registered range, a previous asynchronous launch
    /// reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `ptr` must be the base address of a host range registered with
    /// [`DeviceMemory::register_host`] and must not be unregistered twice.
    pub unsafe fn unregister_host(ptr: *mut ()) -> Result<()> {
        unsafe { try_ffi!(runtime::cudaHostUnregister(ptr.cast())) }
    }

    /// Returns the total amount of memory available to the current context and the amount of memory free on the device.
    /// CUDA is not guaranteed to be able to allocate all of the memory that the OS reports as free.
    /// In a multi-tenant situation, the free-memory estimate is prone to a race condition: an allocation or free by another process or thread between estimation and reporting can make the reported free value differ from actual free memory.
    ///
    /// The integrated GPU on Tegra shares memory with CPU and other component of the SoC.
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
        unsafe {
            try_ffi!(runtime::cudaMemGetInfo(&raw mut free, &raw mut total))?;
        }
        Ok((free as usize, total as usize))
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
        let mut attr_ffi = MaybeUninit::<runtime::cudaPointerAttributes>::uninit();
        unsafe {
            try_ffi!(runtime::cudaPointerGetAttributes(
                attr_ffi.as_mut_ptr(),
                ptr.cast(),
            ))?;
            // Safety: FFI call successful, attr_ffi is initialized.
            Ok(attr_ffi.assume_init().into())
        }
    }

    pub unsafe fn alloc_async(count: usize, stream: &Stream) -> Result<*mut T> {
        let Some(bytes) = count.checked_mul(size_of::<T>()) else {
            return Err(Error::InvalidMemoryAllocationRequest);
        };
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }
        let mut p = ptr::null_mut();
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
        unsafe { try_ffi!(runtime::cudaFreeAsync(ptr.cast(), stream.as_raw())) }
    }

    pub unsafe fn copy_async(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let Some(bytes) = count.checked_mul(size_of::<T>()) else {
            return Err(Error::InvalidMemoryAllocationRequest);
        };
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
        if count == 0 {
            return Ok(());
        }
        let Some(bytes) = count.checked_mul(size_of::<T>()) else {
            return Err(Error::InvalidMemoryAllocationRequest);
        };
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

    /// Prefetches memory to the specified destination location.
    /// `ptr` is the base device pointer of the memory to be prefetched, `location` specifies the destination location, `count` specifies the number of bytes to copy, and `stream` is the stream in which the operation is enqueued.
    /// The memory range must refer to managed memory allocated via [`DeviceMemory::alloc_managed`] or declared via `__managed__` variables. It may also refer to memory allocated from a managed memory pool, or to system-allocated memory on systems where [`DeviceProperties::pageable_memory_access`](crate::device::DeviceProperties::pageable_memory_access) is enabled.
    ///
    /// Setting [`MemoryLocation::kind`](crate::memory::MemoryLocation::kind) to [`MemoryLocationKind::Device`] prefetches memory to the GPU identified by [`MemoryLocation::id`](crate::memory::MemoryLocation::id). That device, and the device associated with `stream`, must support concurrent managed access.
    /// Setting [`MemoryLocation::kind`](crate::memory::MemoryLocation::kind) to [`MemoryLocationKind::Host`] prefetches data to host memory.
    /// Applications can request prefetching memory to a specific host NUMA node by using [`MemoryLocationKind::Numa`] with a valid NUMA node identifier, or to the NUMA node closest to the current thread's CPU by using [`MemoryLocationKind::NumaCurrent`].
    /// When [`MemoryLocation::kind`](crate::memory::MemoryLocation::kind) is [`MemoryLocationKind::Host`] or [`MemoryLocationKind::NumaCurrent`], [`MemoryLocation::id`](crate::memory::MemoryLocation::id) is ignored.
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
}

// Safety: DeviceMemory owns a CUDA allocation and does not provide CPU
// references into the allocation. Moving the owner to another thread is safe
// when `T: Send` because CUDA allocation/free operations are synchronized by
// the CUDA context APIs used by this wrapper. Shared references are safe when
// `T: Sync` because `&DeviceMemory<T>` only exposes raw device pointers and
// read-only metadata; mutation of device memory still requires CUDA operations
// whose safety contracts are enforced by the calling APIs.
unsafe impl<T: Send> Send for DeviceMemory<T> {}
unsafe impl<T: Sync> Sync for DeviceMemory<T> {}

// Safety: managed memory has the same Rust-side ownership properties as
// DeviceMemory. CUDA maintains coherence for managed allocations across host
// threads and devices; Rust references into the allocation are not exposed by
// this wrapper.
unsafe impl<T: DeviceRepr + Send> Send for ManagedMemory<T> {}
unsafe impl<T: DeviceRepr + Sync> Sync for ManagedMemory<T> {}

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
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize) -> Self {
        Self {
            ptr,
            length,
            _phantom: PhantomData,
        }
    }

    pub fn into_raw_parts(self) -> (*mut T, usize) {
        let ptr = self.ptr;
        let length = self.length;
        mem::forget(self);
        (ptr, length)
    }

    pub fn create(length: usize) -> Result<Self> {
        let size_t = size_of::<T>();

        if size_t == 0 {
            if length == 0 {
                return Ok(Self {
                    ptr: ptr::null_mut(), // No allocation needed for ZSTs with count 0
                    length: 0,
                    _phantom: PhantomData,
                });
            }
            return Err(Error::InvalidMemoryAllocationRequest);
        }

        // Ensure allocation size doesn't overflow usize when calculating bytes internally in `alloc`.
        if length > (usize::MAX / size_t) {
            return Err(Error::InvalidMemoryAllocationRequest);
        }

        if length == 0 {
            Ok(Self {
                ptr: ptr::null_mut(),
                length: 0,
                _phantom: PhantomData,
            })
        } else {
            let device_ptr = unsafe { Self::alloc(length)? };

            Ok(Self {
                ptr: device_ptr,
                length,
                _phantom: PhantomData,
            })
        }
    }

    pub fn zeroes(length: usize) -> Result<Self> {
        let mut mem = Self::create(length)?;
        mem.set_zeroes()?;
        Ok(mem)
    }

    pub fn from_slice(v: &[T]) -> Result<Self> {
        let mut mem = Self::create(v.len())?;
        mem.copy_from_host(v)?;
        Ok(mem)
    }

    /// # Safety
    ///
    /// The caller must ensure `v` remains valid and unmodified until `stream`
    /// has completed the transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate device memory or enqueue the
    /// host-to-device copy.
    pub unsafe fn from_slice_async(v: &[T], stream: &Stream) -> Result<Self> {
        let mut mem = Self::create(v.len())?;
        unsafe {
            mem.copy_from_host_async_unchecked(v, stream)?;
        }
        Ok(mem)
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn byte_len(&self) -> usize {
        self.length
            .checked_mul(size_of::<T>())
            .expect("device memory byte length overflow")
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    pub const fn as_device_ptr(&self) -> DevicePtr {
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    pub fn copy_from_host(&mut self, host_slice: &[T]) -> Result<()> {
        if host_slice.len() != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.length == 0 {
            return Ok(());
        }
        unsafe {
            Self::copy(
                self.ptr,
                host_slice.as_ptr(),
                self.length,
                MemoryCopyKind::HostToDevice,
            )
        }
    }

    pub fn copy_from_host_async<'scope, 'env>(
        &mut self,
        host_slice: &'env [T],
        stream: &StreamScope<'scope, 'env>,
    ) -> Result<()> {
        unsafe { self.copy_from_host_async_unchecked(host_slice, stream.stream()) }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until
    /// `stream` has completed the transfer.
    pub unsafe fn copy_from_host_async_unchecked(
        &mut self,
        host_slice: &[T],
        stream: &Stream,
    ) -> Result<()> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
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
    pub unsafe fn copy_from_host_operation<'a>(
        &'a mut self,
        host_slice: &'a [T],
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(unsafe {
            MemoryCopyOperation::new(
                self.as_mut_ptr(),
                host_slice.as_ptr(),
                self.len(),
                MemoryCopyKind::HostToDevice,
            )
        })
    }

    pub fn copy_to_host(&self, host_slice: &mut [T]) -> Result<()> {
        if host_slice.len() != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.length == 0 {
            return Ok(());
        }
        unsafe {
            Self::copy(
                host_slice.as_mut_ptr(),
                self.ptr,
                self.length,
                MemoryCopyKind::DeviceToHost,
            )
        }
    }

    pub fn copy_to_host_async<'scope, 'env>(
        &self,
        host_slice: &'env mut [T],
        stream: &StreamScope<'scope, 'env>,
    ) -> Result<()> {
        unsafe { self.copy_to_host_async_unchecked(host_slice, stream.stream()) }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until
    /// `stream` has completed the transfer.
    pub unsafe fn copy_to_host_async_unchecked(
        &self,
        host_slice: &mut [T],
        stream: &Stream,
    ) -> Result<()> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
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
    pub unsafe fn copy_to_host_operation<'a>(
        &'a self,
        host_slice: &'a mut [T],
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(unsafe {
            MemoryCopyOperation::new(
                host_slice.as_mut_ptr(),
                self.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToHost,
            )
        })
    }

    pub fn copy_to_host_vec(&self) -> Result<Vec<T>> {
        if size_of::<T>() == 0 {
            return Err(Error::InvalidMemoryAllocationRequest);
        }

        if self.length == 0 {
            return Ok(Vec::new());
        }

        let mut host_vec = Vec::<T>::with_capacity(self.length);

        unsafe {
            Self::copy(
                host_vec.as_mut_ptr(),
                self.ptr,
                self.length,
                MemoryCopyKind::DeviceToHost,
            )?;

            host_vec.set_len(self.length);
        }

        Ok(host_vec)
    }

    pub fn copy_from_device(&mut self, src: &Self) -> Result<()> {
        if src.len() != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.length == 0 {
            return Ok(());
        }
        unsafe {
            Self::copy(
                self.ptr,
                src.as_ptr(),
                self.length,
                MemoryCopyKind::DeviceToDevice,
            )
        }
    }

    pub fn copy_from_device_async<'scope, 'env>(
        &mut self,
        src: &Self,
        stream: &StreamScope<'scope, 'env>,
    ) -> Result<()> {
        unsafe { self.copy_from_device_async_unchecked(src, stream.stream()) }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` and `src` both remain valid until
    /// `stream` has completed the transfer.
    pub unsafe fn copy_from_device_async_unchecked(
        &mut self,
        src: &Self,
        stream: &Stream,
    ) -> Result<()> {
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
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
    pub unsafe fn copy_from_device_operation<'a>(
        &'a mut self,
        src: &'a Self,
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(unsafe {
            MemoryCopyOperation::new(
                self.as_mut_ptr(),
                src.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToDevice,
            )
        })
    }

    pub fn set_zeroes(&mut self) -> Result<()> {
        if self.length == 0 {
            return Ok(());
        }
        unsafe { Self::set(self.ptr, 0, self.length) }
    }

    pub fn set_value(&mut self, value: u8) -> Result<()> {
        if self.length == 0 {
            return Ok(());
        }
        unsafe { Self::set(self.ptr, value, self.length) }
    }

    pub fn set_value_async<'scope, 'env>(
        &mut self,
        value: u8,
        stream: &StreamScope<'scope, 'env>,
    ) -> Result<()> {
        unsafe { self.set_value_async_unchecked(value, stream.stream()) }
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
        if self.is_empty() {
            return Ok(());
        }
        unsafe { Self::set_async(self.as_mut_ptr(), value, self.len(), stream) }
    }

    /// Returns a capture operation that fills this device allocation with `value`.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores this allocation's pointer address in the
    /// resulting CUDA graph. The caller must ensure `self` remains valid and
    /// exclusive whenever a captured graph using this operation is launched.
    pub unsafe fn set_value_operation<'a>(&'a mut self, value: u8) -> MemorySetOperation<'a, T> {
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
        if self.is_empty() {
            // Cannot get handle for null pointer / zero size? Check docs.
            return Err(Error::InvalidMemoryAccess);
        }
        let mut handle = MaybeUninit::uninit();
        unsafe {
            try_ffi!(runtime::cudaIpcGetMemHandle(
                handle.as_mut_ptr(),
                self.as_ptr().cast_mut().cast(),
            ))?;
            Ok(IpcMemoryHandle::from_raw(handle.assume_init()))
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        if self.length == 0 || size_of::<T>() == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                length: self.length,
                _phantom: PhantomData,
            });
        }

        let new_mem = Self::create(self.length)?;

        unsafe {
            Self::copy(
                new_mem.as_mut_ptr(),
                self.as_ptr(),
                self.length,
                MemoryCopyKind::DeviceToDevice,
            )?;
        }

        Ok(new_mem)
    }
}

impl<T> Clone for DeviceMemory<T> {
    fn clone(&self) -> Self {
        match self.try_clone() {
            Ok(new_mem) => new_mem,
            Err(err) => {
                #[cfg(debug_assertions)]
                eprintln!("device memory clone failed: {err}");
                Self {
                    ptr: ptr::null_mut(),
                    length: 0,
                    _phantom: PhantomData,
                }
            }
        }
    }
}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        // debug_assert!(
        //     unsafe { free(self.ptr) }.is_ok(),
        //     "failed to free device memory at {:#x}",
        //     self.ptr as usize
        // );
        if let Err(err) = unsafe { Self::free(self.ptr) } {
            #[cfg(debug_assertions)]
            eprintln!("failed to free device memory: {err}");
            return;
        }

        self.ptr = ptr::null_mut();
        self.length = 0;
    }
}

impl<T: DeviceRepr> ManagedMemory<T> {
    /// Allocates typed CUDA managed memory and records its initial attach mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot
    /// allocate managed memory, or a non-empty zero-sized allocation is
    /// requested.
    pub fn create(length: usize, attach_flags: MemoryAttachFlags) -> Result<Self> {
        if size_of::<T>() == 0 {
            return if length == 0 {
                Ok(Self {
                    ptr: ptr::null_mut(),
                    length,
                    attach_flags,
                    _phantom: PhantomData,
                })
            } else {
                Err(Error::InvalidMemoryAllocationRequest)
            };
        }

        let ptr = unsafe { DeviceMemory::<T>::alloc_managed(length, attach_flags)? };
        Ok(Self {
            ptr,
            length,
            attach_flags,
            _phantom: PhantomData,
        })
    }

    pub fn zeroes(length: usize, attach_flags: MemoryAttachFlags) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut memory = Self::create(length, attach_flags)?;
        memory.set_zeroes()?;
        Ok(memory)
    }

    /// Takes ownership of an existing managed allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be null for an empty allocation or point to `length` live
    /// elements allocated by a CUDA allocation function compatible with
    /// [`DeviceMemory::free`]. `length * size_of::<T>()` must fit in `usize`.
    /// No other owner may free the pointer while the returned value is alive.
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        length: usize,
        attach_flags: MemoryAttachFlags,
    ) -> Self {
        Self {
            ptr,
            length,
            attach_flags,
            _phantom: PhantomData,
        }
    }

    pub fn into_raw_parts(self) -> (*mut T, usize, MemoryAttachFlags) {
        let ptr = self.ptr;
        let length = self.length;
        let attach_flags = self.attach_flags;
        mem::forget(self);
        (ptr, length, attach_flags)
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn byte_len(&self) -> usize {
        self.length
            .checked_mul(size_of::<T>())
            .expect("managed memory byte length overflow")
    }

    pub const fn attach_flags(&self) -> MemoryAttachFlags {
        self.attach_flags
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn view(&self) -> DeviceView<'_, T> {
        // The ManagedMemory owner guarantees the pointer remains live for the
        // borrowed view lifetime.
        unsafe { DeviceView::from_raw_parts(self.ptr, self.length) }
    }

    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        // &mut self guarantees unique access to the represented range.
        unsafe { DeviceViewMut::from_raw_parts(self.ptr, self.length) }
    }

    /// Returns a host slice over this managed allocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure no GPU work or other CPU reference can
    /// concurrently mutate the same memory for the returned lifetime, and that
    /// the allocation is accessible from the host at the point of access.
    pub unsafe fn as_host_slice(&self) -> &[T] {
        if self.is_empty() {
            return &[];
        }
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
        unsafe { slice::from_raw_parts_mut(self.ptr, self.length) }
    }

    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        if self.is_empty() {
            return Ok(());
        }
        unsafe { DeviceMemory::<T>::set(self.ptr, 0, self.length) }
    }

    pub fn prefetch_to(&self, location: MemoryLocation, stream: &Stream) -> Result<()> {
        DeviceMemory::<T>::prefetch_async(
            unsafe { DevicePtr::from_raw(self.ptr.cast::<()>()) },
            self.byte_len(),
            location,
            stream,
        )
    }

    pub fn attach_to_stream(&mut self, stream: &Stream, flags: MemoryAttachFlags) -> Result<()> {
        stream.context().bind()?;
        unsafe {
            try_ffi!(runtime::cudaStreamAttachMemAsync(
                stream.as_raw(),
                self.ptr.cast(),
                self.byte_len() as _,
                flags.bits(),
            ))?;
        }
        self.attach_flags = flags;
        Ok(())
    }
}

impl<T: DeviceRepr> DeviceSlice<T> for ManagedMemory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl<T: DeviceRepr> DeviceSliceMut<T> for ManagedMemory<T> {
    fn as_device_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: DeviceRepr> Drop for ManagedMemory<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        if let Err(err) = unsafe { DeviceMemory::<T>::free(self.ptr) } {
            #[cfg(debug_assertions)]
            eprintln!("failed to free managed memory: {err}");
            return;
        }

        self.ptr = ptr::null_mut();
        self.length = 0;
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing;

    #[test]
    fn it_works() -> Result<()> {
        unsafe {
            let host_in = [1, 2, 3];

            let device_ptr = DeviceMemory::alloc(3)?;

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
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

        let host_in = [4_i32, 5, 6];
        let mut device = DeviceMemory::create(host_in.len())?;
        let mut host_out = [0_i32; 3];

        stream.sync_scope(|scope| {
            device.copy_from_host_async(&host_in, scope)?;
            device.copy_to_host_async(&mut host_out, scope)
        })?;

        assert_eq!(host_out, host_in);

        Ok(())
    }

    #[test]
    fn managed_memory_tracks_metadata_and_views() -> Result<()> {
        let mut backing = [1_u32, 2, 3, 4];
        let mut managed = unsafe {
            ManagedMemory::from_raw_parts(
                backing.as_mut_ptr(),
                backing.len(),
                MemoryAttachFlags::HOST,
            )
        };

        assert_eq!(managed.len(), backing.len());
        assert_eq!(managed.byte_len(), backing.len() * size_of::<u32>());
        assert_eq!(managed.attach_flags(), MemoryAttachFlags::HOST);
        assert_eq!(managed.view().len(), backing.len());
        assert_eq!(managed.view_mut().len(), backing.len());

        unsafe {
            assert_eq!(managed.as_host_slice(), &[1, 2, 3, 4]);
            managed.as_host_slice_mut()[2] = 9;
        }
        assert_eq!(backing[2], 9);

        let (ptr, length, flags) = managed.into_raw_parts();
        assert_eq!(ptr, backing.as_mut_ptr());
        assert_eq!(length, backing.len());
        assert_eq!(flags, MemoryAttachFlags::HOST);

        Ok(())
    }
}
