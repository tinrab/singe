//! Device, managed, pinned host, and CUDA array memory.

pub mod array;
pub mod pattern;
pub mod peer;

mod allocation;
mod device;
mod managed;
mod pinned;
mod pitched;
mod pool;
mod registered;
mod stream_ordered;
pub mod transfer;

use std::{
    fmt::{self, Display, Formatter},
    ptr::NonNull,
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::{
    driver::{self, CUmemLocationType as MemoryLocationKind},
    runtime,
};

pub use self::{
    allocation::DeviceAllocation,
    device::{DeviceMemory, RawDeviceMemory},
    managed::{ManagedMemory, RawManagedMemory},
    pinned::{PinnedHost, PinnedStats, PinnedVec, PinnedVecExt},
    pitched::{
        PitchedDeviceMemory, PitchedDeviceVolume, RawPitchedDeviceMemory, RawPitchedDeviceVolume,
    },
    pool::{
        MemoryAccessDescriptor, MemoryAccessFlag, MemoryAllocationType, MemoryPool,
        MemoryPoolAttribute, MemoryPoolAttributeValue, MemoryPoolProps, MemoryPoolUsage,
        RawMemoryPool,
    },
    registered::{RegisteredHostMemory, RegisteredHostSlice},
    stream_ordered::StreamOrderedFree,
};
pub(crate) use self::{
    allocation::{DeferredFree, DeviceDeallocation},
    pinned::PinnedPool,
    registered::mapped_host_device_ptr,
};
use crate::{
    driver::entry_points::cuMemGetLocationInfo,
    error::{Error, Result},
    try_ffi,
    types::DevicePtr,
    utility::to_u64,
    view::{DeviceRepr, HostRegion, HostSlice, HostSliceMut},
};

singe_core::native_enum! {
    /// Direction of a CUDA memory copy.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryCopyKind: runtime::cudaMemcpyKind {
        /// Copies from host memory to host memory.
        HostToHost = cudaMemcpyHostToHost,
        /// Copies from host memory to device memory.
        HostToDevice = cudaMemcpyHostToDevice,
        /// Copies from device memory to host memory.
        DeviceToHost = cudaMemcpyDeviceToHost,
        /// Copies from device memory to device memory.
        DeviceToDevice = cudaMemcpyDeviceToDevice,
        /// Direction of the transfer is inferred from the pointer values.
        /// Requires unified virtual addressing.
        Default = cudaMemcpyDefault,
    }
}

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
    /// Stream visibility of a managed allocation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MemoryAttachFlags: u32 {
        /// Memory can be accessed by any stream on any device.
        const GLOBAL = driver::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL.0;
        /// Memory cannot be accessed by any stream on any device.
        const HOST = driver::CUmemAttach_flags::CU_MEM_ATTACH_HOST.0;
        /// Memory can only be accessed by a single stream on the associated device.
        const SINGLE = driver::CUmemAttach_flags::CU_MEM_ATTACH_SINGLE.0;
    }
}

impl Display for MemoryAttachFlags {
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

singe_core::native_enum! {
    /// Kind of memory a pointer addresses.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryType: runtime::cudaMemoryType {
        /// Unregistered memory.
        Unregistered = cudaMemoryTypeUnregistered,
        /// Host memory.
        Host = cudaMemoryTypeHost,
        /// Device memory.
        Device = cudaMemoryTypeDevice,
        /// Managed memory.
        Managed = cudaMemoryTypeManaged,
    }
}

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

impl TryFrom<runtime::cudaPointerAttributes> for PointerAttributes {
    type Error = Error;

    fn try_from(attr: runtime::cudaPointerAttributes) -> Result<Self> {
        Ok(Self {
            // SAFETY: every field of this union is a plain integer, so reading it is valid for any memory type.
            locality_domain_ordinal: unsafe { attr.__bindgen_anon_1.localityDomainOrdinal },
            memory_type: attr.type_.try_into()?,
            device: attr.device,
            // SAFETY: `DevicePtr` only records the CUDA-reported alias; each use carries its own contract.
            device_pointer: unsafe { DevicePtr::from_raw(attr.devicePointer.cast()) },
            host_pointer: attr.hostPointer.cast(),
        })
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
        attr_ffi.try_into()
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

/// A page-locked host-memory range suitable for efficient CUDA transfers.
///
/// # Safety
///
/// Implementors must guarantee the range returned by [`HostSlice::as_host_ptr`] is page-locked for the lifetime of `&self`.
/// [`Self::host_region`] must report that range, the actual registration flags, and the identity shared by all mapped aliases.
pub unsafe trait PinnedHostSlice<T: DeviceRepr>: HostSlice<T> {
    /// Describes the same bytes as this host slice, preserving identity across mapped aliases.
    fn host_region(&self) -> HostRegion<'_>;
}

/// A mutable page-locked host-memory range suitable for efficient CUDA transfers.
///
/// # Safety
///
/// Implementors must uphold both [`PinnedHostSlice`] and [`HostSliceMut`] contracts.
pub unsafe trait PinnedHostSliceMut<T: DeviceRepr>:
    PinnedHostSlice<T> + HostSliceMut<T>
{
}

unsafe fn free_host_raw(ptr: *mut ()) -> Result<()> {
    // SAFETY: the caller passes a live `cudaMallocHost` block that nothing uses any more.
    unsafe { try_ffi!(runtime::cudaFreeHost(ptr.cast())) }
}

unsafe fn register_host_raw(ptr: *mut (), size: usize, flags: HostRegisterFlags) -> Result<()> {
    // SAFETY: the caller passes `size` live host bytes at `ptr` that stay allocated until unregistered.
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
    // SAFETY: the caller passes the base of a live registration that no queued work still uses.
    unsafe { try_ffi!(runtime::cudaHostUnregister(ptr.cast())) }
}

/// Retrieves memory location information for an address range.
///
/// Returns the most common location in each summary region, sampling at the specified interval.
/// A zero sampling granularity uses the system default.
///
/// # Safety
///
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
    if summary_granularity == 0 {
        return Err(Error::not_positive("summary granularity"));
    }

    if summary_granularity > size {
        return Err(Error::out_of_range("summary granularity"));
    }

    if sampling_granularity > summary_granularity {
        return Err(Error::out_of_range("sampling granularity"));
    }

    let count = size.div_ceil(summary_granularity);
    let mut locations = vec![driver::CUmemLocation::default(); count];
    let address = to_u64(ptr.addr(), "device address")?;
    // SAFETY: the caller keeps the range allocated, and `locations` has one writable entry per summary region.
    unsafe {
        cuMemGetLocationInfo(
            address,
            size as _,
            summary_granularity as _,
            sampling_granularity as _,
            locations.as_mut_ptr(),
        )?;
    }
    Ok(locations.into_iter().map(Into::into).collect())
}

#[cfg(all(test, feature = "testing"))]
mod tests;
