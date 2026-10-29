//! Virtual address reservations, physical allocations and mappings.
use std::{
    mem::ManuallyDrop,
    os::fd::{BorrowedFd, OwnedFd},
    ptr,
    sync::Arc,
};

use bitflags::bitflags;
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    memory::{
        MemoryAccessDescriptor, MemoryAccessFlag, MemoryAllocationHandleType, MemoryLocation,
    },
    try_ffi,
    types::DevicePtr,
    utility::{to_u32, to_usize, usage_bits},
};

bitflags! {
    /// Available export handle types for an allocation or memory pool.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AllocationHandleTypes: u32 {
        /// Allows a file descriptor to represent the allocation for IPC.
        const POSIX_FILE_DESCRIPTOR = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR as _;
        /// Allows a Win32 NT handle to represent the allocation for IPC.
        const WIN32 = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32 as _;
        /// Allows a Win32 KMT handle to represent the allocation for IPC.
        const WIN32_KMT = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32_KMT as _;
        /// Allows a fabric handle to represent the allocation for IPC.
        const FABRIC = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC as _;
    }
    /// Usage restrictions of a physical allocation.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct AllocationUsage: u16 {
        /// Allocation will be used as a tile pool.
        const TILE_POOL = usage_bits(sys::CU_MEM_CREATE_USAGE_TILE_POOL);
        /// Allocation will be used for hardware decompression.
        const HARDWARE_DECOMPRESSION = usage_bits(sys::CU_MEM_CREATE_USAGE_HW_DECOMPRESS);
        /// Permits PCIe GPUDirect RDMA access; valid only when RDMA capability is set.
        const GPU_DIRECT_RDMA_OVER_PCIE = usage_bits(sys::CU_MEM_CREATE_USAGE_GPU_DIRECT_RDMA_OVER_PCIE);
    }
}

/// Specifies the allocation properties for an allocation.
#[derive(Debug, Clone, Copy)]
pub struct AllocationOptions {
    /// Location of allocation.
    pub location: MemoryLocation,
    /// Handle type to export the allocation as.
    pub handle_type: MemoryAllocationHandleType,
    /// Allocating compressible memory.
    pub compressible: bool,
    /// Enables `GPUDirect` RDMA access to this allocation.
    pub gpu_direct_rdma_capable: bool,
    /// Bit mask indicating intended usage for this allocation.
    pub usage: AllocationUsage,
}

impl AllocationOptions {
    /// Selects an allocation location with export, compression, RDMA, and special usage disabled.
    #[must_use]
    pub fn new(location: MemoryLocation) -> Self {
        Self {
            location,
            handle_type: MemoryAllocationHandleType::None,
            compressible: false,
            gpu_direct_rdma_capable: false,
            usage: AllocationUsage::empty(),
        }
    }

    fn raw(self) -> Result<sys::CUmemAllocationProp> {
        if self
            .usage
            .contains(AllocationUsage::GPU_DIRECT_RDMA_OVER_PCIE)
            && !self.gpu_direct_rdma_capable
        {
            return Err(Error::InvalidValue);
        }
        Ok(sys::CUmemAllocationProp {
            type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: self.handle_type.into(),
            location: self.location.into(),
            allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
                compressionType: u8::from(self.compressible),
                gpuDirectRDMACapable: u8::from(self.gpu_direct_rdma_capable),
                usage: self.usage.bits(),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    /// Calculates either the minimal or recommended allocation granularity.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if RDMA usage over PCI Express is selected without enabling RDMA capability. Also propagates native granularity query errors for invalid or unsupported allocation properties and an error if the returned byte granularity does not fit `usize`.
    pub fn granularity(self, recommended: bool) -> Result<usize> {
        let mut size = 0;
        let flag = if recommended {
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        } else {
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM
        };
        // SAFETY: `cuMemGetAllocationGranularity` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMemGetAllocationGranularity(
                &raw mut size,
                &self.raw()?,
                flag
            ))?;
        }
        to_usize(size, "allocation granularity")
    }
}

/// A physical allocation without a device or host mapping.
#[derive(Debug)]
pub struct Allocation {
    handle: sys::CUmemGenericAllocationHandle,
    size: usize,
    context: Arc<Context>,
}

impl Allocation {
    /// Creates a memory allocation on the target device specified by the properties.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` for a zero size, a size not divisible by the minimum allocation granularity, or RDMA usage over PCI Express without RDMA capability. Also propagates context binding, granularity query, and physical allocation errors.
    pub fn create(
        context: &Arc<Context>,
        size: usize,
        options: AllocationOptions,
    ) -> Result<Arc<Self>> {
        context.bind()?;
        let granularity = options.granularity(false)?;
        if size == 0 || !size.is_multiple_of(granularity) {
            return Err(Error::InvalidValue);
        }
        let mut handle = 0;
        // SAFETY: `cuMemCreate` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMemCreate(
                &raw mut handle,
                size as _,
                &options.raw()?,
                0
            ))?;
        }
        Ok(Arc::new(Self {
            handle,
            size,
            context: Arc::clone(context),
        }))
    }

    /// Returns the borrowed native physical-allocation handle without retaining another native reference.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUmemGenericAllocationHandle {
        self.handle
    }

    /// Returns the physical allocation size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the retained context associated with this allocation reference.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Takes ownership of an existing allocation reference.
    ///
    /// # Safety
    /// `handle` must be a live allocation reference owned by the caller, with
    /// at least `size` bytes, valid for `context`. Ownership is transferred.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if `handle` or `size` is zero. The caller must satisfy the remaining allocation validity and ownership requirements.
    pub unsafe fn from_raw(
        context: &Arc<Context>,
        handle: sys::CUmemGenericAllocationHandle,
        size: usize,
    ) -> Result<Arc<Self>> {
        if handle == 0 || size == 0 {
            return Err(Error::InvalidValue);
        }
        Ok(Arc::new(Self {
            handle,
            size,
            context: Arc::clone(context),
        }))
    }

    /// Transfers the allocation reference, size and retained context to the caller.
    #[must_use]
    pub fn into_raw(self) -> (sys::CUmemGenericAllocationHandle, usize, Arc<Context>) {
        let owner = ManuallyDrop::new(self);
        // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
        (owner.handle, owner.size, unsafe {
            ptr::read(&raw const owner.context)
        })
    }

    /// Exports the allocation as an opaque file descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot export this allocation as a POSIX file descriptor, including when it was not created with the required export handle type. The returned descriptor is owned by the caller.
    #[cfg(unix)]
    pub fn export_fd(&self) -> Result<OwnedFd> {
        use std::os::fd::FromRawFd;
        let mut fd = -1;
        // SAFETY: `fd` is a writable local sized for the requested handle type.
        unsafe {
            try_ffi!(sys::cuMemExportToShareableHandle(
                (&raw mut fd).cast(),
                self.handle,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0
            ))?;
            Ok(OwnedFd::from_raw_fd(fd))
        }
    }

    /// Imports an allocation from an opaque file descriptor.
    ///
    /// # Safety
    /// `size` must be the size of the exported allocation. The descriptor must
    /// refer to a CUDA allocation accessible in `context`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if `size` is zero or CUDA returns a zero allocation handle. Also propagates context binding and native import errors. The supplied file descriptor remains borrowed; this method does not take descriptor ownership.
    #[cfg(unix)]
    pub unsafe fn import_fd(
        context: &Arc<Context>,
        fd: BorrowedFd<'_>,
        size: usize,
    ) -> Result<Arc<Self>> {
        use std::os::fd::AsRawFd;
        if size == 0 {
            return Err(Error::InvalidValue);
        }
        context.bind()?;
        let mut handle = 0;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            try_ffi!(sys::cuMemImportFromShareableHandle(
                &raw mut handle,
                fd.as_raw_fd() as usize as *mut _,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            ))?;
            Self::from_raw(context, handle, size)
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: `cuMemRelease` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuMemRelease(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to release cuda memory allocation handle: {error}");
            }
        }
    }
}

/// A reserved virtual address range, without backing memory.
#[derive(Debug)]
pub struct AddressReservation {
    address: sys::CUdeviceptr,
    size: usize,
    context: Arc<Context>,
}

impl AddressReservation {
    /// Reserves an address range with the specified size and alignment.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` for zero `size` or a nonzero alignment that is not a power of two. Also propagates context binding and native reservation errors, including unsupported size or alignment. An alignment of zero requests CUDA's default alignment.
    pub fn create(context: &Arc<Context>, size: usize, alignment: usize) -> Result<Self> {
        if size == 0 || (alignment != 0 && !alignment.is_power_of_two()) {
            return Err(Error::InvalidValue);
        }
        context.bind()?;
        let mut address = 0;
        // SAFETY: `cuMemAddressReserve` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuMemAddressReserve(
                &raw mut address,
                size as _,
                alignment as _,
                0,
                0
            ))?;
        }
        Ok(Self {
            address,
            size,
            context: Arc::clone(context),
        })
    }

    /// Takes ownership of a reserved virtual address range.
    ///
    /// # Safety
    /// The range must be live, uniquely owned, associated with `context`, and unmapped.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if `size` is zero or `address` is null. The caller must satisfy the remaining reservation validity and ownership requirements.
    pub unsafe fn from_raw(
        context: &Arc<Context>,
        address: DevicePtr,
        size: usize,
    ) -> Result<Self> {
        if size == 0 || address.as_ptr().is_null() {
            return Err(Error::InvalidValue);
        }
        Ok(Self {
            address: address.as_ptr() as _,
            size,
            context: Arc::clone(context),
        })
    }

    /// Transfers the reserved range and retained context to the caller.
    #[must_use]
    pub fn into_raw(self) -> (DevicePtr, usize, Arc<Context>) {
        let owner = ManuallyDrop::new(self);
        // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
        (owner.address(), owner.size, unsafe {
            ptr::read(&raw const owner.context)
        })
    }

    /// Returns the base of the reserved virtual address range.
    ///
    /// A reservation alone has no backing storage or memory-access permission.
    #[must_use]
    pub fn address(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.address as *mut ()) }
    }

    /// Returns the reserved address-range size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Maps an allocation into the reserved virtual address range.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if the reservation and allocation have different actual contexts or `offset + reservation_size` overflows or exceeds the physical allocation. Also propagates context binding and native mapping errors, including native granularity or offset restrictions. The consumed reservation is released on failure.
    pub fn map(self, allocation: Arc<Allocation>, offset: usize) -> Result<Mapping> {
        if self.context.as_raw() != allocation.context.as_raw()
            || offset
                .checked_add(self.size)
                .is_none_or(|end| end > allocation.size)
        {
            return Err(Error::InvalidValue);
        }
        self.context.bind()?;
        // SAFETY: `cuMemMap` receives only plain values.
        unsafe {
            try_ffi!(sys::cuMemMap(
                self.address,
                self.size as _,
                offset as _,
                allocation.handle,
                0
            ))?;
        }
        Ok(Mapping {
            reservation: self,
            allocation,
        })
    }
}

impl Drop for AddressReservation {
    fn drop(&mut self) {
        // SAFETY: `cuMemAddressFree` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuMemAddressFree(self.address, self.size as _)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to free cuda virtual address range: {error}");
            }
        }
    }
}

/// A mapping that retains both its virtual reservation and physical allocation.
#[derive(Debug)]
pub struct Mapping {
    reservation: AddressReservation,
    allocation: Arc<Allocation>,
}

impl Mapping {
    /// Returns the mapped device virtual address.
    ///
    /// Set appropriate permissions with [`Self::set_access`] before device use, and keep this mapping alive through every queued access.
    #[must_use]
    pub fn address(&self) -> DevicePtr {
        self.reservation.address()
    }

    /// Returns the mapped range size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.reservation.size
    }

    /// Borrows the physical allocation retained by this mapping.
    #[must_use]
    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }

    /// Sets the access flags for the mapped virtual address range.
    ///
    /// # Safety
    /// No operation may access the mapping while permissions are changed, and
    /// all queued uses must finish before this mapping is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the target locations, access flags, mapped address range, or permission change. No Rust-side access permission or initialization claim is established by the descriptors.
    pub unsafe fn set_access(&mut self, access: &[MemoryAccessDescriptor]) -> Result<()> {
        let raw = access
            .iter()
            .copied()
            .map(sys::CUmemAccessDesc::from)
            .collect::<Vec<_>>();
        // SAFETY: `cuMemSetAccess` receives pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMemSetAccess(
                self.reservation.address,
                self.reservation.size as _,
                raw.as_ptr(),
                raw.len() as _
            ))
        }
    }

    /// Returns the access flags for a location and the mapped address.
    ///
    /// # Errors
    ///
    /// Propagates native access-query errors. Returns an error if the returned flags do not fit `u32`, or `Error::InvalidValue` if they are not a recognized [`MemoryAccessFlag`].
    pub fn access(&self, location: MemoryLocation) -> Result<MemoryAccessFlag> {
        let mut flags = 0;
        // SAFETY: `cuMemGetAccess` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMemGetAccess(
                &raw mut flags,
                &location.into(),
                self.reservation.address
            ))?;
        }
        MemoryAccessFlag::try_from(to_u32(flags, "memory access flags")?)
            .map_err(|_| Error::InvalidValue)
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: `cuMemUnmap` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuMemUnmap(
                self.reservation.address,
                self.reservation.size as _
            )) {
                #[cfg(debug_assertions)]
                eprintln!("failed to unmap cuda virtual memory: {error}");
            }
        }
    }
}
