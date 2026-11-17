//! Virtual address reservations, physical allocations and mappings.

use std::{
    mem::ManuallyDrop,
    os::fd::{BorrowedFd, OwnedFd},
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use bitflags::bitflags;
use singe_core::report_cleanup_failure;
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    memory::{MemoryAccessDescriptor, MemoryAccessFlag, MemoryLocation},
    try_ffi,
    types::DevicePtr,
    utility::{to_c_ulong, to_u32, to_usize, usage_bits},
};

bitflags! {
    /// Available export handle types for an allocation or memory pool.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct AllocationHandleTypes: u32 {
        /// Allows a file descriptor to represent the allocation for IPC.
        const POSIX_FILE_DESCRIPTOR = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR.0;
        /// Allows a Win32 NT handle to represent the allocation for IPC.
        const WIN32 = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32.0;
        /// Allows a Win32 KMT handle to represent the allocation for IPC.
        const WIN32_KMT = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32_KMT.0;
        /// Allows a fabric handle to represent the allocation for IPC.
        const FABRIC = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC.0;
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
    /// Handle types the allocation can be exported as; empty disables export.
    pub handle_types: AllocationHandleTypes,
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
            handle_types: AllocationHandleTypes::empty(),
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
            return Err(Error::out_of_range("allocation usage"));
        }
        Ok(sys::CUmemAllocationProp {
            type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: sys::CUmemAllocationHandleType(self.handle_types.bits()),
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
    /// Returns an error if RDMA usage over PCI Express is selected without enabling RDMA capability, CUDA rejects the allocation properties, or the granularity does not fit `usize`.
    pub fn granularity(self, recommended: bool) -> Result<usize> {
        let mut size = 0;
        let flag = if recommended {
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        } else {
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM
        };
        // SAFETY: `size` is a local output, and the properties are a temporary that lives through the call.
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

/// Raw ownership of a physical allocation reference and its context.
///
/// Release `handle` while `context` is alive, or re-adopt the value with [`Allocation::from_raw`].
#[derive(Debug)]
pub struct RawAllocation {
    /// Native allocation reference.
    pub handle: sys::CUmemGenericAllocationHandle,
    /// Allocation size in bytes.
    pub size: usize,
    /// Context the allocation belongs to.
    pub context: Arc<Context>,
}

/// Raw ownership of a reserved virtual address range and its context.
///
/// Free the range while `context` is alive, or re-adopt the value with [`AddressReservation::from_raw`].
#[derive(Debug)]
pub struct RawAddressReservation {
    /// Base of the reserved range.
    pub address: DevicePtr,
    /// Range size in bytes.
    pub size: usize,
    /// Context the range belongs to.
    pub context: Arc<Context>,
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
    /// Returns an error if `size` is zero or not a multiple of the minimum allocation granularity, RDMA usage over PCI Express is selected without RDMA capability, the context cannot be bound, or CUDA cannot allocate the memory.
    pub fn create(
        context: &Arc<Context>,
        size: usize,
        options: AllocationOptions,
    ) -> Result<Arc<Self>> {
        context.bind()?;
        let granularity = options.granularity(false)?;
        if size == 0 {
            return Err(Error::not_positive("size"));
        }

        if !size.is_multiple_of(granularity) {
            return Err(Error::misaligned("size", granularity));
        }

        let mut handle = 0;
        let native_size = to_c_ulong(size, "allocation size")?;
        // SAFETY: `handle` is a local output, the properties are a temporary that lives through the call, and `context` is bound.
        unsafe {
            try_ffi!(sys::cuMemCreate(
                &raw mut handle,
                native_size,
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
    ///
    /// `raw.handle` must be a live allocation reference owned by the caller, with at least `raw.size` bytes, valid for `raw.context`.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` or `raw.size` is zero.
    pub unsafe fn from_raw(raw: RawAllocation) -> Result<Arc<Self>> {
        if raw.handle == 0 {
            return Err(Error::NullHandle);
        }

        if raw.size == 0 {
            return Err(Error::not_positive("size"));
        }
        Ok(Arc::new(Self {
            handle: raw.handle,
            size: raw.size,
            context: raw.context,
        }))
    }

    /// Transfers the allocation reference, size and context to the caller.
    ///
    /// # Errors
    ///
    /// Returns an error holding the allocation unchanged if another owner, such as a mapping, still retains it.
    pub fn into_raw(this: Arc<Self>) -> StdResult<RawAllocation, Arc<Self>> {
        let owner = ManuallyDrop::new(Arc::try_unwrap(this)?);
        Ok(RawAllocation {
            handle: owner.handle,
            size: owner.size,
            // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
            context: unsafe { ptr::read(&raw const owner.context) },
        })
    }

    /// Exports the allocation as an opaque file descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot export this allocation as a POSIX file descriptor, including when it was not created with the required export handle type.
    /// The returned descriptor is owned by the caller.
    #[cfg(unix)]
    pub fn export_fd(&self) -> Result<OwnedFd> {
        use std::os::fd::FromRawFd;
        let mut fd = -1;
        // SAFETY: `fd` is a local `c_int`, the type CUDA writes for a POSIX file descriptor handle, and `self` owns the live allocation.
        unsafe {
            try_ffi!(sys::cuMemExportToShareableHandle(
                (&raw mut fd).cast(),
                self.handle,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0
            ))?;
        }
        // SAFETY: on success CUDA returns a new open descriptor that nothing else owns.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Imports an allocation from an opaque file descriptor.
    ///
    /// # Safety
    ///
    /// `size` must be the size of the exported allocation.
    /// The descriptor must refer to a CUDA allocation accessible in `context`.
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is zero, the descriptor is negative, the context cannot be bound, or CUDA rejects the import.
    /// The file descriptor stays borrowed; this method does not take its ownership.
    #[cfg(unix)]
    pub unsafe fn import_fd(
        context: &Arc<Context>,
        fd: BorrowedFd<'_>,
        size: usize,
    ) -> Result<Arc<Self>> {
        use std::os::fd::AsRawFd;
        if size == 0 {
            return Err(Error::not_positive("size"));
        }

        context.bind()?;
        let os_handle = ptr::without_provenance_mut(to_usize(fd.as_raw_fd(), "file descriptor")?);
        let mut handle = 0;
        // SAFETY: CUDA reads the descriptor value from the pointer-sized handle, `handle` is a local output, and the caller guarantees the descriptor names an allocation of `size` bytes accessible in `context`.
        unsafe {
            try_ffi!(sys::cuMemImportFromShareableHandle(
                &raw mut handle,
                os_handle,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            ))?;
            Self::from_raw(RawAllocation {
                handle,
                size,
                context: Arc::clone(context),
            })
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // A handle whose context cannot be bound leaks rather than being released under another context.
        // SAFETY: this wrapper owns the allocation reference, and mappings that use it retain this owner, so none is live.
        let result = self
            .context
            .bind()
            .and_then(|()| unsafe { try_ffi!(sys::cuMemRelease(self.handle)) });

        if let Err(error) = result {
            report_cleanup_failure("release", "cuda memory allocation handle", &error);
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
    /// Returns an error if `size` is zero, a nonzero `alignment` is not a power of two, the context cannot be bound, or CUDA cannot reserve the range.
    /// An alignment of zero requests CUDA's default alignment.
    pub fn create(context: &Arc<Context>, size: usize, alignment: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::not_positive("size"));
        }

        if alignment != 0 && !alignment.is_power_of_two() {
            return Err(Error::out_of_range("alignment"));
        }

        context.bind()?;
        let mut address = 0;
        let native_size = to_c_ulong(size, "reservation size")?;
        let native_alignment = to_c_ulong(alignment, "reservation alignment")?;
        // SAFETY: `address` is a local output, and `context` is bound.
        unsafe {
            try_ffi!(sys::cuMemAddressReserve(
                &raw mut address,
                native_size,
                native_alignment,
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
    ///
    /// The range must be live, uniquely owned, associated with `raw.context`, and unmapped.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.size` is zero or `raw.address` is null.
    pub unsafe fn from_raw(raw: RawAddressReservation) -> Result<Self> {
        if raw.size == 0 {
            return Err(Error::not_positive("size"));
        }

        if raw.address.as_ptr().is_null() {
            return Err(Error::null_pointer("address"));
        }
        Ok(Self {
            address: raw.address.as_ptr().addr() as sys::CUdeviceptr,
            size: raw.size,
            context: raw.context,
        })
    }

    /// Transfers the reserved range and context to the caller.
    #[must_use]
    pub fn into_raw(self) -> RawAddressReservation {
        let owner = ManuallyDrop::new(self);
        RawAddressReservation {
            address: owner.address(),
            size: owner.size,
            // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
            context: unsafe { ptr::read(&raw const owner.context) },
        }
    }

    /// Returns the base of the reserved virtual address range.
    ///
    /// A reservation alone has no backing storage or memory-access permission.
    #[must_use]
    pub fn address(&self) -> DevicePtr {
        // SAFETY: the address is the base of the range this reservation owns; `DevicePtr` claims no access to it.
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
    /// Returns an error if the reservation and allocation belong to different contexts, `offset` plus the reservation size overflows or exceeds the allocation, the context cannot be bound, or CUDA rejects the mapping, such as for an offset off the allocation granularity.
    /// The consumed reservation is released on failure.
    pub fn map(self, allocation: Arc<Allocation>, offset: usize) -> Result<Mapping> {
        if self.context.as_raw() != allocation.context.as_raw() {
            return Err(Error::MemoryContextMismatch);
        }

        let end = offset
            .checked_add(self.size)
            .ok_or_else(|| Error::out_of_range("mapping end"))?;
        if end > allocation.size {
            return Err(Error::buffer_too_small(
                "physical allocation",
                end,
                allocation.size,
            ));
        }

        self.context.bind()?;
        let native_size = to_c_ulong(self.size, "mapping size")?;
        let native_offset = to_c_ulong(offset, "mapping offset")?;
        // SAFETY: `self` owns the reserved range and `allocation` the physical memory, both in the bound context, and the mapped span lies inside the allocation, as checked above.
        unsafe {
            try_ffi!(sys::cuMemMap(
                self.address,
                native_size,
                native_offset,
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
        // SAFETY: this wrapper owns the reserved range, and a mapping that used it consumed and unmapped it first.
        let result = self.context.bind().and_then(|()| unsafe {
            try_ffi!(sys::cuMemAddressFree(self.address, self.size as _))
        });

        if let Err(error) = result {
            report_cleanup_failure("free", "cuda virtual address range", &error);
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
    ///
    /// No operation may access the mapping while permissions are changed, and all queued uses must finish before this mapping is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the size or descriptor count does not fit the native type, or CUDA rejects the target locations, access flags, mapped address range, or permission change.
    /// The descriptors establish no Rust-side access permission or initialization claim.
    pub unsafe fn set_access(&mut self, access: &[MemoryAccessDescriptor]) -> Result<()> {
        let raw = access
            .iter()
            .copied()
            .map(sys::CUmemAccessDesc::from)
            .collect::<Vec<_>>();
        let native_size = to_c_ulong(self.reservation.size, "mapping size")?;
        let count = to_c_ulong(raw.len(), "access descriptor count")?;
        // SAFETY: `self` owns the whole mapped range, `raw` holds `count` descriptors for the call, and the caller excludes concurrent access to the mapping.
        unsafe {
            try_ffi!(sys::cuMemSetAccess(
                self.reservation.address,
                native_size,
                raw.as_ptr(),
                count
            ))
        }
    }

    /// Returns the access flags for a location and the mapped address.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the access flags, or they do not fit `u32` or name no [`MemoryAccessFlag`].
    pub fn access(&self, location: MemoryLocation) -> Result<MemoryAccessFlag> {
        let mut flags = 0;
        // SAFETY: `flags` is a local output, the location is a temporary that lives through the call, and `self` owns the mapped address.
        unsafe {
            try_ffi!(sys::cuMemGetAccess(
                &raw mut flags,
                &location.into(),
                self.reservation.address
            ))?;
        }

        let flags = to_u32(flags, "memory access flags")?;
        MemoryAccessFlag::try_from(flags).map_err(|_| Error::UnknownEnumValue {
            name: "memory access flags".into(),
            value: i64::from(flags),
        })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this wrapper owns the mapping, and the caller of `set_access` keeps queued uses from outliving it.
        let result = self.reservation.context.bind().and_then(|()| unsafe {
            try_ffi!(sys::cuMemUnmap(
                self.reservation.address,
                self.reservation.size as _
            ))
        });

        if let Err(error) = result {
            report_cleanup_failure("unmap", "cuda virtual memory", &error);
        }
    }
}
