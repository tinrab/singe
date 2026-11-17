use std::{ptr, result::Result as StdResult, sync::Arc};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};

use singe_cuda_sys::driver;

use crate::{
    context::Context,
    error::{Error, Result},
    memory::MemoryLocation,
    try_ffi,
    utility::{to_u32, to_u64, usage_bits},
    virtual_memory::AllocationHandleTypes,
};

impl MemoryPool {
    /// Creates a CUDA memory pool.
    /// `props` determines the properties of the pool such as the backing device and IPC capabilities.
    ///
    /// To create a memory pool for host memory not targeting a specific NUMA node, applications must set [`MemoryPoolProps::location`] to [`MemoryLocation::Host`].
    /// [`MemoryLocation`] is ignored for such pools.
    /// Pools created with [`MemoryLocation::Host`] are not IPC-capable and [`MemoryPoolProps::handle_types`] must be empty; any other value returns [`crate::error::Status::InvalidValue`].
    /// To create a memory pool targeting a specific host NUMA node, applications must set [`MemoryLocation`] to [`MemoryLocation::Numa`] and [`MemoryLocation`] must specify the NUMA ID of the host memory node.
    /// Specifying [`MemoryLocation::NumaCurrent`] as [`MemoryLocation`] returns [`crate::error::Status::InvalidValue`].
    /// By default, the pool's memory is accessible from the device where it is allocated.
    /// Pools created with [`MemoryLocation::Numa`] or [`MemoryLocation::Host`] are accessible from the host CPU by default.
    /// Applications can control the maximum size of the pool by specifying a non-zero value for [`MemoryPoolProps::max_size`].
    /// A value of 0 uses a system-dependent maximum pool size.
    ///
    /// Callers that intend to use [`AllocationHandleTypes::FABRIC`] based memory sharing must ensure: (1) the `nvidia-caps-imex-channels` character device is created by the driver and is listed under `/proc/devices`; (2) at least one IMEX channel file is accessible to the process.
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
    /// [`MemoryPoolProps::handle_types`] must also be empty because IPC is not supported.
    /// For managed memory pools, [`MemoryPoolProps::location`] is treated as the preferred location for all allocations created from the pool.
    /// An application can also set [`MemoryLocation::Unspecified`] to indicate no preferred location.
    /// [`MemoryPoolProps::max_size`] must be set to zero for managed memory pools.
    /// [`MemoryPoolProps::usage`] must be zero because decompression for managed memory is not supported.
    /// For managed memory pools, every device on the system must report [`DeviceAttribute::ConcurrentManagedAccess`](crate::device::DeviceAttribute::ConcurrentManagedAccess).
    /// If not, this call returns [`crate::error::Status::NotSupported`].
    ///
    /// Empty [`MemoryPoolProps::handle_types`] create a memory pool that does not support IPC.
    ///
    /// # Errors
    ///
    /// Returns an error if `props` describes an unsupported pool, CUDA cannot create the pool, or CUDA returns a null memory-pool handle.
    pub fn create(context: &Arc<Context>, props: MemoryPoolProps) -> Result<Self> {
        context.bind()?;
        let mut handle = ptr::null_mut();
        let props = driver::CUmemPoolProps::from(props);
        // SAFETY: `handle` is a writable local, `props` is a converted local that outlives the call, and `context` is bound.
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

    /// Sets a pool attribute.
    ///
    /// Settable attributes are:
    ///
    /// * [`MemoryPoolAttribute::ReleaseThreshold`]: the reserved bytes the pool keeps before it tries to release memory back to the OS.
    ///   When the pool holds more, it tries to release memory at the next stream, event, or context synchronization.
    ///   The default is 0.
    /// * [`MemoryPoolAttribute::ReuseFollowEventDependencies`]: lets a stream-ordered allocation reuse memory freed in another stream when the allocating stream depends on the free.
    ///   Events and the null stream can create that dependency.
    ///   The default is enabled.
    /// * [`MemoryPoolAttribute::ReuseAllowOpportunistic`]: lets an allocation reuse a completed free that it has no dependency on.
    ///   The default is enabled.
    /// * [`MemoryPoolAttribute::ReuseAllowInternalDependencies`]: lets a stream-ordered allocation insert stream dependencies so it can reuse memory freed in another stream.
    ///   The default is enabled.
    /// * [`MemoryPoolAttribute::ReservedMemoryHigh`]: resets the high watermark of the pool's backing memory, and only accepts zero.
    /// * [`MemoryPoolAttribute::UsedMemoryHigh`]: resets the high watermark of the pool's memory in use, and only accepts zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the value variant does not match the attribute, the context cannot be bound, or CUDA rejects the attribute or value.
    /// Current-usage statistics are read-only, and high watermarks only reset to zero.
    pub fn set_attribute(
        &mut self,
        attribute: MemoryPoolAttribute,
        value: MemoryPoolAttributeValue,
    ) -> Result<()> {
        self.context().bind()?;
        // SAFETY: the pool handle is live and its context is bound, and each `value` is a local of the width CUDA reads for its attribute: `u32` for the reuse flags, `u64` for byte counts.
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
                _ => return Err(Error::out_of_range("memory pool attribute value")),
            }
        }
        Ok(())
    }

    /// Returns a pool attribute.
    ///
    /// Besides the settable attributes described at [`MemoryPool::set_attribute`], it reports:
    ///
    /// * [`MemoryPoolAttribute::ReservedMemoryCurrent`]: the backing memory the pool holds.
    /// * [`MemoryPoolAttribute::ReservedMemoryHigh`]: the high watermark of backing memory since the last reset.
    /// * [`MemoryPoolAttribute::UsedMemoryCurrent`]: the pool's memory in use by the application.
    /// * [`MemoryPoolAttribute::UsedMemoryHigh`]: the high watermark of memory in use since the last reset.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot report the attribute.
    pub fn attribute(&self, attribute: MemoryPoolAttribute) -> Result<MemoryPoolAttributeValue> {
        self.context().bind()?;
        // SAFETY: the pool handle is live and its context is bound, and each `value` is a writable local of the width CUDA writes for its attribute: `u32` for the reuse flags, `u64` for byte counts.
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

    /// Sets which locations can access the pool's memory, and how.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA rejects the access descriptors.
    pub fn set_access(&mut self, access_descs: &[MemoryAccessDescriptor]) -> Result<()> {
        self.context().bind()?;
        let access_descs: Vec<_> = access_descs.iter().copied().map(Into::into).collect();
        let count = to_u64(access_descs.len(), "memory pool access descriptor count")?;
        // SAFETY: the pool handle is live and its context is bound, and `access_descs` is a local vector of `count` converted descriptors.
        unsafe {
            try_ffi!(driver::cuMemPoolSetAccess(
                self.owner.handle,
                access_descs.as_ptr(),
                count,
            ))?;
        }
        Ok(())
    }

    /// Returns the accessibility of the pool's memory from the specified location.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot report access from `location`.
    pub fn access(&self, location: MemoryLocation) -> Result<MemoryAccessFlag> {
        self.context().bind()?;
        let mut flags = driver::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE;
        let mut location = driver::CUmemLocation_st::from(location);
        // SAFETY: `flags` and `location` are writable locals, the pool handle is live, and its context is bound.
        unsafe {
            try_ffi!(driver::cuMemPoolGetAccess(
                &raw mut flags,
                self.owner.handle,
                &raw mut location,
            ))?;
        }
        Ok(flags.try_into()?)
    }

    /// Releases memory back to the OS until the pool holds fewer than `min_bytes_to_keep` reserved bytes or has no more memory it can release.
    /// The allocator cannot release OS allocations that back outstanding asynchronous allocations.
    /// The OS allocations may happen at different granularity from the caller's allocations.
    ///
    /// * Allocations that have not been freed count as outstanding.
    /// * Allocations that have been asynchronously freed but whose completion has not been observed on the host, for example by synchronization, can count as outstanding.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot trim the pool.
    pub fn trim_to(&mut self, min_bytes_to_keep: usize) -> Result<()> {
        let min_bytes_to_keep = to_u64(min_bytes_to_keep, "minimum bytes to keep")?;
        self.context().bind()?;
        // SAFETY: the pool handle is live, its context is bound, and the byte count is a plain integer.
        unsafe {
            try_ffi!(driver::cuMemPoolTrimTo(
                self.owner.handle,
                min_bytes_to_keep
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
    pub unsafe fn from_raw(raw: RawMemoryPool) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(MemoryPoolOwner {
                context: raw.context,
                handle: raw.handle,
                owned: true,
            }),
        })
    }

    /// Transfers an owned handle only when no allocations or other owners retain it.
    /// Driver-owned pools cannot transfer destruction responsibility.
    ///
    /// # Errors
    ///
    /// Returns the wrapper unchanged as the error if the pool is driver-owned or another wrapper or allocation retains it.
    pub fn into_raw(self) -> StdResult<RawMemoryPool, Self> {
        if !self.owner.owned {
            return Err(self);
        }
        match Arc::try_unwrap(self.owner) {
            Ok(mut owner) => {
                owner.owned = false;
                Ok(RawMemoryPool {
                    handle: owner.handle,
                    context: Arc::clone(&owner.context),
                })
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
    /// Returns an error if the context cannot be bound, CUDA has no default pool for the location and allocation type, or CUDA returns a null handle.
    pub fn default_for(
        context: &Arc<Context>,
        location: MemoryLocation,
        allocation_type: MemoryAllocationType,
    ) -> Result<Self> {
        context.bind()?;
        let mut handle = ptr::null_mut();
        let mut location = driver::CUmemLocation::from(location);
        // SAFETY: `handle` and `location` are writable locals, and `context` is bound.
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
    /// This changes CUDA's shared pool selection.
    /// Destroying the selected owned pool restores the driver's default pool.
    /// Existing allocations retain their original backing memory regardless of later selection changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the pool's location or allocation type cannot be read, or CUDA rejects the selection.
    pub fn set_current(&self) -> Result<()> {
        self.context().bind()?;
        let mut location = driver::CUmemLocation::from(self.location()?);
        let allocation_type = self.allocation_type()?;
        // SAFETY: `location` is a writable local, the pool handle is live for `&self`, and its context is bound.
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
    /// Returns an error if the context cannot be bound, CUDA cannot look up the default pool, or CUDA returns a null handle.
    pub fn default_memory_pool(self: &Arc<Self>) -> Result<MemoryPool> {
        MemoryPool::default_for(
            self,
            MemoryLocation::Device(self.device().id()),
            MemoryAllocationType::Pinned,
        )
    }
}

impl PoolAttributeValue for i32 {}

impl PoolAttributeValue for u32 {}

impl PoolAttributeValue for u64 {}

impl MemoryPool {
    /// Returns the locality domain ID of a localized pool, or `None` for a full-device pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot report the attribute, or the identifier is negative other than the full-device value of -1.
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

        Ok(Some(to_u32(value, "locality domain id")?))
    }

    /// Returns the allocation type of the memory pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot report the attribute, or the value is not a known allocation type.
    pub fn allocation_type(&self) -> Result<MemoryAllocationType> {
        let value: u32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_ALLOCATION_TYPE)?;
        MemoryAllocationType::try_from(value).map_err(|_| Error::UnknownEnumValue {
            name: "MemoryAllocationType".into(),
            value: i64::from(value),
        })
    }

    /// Returns the location of the memory pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot report the attributes, the location type is unknown, or an identifier does not fit the localized device representation.
    pub fn location(&self) -> Result<MemoryLocation> {
        let kind = self.location_type()?;
        let id: i32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_LOCATION_ID)?;
        if let Some(locality_domain_id) = self.locality_domain_id()? {
            return Ok(MemoryLocation::DeviceLocalityDomain {
                device_id: id
                    .try_into()
                    .map_err(|_| Error::out_of_range("pool device id"))?,
                locality_domain_id: locality_domain_id
                    .try_into()
                    .map_err(|_| Error::out_of_range("pool locality domain id"))?,
            });
        }
        Ok(driver::CUmemLocation {
            type_: kind,
            __bindgen_anon_1: driver::CUmemLocation_st__bindgen_ty_1 { id },
        }
        .into())
    }

    /// Returns the maximum size of the pool in bytes, including alignment requirements.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot report the attribute.
    pub fn maximum_size(&self) -> Result<u64> {
        self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_MAX_POOL_SIZE)
    }

    /// Returns whether hardware decompression is enabled for allocations from this pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot report the attribute.
    pub fn hardware_decompression_enabled(&self) -> Result<bool> {
        let value: i32 = self
            .read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED)?;
        Ok(value != 0)
    }

    /// Returns the handle types the pool can export.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot report the attribute.
    pub fn export_handle_types(&self) -> Result<AllocationHandleTypes> {
        let value: u32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES)?;
        Ok(AllocationHandleTypes::from_bits_retain(value))
    }

    fn location_type(&self) -> Result<driver::CUmemLocationType> {
        let value: u32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_LOCATION_TYPE)?;
        Ok(driver::CUmemLocationType(value))
    }

    // Reads an integer-valued pool attribute; enum attributes are read as `u32`.
    fn read_attribute<T: PoolAttributeValue>(
        &self,
        attribute: driver::CUmemPool_attribute,
    ) -> Result<T> {
        self.context().bind()?;
        let mut value = T::default();
        // SAFETY: `T` is an integer type whose size matches the attribute's documented representation, `value` is a writable local, the pool handle is live, and its context is bound.
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

singe_core::native_enum! {
    /// Kind of physical memory backing an allocation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryAllocationType: driver::CUmemAllocationType {
        /// Memory that cannot migrate from its current location while the application uses it.
        Pinned = CU_MEM_ALLOCATION_TYPE_PINNED,
        /// This allocation type is managed memory.
        Managed = CU_MEM_ALLOCATION_TYPE_MANAGED,
    }
}

singe_core::native_enum! {
    /// Specifies the memory protection flags for mapping.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryAccessFlag: driver::CUmemAccess_flags {
        /// Default, make the address range not accessible.
        None = CU_MEM_ACCESS_FLAGS_PROT_NONE,
        /// Make the address range read accessible.
        Read = CU_MEM_ACCESS_FLAGS_PROT_READ,
        /// Make the address range read-write accessible.
        ReadWrite = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    }
}

singe_core::native_enum! {
    /// Attribute of a CUDA memory pool, read and written as a [`MemoryPoolAttributeValue`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MemoryPoolAttribute: driver::CUmemPool_attribute {
        /// Allows stream-ordered allocations to reuse memory freed on another stream when the allocating stream depends on that free.
        /// Events and default-stream interactions can create the dependency.
        /// A [`MemoryPoolAttributeValue::Bool`], enabled by default.
        ReuseFollowEventDependencies = CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
        /// Allows reuse of completed frees that the allocation does not depend on.
        /// A [`MemoryPoolAttributeValue::Bool`], enabled by default.
        ReuseAllowOpportunistic = CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        /// Allows stream-ordered allocations to insert stream dependencies so they can reuse memory from stream-ordered frees.
        /// A [`MemoryPoolAttributeValue::Bool`], enabled by default.
        ReuseAllowInternalDependencies = CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
        /// Reserved bytes the pool holds before releasing memory back to the operating system.
        /// Above the threshold, the allocator releases memory at the next stream, event, or context synchronization.
        /// A [`MemoryPoolAttributeValue::Bytes`], zero by default.
        ReleaseThreshold = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
        /// Backing memory currently allocated for the pool, as [`MemoryPoolAttributeValue::Bytes`].
        ReservedMemoryCurrent = CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        /// High watermark of backing memory since the last reset, as [`MemoryPoolAttributeValue::Bytes`].
        /// It can only be reset to zero.
        ReservedMemoryHigh = CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        /// Pool memory currently in use by the application, as [`MemoryPoolAttributeValue::Bytes`].
        UsedMemoryCurrent = CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        /// High watermark of pool memory in use since the last reset, as [`MemoryPoolAttributeValue::Bytes`].
        /// It can only be reset to zero.
        UsedMemoryHigh = CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    }
}

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
    /// Handle types pool memory can be exported as; empty disables IPC export.
    pub handle_types: AllocationHandleTypes,
    /// Backing location, or preferred location for managed allocations.
    pub location: MemoryLocation,
    /// Maximum pool size in bytes; zero uses the system-dependent limit.
    pub max_size: usize,
    /// Additional usage requirements for allocations from the pool.
    pub usage: MemoryPoolUsage,
    /// Enables `GPUDirect` RDMA access for allocations from this pool.
    pub gpu_direct_rdma_capable: bool,
}

/// Raw ownership of a CUDA memory pool and its context.
///
/// Destroy `handle` while `context` is alive, or re-adopt the value with [`MemoryPool::from_raw`].
#[derive(Debug)]
pub struct RawMemoryPool {
    /// Native pool handle.
    pub handle: driver::CUmemoryPool,
    /// Context the pool was created for.
    pub context: Arc<Context>,
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
pub(crate) struct MemoryPoolOwner {
    context: Arc<Context>,
    handle: driver::CUmemoryPool,
    owned: bool,
}

// SAFETY: CUDA driver calls on a `CUmemoryPool` are thread-safe, and each call binds the retained context on the calling thread first.
unsafe impl Send for MemoryPoolOwner {}

// SAFETY: the owner exposes only the immutable handle, and concurrent driver calls on one pool are thread-safe.
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
            handleTypes: driver::CUmemAllocationHandleType(value.handle_types.bits()),
            location: value.location.into(),
            win32SecurityAttributes: ptr::null_mut(),
            maxSize: value.max_size as _,
            usage: value.usage.bits(),
            gpuDirectRDMACapable: value.gpu_direct_rdma_capable.into(),
            reserved: [0; 53],
        }
    }
}

impl Drop for MemoryPoolOwner {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }

        if let Err(error) = self.context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cuda memory pool",
                &error,
            );
            return;
        }

        // SAFETY: this owned pool was created by this wrapper, and the last `MemoryPool` clone and pool-backed allocation are gone.
        if let Err(error) = unsafe { try_ffi!(driver::cuMemPoolDestroy(self.handle)) } {
            report_cleanup_failure("destroy", "cuda memory pool", &error);
        }
    }
}

/// Integer types that memory pool attributes are read into; every bit pattern is valid.
pub(crate) trait PoolAttributeValue: Copy + Default {}

impl_enum_display!(MemoryAllocationType, {
    MemoryAllocationType::Pinned => "CU_MEM_ALLOCATION_TYPE_PINNED",
    MemoryAllocationType::Managed => "CU_MEM_ALLOCATION_TYPE_MANAGED",
});

impl_enum_display!(MemoryAccessFlag, {
    MemoryAccessFlag::None => "CU_MEM_ACCESS_FLAGS_PROT_NONE",
    MemoryAccessFlag::Read => "CU_MEM_ACCESS_FLAGS_PROT_READ",
    MemoryAccessFlag::ReadWrite => "CU_MEM_ACCESS_FLAGS_PROT_READWRITE",
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
