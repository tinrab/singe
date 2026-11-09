use std::{ptr, result::Result as StdResult, sync::Arc};

use singe_cuda_sys::driver;

use super::{
    MemoryAccessDescriptor, MemoryAccessFlag, MemoryAllocationType, MemoryLocation, MemoryPool,
    MemoryPoolAttribute, MemoryPoolAttributeValue, MemoryPoolOwner, MemoryPoolProps,
    PoolAttributeValue,
};
use crate::{
    context::Context,
    error::{Error, Result},
    try_ffi,
    utility::usage_bits,
    virtual_memory::AllocationHandleTypes,
};

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
    /// Returns an error if `props` describes an unsupported pool, CUDA cannot create the pool, or CUDA returns a null memory-pool handle.
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
    ///   When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)
    /// * [`MemoryPoolAttribute::ReuseFollowEventDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to use memory asynchronously freed in another stream as long as a stream ordering dependency of the allocating stream on the free action exists.
    ///   CUDA events and null stream interactions can create the required stream ordered dependencies. (default enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowOpportunistic`]: allows reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowInternalDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to insert new stream dependencies to establish the stream ordering required to reuse a piece of memory released by [`sys::cuMemFreeAsync`](singe_cuda_sys::driver::cuMemFreeAsync) (default enabled).
    /// * [`MemoryPoolAttribute::ReservedMemoryHigh`]: resets the high watermark that tracks the amount of backing memory allocated for the memory pool.
    ///   It is illegal to set this attribute to a non-zero value.
    /// * [`MemoryPoolAttribute::UsedMemoryHigh`]: resets the high watermark that tracks the amount of used memory allocated for the memory pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the value variant does not match the attribute, context binding fails, or CUDA rejects the attribute or value.
    /// Current-usage statistics are read-only, and high-water marks can only be reset to zero.
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
    ///   When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)
    /// * [`MemoryPoolAttribute::ReuseFollowEventDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to use memory asynchronously freed in another stream as long as a stream ordering dependency of the allocating stream on the free action exists.
    ///   CUDA events and null stream interactions can create the required stream ordered dependencies. (default enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowOpportunistic`]: allows reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)
    /// * [`MemoryPoolAttribute::ReuseAllowInternalDependencies`]: allows [`sys::cuMemAllocAsync`](singe_cuda_sys::driver::cuMemAllocAsync) to insert new stream dependencies to establish the stream ordering required to reuse a piece of memory released by [`sys::cuMemFreeAsync`](singe_cuda_sys::driver::cuMemFreeAsync) (default enabled).
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
    /// Propagates context-binding and CUDA default-pool lookup errors, including unsupported location or allocation-type combinations.
    /// Rejects a null returned pool handle.
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
    /// This changes CUDA's shared pool selection.
    /// Destroying the selected owned pool restores the driver's default pool.
    /// Existing allocations retain their original backing memory regardless of later selection changes.
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

impl PoolAttributeValue for i32 {}

impl PoolAttributeValue for u32 {}

impl PoolAttributeValue for u64 {}

impl MemoryPool {
    /// Returns the locality domain ID of a localized pool, or `None` for a full-device pool.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors.
    /// Also rejects an unexpected negative locality-domain identifier other than the full-device sentinel.
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
    /// Propagates context-binding and CUDA pool-attribute query errors.
    /// Also rejects an unknown allocation-type value.
    pub fn allocation_type(&self) -> Result<MemoryAllocationType> {
        let value: u32 =
            self.read_attribute(driver::CUmemPool_attribute::CU_MEMPOOL_ATTR_ALLOCATION_TYPE)?;
        MemoryAllocationType::try_from(value).map_err(|_| Error::InvalidValue)
    }

    /// Returns the location of the memory pool.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA pool-attribute query errors.
    /// Also rejects unknown location types and identifiers that do not fit the localized device representation.
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
