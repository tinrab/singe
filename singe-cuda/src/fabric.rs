//! Fabric multicast allocations and logical endpoints.
use std::{cell::RefCell, rc::Rc, sync::Arc};

use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    device::Device,
    driver::entry_points::{
        cuLogicalEndpointAddDevice, cuLogicalEndpointBindAddr, cuLogicalEndpointBindMem,
        cuLogicalEndpointCreate, cuLogicalEndpointDestroy, cuLogicalEndpointExport,
        cuLogicalEndpointGetLimits, cuLogicalEndpointIdRelease, cuLogicalEndpointIdReserve,
        cuLogicalEndpointImport, cuLogicalEndpointQuery, cuLogicalEndpointUnbind,
    },
    error::{Error, Result, Status},
    try_ffi,
    types::DevicePtr,
    utility::to_usize,
    virtual_memory::{AddressReservation, Allocation, AllocationHandleTypes, Mapping},
};

/// Fabric handle for a memory allocation.
#[derive(Debug, Clone, Copy)]
pub struct MemoryFabricHandle(pub [u8; 64]);

impl Allocation {
    /// Exports an allocation to a fabric handle.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot export the allocation as a fabric handle, including unsupported export properties or insufficient fabric permissions.
    pub fn export_fabric(&self) -> Result<MemoryFabricHandle> {
        let mut raw = sys::CUmemFabricHandle::default();
        // SAFETY: `cuMemExportToShareableHandle` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuMemExportToShareableHandle(
                (&raw mut raw).cast(),
                self.as_raw(),
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC,
                0
            ))?;
        }
        Ok(MemoryFabricHandle(raw.data))
    }

    /// Imports a memory allocation from a fabric handle.
    ///
    /// # Safety
    /// The handle must refer to a live allocation with at least `size` bytes,
    /// exported by an authorized process and accessible to `context`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if `size` is zero or CUDA returns a zero allocation handle. Also propagates context binding and native import errors for invalid handles, unsupported fabric access, or insufficient permissions.
    pub unsafe fn import_fabric(
        context: &Arc<Context>,
        handle: &MemoryFabricHandle,
        size: usize,
    ) -> Result<Arc<Self>> {
        if size == 0 {
            return Err(Error::InvalidValue);
        }
        context.bind()?;
        let mut raw = sys::CUmemFabricHandle { data: handle.0 };
        let mut allocation = 0;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            try_ffi!(sys::cuMemImportFromShareableHandle(
                &raw mut allocation,
                (&raw mut raw).cast(),
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC
            ))?;
            Self::from_raw(context, allocation, size)
        }
    }
}

/// Properties of a multicast object.
#[derive(Debug, Clone, Copy)]
pub struct MulticastOptions {
    /// Number of devices in the multicast team.
    pub devices: u32,
    /// Size of the multicast object in bytes.
    pub size: usize,
    /// Handle types from which this object can be imported and exported.
    pub handle_types: AllocationHandleTypes,
}

impl MulticastOptions {
    fn raw(self) -> sys::CUmulticastObjectProp {
        sys::CUmulticastObjectProp {
            numDevices: self.devices,
            size: self.size as _,
            handleTypes: u64::from(self.handle_types.bits()),
            flags: 0,
        }
    }

    /// Calculates the minimal or recommended granularity for the multicast object.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the multicast properties or cannot query the selected granularity. Also returns an error if the byte granularity does not fit `usize`.
    pub fn granularity(self, recommended: bool) -> Result<usize> {
        let mut size = 0;
        let flag = if recommended {
            sys::CUmulticastGranularity_flags::CU_MULTICAST_GRANULARITY_RECOMMENDED
        } else {
            sys::CUmulticastGranularity_flags::CU_MULTICAST_GRANULARITY_MINIMUM
        };
        // SAFETY: `cuMulticastGetGranularity` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMulticastGetGranularity(
                &raw mut size,
                &self.raw(),
                flag
            ))?;
        }
        to_usize(size, "multicast granularity")
    }
}

/// A multicast object whose physical allocation can be mapped into virtual memory.
#[derive(Debug)]
pub struct Multicast {
    allocation: Arc<Allocation>,
    devices: RefCell<Vec<Device>>,
    required_devices: u32,
}

impl Multicast {
    /// Creates a multicast object with the specified properties.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` for zero devices, zero size, a size not divisible by the minimum multicast granularity, or a zero returned allocation handle. Also propagates granularity query, context binding, and native multicast creation errors.
    pub fn create(context: &Arc<Context>, options: MulticastOptions) -> Result<Rc<Self>> {
        if options.devices == 0
            || options.size == 0
            || !options.size.is_multiple_of(options.granularity(false)?)
        {
            return Err(Error::InvalidValue);
        }
        context.bind()?;
        let mut handle = 0;
        // SAFETY: `cuMulticastCreate` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMulticastCreate(&raw mut handle, &options.raw()))?;
        }
        // SAFETY: `options` is a valid local properties struct, and `handle` is a writable local.
        let allocation = unsafe { Allocation::from_raw(context, handle, options.size)? };
        Ok(Rc::new(Self {
            allocation,
            devices: RefCell::default(),
            required_devices: options.devices,
        }))
    }

    /// Associates a device permanently with the multicast team.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if the device is already registered or the team already has its declared number of devices. Also propagates CUDA errors when adding the device. The local team list changes only after native registration succeeds.
    pub fn add_device(&self, device: Device) -> Result<()> {
        let mut devices = self.devices.borrow_mut();
        if devices.contains(&device) || devices.len() >= self.required_devices as usize {
            return Err(Error::InvalidValue);
        }
        // SAFETY: `cuMulticastAddDevice` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuMulticastAddDevice(
                self.allocation.as_raw(),
                device.id() as _
            ))?;
        }
        devices.push(device);
        Ok(())
    }

    /// Borrows the physical-allocation handle representing this multicast object.
    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }

    fn ready(&self) -> Result<()> {
        if self.devices.borrow().len() != self.required_devices as usize {
            return Err(Status::NotReady.into());
        }
        Ok(())
    }

    /// Maps this multicast object into the reserved virtual address range.
    ///
    /// # Errors
    ///
    /// Returns the native not-ready status if the team has fewer than its declared number of devices. Also propagates reservation-context, size, binding, and native mapping errors from [`AddressReservation::map`].
    pub fn map(self: &Rc<Self>, reservation: AddressReservation) -> Result<MulticastMapping> {
        self.ready()?;
        Ok(MulticastMapping {
            mapping: reservation.map(Arc::clone(&self.allocation), 0)?,
            _owner: Rc::clone(self),
        })
    }

    /// Binds an allocation to this multicast object for the specified device.
    ///
    /// # Errors
    ///
    /// Returns the native not-ready status if the team is incomplete, or `Error::InvalidValue` for zero size, range-end overflow, or a range exceeding either the multicast object or physical allocation. Also propagates native binding errors for device, granularity, or allocation incompatibilities.
    pub fn bind(
        self: &Rc<Self>,
        device: Device,
        offset: usize,
        allocation: Arc<Allocation>,
        allocation_offset: usize,
        size: usize,
    ) -> Result<MulticastBinding> {
        self.ready()?;
        check_range(offset, size, self.allocation.size())?;
        check_range(allocation_offset, size, allocation.size())?;
        // SAFETY: `cuMulticastBindMem_v2` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuMulticastBindMem_v2(
                self.allocation.as_raw(),
                device.id() as _,
                offset as _,
                allocation.as_raw(),
                allocation_offset as _,
                size as _,
                0
            ))?;
        }
        Ok(MulticastBinding {
            owner: Rc::clone(self),
            device,
            offset,
            size,
            allocation: Some(allocation),
            bound: true,
        })
    }

    /// Binds a virtual address to this multicast object.
    ///
    /// # Safety
    /// The address range must be eligible CUDA memory and remain valid until
    /// the binding is dropped. All accesses must finish before unbinding.
    ///
    /// # Errors
    ///
    /// Returns the native not-ready status if the team is incomplete, or `Error::InvalidValue` for zero size, range-end overflow, or a range exceeding the multicast object. Also propagates native address-binding errors; address validity and lifetime remain the caller's responsibility.
    pub unsafe fn bind_address(
        self: &Rc<Self>,
        device: Device,
        offset: usize,
        address: DevicePtr,
        size: usize,
    ) -> Result<MulticastBinding> {
        self.ready()?;
        check_range(offset, size, self.allocation.size())?;
        // SAFETY: `cuMulticastBindAddr_v2` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMulticastBindAddr_v2(
                self.allocation.as_raw(),
                device.id() as _,
                offset as _,
                address.as_ptr() as _,
                size as _,
                0
            ))?;
        }
        Ok(MulticastBinding {
            owner: Rc::clone(self),
            device,
            offset,
            size,
            allocation: None,
            bound: true,
        })
    }
}

/// A mapping retaining its multicast owner.
#[derive(Debug)]
pub struct MulticastMapping {
    /// Mapped virtual range and its retained physical allocation. Set access permissions before device use.
    pub mapping: Mapping,
    _owner: Rc<Multicast>,
}

/// A bound range retaining its multicast object and physical allocation.
#[derive(Debug)]
pub struct MulticastBinding {
    owner: Rc<Multicast>,
    device: Device,
    offset: usize,
    size: usize,
    allocation: Option<Arc<Allocation>>,
    bound: bool,
}

impl MulticastBinding {
    /// Unbinds this range after all device accesses have completed.
    ///
    /// # Errors
    ///
    /// Propagates errors from the native unbind operation. If unbinding fails, dropping the consumed binding attempts it again; the cleanup result is discarded.
    pub fn unbind(mut self) -> Result<()> {
        self.release()
    }

    fn release(&mut self) -> Result<()> {
        if self.bound {
            // SAFETY: `cuMulticastUnbind` receives live handles owned by the wrappers.
            unsafe {
                try_ffi!(sys::cuMulticastUnbind(
                    self.owner.allocation.as_raw(),
                    self.device.id() as _,
                    self.offset as _,
                    self.size as _
                ))?;
            }
            self.bound = false;
        }
        Ok(())
    }

    /// Returns the retained physical allocation for an allocation-backed binding, or `None` for a raw-address binding.
    #[must_use]
    pub fn allocation(&self) -> Option<&Arc<Allocation>> {
        self.allocation.as_ref()
    }
}

impl Drop for MulticastBinding {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

/// Logical endpoint type.
#[derive(Debug, Clone, Copy)]
pub enum EndpointKind {
    /// Unicast logical endpoint associated with a single device.
    Unicast(Device),
    /// Multicast logical endpoint associated with a team of devices.
    Multicast {
        /// Number of devices in the multicast logical-endpoint team.
        devices: u32,
    },
}

/// Properties of a logical endpoint construction.
#[derive(Debug, Clone, Copy)]
pub struct EndpointOptions {
    /// Type of the logical endpoint.
    pub kind: EndpointKind,
    /// Size in bytes of the logical endpoint.
    pub size: u64,
    /// Allow export using a fabric IPC handle.
    pub exportable: bool,
    /// Enable counted operations on the logical endpoint.
    pub counted_operations: bool,
}

impl EndpointOptions {
    fn raw(self) -> sys::CUlogicalEndpointProp {
        let mut raw = sys::CUlogicalEndpointProp {
            size: self.size,
            ipcHandleTypes: u32::from(self.exportable),
            flags: u32::from(self.counted_operations),
            ..Default::default()
        };
        match self.kind {
            EndpointKind::Unicast(device) => {
                raw.type_ = sys::CUlogicalEndpointType::CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
                raw.__bindgen_anon_1.unicast.device = device.id() as _;
            }
            EndpointKind::Multicast { devices } => {
                raw.type_ = sys::CUlogicalEndpointType::CU_LOGICAL_ENDPOINT_TYPE_MULTICAST;
                raw.__bindgen_anon_1.multicast.numDevices = devices;
            }
        }
        raw
    }

    /// Returns binding alignment and maximum size for these properties.
    ///
    /// # Errors
    ///
    /// Returns an error if the required driver entry point is unavailable or CUDA rejects the endpoint properties or cannot query their binding limits.
    pub fn limits(self) -> Result<EndpointLimits> {
        let mut limits = EndpointLimits {
            bind_alignment: 0,
            maximum_size: 0,
        };
        // SAFETY: `cuLogicalEndpointGetLimits` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            cuLogicalEndpointGetLimits(
                &raw mut limits.bind_alignment,
                &raw mut limits.maximum_size,
                &self.raw(),
            )?;
        }
        Ok(limits)
    }
}

/// Limits supported by a logical endpoint configuration.
#[derive(Debug, Clone, Copy)]
pub struct EndpointLimits {
    /// Required alignment for binding offsets and sizes.
    pub bind_alignment: u64,
    /// Maximum supported endpoint size in bytes.
    pub maximum_size: u64,
}

/// Fabric handle for a logical endpoint.
#[derive(Debug, Clone, Copy)]
pub struct EndpointFabricHandle(pub [u8; 64]);

/// A reserved logical endpoint ID and its constructed logical endpoint.
#[derive(Debug)]
pub struct LogicalEndpoint {
    id: u32,
    size: u64,
    context: Arc<Context>,
}

impl LogicalEndpoint {
    /// Reserves an ID and creates a logical endpoint with the requested properties.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if the requested size is zero, exceeds the native maximum, or is not a multiple of the binding alignment. Also propagates context binding, missing driver entry-point, property query, ID reservation, and endpoint creation errors. A reserved ID is released when native creation fails.
    pub fn create(context: &Arc<Context>, options: EndpointOptions) -> Result<Rc<Self>> {
        context.bind()?;
        let limits = options.limits()?;
        if options.size == 0
            || options.size > limits.maximum_size
            || !options.size.is_multiple_of(limits.bind_alignment)
        {
            return Err(Error::InvalidValue);
        }
        let mut id = 0;
        // SAFETY: `cuLogicalEndpointIdReserve`, `cuLogicalEndpointCreate`, `cuLogicalEndpointIdRelease` receive valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            cuLogicalEndpointIdReserve(&raw mut id, 1)?;
            if let Err(error) = cuLogicalEndpointCreate(id, &options.raw()) {
                let _ = cuLogicalEndpointIdRelease(id, 1);
                return Err(error);
            }
        }
        Ok(Rc::new(Self {
            id,
            size: options.size,
            context: Arc::clone(context),
        }))
    }

    /// Returns the locally reserved native logical-endpoint ID.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Associates a device permanently with a multicast logical endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver entry point is unavailable or CUDA rejects adding this device to the endpoint, including an invalid device or incompatible endpoint team.
    pub fn add_device(&self, device: Device) -> Result<()> {
        // SAFETY: `cuLogicalEndpointAddDevice` receives only plain values.
        unsafe { cuLogicalEndpointAddDevice(self.id, device.id() as _) }
    }

    /// Determines whether this endpoint has been successfully constructed.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver entry point is unavailable or CUDA cannot query the endpoint. An endpoint whose construction is still pending returns `Ok(false)`.
    pub fn is_ready(&self) -> Result<bool> {
        let mut ready = 0;
        // SAFETY: `cuLogicalEndpointQuery` receives valid output locations.
        unsafe {
            cuLogicalEndpointQuery(self.id, 1, &raw mut ready)?;
        }
        Ok(ready != 0)
    }

    fn ready(&self) -> Result<()> {
        if self.is_ready()? {
            Ok(())
        } else {
            Err(Status::NotReady.into())
        }
    }

    /// Exports this logical endpoint to a fabric handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver entry point is unavailable or CUDA cannot export the endpoint, including an endpoint not created with fabric export enabled or insufficient fabric permissions.
    pub fn export(&self) -> Result<EndpointFabricHandle> {
        let mut handle = sys::CUlogicalEndpointFabricHandle::default();
        // SAFETY: `cuLogicalEndpointExport` receives valid output locations.
        unsafe {
            cuLogicalEndpointExport(
                (&raw mut handle).cast(),
                self.id,
                sys::CUlogicalEndpointIpcHandleType::CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC,
            )?;
        }
        Ok(EndpointFabricHandle(handle.data))
    }

    /// Imports an exported logical endpoint using a newly reserved local ID.
    ///
    /// # Safety
    /// The handle must identify a valid exported endpoint of the supplied size.
    ///
    /// # Errors
    ///
    /// Propagates context binding, missing driver entry-point, local ID reservation, and native import errors. A reserved ID is released if import fails. Handle validity and the supplied size remain the caller's responsibility.
    pub unsafe fn import(
        context: &Arc<Context>,
        handle: &EndpointFabricHandle,
        size: u64,
    ) -> Result<Rc<Self>> {
        context.bind()?;
        let raw = sys::CUlogicalEndpointFabricHandle { data: handle.0 };
        let mut id = 0;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            cuLogicalEndpointIdReserve(&raw mut id, 1)?;
            if let Err(error) = cuLogicalEndpointImport(
                id,
                (&raw const raw).cast(),
                sys::CUlogicalEndpointIpcHandleType::CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC,
            ) {
                let _ = cuLogicalEndpointIdRelease(id, 1);
                return Err(error);
            }
        }
        Ok(Rc::new(Self {
            id,
            size,
            context: Arc::clone(context),
        }))
    }

    /// Binds a physical allocation to this logical endpoint.
    ///
    /// # Errors
    ///
    /// Returns the native not-ready status if endpoint construction is incomplete, or `Error::InvalidValue` for zero allocation range size, range-end overflow, or a range exceeding the endpoint or physical allocation. Also propagates missing driver entry-point and native binding errors for device, alignment, or allocation incompatibilities.
    pub fn bind(
        self: &Rc<Self>,
        device: Device,
        offset: u64,
        allocation: Arc<Allocation>,
        allocation_offset: usize,
        size: usize,
    ) -> Result<EndpointBinding> {
        self.ready()?;
        check_range(allocation_offset, size, allocation.size())?;
        if offset
            .checked_add(size as u64)
            .is_none_or(|end| end > self.size)
        {
            return Err(Error::InvalidValue);
        }
        // SAFETY: `cuLogicalEndpointBindMem` receives live handles owned by the wrappers.
        unsafe {
            cuLogicalEndpointBindMem(
                self.id,
                device.id() as _,
                offset,
                allocation.as_raw(),
                allocation_offset as _,
                size as _,
                0,
            )?;
        }
        Ok(EndpointBinding {
            owner: Rc::clone(self),
            device,
            offset,
            size: size as _,
            allocation: Some(allocation),
            bound: true,
        })
    }

    /// Binds a virtual address range to this logical endpoint.
    ///
    /// # Safety
    /// The memory must remain valid until the binding is dropped. All accesses
    /// must finish before unbinding.
    ///
    /// # Errors
    ///
    /// Returns the native not-ready status if endpoint construction is incomplete, or `Error::InvalidValue` if `offset + size` overflows or exceeds the endpoint. Also propagates missing driver entry-point and native address-binding errors. Pointer validity and lifetime remain the caller's responsibility.
    pub unsafe fn bind_address(
        self: &Rc<Self>,
        device: Device,
        offset: u64,
        pointer: DevicePtr,
        size: usize,
    ) -> Result<EndpointBinding> {
        self.ready()?;
        if offset
            .checked_add(size as u64)
            .is_none_or(|end| end > self.size)
        {
            return Err(Error::InvalidValue);
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            cuLogicalEndpointBindAddr(
                self.id,
                device.id() as _,
                offset,
                pointer.as_ptr().cast(),
                size as _,
                0,
            )?;
        }
        Ok(EndpointBinding {
            owner: Rc::clone(self),
            device,
            offset,
            size: size as _,
            allocation: None,
            bound: true,
        })
    }
}

impl Drop for LogicalEndpoint {
    fn drop(&mut self) {
        let _ = self.context.bind();
        // SAFETY: `cuLogicalEndpointDestroy`, `cuLogicalEndpointIdRelease` receive only plain values.
        unsafe {
            let _ = cuLogicalEndpointDestroy(self.id);
            let _ = cuLogicalEndpointIdRelease(self.id, 1);
        }
    }
}

/// A bound range retaining its logical endpoint and allocation.
#[derive(Debug)]
pub struct EndpointBinding {
    owner: Rc<LogicalEndpoint>,
    device: Device,
    offset: u64,
    size: u64,
    allocation: Option<Arc<Allocation>>,
    bound: bool,
}

impl EndpointBinding {
    /// Returns the retained physical allocation for an allocation-backed binding, or `None` for a raw-address binding.
    #[must_use]
    pub fn allocation(&self) -> Option<&Arc<Allocation>> {
        self.allocation.as_ref()
    }

    /// Unbinds this range after all device accesses have completed.
    ///
    /// # Errors
    ///
    /// Propagates errors from the native unbind operation. If unbinding fails, dropping the consumed binding attempts it again; the cleanup result is discarded.
    pub fn unbind(mut self) -> Result<()> {
        self.release()
    }

    fn release(&mut self) -> Result<()> {
        if self.bound {
            // SAFETY: `cuLogicalEndpointUnbind` receives only plain values.
            unsafe {
                cuLogicalEndpointUnbind(
                    self.owner.id,
                    self.device.id() as _,
                    self.offset,
                    self.size,
                )?;
            }
            self.bound = false;
        }
        Ok(())
    }
}

impl Drop for EndpointBinding {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

fn check_range(offset: usize, size: usize, total: usize) -> Result<()> {
    if size == 0 || offset.checked_add(size).is_none_or(|end| end > total) {
        Err(Error::InvalidValue)
    } else {
        Ok(())
    }
}
