//! Fabric multicast allocations and logical endpoints.

use std::sync::{Arc, Mutex};

use singe_core::report_cleanup_failure;
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
    utility::{check_range, to_c_ulong, to_u64, to_usize},
    virtual_memory::{
        AddressReservation, Allocation, AllocationHandleTypes, Mapping, RawAllocation,
    },
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
        // SAFETY: `raw` is a local `CUmemFabricHandle`, the type CUDA writes for a fabric handle, and `self` owns the live allocation.
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
    ///
    /// The handle must refer to a live allocation with at least `size` bytes, exported by an authorized process and accessible to `context`.
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is zero, the context cannot be bound, CUDA rejects the import, such as for missing fabric permissions, or CUDA returns a zero handle.
    pub unsafe fn import_fabric(
        context: &Arc<Context>,
        handle: &MemoryFabricHandle,
        size: usize,
    ) -> Result<Arc<Self>> {
        if size == 0 {
            return Err(Error::not_positive("size"));
        }

        context.bind()?;
        let mut raw = sys::CUmemFabricHandle { data: handle.0 };
        let mut allocation = 0;
        // SAFETY: `raw` is a local fabric handle and `allocation` a local output, and the caller guarantees the handle names an allocation of at least `size` bytes accessible in `context`.
        unsafe {
            try_ffi!(sys::cuMemImportFromShareableHandle(
                &raw mut allocation,
                (&raw mut raw).cast(),
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC
            ))?;
            Self::from_raw(RawAllocation {
                handle: allocation,
                size,
                context: Arc::clone(context),
            })
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
    fn raw(self) -> Result<sys::CUmulticastObjectProp> {
        Ok(sys::CUmulticastObjectProp {
            numDevices: self.devices,
            size: to_c_ulong(self.size, "multicast size")?,
            handleTypes: u64::from(self.handle_types.bits()),
            flags: 0,
        })
    }

    /// Calculates the minimal or recommended granularity for the multicast object.
    ///
    /// # Errors
    ///
    /// Returns an error if the size does not fit the native type, CUDA rejects the multicast properties, or the granularity does not fit `usize`.
    pub fn granularity(self, recommended: bool) -> Result<usize> {
        let mut size = 0;
        let flag = if recommended {
            sys::CUmulticastGranularity_flags::CU_MULTICAST_GRANULARITY_RECOMMENDED
        } else {
            sys::CUmulticastGranularity_flags::CU_MULTICAST_GRANULARITY_MINIMUM
        };
        let properties = self.raw()?;
        // SAFETY: `size` is a local output, and `properties` lives through the call.
        unsafe {
            try_ffi!(sys::cuMulticastGetGranularity(
                &raw mut size,
                &raw const properties,
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
    devices: Mutex<Vec<Device>>,
    required_devices: u32,
}

impl Multicast {
    /// Creates a multicast object with the specified properties.
    ///
    /// # Errors
    ///
    /// Returns an error if the device count or size is zero, the size is not a multiple of the minimum multicast granularity, the context cannot be bound, CUDA cannot create the object, or CUDA returns a zero handle.
    pub fn create(context: &Arc<Context>, options: MulticastOptions) -> Result<Arc<Self>> {
        if options.devices == 0 {
            return Err(Error::not_positive("devices"));
        }

        if options.size == 0 {
            return Err(Error::not_positive("size"));
        }

        let granularity = options.granularity(false)?;
        if !options.size.is_multiple_of(granularity) {
            return Err(Error::misaligned("size", granularity));
        }

        context.bind()?;
        let mut handle = 0;
        let properties = options.raw()?;
        // SAFETY: `handle` is a local output, `properties` lives through the call, and `context` is bound.
        unsafe {
            try_ffi!(sys::cuMulticastCreate(
                &raw mut handle,
                &raw const properties
            ))?;
        }

        // SAFETY: CUDA just created this multicast handle of `options.size` bytes in `context`, and nothing else owns it.
        let allocation = unsafe {
            Allocation::from_raw(RawAllocation {
                handle,
                size: options.size,
                context: Arc::clone(context),
            })?
        };
        Ok(Arc::new(Self {
            allocation,
            devices: Mutex::default(),
            required_devices: options.devices,
        }))
    }

    /// Associates a device permanently with the multicast team.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is already registered, the team already has its declared number of devices, or CUDA cannot add the device.
    /// The local team list changes only after native registration succeeds.
    pub fn add_device(&self, device: Device) -> Result<()> {
        let mut devices = self
            .devices
            .lock()
            .map_err(|_| Error::lock_poisoned("multicast devices"))?;
        if devices.contains(&device) {
            return Err(Error::out_of_range("device"));
        }

        if devices.len() >= to_usize(self.required_devices, "multicast devices")? {
            return Err(Error::out_of_range("multicast devices"));
        }

        // SAFETY: `self.allocation` owns the live multicast handle, and the driver validates the device ordinal.
        unsafe {
            try_ffi!(sys::cuMulticastAddDevice(
                self.allocation.as_raw(),
                device.id()
            ))?;
        }

        devices.push(device);
        drop(devices);
        Ok(())
    }

    /// Borrows the physical-allocation handle representing this multicast object.
    #[must_use]
    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }

    fn ready(&self) -> Result<()> {
        let registered = self
            .devices
            .lock()
            .map_err(|_| Error::lock_poisoned("multicast devices"))?
            .len();
        if registered != to_usize(self.required_devices, "multicast devices")? {
            return Err(Status::NotReady.into());
        }
        Ok(())
    }

    /// Maps this multicast object into the reserved virtual address range.
    ///
    /// # Errors
    ///
    /// Returns an error if the team has fewer than its declared number of devices, with [`Status::NotReady`], or if [`AddressReservation::map`] fails.
    pub fn map(self: &Arc<Self>, reservation: AddressReservation) -> Result<MulticastMapping> {
        self.ready()?;
        Ok(MulticastMapping {
            mapping: reservation.map(Arc::clone(&self.allocation), 0)?,
            _owner: Arc::clone(self),
        })
    }

    /// Binds an allocation to this multicast object for the specified device.
    ///
    /// # Errors
    ///
    /// Returns an error if the team is incomplete, `size` is zero, a range end overflows or exceeds the multicast object or the allocation, or CUDA rejects the binding, such as for a device or granularity mismatch.
    pub fn bind(
        self: &Arc<Self>,
        device: Device,
        offset: usize,
        allocation: Arc<Allocation>,
        allocation_offset: usize,
        size: usize,
    ) -> Result<MulticastBinding> {
        self.ready()?;
        check_range(offset, size, self.allocation.size(), "multicast object")?;
        check_range(
            allocation_offset,
            size,
            allocation.size(),
            "physical allocation",
        )?;
        let native_offset = to_c_ulong(offset, "multicast offset")?;
        let native_allocation_offset = to_c_ulong(allocation_offset, "allocation offset")?;
        let native_size = to_c_ulong(size, "binding size")?;
        // SAFETY: both handles are live and owned by their wrappers, which the binding retains, and both ranges were checked against their objects above.
        unsafe {
            try_ffi!(sys::cuMulticastBindMem_v2(
                self.allocation.as_raw(),
                device.id(),
                native_offset,
                allocation.as_raw(),
                native_allocation_offset,
                native_size,
                0
            ))?;
        }
        Ok(MulticastBinding {
            owner: Arc::clone(self),
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
    ///
    /// The address range must be eligible CUDA memory and remain valid until the binding is dropped.
    /// All accesses must finish before unbinding.
    ///
    /// # Errors
    ///
    /// Returns an error if the team is incomplete, `size` is zero, the range end overflows or exceeds the multicast object, or CUDA rejects the binding.
    pub unsafe fn bind_address(
        self: &Arc<Self>,
        device: Device,
        offset: usize,
        address: DevicePtr,
        size: usize,
    ) -> Result<MulticastBinding> {
        self.ready()?;
        check_range(offset, size, self.allocation.size(), "multicast object")?;
        let native_offset = to_c_ulong(offset, "multicast offset")?;
        let native_size = to_c_ulong(size, "binding size")?;
        // SAFETY: the multicast handle is live and retained by the binding, its range was checked above, and the caller keeps the address range valid until the binding drops.
        unsafe {
            try_ffi!(sys::cuMulticastBindAddr_v2(
                self.allocation.as_raw(),
                device.id(),
                native_offset,
                address.as_ptr().addr() as sys::CUdeviceptr,
                native_size,
                0
            ))?;
        }
        Ok(MulticastBinding {
            owner: Arc::clone(self),
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
    /// Mapped virtual range and its retained physical allocation.
    /// Set access permissions before device use.
    pub mapping: Mapping,
    _owner: Arc<Multicast>,
}

/// A bound range retaining its multicast object and physical allocation.
#[derive(Debug)]
pub struct MulticastBinding {
    owner: Arc<Multicast>,
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
    /// Returns an error if the range does not fit the native types or CUDA cannot unbind it.
    /// If unbinding fails, dropping the consumed binding attempts it again and reports a second failure only in debug builds.
    pub fn unbind(mut self) -> Result<()> {
        self.release()
    }

    fn release(&mut self) -> Result<()> {
        if self.bound {
            let offset = to_c_ulong(self.offset, "multicast offset")?;
            let size = to_c_ulong(self.size, "binding size")?;
            // SAFETY: `self.owner` retains the live multicast handle, and this is the range bound for this device.
            unsafe {
                try_ffi!(sys::cuMulticastUnbind(
                    self.owner.allocation.as_raw(),
                    self.device.id(),
                    offset,
                    size
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
        if let Err(error) = self.release() {
            report_cleanup_failure("unbind", "cuda multicast binding", &error);
        }
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
                raw.__bindgen_anon_1.unicast.device = device.id();
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
        let properties = self.raw();
        // SAFETY: both limits are fields of a local output, and `properties` lives through the call.
        unsafe {
            cuLogicalEndpointGetLimits(
                &raw mut limits.bind_alignment,
                &raw mut limits.maximum_size,
                &raw const properties,
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
    /// Returns an error if the context cannot be bound, the driver lacks the entry points, the size is zero, above the native maximum, or not a multiple of the binding alignment, or CUDA cannot reserve an ID or create the endpoint.
    /// A reserved ID is released when native creation fails.
    pub fn create(context: &Arc<Context>, options: EndpointOptions) -> Result<Arc<Self>> {
        context.bind()?;
        let limits = options.limits()?;
        if options.size == 0 {
            return Err(Error::not_positive("size"));
        }

        if options.size > limits.maximum_size {
            return Err(Error::out_of_range("size"));
        }

        if !options.size.is_multiple_of(limits.bind_alignment) {
            return Err(Error::misaligned(
                "size",
                to_usize(limits.bind_alignment, "bind alignment")?,
            ));
        }

        let mut id = 0;
        let properties = options.raw();
        // SAFETY: `id` is a local output for one ID, `properties` lives through the call, `context` is bound, and a failed creation releases only the ID reserved here.
        unsafe {
            cuLogicalEndpointIdReserve(&raw mut id, 1)?;
            if let Err(error) = cuLogicalEndpointCreate(id, &raw const properties) {
                if let Err(release_error) = cuLogicalEndpointIdRelease(id, 1) {
                    report_cleanup_failure("release", "cuda logical endpoint id", &release_error);
                }

                return Err(error);
            }
        }
        Ok(Arc::new(Self {
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
        // SAFETY: `self` owns the live endpoint ID, and the driver validates the device ordinal.
        unsafe { cuLogicalEndpointAddDevice(self.id, device.id()) }
    }

    /// Determines whether this endpoint has been successfully constructed.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver entry point is unavailable or CUDA cannot query the endpoint.
    /// An endpoint whose construction is still pending returns `Ok(false)`.
    pub fn is_ready(&self) -> Result<bool> {
        let mut ready = 0;
        // SAFETY: `self` owns the live endpoint ID, and `ready` is a local output for one endpoint.
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
        // SAFETY: `handle` is a local `CUlogicalEndpointFabricHandle`, the type CUDA writes for a fabric handle, and `self` owns the live endpoint ID.
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
    ///
    /// The handle must identify a valid exported endpoint of the supplied size.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the driver lacks the entry points, or CUDA cannot reserve a local ID or import the endpoint.
    /// A reserved ID is released if import fails.
    pub unsafe fn import(
        context: &Arc<Context>,
        handle: &EndpointFabricHandle,
        size: u64,
    ) -> Result<Arc<Self>> {
        context.bind()?;
        let raw = sys::CUlogicalEndpointFabricHandle { data: handle.0 };
        let mut id = 0;
        // SAFETY: `id` is a local output for one ID, `raw` is a local fabric handle the caller guarantees names a valid exported endpoint, and a failed import releases only the ID reserved here.
        unsafe {
            cuLogicalEndpointIdReserve(&raw mut id, 1)?;
            if let Err(error) = cuLogicalEndpointImport(
                id,
                (&raw const raw).cast(),
                sys::CUlogicalEndpointIpcHandleType::CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC,
            ) {
                if let Err(release_error) = cuLogicalEndpointIdRelease(id, 1) {
                    report_cleanup_failure("release", "cuda logical endpoint id", &release_error);
                }

                return Err(error);
            }
        }
        Ok(Arc::new(Self {
            id,
            size,
            context: Arc::clone(context),
        }))
    }

    /// Binds a physical allocation to this logical endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if endpoint construction is incomplete, `size` is zero, a range end overflows or exceeds the endpoint or the allocation, the driver lacks the entry point, or CUDA rejects the binding, such as for a device or alignment mismatch.
    pub fn bind(
        self: &Arc<Self>,
        device: Device,
        offset: u64,
        allocation: Arc<Allocation>,
        allocation_offset: usize,
        size: usize,
    ) -> Result<EndpointBinding> {
        self.ready()?;
        check_range(
            allocation_offset,
            size,
            allocation.size(),
            "physical allocation",
        )?;
        check_endpoint_range(offset, size, self.size)?;
        let size = to_u64(size, "binding size")?;

        // SAFETY: the endpoint ID and the allocation are live and retained by the binding, and both ranges were checked above.
        unsafe {
            cuLogicalEndpointBindMem(
                self.id,
                device.id(),
                offset,
                allocation.as_raw(),
                to_u64(allocation_offset, "allocation offset")?,
                size,
                0,
            )?;
        }
        Ok(EndpointBinding {
            owner: Arc::clone(self),
            device,
            offset,
            size,
            allocation: Some(allocation),
            bound: true,
        })
    }

    /// Binds a virtual address range to this logical endpoint.
    ///
    /// # Safety
    ///
    /// The memory must remain valid until the binding is dropped.
    /// All accesses must finish before unbinding.
    ///
    /// # Errors
    ///
    /// Returns an error if endpoint construction is incomplete, `offset + size` overflows or exceeds the endpoint, the driver lacks the entry point, or CUDA rejects the binding.
    pub unsafe fn bind_address(
        self: &Arc<Self>,
        device: Device,
        offset: u64,
        pointer: DevicePtr,
        size: usize,
    ) -> Result<EndpointBinding> {
        self.ready()?;
        check_endpoint_range(offset, size, self.size)?;
        let size = to_u64(size, "binding size")?;

        // SAFETY: the endpoint ID is live and retained by the binding, its range was checked above, and the caller keeps the memory valid until the binding drops.
        unsafe {
            cuLogicalEndpointBindAddr(
                self.id,
                device.id(),
                offset,
                pointer.as_ptr().cast(),
                size,
                0,
            )?;
        }
        Ok(EndpointBinding {
            owner: Arc::clone(self),
            device,
            offset,
            size,
            allocation: None,
            bound: true,
        })
    }
}

impl Drop for LogicalEndpoint {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cuda logical endpoint",
                &error,
            );
            return;
        }

        // SAFETY: this wrapper owns the endpoint, its context is bound, and every binding retains it, so none is live.
        if let Err(error) = unsafe { cuLogicalEndpointDestroy(self.id) } {
            // The ID stays reserved while its endpoint may still exist.
            report_cleanup_failure("destroy", "cuda logical endpoint", &error);
            return;
        }

        // SAFETY: the endpoint using this ID was just destroyed, and this wrapper reserved the ID.
        if let Err(error) = unsafe { cuLogicalEndpointIdRelease(self.id, 1) } {
            report_cleanup_failure("release", "cuda logical endpoint id", &error);
        }
    }
}

/// A bound range retaining its logical endpoint and allocation.
#[derive(Debug)]
pub struct EndpointBinding {
    owner: Arc<LogicalEndpoint>,
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
    /// Returns an error if the driver lacks the entry point or CUDA cannot unbind the range.
    /// If unbinding fails, dropping the consumed binding attempts it again and reports a second failure only in debug builds.
    pub fn unbind(mut self) -> Result<()> {
        self.release()
    }

    fn release(&mut self) -> Result<()> {
        if self.bound {
            // SAFETY: `self.owner` retains the live endpoint ID, and this is the range bound for this device.
            unsafe {
                cuLogicalEndpointUnbind(self.owner.id, self.device.id(), self.offset, self.size)?;
            }

            self.bound = false;
        }
        Ok(())
    }
}

impl Drop for EndpointBinding {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            report_cleanup_failure("unbind", "cuda logical endpoint binding", &error);
        }
    }
}

fn check_endpoint_range(offset: u64, size: usize, total: u64) -> Result<()> {
    let end = offset
        .checked_add(to_u64(size, "size")?)
        .ok_or_else(|| Error::out_of_range("range end"))?;
    if end > total {
        return Err(Error::buffer_too_small(
            "logical endpoint",
            to_usize(end, "range end")?,
            to_usize(total, "logical endpoint size")?,
        ));
    }
    Ok(())
}
