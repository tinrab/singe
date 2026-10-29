//! Zero-copy cudarc storage with explicit cross-runtime access and ordering contracts.

use std::{marker::PhantomData, sync::Arc};

use ::cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as ForeignDevicePtr,
    DevicePtrMut as ForeignDevicePtrMut, DeviceSlice as ForeignDeviceSlice, DriverError,
    sys::CUdeviceptr,
};

use crate::{
    context::{Context, RawContextOwnership},
    device::Device,
    error::{Error, Result},
    stream::{Stream, StreamIdentity},
    view::{
        AllocationId, DeviceInput, DeviceOutput, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut, OwnedDeviceRegion,
    },
};

fn foreign_error(error: DriverError) -> Error {
    Error::Operation(Box::new(error))
}

impl Context {
    /// Retains cudarc's actual context, including independently created non-primary contexts.
    pub fn from_cudarc(owner: Arc<CudaContext>) -> Result<Arc<Self>> {
        let handle = owner.cu_ctx().cast();
        let device = Device::new(owner.cu_device());
        // SAFETY: the cudarc context keeps this handle alive, and `owner` retains it.
        unsafe { Self::from_raw(handle, device, RawContextOwnership::Borrowed(owner)) }
    }
}

impl Stream {
    /// Borrows a cudarc stream and retains its native stream and context owners.
    ///
    /// # Safety
    /// Coordinate external submissions with Singe access tracking and capture lifecycle.
    pub unsafe fn from_cudarc(owner: Arc<CudaStream>) -> Result<Self> {
        let context = Context::from_cudarc(Arc::clone(owner.context()))?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::borrow_with_owner(owner.cu_stream().cast(), context, owner) }
    }

    /// Exposes this stream to cudarc without transferring destruction responsibility.
    ///
    /// # Safety
    /// Retain this Singe stream while any returned cudarc clone or allocation uses it.
    /// Cudarc submissions must participate in explicit ordering and exclude conflicting Singe accesses.
    pub unsafe fn to_cudarc(&self, context: &Arc<CudaContext>) -> Result<Arc<CudaStream>> {
        if self.context().as_raw().cast::<()>() != context.cu_ctx().cast::<()>() {
            return Err(Error::StreamContextMismatch);
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { context.wrap_borrowed_stream(self.as_raw().cast()) })
    }
}

/// Retains the original cudarc allocation and the actual stream establishing its readiness.
#[derive(Debug)]
pub struct CudarcAllocation<T: DeviceRepr> {
    owner: Option<Arc<CudaSlice<T>>>,
    stream: Stream,
    retirement_stream: Stream,
    id: AllocationId,
    pointer: *mut T,
    bytes: usize,
}

// SAFETY: device pointers are never dereferenced by the host; typed access and external exclusion govern values.
unsafe impl<T: DeviceRepr + Send> Send for CudarcAllocation<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for CudarcAllocation<T> {}

impl<T: DeviceRepr> CudarcAllocation<T> {
    pub fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn id(&self) -> AllocationId {
        self.id
    }

    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    pub fn as_ptr(&self) -> *mut T {
        self.pointer
    }

    /// # Safety
    /// Foreign accesses must exclude conflicting Singe operations and establish event ordering in both directions.
    pub unsafe fn cudarc_owner(&self) -> &Arc<CudaSlice<T>> {
        self.owner.as_ref().expect("live cudarc allocation owner")
    }
}

// SAFETY: native storage and its actual context/stream stay live through the retained cudarc owner.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcAllocation<T> {
    fn region(&self) -> DeviceRegion<'_> {
        // SAFETY: the retained cudarc owner keeps this allocation, its context and its stream live.
        unsafe {
            DeviceRegion::from_raw_parts(self.context(), self.id, self.pointer.cast(), self.bytes)
                .and_then(|region| region.with_stream(&self.stream))
        }
        .expect("checked cudarc allocation metadata")
    }
}

impl<T: DeviceRepr> Drop for CudarcAllocation<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take()
            && let Err(error) = self.retirement_stream.release_foreign_owner(owner)
        {
            #[cfg(debug_assertions)]
            eprintln!("failed to retire cudarc allocation: {error}");
        }
    }
}

/// Typed cudarc allocation handle exposing Singe views while retaining the foreign owner.
#[derive(Debug)]
pub struct CudarcMemory<T: DeviceRepr> {
    allocation: Arc<CudarcAllocation<T>>,
    length: usize,
}

impl<T: DeviceRepr> CudarcMemory<T> {
    /// Adopts an exclusive cudarc allocation, ordering its prior reads and writes before Singe consumers.
    ///
    /// # Safety
    /// Every element must be valid for `T` and externally retained aliases must exclude conflicting accesses.
    /// External raw CUDA work and capture must obey the same resource and stream contracts as Singe.
    pub unsafe fn from_owned(mut owner: CudaSlice<T>) -> Result<Self> {
        let ready = Arc::clone(owner.stream());
        let (_, order) = owner.device_ptr_mut(&ready);
        drop(order);
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_shared(Arc::new(owner), ready) }
    }

    /// Retains a shared cudarc owner and an explicit readiness stream.
    ///
    /// # Safety
    /// `ready` must order all prior accesses that conflict with the intended Singe uses.
    /// Values must be valid for `T`; exclude conflicting accesses through foreign or separately imported aliases.
    /// Use this bridge's views to preserve allocation identity across Singe accesses.
    /// Foreign and Singe tracking are independent, so order both directions explicitly with events or completion.
    /// Coordinate external capture on both the readiness stream and the allocation's original stream.
    pub unsafe fn from_shared(owner: Arc<CudaSlice<T>>, ready: Arc<CudaStream>) -> Result<Self> {
        if owner.context().cu_ctx() != ready.context().cu_ctx() {
            return Err(Error::MemoryContextMismatch);
        }
        let () = T::LAYOUT;
        let length = owner.len();
        let bytes = length
            .checked_mul(size_of::<T>())
            .ok_or(Error::InvalidMemoryAllocationRequest)?;
        if length != 0 && size_of::<T>() == 0 {
            return Err(Error::InvalidMemoryAllocationRequest);
        }
        let (pointer, order) = owner.device_ptr(&ready);
        drop(order);
        ready.context().check_err().map_err(foreign_error)?;
        // SAFETY: the caller upholds this function's safety contract.
        let stream = unsafe { Stream::from_cudarc(ready)? };
        // Cudarc frees on the allocation's original stream, independently of its readiness stream.
        let retirement_stream =
            if StreamIdentity::new(stream.context(), owner.stream().cu_stream().cast())
                == stream.identity()
            {
                stream.clone()
            } else {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { Stream::from_cudarc(Arc::clone(owner.stream()))? }
            };
        Ok(Self {
            allocation: Arc::new(CudarcAllocation {
                owner: Some(owner),
                stream,
                retirement_stream,
                id: AllocationId::create(),
                pointer: pointer as usize as *mut T,
                bytes,
            }),
            length,
        })
    }

    pub fn allocation(&self) -> &Arc<CudarcAllocation<T>> {
        &self.allocation
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn view(&self) -> DeviceView<'_, T> {
        // SAFETY: the view covers exactly the elements of the borrowed allocation and lives no longer than the borrow.
        unsafe { DeviceView::from_raw_parts(self.region(), self.length) }
            .expect("checked cudarc allocation view")
    }

    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        // SAFETY: the view covers exactly the elements of the borrowed allocation and lives no longer than the borrow.
        unsafe { DeviceViewMut::from_raw_parts(self.region(), self.length) }
            .expect("exclusive cudarc allocation view")
    }

    pub fn retained_region(&self) -> Result<OwnedDeviceRegion<CudarcAllocation<T>>> {
        OwnedDeviceRegion::from_owner(Arc::clone(&self.allocation), 0..self.allocation.bytes)
    }
}

// SAFETY: metadata and initialized-value guarantees follow the import contract and retained owner.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for CudarcMemory<T> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceSlice<T> for CudarcMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for CudarcMemory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.pointer
    }
}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for CudarcMemory<T> {}

/// Read-only borrowed foreign view tied to a checked root allocation identity.
#[derive(Debug)]
pub struct CudarcView<'a, T: DeviceRepr> {
    view: DeviceView<'a, T>,
    foreign: PhantomData<&'a CudaSlice<T>>,
}

impl<'a, T: DeviceRepr> CudarcView<'a, T> {
    /// Borrows a cudarc subview using an existing Singe bridge's root identity and bounds.
    ///
    /// # Safety
    /// The foreign view must alias this root and remain valid and unmodified through the borrow.
    /// Order prior foreign producers on `ready` and order Singe consumers after that stream.
    pub unsafe fn create(
        view: CudaView<'a, T>,
        root: DeviceRegion<'a>,
        ready: &'a Stream,
    ) -> Result<Self> {
        let foreign = Arc::clone(ForeignDeviceSlice::stream(&view));
        check_ready(root, ready, &foreign)?;
        let length = view.len();
        let (pointer, order) = view.device_ptr(&foreign);
        drop(order);
        foreign.context().check_err().map_err(foreign_error)?;
        let region = subregion::<T>(root, pointer, length)?;
        // SAFETY: the caller upholds this function's safety contract.
        let region = unsafe { region.with_stream(ready)? };
        Ok(Self {
            // SAFETY: the caller upholds this function's safety contract.
            view: unsafe { DeviceView::from_raw_parts(region, length)? },
            foreign: PhantomData,
        })
    }

    pub fn view(&self) -> DeviceView<'_, T> {
        self.view
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcView<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.view.region()
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for CudarcView<'_, T> {
    fn len(&self) -> usize {
        self.view.len()
    }
}

unsafe impl<T: DeviceRepr> DeviceSlice<T> for CudarcView<'_, T> {}

/// Exclusive borrowed foreign view retaining the native slice borrow and shared allocation identity.
#[derive(Debug)]
pub struct CudarcViewMut<'a, T: DeviceRepr> {
    view: DeviceViewMut<'a, T>,
    foreign: PhantomData<&'a mut CudaSlice<T>>,
}

impl<'a, T: DeviceRepr> CudarcViewMut<'a, T> {
    /// Borrows a mutable cudarc subview using its original allocation's identity and exact stream.
    ///
    /// # Safety
    /// Values must be valid for `T`, and `root` must describe this view's actual backing allocation.
    /// Exclude foreign aliases until this borrow ends and all Singe work using it completes.
    pub unsafe fn create(
        mut view: CudaViewMut<'a, T>,
        root: DeviceRegion<'a>,
        ready: &'a Stream,
    ) -> Result<Self> {
        let foreign = Arc::clone(ForeignDeviceSlice::stream(&view));
        check_ready(root, ready, &foreign)?;
        let length = view.len();
        let (pointer, order) = view.device_ptr_mut(&foreign);
        drop(order);
        foreign.context().check_err().map_err(foreign_error)?;
        let region = subregion::<T>(root, pointer, length)?;
        // SAFETY: the caller upholds this function's safety contract.
        let region = unsafe { region.with_stream(ready)? };
        Ok(Self {
            // SAFETY: the caller upholds this function's safety contract.
            view: unsafe { DeviceViewMut::from_raw_parts(region, length)? },
            foreign: PhantomData,
        })
    }

    pub fn view(&self) -> DeviceView<'_, T> {
        self.view.as_view()
    }

    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        self.view.reborrow()
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcViewMut<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.view.region()
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for CudarcViewMut<'_, T> {
    fn len(&self) -> usize {
        self.view.len()
    }
}

unsafe impl<T: DeviceRepr> DeviceSlice<T> for CudarcViewMut<'_, T> {}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for CudarcViewMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.view.as_mut_ptr()
    }
}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for CudarcViewMut<'_, T> {}

fn check_ready(root: DeviceRegion<'_>, ready: &Stream, foreign: &CudaStream) -> Result<()> {
    if root.context() != ready.context()
        || root.context().as_raw().cast::<()>() != foreign.context().cu_ctx().cast::<()>()
    {
        return Err(Error::MemoryContextMismatch);
    }
    if ready.identity() != StreamIdentity::new(ready.context(), foreign.cu_stream().cast()) {
        return Err(Error::ForeignStreamMismatch);
    }
    Ok(())
}

fn subregion<T>(
    root: DeviceRegion<'_>,
    pointer: CUdeviceptr,
    length: usize,
) -> Result<DeviceRegion<'_>> {
    let start = (pointer as usize)
        .checked_sub(root.as_ptr().addr())
        .ok_or(Error::InvalidMemoryAccess)?;
    let bytes = length
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidMemoryAccess)?;
    root.slice(start..start.checked_add(bytes).ok_or(Error::InvalidMemoryAccess)?)
}
