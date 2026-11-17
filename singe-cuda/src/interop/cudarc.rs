//! Zero-copy cudarc storage with explicit cross-runtime access and ordering contracts.

use std::{marker::PhantomData, ptr, sync::Arc};

use ::cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as ForeignDevicePtr,
    DevicePtrMut as ForeignDevicePtrMut, DeviceSlice as ForeignDeviceSlice, DriverError,
    sys::CUdeviceptr,
};
use singe_core::report_cleanup_failure;

use crate::{
    context::{Context, RawContext, RawContextOwnership},
    device::Device,
    error::{Error, Result},
    stream::{Stream, StreamIdentity},
    utility::{checked_bytes, to_usize},
    view::{
        AllocationId, DeviceOutput, DeviceRange, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut, OwnedDeviceRegion,
    },
};

fn foreign_error(error: DriverError) -> Error {
    Error::Operation(Box::new(error))
}

impl Context {
    /// Retains cudarc's actual context, including independently created non-primary contexts.
    ///
    /// # Errors
    ///
    /// Returns an error if cudarc's context handle is null.
    pub fn from_cudarc(owner: Arc<CudaContext>) -> Result<Arc<Self>> {
        let handle = owner.cu_ctx().cast();
        let device = Device::new(owner.cu_device());
        // SAFETY: `owner` keeps cudarc's live context, which belongs to `device`, alive for as long as the returned context.
        unsafe {
            Self::from_raw(RawContext {
                handle,
                device,
                ownership: RawContextOwnership::Borrowed(owner),
            })
        }
    }
}

impl Stream {
    /// Borrows a cudarc stream and retains its native stream and context owners.
    ///
    /// # Safety
    ///
    /// Coordinate external submissions with Singe access tracking and capture lifecycle.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is a per-thread default stream or its context cannot be retained or bound.
    pub unsafe fn from_cudarc(owner: Arc<CudaStream>) -> Result<Self> {
        let context = Context::from_cudarc(Arc::clone(owner.context()))?;
        // SAFETY: the stream belongs to cudarc's context, which `context` retains, `owner` keeps the stream alive, and the caller coordinates external submissions.
        unsafe { Self::borrow_with_owner(owner.cu_stream().cast(), context, owner) }
    }

    /// Exposes this stream to cudarc without transferring destruction responsibility.
    ///
    /// # Safety
    ///
    /// Retain this Singe stream while any returned cudarc clone or allocation uses it.
    /// Cudarc submissions must participate in explicit ordering and exclude conflicting Singe accesses.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` is not this stream's context.
    pub unsafe fn to_cudarc(&self, context: &Arc<CudaContext>) -> Result<Arc<CudaStream>> {
        if self.context().as_raw().cast::<()>() != context.cu_ctx().cast::<()>() {
            return Err(Error::StreamContextMismatch);
        }
        // SAFETY: the stream belongs to `context`, as checked above, and the caller keeps it alive and orders cudarc work with Singe work.
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

// SAFETY: the device pointer is never dereferenced on the host, and the cudarc owner and the streams may move between threads; values move as `T` does.
unsafe impl<T: DeviceRepr + Send> Send for CudarcAllocation<T> {}

// SAFETY: shared references expose only the pointer, metadata, and `&` access to thread-safe owners; values are shared as `T` is.
unsafe impl<T: DeviceRepr + Sync> Sync for CudarcAllocation<T> {}

impl<T: DeviceRepr> CudarcAllocation<T> {
    /// Returns the context of the allocation's readiness stream.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    /// Returns the stream that establishes the allocation's readiness.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns the allocation identity shared by every view of this allocation.
    #[must_use]
    pub fn id(&self) -> AllocationId {
        self.id
    }

    /// Returns the allocation's size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Returns the allocation's device address without granting access to its values.
    #[must_use]
    pub fn as_ptr(&self) -> *mut T {
        self.pointer
    }

    /// Returns the retained cudarc allocation.
    ///
    /// # Safety
    ///
    /// Foreign accesses must exclude conflicting Singe operations and establish event ordering in both directions.
    #[must_use]
    pub unsafe fn cudarc_owner(&self) -> &Arc<CudaSlice<T>> {
        self.owner.as_ref().expect("live cudarc allocation owner")
    }
}

// SAFETY: the retained cudarc owner keeps the storage live, and `id`, `pointer`, and `bytes` describe it in the retained stream's context.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcAllocation<T> {
    fn region(&self) -> DeviceRegion<'_> {
        // SAFETY: the retained cudarc owner keeps these `bytes` bytes live in this context, `id` is unique to the allocation, and its pending writes are ordered on `self.stream`.
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
            report_cleanup_failure("retire", "cudarc allocation", &error);
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
    ///
    /// Every element must be valid for `T` and externally retained aliases must exclude conflicting accesses.
    /// External raw CUDA work and capture must obey the same resource and stream contracts as Singe.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::from_shared`] fails for the allocation and its own stream.
    pub unsafe fn from_owned(mut owner: CudaSlice<T>) -> Result<Self> {
        let ready = Arc::clone(owner.stream());
        let (_, order) = owner.device_ptr_mut(&ready);
        drop(order);
        // SAFETY: the mutable access above ordered prior cudarc work on `ready`, and the caller guarantees valid values and excluded aliases.
        unsafe { Self::from_shared(Arc::new(owner), ready) }
    }

    /// Retains a shared cudarc owner and an explicit readiness stream.
    ///
    /// # Safety
    ///
    /// `ready` must order all prior accesses that conflict with the intended Singe uses.
    /// Values must be valid for `T`; exclude conflicting accesses through foreign or separately imported aliases.
    /// Use this bridge's views to preserve allocation identity across Singe accesses.
    /// Foreign and Singe tracking are independent, so order both directions explicitly with events or completion.
    /// Coordinate external capture on both the readiness stream and the allocation's original stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `owner` and `ready` belong to different contexts, the byte size overflows, cudarc reports an error, or a stream cannot be retained.
    pub unsafe fn from_shared(owner: Arc<CudaSlice<T>>, ready: Arc<CudaStream>) -> Result<Self> {
        if owner.context().cu_ctx() != ready.context().cu_ctx() {
            return Err(Error::MemoryContextMismatch);
        }

        let () = T::LAYOUT;
        let length = owner.len();
        let bytes = checked_bytes::<T>(length, "cudarc slice bytes")?;

        let (pointer, order) = owner.device_ptr(&ready);
        drop(order);
        ready.context().check_err().map_err(foreign_error)?;
        // SAFETY: the caller coordinates external submissions on `ready` with Singe tracking and capture.
        let stream = unsafe { Stream::from_cudarc(ready)? };
        // Cudarc frees on the allocation's original stream, independently of its readiness stream.
        let retirement_stream =
            if StreamIdentity::new(stream.context(), owner.stream().cu_stream().cast())
                == stream.identity()
            {
                stream.clone()
            } else {
                // SAFETY: the caller coordinates external capture and submissions on the allocation's original stream.
                unsafe { Stream::from_cudarc(Arc::clone(owner.stream()))? }
            };
        Ok(Self {
            allocation: Arc::new(CudarcAllocation {
                owner: Some(owner),
                stream,
                retirement_stream,
                id: AllocationId::create(),
                pointer: ptr::without_provenance_mut(to_usize(pointer, "cudarc device pointer")?),
                bytes,
            }),
            length,
        })
    }

    /// Returns the shared allocation owner.
    #[must_use]
    pub fn allocation(&self) -> &Arc<CudarcAllocation<T>> {
        &self.allocation
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the memory holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Borrows the elements as a typed device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        // SAFETY: the region covers the `length` elements `from_shared`'s caller guaranteed valid, and the view lives no longer than `&self`.
        unsafe { DeviceView::from_raw_parts(self.region(), self.length) }
            .expect("checked cudarc allocation view")
    }

    /// Exclusively borrows the elements as a mutable typed device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        // SAFETY: the region covers the `length` valid elements, and `&mut self` excludes other Singe aliases for the view's lifetime.
        unsafe { DeviceViewMut::from_raw_parts(self.region(), self.length) }
            .expect("exclusive cudarc allocation view")
    }

    /// Returns a region that retains the allocation owner.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation's byte range cannot be described as an owned region.
    pub fn retained_region(&self) -> Result<OwnedDeviceRegion<CudarcAllocation<T>>> {
        OwnedDeviceRegion::from_owner(Arc::clone(&self.allocation), 0..self.allocation.bytes)
    }
}

// SAFETY: the region comes from the retained allocation owner, which keeps its identity and bounds for the borrow.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation.region()
    }
}

// SAFETY: `length` elements of `T` span exactly the allocation's checked byte size.
unsafe impl<T: DeviceRepr> DeviceRange<T> for CudarcMemory<T> {
    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: `from_shared`'s caller guarantees every element is a valid `T`.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for CudarcMemory<T> {}

// SAFETY: the allocation is writable device storage for `length` elements, borrowed exclusively through `&mut self`.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for CudarcMemory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.pointer
    }
}

// SAFETY: `&mut self` excludes other Singe aliases, and `from_shared`'s caller excludes conflicting foreign aliases.
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
    ///
    /// The foreign view must alias this root and remain valid and unmodified through the borrow.
    /// Order prior foreign producers on `ready` and order Singe consumers after that stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `root`, `ready`, and the view's stream disagree on context or stream, cudarc reports an error, or the view lies outside `root`.
    pub unsafe fn create(
        view: &CudaView<'a, T>,
        root: DeviceRegion<'a>,
        ready: &'a Stream,
    ) -> Result<Self> {
        let foreign = Arc::clone(ForeignDeviceSlice::stream(view));
        check_ready(root, ready, &foreign)?;
        let length = view.len();
        let (pointer, order) = view.device_ptr(&foreign);
        drop(order);
        foreign.context().check_err().map_err(foreign_error)?;
        let region = subregion::<T>(root, pointer, length)?;
        // SAFETY: the caller orders prior foreign producers on `ready`, which `check_ready` matched to the view's stream.
        let region = unsafe { region.with_stream(ready)? };
        Ok(Self {
            // SAFETY: the caller guarantees the aliased values stay valid and unmodified for `'a`.
            view: unsafe { DeviceView::from_raw_parts(region, length)? },
            foreign: PhantomData,
        })
    }

    /// Returns the Singe view of the borrowed elements.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        self.view
    }
}

// SAFETY: the region is the checked subregion of the borrowed root, which stays live for the view's lifetime.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcView<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.view.region()
    }
}

// SAFETY: the length is the inner view's, which `DeviceView::from_raw_parts` matched to the region's byte length.
unsafe impl<T: DeviceRepr> DeviceRange<T> for CudarcView<'_, T> {
    fn len(&self) -> usize {
        self.view.len()
    }
}

// SAFETY: `create`'s caller guarantees the values are valid and unmodified for the borrow.
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
    ///
    /// Values must be valid for `T`, and `root` must describe this view's actual backing allocation.
    /// Exclude foreign aliases until this borrow ends and all Singe work using it completes.
    ///
    /// # Errors
    ///
    /// Returns an error if `root`, `ready`, and the view's stream disagree on context or stream, cudarc reports an error, or the view lies outside `root`.
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
        // SAFETY: the mutable access above ordered prior cudarc work on the view's stream, which `check_ready` matched to `ready`.
        let region = unsafe { region.with_stream(ready)? };
        Ok(Self {
            // SAFETY: the caller guarantees valid values and excludes foreign aliases until the borrow and its Singe work end.
            view: unsafe { DeviceViewMut::from_raw_parts(region, length)? },
            foreign: PhantomData,
        })
    }

    /// Borrows the elements as a read-only Singe view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        self.view.as_view()
    }

    /// Reborrows the elements as a mutable Singe view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        self.view.reborrow()
    }
}

// SAFETY: the region is the checked subregion of the borrowed root, which stays live for the view's lifetime.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for CudarcViewMut<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.view.region()
    }
}

// SAFETY: the length is the inner view's, which `DeviceViewMut::from_raw_parts` matched to the region's byte length.
unsafe impl<T: DeviceRepr> DeviceRange<T> for CudarcViewMut<'_, T> {
    fn len(&self) -> usize {
        self.view.len()
    }
}

// SAFETY: `create`'s caller guarantees the values are valid for the borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for CudarcViewMut<'_, T> {}

// SAFETY: the inner exclusive view is writable storage for its elements, borrowed through `&mut self`.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for CudarcViewMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.view.as_mut_ptr()
    }
}

// SAFETY: the inner view is exclusive, and `create`'s caller excludes foreign aliases until the borrow ends.
unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for CudarcViewMut<'_, T> {}

fn check_ready(root: DeviceRegion<'_>, ready: &Stream, foreign: &CudaStream) -> Result<()> {
    let foreign_context = foreign.context().cu_ctx().cast::<()>();
    if root.context() != ready.context() || root.context().as_raw().cast::<()>() != foreign_context
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
    let start = usize::try_from(pointer)
        .map_err(|_| Error::out_of_range("cudarc view address"))?
        .checked_sub(root.as_ptr().addr())
        .ok_or_else(|| Error::out_of_range("cudarc view offset"))?;
    let bytes = length
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::out_of_range("cudarc view bytes"))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| Error::out_of_range("cudarc view end"))?;
    root.slice(start..end)
}
