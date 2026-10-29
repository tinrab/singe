use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    memory::{DeviceMemory, transfer},
    types::{Complex32, f16},
    view::{
        AllocationId, AnyBitPatternDeviceRepr, DeviceOutput, DeviceOutputView, DeviceRegion,
        DeviceRegionProvider, DeviceRepr,
    },
};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        raw::exchange::{self, SetC1},
        view::{
            AC4, C1, C2, C3, C4, ChannelLayout, ImageOutputView, ImageView, ImageViewMut,
            image_byte_len,
        },
    },
    memory::{Allocation, AllocationKind, RawAllocation},
    types::{ComplexI16, ComplexI32, DataTypeLike, Size},
    utility::checked_len,
};

/// RAII owner for pitched image memory allocated by NPP.
///
/// The shared [`Allocation`] retains the actual CUDA context, allocation identity,
/// and full pitched capacity. The last allocation owner releases the native base
/// with [`sys::nppiFree`].
/// The raw pointer must not be freed by callers.
/// `step` is the NPP row pitch in bytes and can be larger than the logical ROI
/// row size.
#[derive(Debug)]
pub struct Image<T, L = C1> {
    allocation: Arc<Allocation>,
    size: Size,
    step: i32,
    _t: PhantomData<(T, L)>,
}

/// Marker for image element/layout combinations that NPP can allocate.
pub trait SupportedImage<Layout>: DataTypeLike + private::Allocate<Layout>
where
    Layout: ChannelLayout,
{
}

impl<T, Layout> SupportedImage<Layout> for T
where
    T: DataTypeLike + private::Allocate<Layout>,
    Layout: ChannelLayout,
{
}

mod private {
    pub trait Allocate<Layout>: Sized {
        unsafe fn allocate(width: i32, height: i32, step: *mut i32) -> *mut Self;
    }
}

macro_rules! impl_image_create {
    ($layout:ty, [$($ty:ty => $ffi:ident),* $(,)?]) => {
        $(
            impl private::Allocate<$layout> for $ty {
                unsafe fn allocate(width: i32, height: i32, step: *mut i32) -> *mut Self {
                    // SAFETY: sealed dispatch maps this element and channel layout to the native allocator.
                    unsafe { sys::$ffi(width, height, step).cast() }
                }
            }
        )*
    };
}

impl<T, L> Image<T, L>
where
    L: ChannelLayout,
{
    fn from_native(
        context: &Arc<CudaContext>,
        pointer: *mut T,
        size: Size,
        step: i32,
    ) -> Result<Self> {
        let bounds = (|| {
            let pitch =
                usize::try_from(step).map_err(|_| Error::out_of_range("image allocation size"))?;
            let height = usize::try_from(size.height)
                .map_err(|_| Error::out_of_range("image allocation height"))?;
            let bytes = pitch
                .checked_mul(height)
                .ok_or_else(|| Error::out_of_range("image allocation size"))?;
            // SAFETY: validate pixel geometry against the complete native pitched allocation.
            let region = unsafe {
                DeviceRegion::from_raw_parts(
                    context,
                    AllocationId::create(),
                    pointer.cast(),
                    bytes,
                )?
            };
            // SAFETY: the region covers the whole native allocation; this call only validates the geometry.
            let _ = unsafe { ImageOutputView::<T, L>::from_raw_parts(region, size, step)? };
            Ok::<_, Error>((bytes, region.allocation_id()))
        })();
        let (bytes, id) = match bounds {
            Ok(bounds) => bounds,
            Err(error) => {
                // SAFETY: this pointer just came from nppiMalloc in the bound context.
                unsafe { sys::nppiFree(pointer.cast()) };
                return Err(error);
            }
        };
        let parts = RawAllocation {
            pointer: pointer.cast(),
            bytes,
            context: Arc::clone(context),
            id,
            kind: AllocationKind::Image,
        };
        // SAFETY: native allocation establishes the complete validated pitched capacity.
        let allocation = Arc::new(unsafe { Allocation::from_raw(parts)? });
        Ok(Self {
            allocation,
            size,
            step,
            _t: PhantomData,
        })
    }

    /// Returns the shared native allocation owner and its full pitched byte capacity.
    ///
    /// Cloning this owner retains storage but grants no additional memory-access permissions.
    #[must_use]
    pub fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }

    /// Transfers unique native ownership with the full capacity and logical image geometry.
    ///
    /// # Errors
    ///
    /// Returns an error if another owner still shares the underlying allocation.
    pub fn into_raw(self) -> Result<(RawAllocation, Size, i32)> {
        let allocation = Arc::try_unwrap(self.allocation).map_err(|_| Error::OwnershipShared)?;
        Ok((allocation.into_raw(), self.size, self.step))
    }

    /// Adopts native image ownership and validates its pixel geometry.
    ///
    /// # Errors
    ///
    /// Returns an error if the raw allocation metadata is invalid, its allocator family is not
    /// `Image`, or its dimensions, byte pitch, alignment, or capacity cannot form a valid image.
    ///
    /// # Safety
    ///
    /// The allocation must satisfy [`Allocation::from_raw`] and contain valid `T` values in every
    /// logical pixel channel, including alpha for AC4. Padding need not be initialized. Preserve
    /// allocation identity and context, retain owners through queued access and replay, and
    /// order producing work before consumers and readback.
    pub unsafe fn from_raw(parts: RawAllocation, size: Size, step: i32) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: caller transfers unique native ownership in the declared context.
        let allocation = Arc::new(unsafe { Allocation::from_raw(parts)? });
        let () = T::LAYOUT;
        if allocation.kind() != AllocationKind::Image {
            return Err(Error::out_of_range("native image allocation kind"));
        }
        // SAFETY: the caller upholds this function's safety contract.
        let _ = unsafe { ImageView::<T, L>::from_raw_parts(allocation.region(), size, step)? };
        Ok(Self {
            allocation,
            size,
            step,
            _t: PhantomData,
        })
    }

    /// Borrows logical pixels for production without claiming their old values.
    ///
    /// # Errors
    ///
    /// Returns an error if stored dimensions, byte stride, element alignment, or allocation
    /// bounds cannot form a valid output view.
    pub fn output_view(&mut self) -> Result<ImageOutputView<'_, T, L>> {
        // SAFETY: exclusive ownership grants writable pixel storage.
        unsafe { ImageOutputView::from_raw_parts(self.region(), self.size, self.step) }
    }

    /// Allocates and fills every physical channel with the same value, including alpha.
    /// Completes native production before returning initialized pixels; pitch padding stays uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if allocation or view construction fails,
    /// if context binding, region preparation, or producer waits fail, or if the native fill
    /// or stream synchronization fails.
    pub fn filled(
        stream_context: &StreamContext,
        size: Size,
        value: <T as SetC1<C1>>::Value,
    ) -> Result<Self>
    where
        T: SupportedImage<L> + SetC1<C1>,
    {
        stream_context.stream().ensure_not_capturing()?;
        let mut image = Self::create_uninit(stream_context.cuda_context(), size)?;
        {
            let mut samples = image.output()?.into_samples()?;
            stream_context.sync_scope(|| {
                stream_context.prepare_region(&samples)?;
                // SAFETY: the native scalar fill writes every physical sample and the scope
                // retains the exclusive destination through completion.
                unsafe { exchange::set_c1(stream_context, value, &mut samples) }
            })?;
        }
        // SAFETY: completed fill initialized every logical sample, including alpha.
        Ok(unsafe { image.assume_init() })
    }

    /// Allocates pitched storage without claiming initialized logical pixels or padding.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are nonpositive, CUDA context binding fails, the native
    /// allocator returns null, allocation-size arithmetic overflows, or the returned byte pitch
    /// and allocation bounds cannot form a valid image.
    pub fn create_uninit(context: &Arc<CudaContext>, size: Size) -> Result<Image<MaybeUninit<T>, L>>
    where
        T: SupportedImage<L>,
    {
        let () = T::LAYOUT;
        size.validate()?;
        context.bind()?;
        let mut step = 0;
        // SAFETY: the sealed allocator has T's ABI and L's physical channel count.
        let pointer = unsafe { T::allocate(size.width, size.height, &raw mut step) };
        let pointer = NonNull::new(pointer).ok_or(Error::NullHandle)?;
        Image::from_native(context, pointer.as_ptr().cast(), size, step)
    }

    /// Uploads every logical channel from compact host pixels, leaving row padding uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error unless `pixels` has exactly width times height times physical channel
    /// count elements. Dimension or size arithmetic, device allocation, output-view construction,
    /// row slicing, host-access coordination, or row upload can also fail.
    pub fn from_slice(context: &Arc<CudaContext>, size: Size, pixels: &[T]) -> Result<Self>
    where
        T: SupportedImage<L>,
    {
        let expected = checked_len(size, L::CHANNELS)?;
        if pixels.len() != expected {
            return Err(Error::LengthMismatch {
                name: "image pixels".into(),
                expected,
                actual: pixels.len(),
            });
        }
        let mut image = Self::create_uninit(context, size)?;
        let row_len = expected / size.height as usize;
        {
            let mut output = image.output()?;
            for (row, pixels) in pixels.chunks_exact(row_len).enumerate() {
                transfer::copy_from_host(&mut output.row_output(row)?, pixels)?;
            }
        }
        // SAFETY: completed row uploads initialized every logical channel, including alpha.
        Ok(unsafe { image.assume_init() })
    }

    /// Copies all logical channels into a completed native image, including alpha for AC4.
    ///
    /// # Errors
    ///
    /// Returns an error if allocating the destination or constructing its output view fails,
    /// or if the completed image copy or its host-access coordination fails.
    pub fn from_view(source: &ImageView<'_, T, L>) -> Result<Self>
    where
        T: SupportedImage<L>,
    {
        let mut image = Self::create_uninit(source.context(), source.size())?;
        image.output()?.copy_from(source)?;
        // SAFETY: the completed copy initialized every logical channel without exposing pitch padding.
        Ok(unsafe { image.assume_init() })
    }

    /// Returns a completed compact device copy of all logical image samples.
    /// Includes every physical channel, including alpha, and excludes native row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if compact sample-count arithmetic, destination allocation, view or
    /// row construction, host-access coordination, or a device copy fails.
    pub fn copy_to_device_memory(&self) -> Result<DeviceMemory<T>>
    where
        T: DeviceRepr,
    {
        let len = checked_len(self.size, L::CHANNELS)?;
        let mut destination = DeviceMemory::<T>::create_uninit(self.context(), len)?;
        self.copy_into_device_memory(&mut destination)?;
        // SAFETY: the completed operation above wrote every element.
        Ok(unsafe { destination.assume_init() })
    }

    /// Copies logical pixel channels into compact host storage, excluding row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if compact sample-count arithmetic, image-view or row construction,
    /// host-access coordination, or downloading any row fails.
    pub fn copy_to_host_vec(&self) -> Result<Vec<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let len = checked_len(self.size, L::CHANNELS)?;
        let image = self.view()?;
        let mut host = Vec::<T>::with_capacity(len);
        for row in 0..self.size.height as usize {
            let source = image.row(row)?;
            let initialized = host.len();
            transfer::copy_to_host_uninit(&source, &mut host.spare_capacity_mut()[..source.len()])?;
            // SAFETY: the completed row transfer established these additional T values.
            unsafe { host.set_len(initialized + source.len()) };
        }
        Ok(host)
    }

    /// Completes a compact copy of logical pixels into any CUDA output buffer.
    ///
    /// # Errors
    ///
    /// Returns an error unless the destination has the same CUDA context and exactly width
    /// times height times physical channel count elements. Size arithmetic, view construction,
    /// row slicing, host-access coordination, and device copies can also fail.
    pub fn copy_into_device_memory(
        &self,
        destination: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<()>
    where
        T: DeviceRepr,
    {
        if destination.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        let len = checked_len(self.size, L::CHANNELS)?;
        if destination.len() != len {
            return Err(Error::LengthMismatch {
                name: "image memory".into(),
                expected: len,
                actual: destination.len(),
            });
        }
        let image = self.view()?;
        let mut output = DeviceOutputView::from_output(destination);
        let row_len = self.size.width as usize * L::CHANNELS;
        for row in 0..self.size.height as usize {
            let start = row * row_len;
            let mut destination = output.reborrow().slice(start..start + row_len)?;
            transfer::copy_device(&mut destination, &image.row(row)?)?;
        }
        Ok(())
    }

    /// Returns the context that owns the native NPP allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.allocation.context()
    }

    /// Returns the image ROI size requested at allocation time.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the NPP row pitch in bytes.
    #[must_use]
    pub const fn step(&self) -> i32 {
        self.step
    }

    /// Borrows this allocation as an immutable NPP image view.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored size or pitch is not valid for an NPP image
    /// view.
    pub fn view(&self) -> Result<ImageView<'_, T, L>> {
        // SAFETY: the image owns this region and was allocated with this size and step.
        unsafe { ImageView::from_raw_parts(self.region(), self.size, self.step) }
    }

    /// Borrows this allocation as a mutable NPP image view.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored size or pitch is not valid for an NPP image
    /// view.
    pub fn view_mut(&mut self) -> Result<ImageViewMut<'_, T, L>> {
        // SAFETY: the image owns this region and was allocated with this size and step.
        unsafe { ImageViewMut::from_raw_parts(self.region(), self.size, self.step) }
    }

    /// Returns the raw device pointer owned by this allocation.
    ///
    /// The pointer remains owned by `self` and is freed with `nppiFree` when the
    /// last allocation owner is dropped.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.allocation.as_ptr().cast()
    }

    /// Returns the mutable raw device pointer owned by this allocation.
    ///
    /// The pointer remains owned by `self` and is freed with `nppiFree` when the
    /// last allocation owner is dropped.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.as_ptr().cast()
    }
}

impl_image_create!(C1, [
    u8 => nppiMalloc_8u_C1,
    i8 => nppiMalloc_8u_C1,
    u16 => nppiMalloc_16u_C1,
    f16 => nppiMalloc_16u_C1,
    i16 => nppiMalloc_16s_C1,
    ComplexI16 => nppiMalloc_16sc_C1,
    i32 => nppiMalloc_32s_C1,
    u32 => nppiMalloc_32s_C1,
    ComplexI32 => nppiMalloc_32sc_C1,
    f32 => nppiMalloc_32f_C1,
    Complex32 => nppiMalloc_32fc_C1,
]);

impl_image_create!(C2, [
    u8 => nppiMalloc_8u_C2,
    i8 => nppiMalloc_8u_C2,
    u16 => nppiMalloc_16u_C2,
    f16 => nppiMalloc_16u_C2,
    i16 => nppiMalloc_16s_C2,
    ComplexI16 => nppiMalloc_16sc_C2,
    ComplexI32 => nppiMalloc_32sc_C2,
    f32 => nppiMalloc_32f_C2,
    Complex32 => nppiMalloc_32fc_C2,
]);

impl_image_create!(C3, [
    u8 => nppiMalloc_8u_C3,
    i8 => nppiMalloc_8u_C3,
    u16 => nppiMalloc_16u_C3,
    f16 => nppiMalloc_16u_C3,
    i16 => nppiMalloc_16u_C3,
    ComplexI16 => nppiMalloc_16sc_C3,
    i32 => nppiMalloc_32s_C3,
    ComplexI32 => nppiMalloc_32sc_C3,
    f32 => nppiMalloc_32f_C3,
    Complex32 => nppiMalloc_32fc_C3,
]);

impl_image_create!(C4, [
    u8 => nppiMalloc_8u_C4,
    i8 => nppiMalloc_8u_C4,
    u16 => nppiMalloc_16u_C4,
    f16 => nppiMalloc_16u_C4,
    i16 => nppiMalloc_16s_C4,
    ComplexI16 => nppiMalloc_16sc_C4,
    i32 => nppiMalloc_32s_C4,
    ComplexI32 => nppiMalloc_32sc_C4,
    f32 => nppiMalloc_32f_C4,
    Complex32 => nppiMalloc_32fc_C4,
]);

impl_image_create!(AC4, [
    u8 => nppiMalloc_8u_C4,
    i8 => nppiMalloc_8u_C4,
    u16 => nppiMalloc_16u_C4,
    f16 => nppiMalloc_16u_C4,
    i16 => nppiMalloc_16s_C4,
    ComplexI16 => nppiMalloc_16sc_C4,
    i32 => nppiMalloc_32s_C4,
    u32 => nppiMalloc_32s_C4,
    ComplexI32 => nppiMalloc_32sc_C4,
    f32 => nppiMalloc_32f_C4,
    Complex32 => nppiMalloc_32fc_C4,
]);

/// Allocates pitched image storage without initializing logical pixels or padding.
///
/// # Errors
///
/// Returns an error if dimensions are nonpositive, CUDA context binding fails, the native
/// allocator returns null, allocation-size arithmetic overflows, or the returned byte pitch
/// and allocation bounds cannot form a valid image.
pub fn create_uninit<T, L>(
    allocation_context: &Arc<CudaContext>,
    size: Size,
) -> Result<Image<MaybeUninit<T>, L>>
where
    T: SupportedImage<L>,
    L: ChannelLayout,
{
    Image::<T, L>::create_uninit(allocation_context, size)
}

// SAFETY: the native owner retains full physical capacity; this region narrows it
// to the pixel envelope without asserting initialization of intervening padding.
unsafe impl<T, L: ChannelLayout> DeviceRegionProvider for Image<T, L> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation
            .region()
            .slice(
                ..image_byte_len::<T, L>(self.size, self.step)
                    .expect("owned image geometry is valid"),
            )
            .expect("owned image fits its allocation")
    }
}

impl<T, L: ChannelLayout> Image<MaybeUninit<T>, L> {
    /// Borrows potentially uninitialized pixels for a producer of T values.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored dimensions, byte stride, element alignment, or allocation
    /// bounds cannot form a valid output view.
    pub fn output(&mut self) -> Result<ImageOutputView<'_, T, L>> {
        // SAFETY: MaybeUninit<T> has T's layout and the owner is exclusively borrowed.
        unsafe { ImageOutputView::from_raw_parts(self.region(), self.size, self.step) }
    }

    /// Promotes completed logical pixels without promoting pitch padding.
    ///
    /// # Safety
    /// Every logical pixel channel must contain a valid T, including alpha for
    /// AC4, and production must be ordered before every consumer and readback.
    /// Native submission success alone does not establish completed initialization.
    #[must_use]
    pub unsafe fn assume_init(self) -> Image<T, L> {
        Image {
            allocation: self.allocation,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        }
    }
}
