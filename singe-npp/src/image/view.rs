//! Pitched image borrows carrying actual CUDA allocation metadata.
use singe_cuda::memory::transfer::copy_device;

use crate::image::exchange_validation::validate_same_size;

mod window;

use std::{
    marker::PhantomData,
    mem::{MaybeUninit, align_of},
    sync::Arc,
};

use singe_cuda::{
    context::Context,
    view::{
        DeviceOutput, DeviceOutputView, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut,
    },
};
use singe_npp_sys as sys;
pub use window::ImageWindow;

use crate::{
    error::{Error, Result},
    types::Size,
    utility::{checked_step, ensure_positive, to_usize},
};

/// Physical component count and NPP channel-processing semantics of an interleaved pixel layout.
pub trait ChannelLayout: Copy + 'static {
    /// Number of physical components stored for each pixel.
    const CHANNELS: usize;
    /// Whether NPP AC4 operations ignore the alpha component.
    ///
    /// This changes channel-processing semantics without removing alpha storage or its initialization requirement.
    const IGNORES_ALPHA: bool = false;
}

/// Layouts whose unmasked NPP operations process every physical channel.
/// AC4 operations leave alpha untouched and therefore cannot initialize a fresh image alone.
pub trait AllChannels: ChannelLayout + private::AllChannels {}

mod private {
    pub trait AllChannels {}
}

macro_rules! all_channels {
    ($($layout:ty),+ $(,)?) => {$(
        impl private::AllChannels for $layout {}
        impl AllChannels for $layout {}
    )+};
}

all_channels!(C1, C2, C3, C4);

/// One interleaved component per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C1;

impl ChannelLayout for C1 {
    const CHANNELS: usize = 1;
}

/// Two interleaved components per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C2;

impl ChannelLayout for C2 {
    const CHANNELS: usize = 2;
}

/// Three interleaved components per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C3;

impl ChannelLayout for C3 {
    const CHANNELS: usize = 3;
}

/// Four interleaved components per pixel, all processed by ordinary C4 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct C4;

impl ChannelLayout for C4 {
    const CHANNELS: usize = 4;
}

/// Four interleaved components per pixel with alpha ignored by AC4 operations.
///
/// The alpha component still occupies storage and must be initialized before the image is read as initialized values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AC4;

impl ChannelLayout for AC4 {
    const CHANNELS: usize = 4;
    const IGNORES_ALPHA: bool = true;
}

/// Conventional name for the single-channel [`C1`] layout.
pub type Gray = C1;

/// Conventional name for the two-component [`C2`] layout.
pub type Channels2 = C2;

/// Conventional RGB name for the three-component [`C3`] layout.
pub type Rgb = C3;

/// Conventional RGBA name for the four-component [`C4`] layout.
pub type Rgba = C4;

/// Conventional RGBA name for the alpha-ignored [`AC4`] layout.
pub type AlphaIgnoredRgba = AC4;

/// Initialized single-channel 8-bit image view commonly used for masks.
pub type MaskView<'a> = ImageView<'a, u8, C1>;

/// Exclusively borrowed initialized single-channel 8-bit image view commonly used for masks.
pub type MaskViewMut<'a> = ImageViewMut<'a, u8, C1>;

/// Writable single-channel 8-bit image destination commonly used for masks.
pub type MaskOutputView<'a> = ImageOutputView<'a, u8, C1>;

fn sample_size<L: ChannelLayout>(size: Size) -> Result<Size> {
    let width =
        usize::try_from(size.width).map_err(|_| Error::out_of_range("image sample width"))?;
    let width = width
        .checked_mul(L::CHANNELS)
        .ok_or_else(|| Error::out_of_range("image sample width"))?;
    let width = i32::try_from(width).map_err(|_| Error::out_of_range("image sample width"))?;
    Ok(Size::new(width, size.height))
}

fn checked_region<T, L: ChannelLayout>(
    region: DeviceRegion<'_>,
    size: Size,
    step: i32,
) -> Result<DeviceRegion<'_>> {
    validate_image_shape::<T, L>(size, step)?;
    if !region.as_ptr().addr().is_multiple_of(align_of::<T>())
        || !(step as usize).is_multiple_of(align_of::<T>())
    {
        return Err(Error::out_of_range("image alignment"));
    }
    let bytes = image_byte_len::<T, L>(size, step)?;
    if bytes > region.byte_len() {
        return Err(Error::LengthMismatch {
            name: "image memory".into(),
            expected: bytes,
            actual: region.byte_len(),
        });
    }
    Ok(region.slice(..bytes)?)
}

fn row_region<T, L: ChannelLayout>(
    region: DeviceRegion<'_>,
    size: Size,
    step: i32,
    row: usize,
) -> Result<DeviceRegion<'_>> {
    if row >= size.height as usize {
        return Err(Error::out_of_range("image row"));
    }
    let start = row
        .checked_mul(step as usize)
        .ok_or_else(|| Error::out_of_range("image row offset"))?;
    let bytes = (size.width as usize)
        .checked_mul(L::CHANNELS)
        .and_then(|n| n.checked_mul(size_of::<T>()))
        .ok_or_else(|| Error::out_of_range("image row size"))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| Error::out_of_range("image row end"))?;
    Ok(region.slice(start..end)?)
}

/// Initialized logical pixels; row padding carries no initialization guarantee.
#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a, T, L = C1> {
    region: DeviceRegion<'a>,
    size: Size,
    step: i32,
    _t: PhantomData<(&'a T, L)>,
}

impl<'a, T, L: ChannelLayout> ImageView<'a, T, L> {
    /// Views every stored channel as a scalar sample while preserving pixels, pitch, and initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if multiplying the pixel width by the physical channel count overflows or the resulting sample width cannot be represented as `i32`.
    pub fn into_samples(self) -> Result<ImageView<'a, T, C1>> {
        Ok(ImageView {
            region: self.region,
            size: sample_size::<L>(self.size)?,
            step: self.step,
            _t: PhantomData,
        })
    }

    /// Changes channel-processing semantics while preserving the physical channel count and initialized borrow.
    ///
    /// # Errors
    ///
    /// Returns an error if the new layout has a different physical channel count.
    pub fn with_layout<M: ChannelLayout>(self) -> Result<ImageView<'a, T, M>> {
        if M::CHANNELS != L::CHANNELS {
            return Err(Error::out_of_range("image channel layout"));
        }
        Ok(ImageView {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        })
    }

    /// Borrows device-view storage as initialized read-only image pixels.
    /// `size` gives pixel dimensions; the row stride is the tightly packed pixel-row size.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_view(view: DeviceView<'a, T>, size: Size) -> Result<Self>
    where
        T: DeviceRepr,
    {
        Self::from_view_with_step(
            view,
            size,
            checked_step(size.width, L::CHANNELS, size_of::<T>())?,
        )
    }

    /// Borrows device-view storage as initialized read-only image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_view_with_step(view: DeviceView<'a, T>, size: Size, step: i32) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: the CUDA input view establishes initialized access to its full range.
        unsafe { Self::from_raw_parts(view.region(), size, step) }
    }

    /// Borrows buffer storage as initialized read-only image pixels.
    /// `size` gives pixel dimensions; the row stride is the tightly packed pixel-row size.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_buffer<B>(buffer: &'a B, size: Size) -> Result<Self>
    where
        T: DeviceRepr,
        B: DeviceSlice<T> + ?Sized,
    {
        Self::from_view(DeviceView::from_buffer(buffer), size)
    }

    /// Borrows buffer storage as initialized read-only image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_buffer_with_step<B>(buffer: &'a B, size: Size, step: i32) -> Result<Self>
    where
        T: DeviceRepr,
        B: DeviceSlice<T> + ?Sized,
    {
        Self::from_view_with_step(DeviceView::from_buffer(buffer), size, step)
    }

    /// Borrows allocation metadata as initialized read-only image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    ///
    /// # Safety
    /// Every logical pixel channel must remain initialized and readable for the view lifetime.
    /// T must have a stable plain device representation. Row padding need not be initialized.
    /// Retain the actual allocation through queued accesses and captured replay.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, size: Size, step: i32) -> Result<Self> {
        let region = checked_region::<T, L>(region, size, step)?;
        Ok(Self {
            region,
            size,
            step,
            _t: PhantomData,
        })
    }

    /// Returns the logical width and height in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the byte stride between successive image rows.
    #[must_use]
    pub const fn step(&self) -> i32 {
        self.step
    }

    /// Returns the byte span through the last logical pixel of the last row.
    ///
    /// The span is `(height - 1) * step + width * channels * size_of::<T>()`.
    /// It includes padding between rows and excludes padding after the final row.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions or byte stride are nonpositive, or computing the logical row size or pitched span overflows `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        image_byte_len::<T, L>(self.size, self.step)
    }

    /// Returns the pitched byte region and its actual allocation metadata.
    ///
    /// The region may include row padding that has no initialized-value guarantee.
    #[must_use]
    pub const fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Returns the CUDA context that owns the backing allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Returns the device pointer to the first logical pixel component.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.region.as_ptr().cast()
    }

    /// Borrows one initialized pixel row, excluding pitch padding.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` is outside the image height, row-offset or length arithmetic overflows, or the selected byte range cannot form the requested device view.
    pub fn row(&self, row: usize) -> Result<DeviceView<'_, T>>
    where
        T: DeviceRepr,
    {
        let region = row_region::<T, L>(self.region, self.size, self.step, row)?;
        // SAFETY: input construction establishes all logical values in this row.
        Ok(unsafe { DeviceView::from_raw_parts(region, self.size.width as usize * L::CHANNELS)? })
    }

    /// Returns a non-owning native descriptor for this pitched image.
    ///
    /// The descriptor does not retain allocation ownership or enforce Rust borrow lifetimes.
    /// Retain the allocation and the required access permissions through native use.
    /// The native descriptor contains a mutable pointer, but this image borrow permits read access only.
    #[must_use]
    pub fn descriptor(&self) -> sys::NppiImageDescriptor {
        sys::NppiImageDescriptor {
            pData: self.as_ptr().cast_mut().cast(),
            nStep: self.step,
            oSize: self.size.into(),
        }
    }
}

// SAFETY: checked constructors preserve the full pixel envelope and actual allocation metadata.
unsafe impl<T, L> DeviceRegionProvider for ImageView<'_, T, L> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

/// Exclusive initialized logical pixels, including alpha for four-channel layouts.
#[derive(Debug)]
pub struct ImageViewMut<'a, T, L = C1> {
    region: DeviceRegion<'a>,
    size: Size,
    step: i32,
    _t: PhantomData<(&'a mut T, L)>,
}

impl<'a, T, L: ChannelLayout> ImageViewMut<'a, T, L> {
    /// Flattens physical channels into exclusively borrowed initialized samples.
    ///
    /// # Errors
    ///
    /// Returns an error if multiplying the pixel width by the physical channel count overflows or the resulting sample width cannot be represented as `i32`.
    pub fn into_samples(self) -> Result<ImageViewMut<'a, T, C1>> {
        Ok(ImageViewMut {
            region: self.region,
            size: sample_size::<L>(self.size)?,
            step: self.step,
            _t: PhantomData,
        })
    }

    /// Changes channel-processing semantics while retaining the exclusive initialized borrow.
    ///
    /// # Errors
    ///
    /// Returns an error if the new layout has a different physical channel count.
    pub fn with_layout<M: ChannelLayout>(self) -> Result<ImageViewMut<'a, T, M>> {
        if M::CHANNELS != L::CHANNELS {
            return Err(Error::out_of_range("image channel layout"));
        }
        Ok(ImageViewMut {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        })
    }

    /// Borrows device-view storage as exclusive initialized image pixels.
    /// `size` gives pixel dimensions; the row stride is the tightly packed pixel-row size.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_view(view: DeviceViewMut<'a, T>, size: Size) -> Result<Self>
    where
        T: DeviceRepr,
    {
        Self::from_view_with_step(
            view,
            size,
            checked_step(size.width, L::CHANNELS, size_of::<T>())?,
        )
    }

    /// Borrows device-view storage as exclusive initialized image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_view_with_step(view: DeviceViewMut<'a, T>, size: Size, step: i32) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: consume the exclusive initialized borrow while preserving its original lifetime.
        unsafe { Self::from_raw_parts(view.into_region(), size, step) }
    }

    /// Borrows buffer storage as exclusive initialized image pixels.
    /// `size` gives pixel dimensions; the row stride is the tightly packed pixel-row size.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_buffer<B>(buffer: &'a mut B, size: Size) -> Result<Self>
    where
        T: DeviceRepr,
        B: DeviceSliceMut<T> + ?Sized,
    {
        Self::from_view(DeviceViewMut::from_buffer(buffer), size)
    }

    /// Borrows buffer storage as exclusive initialized image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_buffer_with_step<B>(buffer: &'a mut B, size: Size, step: i32) -> Result<Self>
    where
        T: DeviceRepr,
        B: DeviceSliceMut<T> + ?Sized,
    {
        Self::from_view_with_step(DeviceViewMut::from_buffer(buffer), size, step)
    }

    /// Borrows allocation metadata as exclusive initialized image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    ///
    /// # Safety
    /// Every logical pixel channel must remain initialized and exclusively accessible for the view lifetime.
    /// T must have a stable plain device representation. Row padding need not be initialized.
    /// Retain the actual allocation through queued accesses and captured replay.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, size: Size, step: i32) -> Result<Self> {
        let region = checked_region::<T, L>(region, size, step)?;
        Ok(Self {
            region,
            size,
            step,
            _t: PhantomData,
        })
    }

    /// Returns the logical width and height in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the byte stride between successive image rows.
    #[must_use]
    pub const fn step(&self) -> i32 {
        self.step
    }

    /// Returns the byte span through the last logical pixel of the last row.
    ///
    /// The span is `(height - 1) * step + width * channels * size_of::<T>()`.
    /// It includes padding between rows and excludes padding after the final row.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions or byte stride are nonpositive, or computing the logical row size or pitched span overflows `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        image_byte_len::<T, L>(self.size, self.step)
    }

    /// Returns the pitched byte region and its actual allocation metadata.
    ///
    /// The region may include row padding that has no initialized-value guarantee.
    #[must_use]
    pub const fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Returns the CUDA context that owns the backing allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Returns a writable device pointer to the first logical pixel component.
    ///
    /// Retain the allocation and its exclusive access through native use of the pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.region.as_ptr().cast()
    }

    /// Reborrows the logical pixels as an exclusive output destination.
    ///
    /// The returned view does not claim that every logical component contains an initialized value.
    pub fn output(&mut self) -> ImageOutputView<'_, T, L> {
        ImageOutputView {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        }
    }

    /// Borrows one writable output row, excluding pitch padding.
    ///
    /// The row contains `width * channels` physical components, including alpha for AC4 layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` is outside the image height, row-offset or length arithmetic overflows, or the selected byte range cannot form the requested device view.
    pub fn row_output(&mut self, row: usize) -> Result<DeviceOutputView<'_, T>>
    where
        T: DeviceRepr,
    {
        let region = row_region::<T, L>(self.region, self.size, self.step, row)?;
        // SAFETY: this exclusively borrowed row contains writable storage for the logical pixels.
        Ok(unsafe {
            DeviceOutputView::from_raw_parts(region, self.size.width as usize * L::CHANNELS)?
        })
    }

    /// Returns a non-owning native descriptor for this pitched image.
    ///
    /// The descriptor does not retain allocation ownership or enforce Rust borrow lifetimes.
    /// Retain the allocation and the required access permissions through native use.
    pub fn descriptor(&mut self) -> sys::NppiImageDescriptor {
        sys::NppiImageDescriptor {
            pData: self.as_mut_ptr().cast(),
            nStep: self.step,
            oSize: self.size.into(),
        }
    }

    /// Reborrows the initialized logical pixels for read-only access.
    #[must_use]
    pub fn as_view(&self) -> ImageView<'_, T, L> {
        ImageView {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        }
    }

    /// Reborrows the initialized logical pixels with exclusive access.
    pub fn reborrow(&mut self) -> ImageViewMut<'_, T, L> {
        ImageViewMut {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        }
    }

    /// Borrows one exclusively accessible initialized row, excluding pitch padding.
    ///
    /// The row contains `width * channels` physical components, including alpha for AC4 layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` is outside the image height, row-offset or length arithmetic overflows, or the selected byte range cannot form the requested device view.
    pub fn row_mut(&mut self, row: usize) -> Result<DeviceViewMut<'_, T>>
    where
        T: DeviceRepr,
    {
        let region = row_region::<T, L>(self.region, self.size, self.step, row)?;
        // SAFETY: the exclusive initialized image borrow covers this logical row.
        Ok(unsafe {
            DeviceViewMut::from_raw_parts(region, self.size.width as usize * L::CHANNELS)?
        })
    }
}

// SAFETY: checked constructors preserve the full pixel envelope and actual allocation metadata.
unsafe impl<T, L> DeviceRegionProvider for ImageViewMut<'_, T, L> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

/// Exclusive pixel destinations without an initialized-value guarantee.
///
/// Native producers may write only a subset of pixels or channels.
/// Initialization promotion requires every logical channel, including alpha,
/// regardless of the native operation's AC4 naming.
#[derive(Debug)]
pub struct ImageOutputView<'a, T, L = C1> {
    region: DeviceRegion<'a>,
    size: Size,
    step: i32,
    _t: PhantomData<(&'a mut MaybeUninit<T>, L)>,
}

impl<'a, T, L: ChannelLayout> ImageOutputView<'a, T, L> {
    /// Flattens physical channels into writable samples without claiming initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if multiplying the pixel width by the physical channel count overflows or the resulting sample width cannot be represented as `i32`.
    pub fn into_samples(self) -> Result<ImageOutputView<'a, T, C1>> {
        Ok(ImageOutputView {
            region: self.region,
            size: sample_size::<L>(self.size)?,
            step: self.step,
            _t: PhantomData,
        })
    }

    /// Changes channel-processing semantics without changing physical channels or claiming initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the new layout has a different physical channel count.
    pub fn with_layout<M: ChannelLayout>(self) -> Result<ImageOutputView<'a, T, M>> {
        if M::CHANNELS != L::CHANNELS {
            return Err(Error::out_of_range("image channel layout"));
        }
        Ok(ImageOutputView {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        })
    }

    /// Completes a copy of every logical channel, including alpha, without copying row padding.
    ///
    /// # Errors
    ///
    /// Returns an error if source and destination dimensions or CUDA contexts differ, corresponding row ranges overlap, or a row range cannot be formed.
    /// Host-access coordination, capture checks, CUDA context binding, device copies, and synchronization can also fail.
    pub fn copy_from(&mut self, source: &ImageView<'_, T, L>) -> Result<()>
    where
        T: DeviceRepr,
    {
        validate_same_size(source.size(), self.size)?;
        for row in 0..self.size.height as usize {
            copy_device(&mut self.row_output(row)?, &source.row(row)?)?;
        }
        Ok(())
    }

    /// Borrows device output storage as writable output image pixels.
    /// `size` gives pixel dimensions; the row stride is the tightly packed pixel-row size.
    /// The resulting view grants write access without claiming initialized pixel values.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_output(view: DeviceOutputView<'a, T>, size: Size) -> Result<Self>
    where
        T: DeviceRepr,
    {
        Self::from_output_with_step(
            view,
            size,
            checked_step(size.width, L::CHANNELS, size_of::<T>())?,
        )
    }

    /// Borrows device output storage as writable output image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    /// The resulting view grants write access without claiming initialized pixel values.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_output_with_step(
        view: DeviceOutputView<'a, T>,
        size: Size,
        step: i32,
    ) -> Result<Self>
    where
        T: DeviceRepr,
    {
        // SAFETY: consume the exclusive output borrow without asserting initialized values.
        unsafe { Self::from_raw_parts(view.into_region(), size, step) }
    }

    /// Borrows buffer storage as writable output image pixels.
    /// `size` gives pixel dimensions; the row stride is the tightly packed pixel-row size.
    /// The resulting view grants write access without claiming initialized pixel values.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_buffer<B: DeviceOutput<T> + ?Sized>(buffer: &'a mut B, size: Size) -> Result<Self>
    where
        T: DeviceRepr,
    {
        Self::from_output(DeviceOutputView::from_output(buffer), size)
    }

    /// Borrows buffer storage as writable output image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    /// The resulting view grants write access without claiming initialized pixel values.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    pub fn from_buffer_with_step<B: DeviceOutput<T> + ?Sized>(
        buffer: &'a mut B,
        size: Size,
        step: i32,
    ) -> Result<Self>
    where
        T: DeviceRepr,
    {
        Self::from_output_with_step(DeviceOutputView::from_output(buffer), size, step)
    }

    /// Borrows allocation metadata as writable output image pixels.
    /// `size` gives pixel dimensions; `step` is the byte stride between rows.
    /// The resulting view grants write access without claiming initialized pixel values.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are nonpositive, the channel count or element size is zero, or the byte stride is smaller than a logical pixel row.
    /// The base address and stride must be aligned for `T`, and the backing region must cover the pitched image span.
    /// Row-byte counts must fit `i32`; size or range arithmetic can also overflow.
    ///
    /// # Safety
    /// Every logical pixel channel must remain exclusively writable for the view lifetime.
    /// T must have a stable plain device representation. Row padding need not be initialized.
    /// Retain the actual allocation through queued accesses and captured replay.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, size: Size, step: i32) -> Result<Self> {
        let region = checked_region::<T, L>(region, size, step)?;
        Ok(Self {
            region,
            size,
            step,
            _t: PhantomData,
        })
    }

    /// Returns the logical width and height in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the byte stride between successive image rows.
    #[must_use]
    pub const fn step(&self) -> i32 {
        self.step
    }

    /// Returns the byte span through the last logical pixel of the last row.
    ///
    /// The span is `(height - 1) * step + width * channels * size_of::<T>()`.
    /// It includes padding between rows and excludes padding after the final row.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions or byte stride are nonpositive, or computing the logical row size or pitched span overflows `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        image_byte_len::<T, L>(self.size, self.step)
    }

    /// Returns the pitched byte region and its actual allocation metadata.
    ///
    /// The region may include row padding that has no initialized-value guarantee.
    #[must_use]
    pub const fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Returns the CUDA context that owns the backing allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Returns a writable device pointer to the first logical pixel component.
    ///
    /// Retain the allocation and its exclusive access through native use of the pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.region.as_ptr().cast()
    }

    /// Reborrows the logical pixels as an exclusive output destination.
    ///
    /// The returned view does not claim that every logical component contains an initialized value.
    pub fn output(&mut self) -> ImageOutputView<'_, T, L> {
        ImageOutputView {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        }
    }

    /// Borrows one writable output row, excluding pitch padding.
    ///
    /// The row contains `width * channels` physical components, including alpha for AC4 layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` is outside the image height, row-offset or length arithmetic overflows, or the selected byte range cannot form the requested device view.
    pub fn row_output(&mut self, row: usize) -> Result<DeviceOutputView<'_, T>>
    where
        T: DeviceRepr,
    {
        let region = row_region::<T, L>(self.region, self.size, self.step, row)?;
        // SAFETY: this exclusively borrowed row contains writable storage for the logical pixels.
        Ok(unsafe {
            DeviceOutputView::from_raw_parts(region, self.size.width as usize * L::CHANNELS)?
        })
    }

    /// Returns a non-owning native descriptor for this pitched image.
    ///
    /// The descriptor does not retain allocation ownership or enforce Rust borrow lifetimes.
    /// Retain the allocation and the required access permissions through native use.
    pub fn descriptor(&mut self) -> sys::NppiImageDescriptor {
        sys::NppiImageDescriptor {
            pData: self.as_mut_ptr().cast(),
            nStep: self.step,
            oSize: self.size.into(),
        }
    }

    /// Promotes the output destination to an exclusive initialized image view.
    ///
    /// The conversion relies on the caller's initialization guarantee and performs no device synchronization or pixel validation.
    ///
    /// # Safety
    /// Every logical channel of every pixel must contain a valid T, including
    /// alpha for AC4, with production ordered before subsequent consumers.
    #[must_use]
    pub unsafe fn assume_init(self) -> ImageViewMut<'a, T, L> {
        ImageViewMut {
            region: self.region,
            size: self.size,
            step: self.step,
            _t: PhantomData,
        }
    }
}

// SAFETY: checked constructors preserve the full pixel envelope and actual allocation metadata.
unsafe impl<T, L> DeviceRegionProvider for ImageOutputView<'_, T, L> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

/// A borrowed collection of two to four initialized single-channel image planes.
#[derive(Debug, Clone, Copy)]
pub struct PlanarImageView<'a, T, const C: usize> {
    planes: [ImageView<'a, T, C1>; C],
}

impl<'a, T, const C: usize> PlanarImageView<'a, T, C> {
    /// Combines initialized single-channel planes after validating their shared geometry.
    ///
    /// # Errors
    ///
    /// Returns an error unless `C` is between two and four and all planes have the same CUDA context, dimensions, and byte stride.
    /// Dimension arithmetic and stride conversions used to report mismatches can also fail.
    pub fn create(planes: [ImageView<'a, T, C1>; C]) -> Result<Self> {
        validate_plane_count(C)?;
        validate_same_planes(&planes)?;
        Ok(Self { planes })
    }

    /// Returns the borrowed array of single-channel planes.
    #[must_use]
    pub const fn planes(&self) -> &[ImageView<'a, T, C1>; C] {
        &self.planes
    }
}

/// An exclusively borrowed collection of two to four initialized single-channel image planes.
#[derive(Debug)]
pub struct PlanarImageViewMut<'a, T, const C: usize> {
    planes: [ImageViewMut<'a, T, C1>; C],
}

impl<'a, T, const C: usize> PlanarImageViewMut<'a, T, C> {
    /// Combines exclusive initialized single-channel planes after validating their shared geometry.
    ///
    /// # Errors
    ///
    /// Returns an error unless `C` is between two and four and all planes have the same CUDA context, dimensions, and byte stride.
    /// Dimension arithmetic and stride conversions used to report mismatches can also fail.
    pub fn create(planes: [ImageViewMut<'a, T, C1>; C]) -> Result<Self> {
        validate_plane_count(C)?;
        validate_same_planes(&planes)?;
        Ok(Self { planes })
    }

    /// Returns the borrowed array of single-channel planes.
    #[must_use]
    pub const fn planes(&self) -> &[ImageViewMut<'a, T, C1>; C] {
        &self.planes
    }

    /// Returns mutable access to the array of single-channel planes.
    ///
    /// Replacing a plane does not repeat the constructor's shared-geometry checks.
    pub fn planes_mut(&mut self) -> &mut [ImageViewMut<'a, T, C1>; C] {
        &mut self.planes
    }
}

/// Exclusive, potentially uninitialized image planes.
#[derive(Debug)]
pub struct PlanarImageOutputView<'a, T, const C: usize> {
    planes: [ImageOutputView<'a, T, C1>; C],
}

impl<'a, T, const C: usize> PlanarImageOutputView<'a, T, C> {
    /// Combines writable output single-channel planes after validating their shared geometry.
    ///
    /// # Errors
    ///
    /// Returns an error unless `C` is between two and four and all planes have the same CUDA context, dimensions, and byte stride.
    /// Dimension arithmetic and stride conversions used to report mismatches can also fail.
    pub fn create(planes: [ImageOutputView<'a, T, C1>; C]) -> Result<Self> {
        validate_plane_count(C)?;
        validate_same_planes(&planes)?;
        Ok(Self { planes })
    }

    /// Returns the borrowed array of single-channel planes.
    #[must_use]
    pub const fn planes(&self) -> &[ImageOutputView<'a, T, C1>; C] {
        &self.planes
    }

    /// Returns mutable access to the array of single-channel planes.
    ///
    /// Replacing a plane does not repeat the constructor's shared-geometry checks.
    pub fn planes_mut(&mut self) -> &mut [ImageOutputView<'a, T, C1>; C] {
        &mut self.planes
    }
}

fn validate_image_shape<T, L>(size: Size, step: i32) -> Result<()>
where
    L: ChannelLayout,
{
    size.validate()?;

    let element_size = size_of::<T>();
    if element_size == 0 {
        return Err(Error::out_of_range("element size"));
    }

    if L::CHANNELS == 0 {
        return Err(Error::out_of_range("image channels"));
    }
    let minimum_step = checked_step(size.width, L::CHANNELS, element_size)?;
    if step < minimum_step {
        return Err(Error::out_of_range("step"));
    }

    Ok(())
}

pub(crate) fn image_byte_len<T, L>(size: Size, step: i32) -> Result<usize>
where
    L: ChannelLayout,
{
    size.validate()?;
    ensure_positive(step, "step")?;

    let row_bytes = (size.width as usize)
        .checked_mul(L::CHANNELS)
        .and_then(|value| value.checked_mul(size_of::<T>()))
        .ok_or_else(|| Error::out_of_range("byte_len"))?;
    (size.height as usize - 1)
        .checked_mul(step as usize)
        .and_then(|value| value.checked_add(row_bytes))
        .ok_or_else(|| Error::out_of_range("byte_len"))
}

fn validate_plane_count(count: usize) -> Result<()> {
    match count {
        2..=4 => Ok(()),
        _ => Err(Error::out_of_range("plane count")),
    }
}

fn validate_same_planes<T, P, const C: usize>(planes: &[P; C]) -> Result<()>
where
    P: PlaneInfo<T>,
{
    let Some(first) = planes.first() else {
        return Ok(());
    };

    for plane in planes.iter().skip(1) {
        if plane.context() != first.context() {
            return Err(Error::StreamContextMismatch);
        }
        if plane.size() != first.size() {
            let expected = (first.size().width as usize)
                .checked_mul(first.size().height as usize)
                .ok_or_else(|| Error::out_of_range("len"))?;
            let actual = (plane.size().width as usize)
                .checked_mul(plane.size().height as usize)
                .ok_or_else(|| Error::out_of_range("len"))?;
            return Err(Error::LengthMismatch {
                name: "plane size".into(),
                expected,
                actual,
            });
        }

        if plane.step() != first.step() {
            return Err(Error::LengthMismatch {
                name: "plane step".into(),
                expected: to_usize(first.step(), "plane step")?,
                actual: to_usize(plane.step(), "plane step")?,
            });
        }
    }

    Ok(())
}

trait PlaneInfo<T> {
    fn context(&self) -> &Arc<Context>;

    fn size(&self) -> Size;

    fn step(&self) -> i32;
}

impl<T> PlaneInfo<T> for ImageView<'_, T, C1> {
    fn context(&self) -> &Arc<Context> {
        self.context()
    }

    fn size(&self) -> Size {
        self.size()
    }

    fn step(&self) -> i32 {
        self.step()
    }
}

impl<T> PlaneInfo<T> for ImageViewMut<'_, T, C1> {
    fn context(&self) -> &Arc<Context> {
        self.context()
    }

    fn size(&self) -> Size {
        self.size()
    }

    fn step(&self) -> i32 {
        self.step()
    }
}

pub(crate) fn image_descriptors<T, L>(
    images: &[ImageView<'_, T, L>],
) -> Vec<sys::NppiImageDescriptor>
where
    L: ChannelLayout,
{
    images.iter().map(ImageView::descriptor).collect()
}

pub(crate) trait ImageDescriptorMut {
    fn image_descriptor(&mut self) -> sys::NppiImageDescriptor;
}

impl<T, L: ChannelLayout> ImageDescriptorMut for ImageViewMut<'_, T, L> {
    fn image_descriptor(&mut self) -> sys::NppiImageDescriptor {
        self.descriptor()
    }
}

impl<T, L: ChannelLayout> ImageDescriptorMut for ImageOutputView<'_, T, L> {
    fn image_descriptor(&mut self) -> sys::NppiImageDescriptor {
        self.descriptor()
    }
}

pub(crate) fn image_descriptors_mut(
    images: &mut [impl ImageDescriptorMut],
) -> Vec<sys::NppiImageDescriptor> {
    images
        .iter_mut()
        .map(ImageDescriptorMut::image_descriptor)
        .collect()
}

impl<T> PlaneInfo<T> for ImageOutputView<'_, T, C1> {
    fn context(&self) -> &Arc<Context> {
        self.context()
    }

    fn size(&self) -> Size {
        self.size()
    }

    fn step(&self) -> i32 {
        self.step()
    }
}
