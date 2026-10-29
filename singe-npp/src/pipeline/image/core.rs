use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::view::{DeviceRepr, DeviceSlice};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        SupportedImage,
        memory::Image,
        view::{AllChannels, C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{Workspace, image::fill},
    types::Size,
};

/// Current storage behind a fluent image pipeline.
#[derive(Debug)]
#[non_exhaustive]
pub enum ImageBacking<'a, T, L = C1> {
    /// A readable image view borrowed from caller-owned storage.
    Borrowed(ImageView<'a, T, L>),
    /// An owned device image allocation.
    Owned(Image<T, L>),
}

/// Eager, stream-aware image processing pipeline.
///
/// Numeric conversion, scaling, and border construction process every physical channel, including alpha.
/// Owning geometric methods transform an initialized destination supplied by the caller.
/// Non-border box, median, min/max filters, and basic morphology operate on valid source windows.
/// Their output dimensions exclude edges without complete mask support; AC4 preserves center alpha.
#[derive(Debug)]
pub struct ImagePipeline<'a, T, L = C1> {
    pub(crate) stream_context: &'a StreamContext,
    pub(crate) workspace: &'a mut Workspace,
    pub(crate) backing: ImageBacking<'a, T, L>,
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Allocates an uninitialized image and retains the stream context and workspace borrow.
    ///
    /// Returns a pending pipeline whose output can be initialized before readable operations.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if image dimensions or allocation arithmetic
    /// are invalid, or if allocating device image storage fails.
    pub fn create_uninit(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        size: Size,
    ) -> Result<UninitImagePipeline<'a, T, L>>
    where
        T: SupportedImage<L>,
    {
        stream_context.stream().ensure_not_capturing()?;
        let image = Image::<T, L>::create_uninit(stream_context.cuda_context(), size)?;
        Ok(UninitImagePipeline {
            stream_context,
            workspace,
            image,
        })
    }

    /// Creates a pipeline over a borrowed image without copying its pixels.
    ///
    /// Checks the CUDA context without waiting for pending producers or rejecting capture here.
    /// Individual eager operations prepare their regions as documented.
    ///
    /// # Errors
    ///
    /// Returns an error if `source` belongs to another CUDA context.
    pub fn from_view(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        source: ImageView<'a, T, L>,
    ) -> Result<Self> {
        if source.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        Ok(Self {
            stream_context,
            workspace,
            backing: ImageBacking::Borrowed(source),
        })
    }

    /// Creates a pipeline by taking ownership of `image` without copying its pixels.
    ///
    /// Rejects capture and checks the CUDA context without synchronizing pending producers.
    /// If rejected during capture, retains the allocation until capture ends.
    ///
    /// # Errors
    ///
    /// Returns an error if capture-retention setup or the capture check fails, or if `image`
    /// belongs to another CUDA context.
    pub fn from_owned(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        image: Image<T, L>,
    ) -> Result<Self> {
        stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(image.allocation()))?;
        stream_context.stream().ensure_not_capturing()?;
        if image.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        Ok(Self {
            stream_context,
            workspace,
            backing: ImageBacking::Owned(image),
        })
    }

    /// Borrows a tightly packed image with pixel dimensions `size` from `memory`.
    ///
    /// Does not copy pixels or wait for pending producers.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions, channel geometry, byte-size arithmetic, alignment,
    /// or buffer capacity are invalid, or if the buffer belongs to another CUDA context.
    pub fn from_buffer<B>(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        memory: &'a B,
        size: Size,
    ) -> Result<Self>
    where
        T: DeviceRepr,
        B: DeviceSlice<T> + ?Sized,
    {
        Self::from_view(
            stream_context,
            workspace,
            ImageView::from_buffer(memory, size)?,
        )
    }

    /// Borrows an image using `step` bytes between successive row starts.
    ///
    /// `size` counts pixels. Does not copy pixels or wait for pending producers.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, channel geometry, byte-size arithmetic, alignment, byte
    /// stride, or buffer capacity are invalid, or if the buffer belongs to another CUDA context.
    pub fn from_buffer_with_step<B>(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        memory: &'a B,
        size: Size,
        step: i32,
    ) -> Result<Self>
    where
        T: DeviceRepr,
        B: DeviceSlice<T> + ?Sized,
    {
        Self::from_view(
            stream_context,
            workspace,
            ImageView::from_buffer_with_step(memory, size, step)?,
        )
    }

    pub(crate) fn ensure_eager(&self) -> Result<()> {
        if let ImageBacking::Owned(image) = &self.backing {
            self.stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(image.allocation()))?;
        }
        self.stream_context.stream().ensure_not_capturing()?;
        if let Some(producer) = self.view()?.region().stream() {
            producer.ensure_not_capturing()?;
        }
        Ok(())
    }

    /// Returns the pixel dimensions of the current backing.
    #[must_use]
    pub const fn size(&self) -> Size {
        match &self.backing {
            ImageBacking::Borrowed(view) => view.size(),
            ImageBacking::Owned(image) => image.size(),
        }
    }

    /// Returns whether this pipeline owns its image storage.
    #[must_use]
    pub const fn is_owned(&self) -> bool {
        matches!(self.backing, ImageBacking::Owned(_))
    }

    pub(crate) fn view(&self) -> Result<ImageView<'_, T, L>> {
        match &self.backing {
            ImageBacking::Borrowed(view) => Ok(*view),
            ImageBacking::Owned(image) => image.view(),
        }
    }

    /// Consumes the pipeline and returns its borrowed or owned image storage.
    ///
    /// Does not copy pixels or synchronize pending producers.
    #[must_use]
    pub fn into_backing(self) -> ImageBacking<'a, T, L> {
        self.backing
    }
}

/// Write-only image storage that becomes a readable pipeline after complete pixel production.
#[derive(Debug)]
pub struct UninitImagePipeline<'a, T, L = C1> {
    stream_context: &'a StreamContext,
    workspace: &'a mut Workspace,
    image: Image<MaybeUninit<T>, L>,
}

impl<'a, T: Copy, L: ChannelLayout> UninitImagePipeline<'a, T, L> {
    /// Initializes every channel and completes the set before exposing readable pixels.
    ///
    /// Uses the selected fill implementation and returns an owned readable pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if capture-retention setup, capture checks, view construction, CUDA
    /// context validation, region preparation, producer waits, native filling, or stream
    /// synchronization fails.
    pub fn set(
        mut self,
        value: <ImagePipeline<'a, T, L> as fill::SetImage<T, L>>::Value,
    ) -> Result<ImagePipeline<'a, T, L>>
    where
        L: AllChannels,
        ImagePipeline<'a, T, L>: fill::SetImage<T, L>,
    {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(self.image.allocation()))?;
        self.stream_context.stream().ensure_not_capturing()?;
        <ImagePipeline<'a, T, L> as fill::SetImage<T, L>>::set_image(
            self.stream_context,
            value,
            &mut self.image.output()?,
        )?;
        // SAFETY: AllChannels excludes ignored alpha, and completed dispatch initialized every logical pixel.
        let image = unsafe { self.image.assume_init() };
        ImagePipeline::from_owned(self.stream_context, self.workspace, image)
    }

    /// Borrows the pending image as exclusive writable output without initializing its pixels.
    ///
    /// Does not submit device work or synchronize producers.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored image geometry, byte stride, alignment, or allocation span
    /// cannot form a valid output view.
    pub fn output(&mut self) -> Result<ImageOutputView<'_, T, L>> {
        self.image.output()
    }

    /// Consumes the pending pipeline and returns its uninitialized image allocation.
    ///
    /// Preserves the current initialization state without copying, synchronizing, or promoting
    /// samples to readable `T` values.
    #[must_use]
    pub fn into_image(self) -> Image<MaybeUninit<T>, L> {
        self.image
    }
}
