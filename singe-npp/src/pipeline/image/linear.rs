use singe_cuda::types::Complex32;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, linear,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
};

type ComplexMagnitudeImageOperation = fn(
    &StreamContext,
    &ImageView<'_, Complex32, C1>,
    &mut ImageOutputView<'_, f32, C1>,
) -> Result<()>;

/// # Safety
/// Complete borrowed native work on every path, preserve the native ABI, and initialize every output pixel on success.
pub unsafe trait MagnitudeImage {
    fn magnitude_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
    ) -> Result<()>;

    fn magnitude_squared_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
    ) -> Result<()>;
}

unsafe impl MagnitudeImage for ImagePipeline<'_, Complex32, C1> {
    fn magnitude_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { linear::magnitude(stream_context, source, destination) })
    }

    fn magnitude_squared_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            linear::magnitude_squared(stream_context, source, destination)
        })
    }
}

impl ImagePipeline<'_, Complex32, C1>
where
    Self: MagnitudeImage,
{
    /// Computes complex magnitudes as C1 `f32` samples.
    /// Each native result is the square root of the sum of the squared real and imaginary components.
    /// Writes every logical sample of the equally sized destination.
    /// Prepares image regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    pub fn magnitude_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
    ) -> Result<()> {
        <Self as MagnitudeImage>::magnitude_image(stream_context, source, destination)
    }

    /// Computes squared complex magnitudes as C1 `f32` samples.
    /// Each native result is the squared real component plus the squared imaginary component.
    /// Writes every logical sample of the equally sized destination.
    /// Prepares image regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    pub fn magnitude_squared_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, Complex32, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
    ) -> Result<()> {
        <Self as MagnitudeImage>::magnitude_squared_image(stream_context, source, destination)
    }
}

impl<'a> ImagePipeline<'a, Complex32, C1>
where
    f32: SupportedImage<C1>,
    Self: MagnitudeImage,
{
    /// Computes complex magnitudes as C1 `f32` samples.
    /// Each native result is the square root of the sum of the squared real and imaginary components.
    /// Always allocates a new owned C1 image with the source pixel dimensions.
    /// Prepares image regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Image allocation, view construction can also fail.
    pub fn magnitude(self) -> Result<ImagePipeline<'a, f32, C1>> {
        self.linear_to_f32(<Self as MagnitudeImage>::magnitude_image)
    }

    /// Computes squared complex magnitudes as C1 `f32` samples.
    /// Each native result is the squared real component plus the squared imaginary component.
    /// Always allocates a new owned C1 image with the source pixel dimensions.
    /// Prepares image regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Image allocation, view construction can also fail.
    pub fn magnitude_squared(self) -> Result<ImagePipeline<'a, f32, C1>> {
        self.linear_to_f32(<Self as MagnitudeImage>::magnitude_squared_image)
    }

    fn linear_to_f32(
        self,
        operation: ComplexMagnitudeImageOperation,
    ) -> Result<ImagePipeline<'a, f32, C1>> {
        self.ensure_eager()?;
        let mut destination =
            Image::<f32, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            operation(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
