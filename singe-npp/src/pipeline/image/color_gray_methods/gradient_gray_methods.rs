use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, C3, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, color::GradientColorToGrayImage, color_gray_methods::ImagePipeline,
    },
    types::ImageNormalization,
};

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: GradientColorToGrayImage<T>,
{
    /// Converts packed RGB to a one-channel gray gradient.
    /// `normalization` selects the native gradient distance method.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    pub fn gradient_color_to_gray_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageOutputView<'_, T, C1>,
        normalization: ImageNormalization,
    ) -> Result<()> {
        <Self as GradientColorToGrayImage<T>>::gradient_color_to_gray_image(
            stream_context,
            source,
            destination,
            normalization,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy + SupportedImage<C1>,
    Self: GradientColorToGrayImage<T>,
{
    /// Converts packed RGB to a one-channel gray gradient.
    /// `normalization` selects the native gradient distance method.
    /// Allocates a new owned C1 image with the source dimensions.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    /// Allocation or image view construction can also fail.
    pub fn gradient_color_to_gray(
        self,
        normalization: ImageNormalization,
    ) -> Result<ImagePipeline<'a, T, C1>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as GradientColorToGrayImage<T>>::gradient_color_to_gray_image(
                self.stream_context,
                &source,
                &mut destination_view,
                normalization,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
