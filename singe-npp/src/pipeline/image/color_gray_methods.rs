use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            color::{ColorToGrayImage, RgbToGrayImage},
        },
    },
};

mod cfa_methods;
mod gradient_gray_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: RgbToGrayImage<T, L>,
{
    /// Converts packed RGB to one-channel gray using the native fixed weights.
    /// Uses `0.299 * R + 0.587 * G + 0.114 * B` for gamma-corrected RGB.
    /// AC4 input alpha is ignored.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    pub fn rgb_to_gray_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, C1>,
    ) -> Result<()> {
        <Self as RgbToGrayImage<T, L>>::rgb_to_gray_image(stream_context, source, destination)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy + SupportedImage<C1>,
    L: ChannelLayout,
    Self: RgbToGrayImage<T, L>,
{
    /// Converts packed RGB to one-channel gray using the native fixed weights.
    /// Uses `0.299 * R + 0.587 * G + 0.114 * B` for gamma-corrected RGB.
    /// AC4 input alpha is ignored.
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
    pub fn rgb_to_gray(self) -> Result<ImagePipeline<'a, T, C1>> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as RgbToGrayImage<T, L>>::rgb_to_gray_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed dispatch initialized every output pixel, including preserved alpha where required.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Converts packed color to one-channel gray using `coefficients`.
    /// Passes the host coefficient array directly to NPP without validating its values.
    /// Each output gray sample is the sum of the processed channels multiplied by their weights.
    /// The provided C3 and AC4 implementations use three weights, ignoring AC4 input alpha;
    /// C4 uses four weights and includes its fourth channel.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    pub fn color_to_gray_into<const COEFFICIENTS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, C1>,
        coefficients: &[f32; COEFFICIENTS],
    ) -> Result<()>
    where
        Self: ColorToGrayImage<T, L, COEFFICIENTS>,
    {
        <Self as ColorToGrayImage<T, L, COEFFICIENTS>>::color_to_gray_image(
            stream_context,
            source,
            destination,
            coefficients,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy + SupportedImage<C1>,
    L: ChannelLayout,
{
    /// Converts packed color to one-channel gray using `coefficients`.
    /// Passes the host coefficient array directly to NPP without validating its values.
    /// Each output gray sample is the sum of the processed channels multiplied by their weights.
    /// The provided C3 and AC4 implementations use three weights, ignoring AC4 input alpha;
    /// C4 uses four weights and includes its fourth channel.
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
    pub fn color_to_gray<const COEFFICIENTS: usize>(
        self,
        coefficients: [f32; COEFFICIENTS],
    ) -> Result<ImagePipeline<'a, T, C1>>
    where
        Self: ColorToGrayImage<T, L, COEFFICIENTS>,
    {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ColorToGrayImage<T, L, COEFFICIENTS>>::color_to_gray_image(
                self.stream_context,
                &source,
                &mut destination_view,
                &coefficients,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed dispatch initialized every output pixel, including preserved alpha where required.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
