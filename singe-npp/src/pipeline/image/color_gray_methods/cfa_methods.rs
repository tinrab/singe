use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{AC4, C1, C3, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        color::{CfaToRgbImage, CfaToRgbaImage},
        color_gray_methods::ImagePipeline,
    },
    types::{BayerGridPosition, Rectangle},
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
{
    /// Converts a one-channel Bayer CFA image to packed RGB.
    /// `source_roi` selects pixels in the full source image. `grid` specifies the Bayer
    /// registration at the ROI origin. Uses the native required undefined interpolation setting.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside
    /// the source without coordinate overflow, and match the destination dimensions.
    pub fn cfa_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageOutputView<'_, T, C3>,
        grid: BayerGridPosition,
    ) -> Result<()>
    where
        Self: CfaToRgbImage<T>,
    {
        <Self as CfaToRgbImage<T>>::cfa_to_rgb_image(
            stream_context,
            source,
            source_roi,
            destination,
            grid,
        )
    }

    /// Converts a one-channel Bayer CFA image to packed RGB with supplied alpha.
    /// `source_roi` selects pixels in the full source image. `grid` specifies the Bayer
    /// registration at the ROI origin. Uses the native required undefined interpolation setting.
    /// Writes `alpha` to every destination alpha channel, including the AC4 layout.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside
    /// the source without coordinate overflow, and match the destination dimensions.
    pub fn cfa_to_rgba_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageOutputView<'_, T, AC4>,
        grid: BayerGridPosition,
        alpha: T,
    ) -> Result<()>
    where
        Self: CfaToRgbaImage<T>,
    {
        <Self as CfaToRgbaImage<T>>::cfa_to_rgba_image(
            stream_context,
            source,
            source_roi,
            destination,
            grid,
            alpha,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy + SupportedImage<C3>,
    Self: CfaToRgbImage<T>,
{
    /// Converts a one-channel Bayer CFA image to packed RGB.
    /// `source_roi` selects pixels in the full source image. `grid` specifies the Bayer
    /// registration at the ROI origin. Uses the native required undefined interpolation setting.
    /// Allocates a new owned image with the source ROI dimensions.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside
    /// the source without coordinate overflow, and match the destination dimensions.
    /// Allocation or image view construction can also fail.
    pub fn cfa_to_rgb(
        self,
        source_roi: Rectangle,
        grid: BayerGridPosition,
    ) -> Result<ImagePipeline<'a, T, C3>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C3>::create_uninit(self.stream_context.cuda_context(), source_roi.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as CfaToRgbImage<T>>::cfa_to_rgb_image(
                self.stream_context,
                &source,
                source_roi,
                &mut destination_view,
                grid,
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

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy + SupportedImage<AC4>,
    Self: CfaToRgbaImage<T>,
{
    /// Converts a one-channel Bayer CFA image to packed RGB with supplied alpha.
    /// `source_roi` selects pixels in the full source image. `grid` specifies the Bayer
    /// registration at the ROI origin. Uses the native required undefined interpolation setting.
    /// Writes `alpha` to every destination alpha channel, including the AC4 layout.
    /// Allocates a new owned image with the source ROI dimensions.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside
    /// the source without coordinate overflow, and match the destination dimensions.
    /// Allocation or image view construction can also fail.
    pub fn cfa_to_rgba(
        self,
        source_roi: Rectangle,
        grid: BayerGridPosition,
        alpha: T,
    ) -> Result<ImagePipeline<'a, T, AC4>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, AC4>::create_uninit(self.stream_context.cuda_context(), source_roi.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as CfaToRgbaImage<T>>::cfa_to_rgba_image(
                self.stream_context,
                &source,
                source_roi,
                &mut destination_view,
                grid,
                alpha,
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
