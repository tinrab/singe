use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        filtering_gaussian_methods::{BilateralGaussBorderFilterImage, ImagePipeline},
    },
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BilateralGaussBorderFilterImage<T, L>,
{
    /// Applies bilateral Gaussian filtering with replicated borders.
    /// `radius` gives the neighborhood radius; `step_between_source_pixels` gives the spacing
    /// between sampled source pixels. `value_square_sigma` and `position_square_sigma` are the
    /// squared sigmas for relative intensity and geometric distances respectively.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions set its extent.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// `radius` must be in `1..=32`, the sample step must be positive, both squared sigmas must
    /// compare greater than zero, and `border_type` must be `BorderType::Replicate`.
    pub fn filter_bilateral_gauss_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as BilateralGaussBorderFilterImage<T, L>>::filter_bilateral_gauss_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            radius,
            step_between_source_pixels,
            value_square_sigma,
            position_square_sigma,
            border_type,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: BilateralGaussBorderFilterImage<T, L>,
{
    /// Applies bilateral Gaussian filtering with replicated borders.
    /// `radius` gives the neighborhood radius; `step_between_source_pixels` gives the spacing
    /// between sampled source pixels. `value_square_sigma` and `position_square_sigma` are the
    /// squared sigmas for relative intensity and geometric distances respectively.
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned output from that offset to the lower-right source edge.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// `radius` must be in `1..=32`, the sample step must be positive, both squared sigmas must
    /// compare greater than zero, and `border_type` must be `BorderType::Replicate`.
    /// Image allocation or view construction can also fail.
    pub fn filter_bilateral_gauss_border(
        self,
        source_offset: Point,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_source = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_source.view())?;
            }
            <Self as BilateralGaussBorderFilterImage<T, L>>::filter_bilateral_gauss_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                radius,
                step_between_source_pixels,
                value_square_sigma,
                position_square_sigma,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
