use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{
        ImageBacking, ImagePipeline,
        filtering::{DirectionalBorderFilterImage, UnsharpBorderFilterImage},
    },
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: DirectionalBorderFilterImage<T, L>,
{
    /// Applies the native sharpening filter with border control.
    /// `source_offset` locates the output ROI in the full source. Allocates a new owned
    /// output spanning from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_sharpen_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_window = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_window.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_window.view())?;
            }
            <Self as DirectionalBorderFilterImage<T, L>>::filter_sharpen_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: UnsharpBorderFilterImage<T, L>,
{
    /// Sharpens the image with a Gaussian unsharp mask and border control.
    /// `radius` is the Gaussian radius in pixels, excluding the center; `sigma` is its
    /// standard deviation in pixels. Adds `weight * (source - blurred)` where the absolute
    /// difference is at least `threshold`. `weight` and `threshold` are forwarded unchecked.
    /// The native routine supports replicated borders; the wrapper forwards `border_type`
    /// without validating it. Queries and allocates temporary device scratch.
    /// `source_offset` locates the output ROI in the full source. Allocates a new owned
    /// output spanning from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// `radius` and `sigma` must compare greater than zero. The scratch-size query and device
    /// scratch allocation can also fail.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_unsharp_border(
        self,
        source_offset: Point,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_window = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_window.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_window.view())?;
            }
            <Self as UnsharpBorderFilterImage<T, L>>::filter_unsharp_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                radius,
                sigma,
                weight,
                threshold,
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
