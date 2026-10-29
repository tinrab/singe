use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        scale::{ImageBacking, ImagePipeline},
        scale_dispatch::ScaleToI16Image,
    },
};

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    Self: ScaleToI16Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// The provided implementations map the full `u8` range `0..=255` onto the full
    /// numeric range of `i16` output, including the negative half of signed
    /// destination ranges.
    /// Supports C1, C3, C4, and AC4 layouts. AC4 is dispatched as physical C4, so alpha
    /// is scaled along with the color channels and every logical output channel is written.
    /// Source and destination pixel dimensions must match.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful native execution and stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, physical-layout view construction, native scaling, or synchronization
    /// fail. Source and destination dimensions must match.
    pub fn scale_to_i16_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, i16, L>,
    ) -> Result<()> {
        <Self as ScaleToI16Image<L>>::scale_to_i16_image(stream_context, source, destination)
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    i16: SupportedImage<L>,
    Self: ScaleToI16Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// The provided implementations map the full `u8` range `0..=255` onto the full
    /// numeric range of `i16` output, including the negative half of signed
    /// destination ranges.
    /// Supports C1, C3, C4, and AC4 layouts. AC4 is dispatched as physical C4, so alpha
    /// is scaled along with the color channels and every logical output channel is written.
    /// Allocates a new source-sized, owned image with the same channel layout.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// returns after successful native execution and stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, physical-layout view construction, native scaling, or synchronization
    /// fail. Source and destination dimensions must match.
    /// Output allocation or image view construction can also fail.
    pub fn scale_to_i16(self) -> Result<ImagePipeline<'a, i16, L>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<i16, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ScaleToI16Image<L>>::scale_to_i16_image(
                self.stream_context,
                &source,
                &mut destination_view,
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
