use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        scale::{ImageBacking, ImagePipeline},
        scale_to_u8_dispatch::ScaleToU8Image,
    },
};

mod integer_methods;

impl<L> ImagePipeline<'_, f32, L>
where
    L: ChannelLayout,
    Self: ScaleToU8Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the `f32` source range `min..=max` into `0..=255` in `u8` output.
    /// NPP saturates and clamps results to the destination range.
    /// `min` and `max` must be finite with `min < max`.
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
    /// fail. Source and destination dimensions must match. Nonfinite or unordered bounds
    /// are rejected.
    pub fn scale_to_u8_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ScaleToU8Image<L>>::scale_to_u8_image(
            stream_context,
            source,
            destination,
            min,
            max,
        )
    }
}

impl<'a, L> ImagePipeline<'a, f32, L>
where
    L: ChannelLayout,
    u8: SupportedImage<L>,
    Self: ScaleToU8Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the `f32` source range `min..=max` into `0..=255` in `u8` output.
    /// NPP saturates and clamps results to the destination range.
    /// `min` and `max` must be finite with `min < max`.
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
    /// fail. Source and destination dimensions must match. Nonfinite or unordered bounds
    /// are rejected.
    /// Output allocation or image view construction can also fail.
    pub fn scale_to_u8(self, min: f32, max: f32) -> Result<ImagePipeline<'a, u8, L>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ScaleToU8Image<L>>::scale_to_u8_image(
                self.stream_context,
                &source,
                &mut destination_view,
                min,
                max,
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
