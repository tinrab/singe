use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, scale::ImagePipeline, scale_dispatch::ScaleToF32Image},
};

mod scale_to_i16_methods;
mod scale_to_integer_methods;

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    Self: ScaleToF32Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full `u8` range `0..=255` into the `f32` destination range `min..=max`.
    /// NPP clamps results to that floating-point range.
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
    pub fn scale_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ScaleToF32Image<L>>::scale_to_f32_image(
            stream_context,
            source,
            destination,
            min,
            max,
        )
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    f32: SupportedImage<L>,
    Self: ScaleToF32Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full `u8` range `0..=255` into the `f32` destination range `min..=max`.
    /// NPP clamps results to that floating-point range.
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
    pub fn scale_to_f32(self, min: f32, max: f32) -> Result<ImagePipeline<'a, f32, L>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<f32, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ScaleToF32Image<L>>::scale_to_f32_image(
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
