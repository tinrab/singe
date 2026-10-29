use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline, scale_dispatch::ScaleRangeImage},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// The provided implementations map `u8` values from `0..=255` into `min..=max` in
    /// `f32` output, or map `f32` values from `min..=max` into `0..=255` in `u8` output.
    /// NPP clamps results to the destination range.
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
    pub fn scale_range_to_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, U, L>,
        min: f32,
        max: f32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaleRangeImage<T, U, L>,
    {
        <Self as ScaleRangeImage<T, U, L>>::scale_range_image(
            stream_context,
            source,
            destination,
            min,
            max,
        )
    }

    /// Linearly scales image samples into the destination numeric range.
    ///
    /// The provided implementations map `u8` values from `0..=255` into `min..=max` in
    /// `f32` output, or map `f32` values from `min..=max` into `0..=255` in `u8` output.
    /// NPP clamps results to the destination range.
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
    pub fn scale_range_to<U>(self, min: f32, max: f32) -> Result<ImagePipeline<'a, U, L>>
    where
        U: Copy + SupportedImage<L>,
        Self: ScaleRangeImage<T, U, L>,
    {
        self.ensure_eager()?;

        let mut destination =
            Image::<U, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ScaleRangeImage<T, U, L>>::scale_range_image(
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
