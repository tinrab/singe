use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline, scale_dispatch::ScaleHintImage},
    types::HintAlgorithm,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full source integer numeric range into `0..=255` in `u8` output.
    /// The provided implementations accept `i16`, `u16`, or `i32` sources.
    /// Forwards `hint` as the performance or accuracy preference; NPP documents this
    /// selector as currently ignored.
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
    pub fn scale_hint_to_into<U>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, U, L>,
        hint: HintAlgorithm,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaleHintImage<T, U, L>,
    {
        <Self as ScaleHintImage<T, U, L>>::scale_hint_image(
            stream_context,
            source,
            destination,
            hint,
        )
    }

    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full source integer numeric range into `0..=255` in `u8` output.
    /// The provided implementations accept `i16`, `u16`, or `i32` sources.
    /// Forwards `hint` as the performance or accuracy preference; NPP documents this
    /// selector as currently ignored.
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
    pub fn scale_hint_to<U>(self, hint: HintAlgorithm) -> Result<ImagePipeline<'a, U, L>>
    where
        U: Copy + SupportedImage<L>,
        Self: ScaleHintImage<T, U, L>,
    {
        self.ensure_eager()?;

        let mut destination =
            Image::<U, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ScaleHintImage<T, U, L>>::scale_hint_image(
                self.stream_context,
                &source,
                &mut destination_view,
                hint,
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
