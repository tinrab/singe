use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        scale::{ImageBacking, ImagePipeline},
        scale_to_u8_dispatch::{ScaleI32ToU8Image, ScaleU16ToU8Image},
    },
    types::HintAlgorithm,
};

mod i16_methods;

impl<L> ImagePipeline<'_, i32, L>
where
    L: ChannelLayout,
    Self: ScaleI32ToU8Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full source integer numeric range into `0..=255` in `u8` output.
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
    pub fn scale_to_u8_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, i32, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()> {
        <Self as ScaleI32ToU8Image<L>>::scale_i32_to_u8_image(
            stream_context,
            source,
            destination,
            hint,
        )
    }
}

impl<'a, L> ImagePipeline<'a, i32, L>
where
    L: ChannelLayout,
    u8: SupportedImage<L>,
    Self: ScaleI32ToU8Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full source integer numeric range into `0..=255` in `u8` output.
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
    pub fn scale_to_u8(self, hint: HintAlgorithm) -> Result<ImagePipeline<'a, u8, L>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ScaleI32ToU8Image<L>>::scale_i32_to_u8_image(
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

impl<L> ImagePipeline<'_, u16, L>
where
    L: ChannelLayout,
    Self: ScaleU16ToU8Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full source integer numeric range into `0..=255` in `u8` output.
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
    pub fn scale_to_u8_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u16, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()> {
        <Self as ScaleU16ToU8Image<L>>::scale_u16_to_u8_image(
            stream_context,
            source,
            destination,
            hint,
        )
    }
}

impl<'a, L> ImagePipeline<'a, u16, L>
where
    L: ChannelLayout,
    u8: SupportedImage<L>,
    Self: ScaleU16ToU8Image<L>,
{
    /// Linearly scales image samples into the destination numeric range.
    ///
    /// Maps the full source integer numeric range into `0..=255` in `u8` output.
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
    pub fn scale_to_u8(self, hint: HintAlgorithm) -> Result<ImagePipeline<'a, u8, L>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ScaleU16ToU8Image<L>>::scale_u16_to_u8_image(
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
