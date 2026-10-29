use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView},
    },
    pipeline::image::channels::{CopyImage, ImageBacking, ImagePipeline, SetChannelImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SetChannelImage<T, L>,
{
    /// Sets each sample of the zero-based destination `channel` to `value`.
    /// Writes the supplied output image.
    /// Only the selected destination channel is written. Other channels retain their previous
    /// initialization state; this operation does not initialize them.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each channel index must be within its corresponding native packed layout.
    pub fn set_channel_into(
        stream_context: &StreamContext,
        destination: &mut ImageOutputView<'_, T, L>,
        value: T,
        channel: usize,
    ) -> Result<()> {
        <Self as SetChannelImage<T, L>>::set_channel_image(
            stream_context,
            value,
            destination,
            channel,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CopyImage<T, L> + SetChannelImage<T, L>,
{
    /// Sets each sample of the zero-based destination `channel` to `value`.
    /// Reuses owned image storage, or copies borrowed backing into an owned image before
    /// updating the selected channel. All unselected channels retain their existing values.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each channel index must be within its corresponding native packed layout.
    /// Participating image pixel dimensions must match.
    /// Output allocation, image-view construction, or copying existing pixels can also fail.
    pub fn set_channel(mut self, value: T, channel: usize) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.output_view()?;
                <Self as SetChannelImage<T, L>>::set_channel_image(
                    self.stream_context,
                    value,
                    &mut image_view,
                    channel,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                <Self as SetChannelImage<T, L>>::set_channel_image(
                    self.stream_context,
                    value,
                    &mut destination_view,
                    channel,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
