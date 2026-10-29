use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        channels::{CopyChannelImage, CopyImage, ImagePipeline},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CopyChannelImage<T, L>,
{
    /// Copies the zero-based `source_channel` into `destination_channel` for every pixel.
    /// Writes the supplied output image. All participating images must have matching
    /// pixel dimensions.
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
    /// Participating image pixel dimensions must match.
    pub fn copy_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_channel: usize,
        destination: &mut ImageOutputView<'_, T, L>,
        destination_channel: usize,
    ) -> Result<()> {
        <Self as CopyChannelImage<T, L>>::copy_channel_image(
            stream_context,
            source,
            source_channel,
            destination,
            destination_channel,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CopyImage<T, L> + CopyChannelImage<T, L>,
{
    /// Copies the zero-based `source_channel` into `destination_channel` for every pixel.
    /// Allocates and copies the current image, then replaces the selected destination channel
    /// from the original source channel. All other channels retain their source values.
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
    pub fn copy_channel(self, source_channel: usize, destination_channel: usize) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as CopyImage<T, L>>::copy(self.stream_context, &source, &mut destination_view)?;
            <Self as CopyChannelImage<T, L>>::copy_channel_image(
                self.stream_context,
                &source,
                source_channel,
                &mut destination_view,
                destination_channel,
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
