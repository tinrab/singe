use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{CopyImage, ImageBacking, ImagePipeline},
};

mod ac4_swap_methods;
mod channels_dispatch;
mod copy_channel_methods;
mod set_channel_methods;
mod traits;

pub use ac4_swap_methods::SwapChannelsAc4Image;
pub use traits::{CopyChannelImage, ExtractChannelImage, InsertChannelImage, SetChannelImage};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ExtractChannelImage<T, L>,
{
    /// Extracts the zero-based source `channel` into a single-channel output image.
    /// Writes the supplied output image. All participating images must have matching
    /// pixel dimensions.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each channel index must be within its corresponding native packed layout.
    /// Participating image pixel dimensions must match.
    pub fn extract_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, C1>,
        channel: usize,
    ) -> Result<()> {
        <Self as ExtractChannelImage<T, L>>::extract_channel_image(
            stream_context,
            source,
            destination,
            channel,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy + SupportedImage<C1>,
    L: ChannelLayout,
    Self: ExtractChannelImage<T, L>,
{
    /// Extracts the zero-based source `channel` into a single-channel output image.
    /// Allocates a new owned C1 image with the same pixel dimensions.
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
    pub fn extract_channel(self, channel: usize) -> Result<ImagePipeline<'a, T, C1>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ExtractChannelImage<T, L>>::extract_channel_image(
                self.stream_context,
                &source,
                &mut destination_view,
                channel,
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: InsertChannelImage<T, L>,
{
    /// Inserts the single-channel `source` into the zero-based destination `channel`.
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
    pub fn insert_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, L>,
        channel: usize,
    ) -> Result<()> {
        <Self as InsertChannelImage<T, L>>::insert_channel_image(
            stream_context,
            source,
            destination,
            channel,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CopyImage<T, L> + InsertChannelImage<T, L>,
{
    /// Inserts the single-channel `source` into the zero-based destination `channel`.
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
    pub fn insert_channel(mut self, source: &ImageView<'_, T, C1>, channel: usize) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.output_view()?;
                <Self as InsertChannelImage<T, L>>::insert_channel_image(
                    self.stream_context,
                    source,
                    &mut image_view,
                    channel,
                )?;
            }
            ImageBacking::Borrowed(current_source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    current_source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    current_source,
                    &mut destination_view,
                )?;
                <Self as InsertChannelImage<T, L>>::insert_channel_image(
                    self.stream_context,
                    source,
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
