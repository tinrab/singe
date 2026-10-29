use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImageBacking, ImagePipeline, operation_traits::LogicalNotImage},
};

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    Self: LogicalNotImage<u8, L>,
{
    /// Complements every bit of each processed `u8` channel.
    ///
    /// Computes the bitwise complement of each sample.
    /// All participating images must have matching pixel dimensions. For AC4, writes only
    /// the color channels and leaves destination alpha untouched.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Participating image dimensions must match.
    pub fn logical_not_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
    ) -> Result<()> {
        <Self as LogicalNotImage<u8, L>>::logical_not_image(stream_context, source, destination)
    }

    /// Complements every bit of each processed `u8` channel.
    ///
    /// Computes the bitwise complement of each sample.
    /// Updates the image in place and preserves AC4 alpha.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    pub fn logical_not_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as LogicalNotImage<u8, L>>::logical_not_image_in_place(
            stream_context,
            source_destination,
        )
    }
}

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    u8: SupportedImage<L>,
    Self: LogicalNotImage<u8, L>,
{
    /// Complements every bit of each processed `u8` channel.
    ///
    /// Computes the bitwise complement of each sample.
    /// Reuses an owned pipeline image in place; for borrowed input, allocates a new
    /// source-sized image.
    /// For AC4, preserves source alpha, copying the source before out-of-place execution
    /// when necessary.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Image view construction, output allocation, or copying source alpha can also fail.
    pub fn logical_not(mut self) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as LogicalNotImage<u8, L>>::logical_not_image_in_place(
                    self.stream_context,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<u8, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                <Self as LogicalNotImage<u8, L>>::logical_not_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
