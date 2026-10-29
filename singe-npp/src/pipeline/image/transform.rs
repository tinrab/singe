use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
    types::Axis,
};

mod mirror_dispatch;

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful out-of-place dispatch initializes every ROI channel except ignored AC4 alpha.
/// In-place dispatch preserves initialized values in all channels.
pub unsafe trait MirrorImage<T, L> {
    fn mirror_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        axis: Axis,
    ) -> Result<()>;

    fn mirror_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        axis: Axis,
    ) -> Result<()>;
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MirrorImage<T, L>,
{
    /// Mirrors image samples about the selected `axis`.
    ///
    /// Source and destination dimensions must match. For AC4, destination alpha remains
    /// unwritten.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Source and destination dimensions must match.
    pub fn mirror_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        axis: Axis,
    ) -> Result<()> {
        <Self as MirrorImage<T, L>>::mirror_image(stream_context, source, destination, axis)
    }

    /// Mirrors image samples about the selected `axis`.
    ///
    /// Updates the supplied initialized image in place, preserving ignored AC4 alpha.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    pub fn mirror_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        axis: Axis,
    ) -> Result<()> {
        <Self as MirrorImage<T, L>>::mirror_image_in_place(stream_context, image, axis)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MirrorImage<T, L>,
{
    /// Mirrors image samples about the selected `axis`.
    ///
    /// Reuses an owned image in place. For borrowed input, allocates a source-sized image,
    /// copying the source first for AC4 to preserve alpha before native mirroring.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Image view construction, output allocation, or copying source alpha can also fail.
    pub fn mirror(mut self, axis: Axis) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as MirrorImage<T, L>>::mirror_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    axis,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                <Self as MirrorImage<T, L>>::mirror_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    axis,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
