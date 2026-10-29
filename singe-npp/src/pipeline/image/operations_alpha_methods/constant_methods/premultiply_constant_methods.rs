use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::operations_alpha_methods::{
        AlphaPremultiplyConstantImage, ImageBacking, ImagePipeline,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AlphaPremultiplyConstantImage<T, L>,
    <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha: Copy,
{
    /// Premultiplies image samples by the constant opacity `alpha` using native scaling.
    ///
    /// The provided implementations accept `u8` or `u16` samples in C1, C3, C4, or AC4
    /// layouts. Opacity ranges from zero to the maximum channel value of the sample type.
    /// Uses the corresponding native constant-alpha premultiplication operation.
    /// Source and destination pixel dimensions must match.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native alpha processing, or stream synchronization fail.
    /// Participating image dimensions must match.
    pub fn alpha_premultiply_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        alpha: <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyConstantImage<T, L>>::alpha_premultiply_constant_image(
            stream_context,
            source,
            alpha,
            destination,
        )
    }

    /// Premultiplies image samples by the constant opacity `alpha` using native scaling.
    ///
    /// The provided implementations accept `u8` or `u16` samples in C1, C3, C4, or AC4
    /// layouts. Opacity ranges from zero to the maximum channel value of the sample type.
    /// Uses the corresponding native constant-alpha premultiplication operation.
    /// Updates the supplied initialized image in place.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native alpha processing, or stream synchronization fail.
    pub fn alpha_premultiply_constant_in_place(
        stream_context: &StreamContext,
        alpha: <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyConstantImage<T, L>>::alpha_premultiply_constant_image_in_place(
            stream_context,
            alpha,
            source_destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: AlphaPremultiplyConstantImage<T, L>,
    <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha: Copy,
{
    /// Premultiplies image samples by the constant opacity `alpha` using native scaling.
    ///
    /// The provided implementations accept `u8` or `u16` samples in C1, C3, C4, or AC4
    /// layouts. Opacity ranges from zero to the maximum channel value of the sample type.
    /// Uses the corresponding native constant-alpha premultiplication operation.
    /// Reuses owned image storage in place. For borrowed input, allocates a new source-sized
    /// image, copying the source first for AC4 before native dispatch.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native alpha processing, or stream synchronization fail.
    /// Participating image dimensions must match.
    /// Output allocation, image view construction, or copying the source can also fail.
    pub fn alpha_premultiply_constant(
        mut self,
        alpha: <Self as AlphaPremultiplyConstantImage<T, L>>::Alpha,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as AlphaPremultiplyConstantImage<
                    T,
                    L,
                >>::alpha_premultiply_constant_image_in_place(
                    self.stream_context,
                    alpha,
                    &mut image_view,
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
                <Self as AlphaPremultiplyConstantImage<T, L>>::alpha_premultiply_constant_image(
                    self.stream_context,
                    source,
                    alpha,
                    &mut destination_view,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
