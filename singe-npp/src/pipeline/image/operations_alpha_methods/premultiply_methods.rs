use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{AC4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImageBacking, ImagePipeline, operation_traits::AlphaPremultiplyImage},
};

impl<T> ImagePipeline<'_, T, AC4>
where
    T: Copy,
    Self: AlphaPremultiplyImage<T>,
{
    /// Premultiplies packed color samples by their per-pixel alpha using native scaling.
    ///
    /// The provided implementations accept AC4 `u8` or `u16` images. Pixel alpha uses the
    /// range from zero to the maximum channel value of the sample type. Uses the native
    /// pixel-alpha premultiplication operation.
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
    pub fn alpha_premultiply_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageOutputView<'_, T, AC4>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image(
            stream_context,
            source,
            destination,
        )
    }

    /// Premultiplies packed color samples by their per-pixel alpha using native scaling.
    ///
    /// The provided implementations accept AC4 `u8` or `u16` images. Pixel alpha uses the
    /// range from zero to the maximum channel value of the sample type. Uses the native
    /// pixel-alpha premultiplication operation.
    /// Updates the supplied initialized image in place.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native alpha processing, or stream synchronization fail.
    pub fn alpha_premultiply_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()> {
        <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image_in_place(
            stream_context,
            source_destination,
        )
    }
}

impl<T> ImagePipeline<'_, T, AC4>
where
    T: Copy + SupportedImage<AC4>,
    Self: AlphaPremultiplyImage<T>,
{
    /// Premultiplies packed color samples by their per-pixel alpha using native scaling.
    ///
    /// The provided implementations accept AC4 `u8` or `u16` images. Pixel alpha uses the
    /// range from zero to the maximum channel value of the sample type. Uses the native
    /// pixel-alpha premultiplication operation.
    /// Reuses owned image storage in place. For borrowed input, allocates a new source-sized
    /// image before native dispatch.
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
    pub fn alpha_premultiply(mut self) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image_in_place(
                    self.stream_context,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, AC4>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as AlphaPremultiplyImage<T>>::alpha_premultiply_image(
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
