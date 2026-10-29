use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, exchange,
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, MaskView},
    },
    pipeline::image::masked::{ImageBacking, ImagePipeline},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait MaskedCopyImage<T, L> {
    fn copy_masked_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()>;
}

macro_rules! impl_masked_copy_image {
    ($ty:ty, $layout:ty, $copy:path) => {
        unsafe impl<'a> MaskedCopyImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy_masked_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &MaskView<'_>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(mask)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $copy(stream_context, source, destination, mask) })
            }
        }
    };
}

impl_masked_copy_image!(u8, C1, exchange::copy_masked_c1);

impl_masked_copy_image!(u8, C3, exchange::copy_masked_c3);

impl_masked_copy_image!(u8, C4, exchange::copy_masked_c4);

impl_masked_copy_image!(u8, AC4, exchange::copy_masked_ac4);

impl_masked_copy_image!(u16, C1, exchange::copy_masked_c1);

impl_masked_copy_image!(u16, C3, exchange::copy_masked_c3);

impl_masked_copy_image!(u16, C4, exchange::copy_masked_c4);

impl_masked_copy_image!(u16, AC4, exchange::copy_masked_ac4);

impl_masked_copy_image!(i16, C1, exchange::copy_masked_c1);

impl_masked_copy_image!(i16, C3, exchange::copy_masked_c3);

impl_masked_copy_image!(i16, C4, exchange::copy_masked_c4);

impl_masked_copy_image!(i16, AC4, exchange::copy_masked_ac4);

impl_masked_copy_image!(i32, C1, exchange::copy_masked_c1);

impl_masked_copy_image!(i32, C3, exchange::copy_masked_c3);

impl_masked_copy_image!(i32, C4, exchange::copy_masked_c4);

impl_masked_copy_image!(i32, AC4, exchange::copy_masked_ac4);

impl_masked_copy_image!(f32, C1, exchange::copy_masked_c1);

impl_masked_copy_image!(f32, C3, exchange::copy_masked_c3);

impl_masked_copy_image!(f32, C4, exchange::copy_masked_c4);

impl_masked_copy_image!(f32, AC4, exchange::copy_masked_ac4);

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedCopyImage<T, L>,
{
    /// Copies source pixels into the destination where the mask byte is nonzero.
    ///
    /// Source, destination, and mask pixel dimensions must match.
    /// Zero mask pixels and AC4 alpha remain unwritten; any initialized values there are
    /// preserved. Initialize those locations separately before reading a complete image.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail. Participating image and
    /// mask dimensions must match.
    pub fn copy_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()> {
        <Self as MaskedCopyImage<T, L>>::copy_masked_image(
            stream_context,
            source,
            destination,
            mask,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MaskedCopyImage<T, L>,
{
    /// Copies source pixels into the destination where the mask byte is nonzero.
    ///
    /// Source, destination, and mask pixel dimensions must match.
    /// Reuses an owned pipeline image in place. For borrowed input, copies the pipeline
    /// image into a new owned image first. Zero mask pixels and AC4 alpha preserve the
    /// pipeline image's previous values.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail. Participating image and
    /// mask dimensions must match.
    /// Output allocation, copying borrowed input, or image view construction can also fail.
    pub fn copy_masked_from(
        mut self,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.output_view()?;
                <Self as MaskedCopyImage<T, L>>::copy_masked_image(
                    self.stream_context,
                    source,
                    &mut image_view,
                    mask,
                )?;
            }
            ImageBacking::Borrowed(destination_source) => {
                let mut destination = Image::<T, L>::from_view(destination_source)?;
                let mut destination_view = destination.output_view()?;
                <Self as MaskedCopyImage<T, L>>::copy_masked_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    mask,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
