use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, exchange,
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, MaskView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait MaskedSetImage<T, L> {
    type Value;

    fn set_masked_image(
        stream_context: &StreamContext,
        value: Self::Value,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()>;
}

macro_rules! impl_masked_set_image {
    ($ty:ty, $layout:ty, $value_ty:ty, $set:path) => {
        unsafe impl<'a> MaskedSetImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Value = $value_ty;

            fn set_masked_image(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &MaskView<'_>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(mask)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $set(stream_context, value, destination, mask) })
            }
        }
    };
}

impl_masked_set_image!(u8, C1, u8, exchange::set_masked_c1);

impl_masked_set_image!(u8, C3, [u8; 3], exchange::set_masked_c3);

impl_masked_set_image!(u8, C4, [u8; 4], exchange::set_masked_c4);

impl_masked_set_image!(u8, AC4, [u8; 3], exchange::set_masked_ac4);

impl_masked_set_image!(u16, C1, u16, exchange::set_masked_c1);

impl_masked_set_image!(u16, C3, [u16; 3], exchange::set_masked_c3);

impl_masked_set_image!(u16, C4, [u16; 4], exchange::set_masked_c4);

impl_masked_set_image!(u16, AC4, [u16; 3], exchange::set_masked_ac4);

impl_masked_set_image!(i16, C1, i16, exchange::set_masked_c1);

impl_masked_set_image!(i16, C3, [i16; 3], exchange::set_masked_c3);

impl_masked_set_image!(i16, C4, [i16; 4], exchange::set_masked_c4);

impl_masked_set_image!(i16, AC4, [i16; 3], exchange::set_masked_ac4);

impl_masked_set_image!(i32, C1, i32, exchange::set_masked_c1);

impl_masked_set_image!(i32, C3, [i32; 3], exchange::set_masked_c3);

impl_masked_set_image!(i32, C4, [i32; 4], exchange::set_masked_c4);

impl_masked_set_image!(i32, AC4, [i32; 3], exchange::set_masked_ac4);

impl_masked_set_image!(f32, C1, f32, exchange::set_masked_c1);

impl_masked_set_image!(f32, C3, [f32; 3], exchange::set_masked_c3);

impl_masked_set_image!(f32, C4, [f32; 4], exchange::set_masked_c4);

impl_masked_set_image!(f32, AC4, [f32; 3], exchange::set_masked_ac4);

mod masked_copy_methods;

mod masked_accumulate_methods;
mod masked_accumulate_product_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedSetImage<T, L>,
    <Self as MaskedSetImage<T, L>>::Value: Copy,
{
    /// Sets destination pixels to `value` where the mask byte is nonzero.
    ///
    /// Destination and mask pixel dimensions must match. `value` is a scalar for C1 or
    /// a per-channel array for C3, C4, or AC4; AC4 uses three color values.
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
    pub fn set_masked_into(
        stream_context: &StreamContext,
        value: <Self as MaskedSetImage<T, L>>::Value,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()> {
        <Self as MaskedSetImage<T, L>>::set_masked_image(stream_context, value, destination, mask)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MaskedSetImage<T, L>,
    <Self as MaskedSetImage<T, L>>::Value: Copy,
{
    /// Sets destination pixels to `value` where the mask byte is nonzero.
    ///
    /// Destination and mask pixel dimensions must match. `value` is a scalar for C1 or
    /// a per-channel array for C3, C4, or AC4; AC4 uses three color values.
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
    pub fn set_masked(
        mut self,
        value: <Self as MaskedSetImage<T, L>>::Value,
        mask: &MaskView<'_>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.output_view()?;
                <Self as MaskedSetImage<T, L>>::set_masked_image(
                    self.stream_context,
                    value,
                    &mut image_view,
                    mask,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::from_view(source)?;
                let mut destination_view = destination.output_view()?;
                <Self as MaskedSetImage<T, L>>::set_masked_image(
                    self.stream_context,
                    value,
                    &mut destination_view,
                    mask,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
