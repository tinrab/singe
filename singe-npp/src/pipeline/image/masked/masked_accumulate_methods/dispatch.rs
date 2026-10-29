use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, ImageView, ImageViewMut, MaskView},
    },
    pipeline::ImagePipeline,
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait MaskedAccumulateSquareImage<T> {
    fn accumulate_square_masked_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
        source_destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait MaskedAccumulateProductImage<T> {
    fn accumulate_product_masked_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, C1>,
        source2: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
        source_destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait MaskedAccumulateWeightedImage<T> {
    fn accumulate_weighted_masked_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
        source_destination: &mut ImageViewMut<'_, f32, C1>,
        alpha: f32,
    ) -> Result<()>;
}

macro_rules! impl_masked_accumulate_square_image {
    ($source_ty:ty, $accumulate:path) => {
        unsafe impl<'a> MaskedAccumulateSquareImage<$source_ty> for ImagePipeline<'a, f32, C1> {
            fn accumulate_square_masked_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                mask: &MaskView<'_>,
                source_destination: &mut ImageViewMut<'_, f32, C1>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $accumulate(stream_context, source, mask, source_destination)
                })
            }
        }
    };
}

macro_rules! impl_masked_accumulate_product_image {
    ($source_ty:ty, $accumulate:path) => {
        unsafe impl<'a> MaskedAccumulateProductImage<$source_ty> for ImagePipeline<'a, f32, C1> {
            fn accumulate_product_masked_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, $source_ty, C1>,
                source2: &ImageView<'_, $source_ty, C1>,
                mask: &MaskView<'_>,
                source_destination: &mut ImageViewMut<'_, f32, C1>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source1)?;
                stream_context.prepare_region(source2)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $accumulate(stream_context, source1, source2, mask, source_destination)
                })
            }
        }
    };
}

macro_rules! impl_masked_accumulate_weighted_image {
    ($source_ty:ty, $accumulate:path) => {
        unsafe impl<'a> MaskedAccumulateWeightedImage<$source_ty> for ImagePipeline<'a, f32, C1> {
            fn accumulate_weighted_masked_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                mask: &MaskView<'_>,
                source_destination: &mut ImageViewMut<'_, f32, C1>,
                alpha: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $accumulate(stream_context, source, mask, source_destination, alpha)
                })
            }
        }
    };
}

impl_masked_accumulate_square_image!(u8, arithmetic::accumulate_square_u8_to_f32_c1_masked);

impl_masked_accumulate_square_image!(u16, arithmetic::accumulate_square_u16_to_f32_c1_masked);

impl_masked_accumulate_square_image!(f32, arithmetic::accumulate_square_f32_c1_masked);

impl_masked_accumulate_product_image!(u8, arithmetic::accumulate_product_u8_to_f32_c1_masked);

impl_masked_accumulate_product_image!(u16, arithmetic::accumulate_product_u16_to_f32_c1_masked);

impl_masked_accumulate_product_image!(f32, arithmetic::accumulate_product_f32_c1_masked);

impl_masked_accumulate_weighted_image!(u8, arithmetic::accumulate_weighted_u8_to_f32_c1_masked);

impl_masked_accumulate_weighted_image!(u16, arithmetic::accumulate_weighted_u16_to_f32_c1_masked);

impl_masked_accumulate_weighted_image!(f32, arithmetic::accumulate_weighted_f32_c1_masked);
