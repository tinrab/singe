use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::{ImagePipeline, filtering_traits::SignedDistanceTransformPbaImage},
};

macro_rules! impl_signed_distance_transform_pba_image {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $shift_ty:ty,
        $distance_transform:path,
        $distance_transform_abs:path
    ) => {
        unsafe impl<'a> SignedDistanceTransformPbaImage<$source_ty, $destination_ty>
            for ImagePipeline<'a, $source_ty, C1>
        {
            #[allow(clippy::cast_possible_truncation, reason = "each dispatch specialization converts shifts to the native f32 or f64 precision")]
            fn signed_distance_transform_pba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination: &mut ImageOutputView<'_, $destination_ty, C1>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $distance_transform(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset as $shift_ty,
                        vertical_subpixel_offset as $shift_ty,
                        None,
                        None,
                        None,
                        Some(destination),
                    )
                }
            }

            #[allow(clippy::cast_possible_truncation, reason = "each dispatch specialization converts shifts to the native f32 or f64 precision")]
            fn signed_distance_transform_abs_pba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination: &mut ImageOutputView<'_, $destination_ty, C1>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $distance_transform_abs(
                        stream_context,
                        source,
                        cutoff_value,
                        horizontal_subpixel_offset as $shift_ty,
                        vertical_subpixel_offset as $shift_ty,
                        None,
                        None,
                        None,
                        Some(destination),
                    )
                }
            }
        }
    };
}

mod antialiasing;

impl_signed_distance_transform_pba_image!(
    f32,
    f32,
    f32,
    filtering::signed_distance_transform_pba_f32_c1,
    filtering::signed_distance_transform_abs_pba_f32_c1
);

impl_signed_distance_transform_pba_image!(
    f32,
    f64,
    f64,
    filtering::signed_distance_transform_pba_f32_to_f64_c1,
    filtering::signed_distance_transform_abs_pba_f32_to_f64_c1
);

impl_signed_distance_transform_pba_image!(
    f64,
    f64,
    f64,
    filtering::signed_distance_transform_pba_f64_c1,
    filtering::signed_distance_transform_abs_pba_f64_c1
);
