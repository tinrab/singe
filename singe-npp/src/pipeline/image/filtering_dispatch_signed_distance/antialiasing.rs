use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImagePipeline, filtering_traits::SignedDistanceTransformPbaAntialiasingImage,
    },
};

macro_rules! impl_signed_distance_transform_pba_antialiasing_image {
    (
        $source_ty:ty,
        $distance_transform:path,
        $distance_transform_abs:path
    ) => {
        unsafe impl<'a> SignedDistanceTransformPbaAntialiasingImage<$source_ty>
            for ImagePipeline<'a, $source_ty, C1>
        {
            fn signed_distance_transform_pba_antialiasing_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination: &mut ImageOutputView<'_, f64, C1>,
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
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
                        None,
                        None,
                        None,
                        Some(destination),
                    )
                }
            }

            fn signed_distance_transform_abs_pba_antialiasing_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                horizontal_subpixel_offset: f64,
                vertical_subpixel_offset: f64,
                destination: &mut ImageOutputView<'_, f64, C1>,
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
                        horizontal_subpixel_offset,
                        vertical_subpixel_offset,
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

impl_signed_distance_transform_pba_antialiasing_image!(
    f32,
    filtering::signed_distance_transform_pba_f32_to_f64_c1_antialiasing,
    filtering::signed_distance_transform_abs_pba_f32_to_f64_c1_antialiasing
);

impl_signed_distance_transform_pba_antialiasing_image!(
    f64,
    filtering::signed_distance_transform_pba_f64_c1_antialiasing,
    filtering::signed_distance_transform_abs_pba_f64_c1_antialiasing
);
