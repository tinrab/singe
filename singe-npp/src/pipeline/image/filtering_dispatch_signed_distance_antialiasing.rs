use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_signed_distance_transform_pba_antialiasing_image {
    (
        $source_ty:ty,
        $distance_transform:path,
        $distance_transform_abs:path
    ) => {
        impl<'a> SignedDistanceTransformPbaAntialiasingImage<$source_ty>
            for ImagePipeline<'a, $source_ty, C1>
        {
            fn signed_distance_transform_pba_antialiasing_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination: &mut ImageViewMut<'_, f64, C1>,
            ) -> Result<()> {
                $distance_transform(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    None,
                    None,
                    None,
                    Some(destination),
                )
            }

            fn signed_distance_transform_abs_pba_antialiasing_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination: &mut ImageViewMut<'_, f64, C1>,
            ) -> Result<()> {
                $distance_transform_abs(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift,
                    subpixel_y_shift,
                    None,
                    None,
                    None,
                    Some(destination),
                )
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
