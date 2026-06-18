use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, filtering_traits::DistanceTransformPbaAntialiasingImage};

macro_rules! impl_distance_transform_pba_antialiasing_image {
    (
        $source_ty:ty,
        $distance_transform:path,
        $distance_transform_abs:path
    ) => {
        impl<'a> DistanceTransformPbaAntialiasingImage<$source_ty>
            for ImagePipeline<'a, $source_ty, C1>
        {
            fn distance_transform_pba_antialiasing_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                minimum_site_value: $source_ty,
                maximum_site_value: $source_ty,
                destination: &mut ImageViewMut<'_, f64, C1>,
            ) -> Result<()> {
                $distance_transform(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    None,
                    None,
                    None,
                    Some(destination),
                )
            }

            fn distance_transform_abs_pba_antialiasing_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                minimum_site_value: $source_ty,
                maximum_site_value: $source_ty,
                destination: &mut ImageViewMut<'_, f64, C1>,
            ) -> Result<()> {
                $distance_transform_abs(
                    stream_context,
                    source,
                    minimum_site_value,
                    maximum_site_value,
                    None,
                    None,
                    None,
                    Some(destination),
                )
            }
        }
    };
}

impl_distance_transform_pba_antialiasing_image!(
    u8,
    filtering::distance_transform_pba_u8_to_f64_c1_antialiasing,
    filtering::distance_transform_abs_pba_u8_to_f64_c1_antialiasing
);
impl_distance_transform_pba_antialiasing_image!(
    i8,
    filtering::distance_transform_pba_i8_to_f64_c1_antialiasing,
    filtering::distance_transform_abs_pba_i8_to_f64_c1_antialiasing
);
impl_distance_transform_pba_antialiasing_image!(
    u16,
    filtering::distance_transform_pba_u16_to_f64_c1_antialiasing,
    filtering::distance_transform_abs_pba_u16_to_f64_c1_antialiasing
);
impl_distance_transform_pba_antialiasing_image!(
    i16,
    filtering::distance_transform_pba_i16_to_f64_c1_antialiasing,
    filtering::distance_transform_abs_pba_i16_to_f64_c1_antialiasing
);
impl_distance_transform_pba_antialiasing_image!(
    f32,
    filtering::distance_transform_pba_f32_to_f64_c1_antialiasing,
    filtering::distance_transform_abs_pba_f32_to_f64_c1_antialiasing
);
impl_distance_transform_pba_antialiasing_image!(
    f64,
    filtering::distance_transform_pba_f64_c1_antialiasing,
    filtering::distance_transform_abs_pba_f64_c1_antialiasing
);
