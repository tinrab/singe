use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, filtering_traits::*};

#[path = "filtering_dispatch_distance_antialiasing.rs"]
mod antialiasing;
#[path = "filtering_dispatch_distance_f64.rs"]
mod f64_distance;

macro_rules! impl_distance_transform_pba_image {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $distance_transform:path,
        $distance_transform_abs:path
    ) => {
        impl<'a> DistanceTransformPbaImage<$source_ty, $destination_ty>
            for ImagePipeline<'a, $source_ty, C1>
        {
            fn distance_transform_pba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                minimum_site_value: $source_ty,
                maximum_site_value: $source_ty,
                destination: &mut ImageViewMut<'_, $destination_ty, C1>,
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

            fn distance_transform_abs_pba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                minimum_site_value: $source_ty,
                maximum_site_value: $source_ty,
                destination: &mut ImageViewMut<'_, $destination_ty, C1>,
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

impl_distance_transform_pba_image!(
    i8,
    u16,
    filtering::distance_transform_pba_i8_to_u16_c1,
    filtering::distance_transform_abs_pba_i8_to_u16_c1
);
impl_distance_transform_pba_image!(
    i8,
    f32,
    filtering::distance_transform_pba_i8_to_f32_c1,
    filtering::distance_transform_abs_pba_i8_to_f32_c1
);
impl_distance_transform_pba_image!(
    u8,
    u16,
    filtering::distance_transform_pba_u8_to_u16_c1,
    filtering::distance_transform_abs_pba_u8_to_u16_c1
);
impl_distance_transform_pba_image!(
    u8,
    f32,
    filtering::distance_transform_pba_u8_to_f32_c1,
    filtering::distance_transform_abs_pba_u8_to_f32_c1
);
impl_distance_transform_pba_image!(
    u16,
    u16,
    filtering::distance_transform_pba_u16_to_u16_c1,
    filtering::distance_transform_abs_pba_u16_to_u16_c1
);
impl_distance_transform_pba_image!(
    u16,
    f32,
    filtering::distance_transform_pba_u16_to_f32_c1,
    filtering::distance_transform_abs_pba_u16_to_f32_c1
);
impl_distance_transform_pba_image!(
    i16,
    u16,
    filtering::distance_transform_pba_i16_to_u16_c1,
    filtering::distance_transform_abs_pba_i16_to_u16_c1
);
impl_distance_transform_pba_image!(
    i16,
    f32,
    filtering::distance_transform_pba_i16_to_f32_c1,
    filtering::distance_transform_abs_pba_i16_to_f32_c1
);
