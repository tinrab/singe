use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_signed_distance_transform_pba_image {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $shift_ty:ty,
        $distance_transform:path,
        $distance_transform_abs:path
    ) => {
        impl<'a> SignedDistanceTransformPbaImage<$source_ty, $destination_ty>
            for ImagePipeline<'a, $source_ty, C1>
        {
            fn signed_distance_transform_pba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination: &mut ImageViewMut<'_, $destination_ty, C1>,
            ) -> Result<()> {
                $distance_transform(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift as $shift_ty,
                    subpixel_y_shift as $shift_ty,
                    None,
                    None,
                    None,
                    Some(destination),
                )
            }

            fn signed_distance_transform_abs_pba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, C1>,
                cutoff_value: $source_ty,
                subpixel_x_shift: f64,
                subpixel_y_shift: f64,
                destination: &mut ImageViewMut<'_, $destination_ty, C1>,
            ) -> Result<()> {
                $distance_transform_abs(
                    stream_context,
                    source,
                    cutoff_value,
                    subpixel_x_shift as $shift_ty,
                    subpixel_y_shift as $shift_ty,
                    None,
                    None,
                    None,
                    Some(destination),
                )
            }
        }
    };
}

#[path = "filtering_dispatch_signed_distance_antialiasing.rs"]
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
