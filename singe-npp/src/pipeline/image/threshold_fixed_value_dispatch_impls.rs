use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, threshold_dispatch::FixedValueThresholdImage};

macro_rules! impl_fixed_value_threshold_image {
    ($ty:ty, $greater:path, $greater_in_place:path, $less:path, $less_in_place:path) => {
        impl<'a> FixedValueThresholdImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn threshold_greater_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
            ) -> Result<()> {
                $greater(stream_context, source, destination, threshold, value)
            }

            fn threshold_greater_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
            ) -> Result<()> {
                $greater_in_place(stream_context, image, threshold, value)
            }

            fn threshold_less_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
            ) -> Result<()> {
                $less(stream_context, source, destination, threshold, value)
            }

            fn threshold_less_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
            ) -> Result<()> {
                $less_in_place(stream_context, image, threshold, value)
            }
        }
    };
}

#[path = "threshold_packed_fixed_value_dispatch_impls.rs"]
mod packed;

impl_fixed_value_threshold_image!(
    u8,
    npp_threshold::threshold_greater_value_u8_c1,
    npp_threshold::threshold_greater_value_u8_c1_in_place,
    npp_threshold::threshold_less_value_u8_c1,
    npp_threshold::threshold_less_value_u8_c1_in_place
);
impl_fixed_value_threshold_image!(
    u16,
    npp_threshold::threshold_greater_value_u16_c1,
    npp_threshold::threshold_greater_value_u16_c1_in_place,
    npp_threshold::threshold_less_value_u16_c1,
    npp_threshold::threshold_less_value_u16_c1_in_place
);
impl_fixed_value_threshold_image!(
    i16,
    npp_threshold::threshold_greater_value_i16_c1,
    npp_threshold::threshold_greater_value_i16_c1_in_place,
    npp_threshold::threshold_less_value_i16_c1,
    npp_threshold::threshold_less_value_i16_c1_in_place
);
impl_fixed_value_threshold_image!(
    f32,
    npp_threshold::threshold_greater_value_f32_c1,
    npp_threshold::threshold_greater_value_f32_c1_in_place,
    npp_threshold::threshold_less_value_f32_c1,
    npp_threshold::threshold_less_value_f32_c1_in_place
);
