use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{AC4, C3, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, threshold_dispatch::PackedFixedValueThresholdImage};

macro_rules! impl_packed_fixed_value_threshold_image {
    (
        $ty:ty,
        $layout:ty,
        $channels:literal,
        $greater:path,
        $greater_in_place:path,
        $less:path,
        $less_in_place:path
    ) => {
        impl<'a> PackedFixedValueThresholdImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn threshold_channels_greater_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
            ) -> Result<()> {
                $greater(stream_context, source, destination, thresholds, values)
            }

            fn threshold_channels_greater_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
            ) -> Result<()> {
                $greater_in_place(stream_context, image, thresholds, values)
            }

            fn threshold_channels_less_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
            ) -> Result<()> {
                $less(stream_context, source, destination, thresholds, values)
            }

            fn threshold_channels_less_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
            ) -> Result<()> {
                $less_in_place(stream_context, image, thresholds, values)
            }
        }
    };
}

impl_packed_fixed_value_threshold_image!(
    u8,
    C3,
    3,
    npp_threshold::threshold_greater_value_u8_c3,
    npp_threshold::threshold_greater_value_u8_c3_in_place,
    npp_threshold::threshold_less_value_u8_c3,
    npp_threshold::threshold_less_value_u8_c3_in_place
);
impl_packed_fixed_value_threshold_image!(
    u8,
    AC4,
    3,
    npp_threshold::threshold_greater_value_u8_ac4,
    npp_threshold::threshold_greater_value_u8_ac4_in_place,
    npp_threshold::threshold_less_value_u8_ac4,
    npp_threshold::threshold_less_value_u8_ac4_in_place
);
impl_packed_fixed_value_threshold_image!(
    u16,
    C3,
    3,
    npp_threshold::threshold_greater_value_u16_c3,
    npp_threshold::threshold_greater_value_u16_c3_in_place,
    npp_threshold::threshold_less_value_u16_c3,
    npp_threshold::threshold_less_value_u16_c3_in_place
);
impl_packed_fixed_value_threshold_image!(
    u16,
    AC4,
    3,
    npp_threshold::threshold_greater_value_u16_ac4,
    npp_threshold::threshold_greater_value_u16_ac4_in_place,
    npp_threshold::threshold_less_value_u16_ac4,
    npp_threshold::threshold_less_value_u16_ac4_in_place
);
impl_packed_fixed_value_threshold_image!(
    i16,
    C3,
    3,
    npp_threshold::threshold_greater_value_i16_c3,
    npp_threshold::threshold_greater_value_i16_c3_in_place,
    npp_threshold::threshold_less_value_i16_c3,
    npp_threshold::threshold_less_value_i16_c3_in_place
);
impl_packed_fixed_value_threshold_image!(
    i16,
    AC4,
    3,
    npp_threshold::threshold_greater_value_i16_ac4,
    npp_threshold::threshold_greater_value_i16_ac4_in_place,
    npp_threshold::threshold_less_value_i16_ac4,
    npp_threshold::threshold_less_value_i16_ac4_in_place
);
impl_packed_fixed_value_threshold_image!(
    f32,
    C3,
    3,
    npp_threshold::threshold_greater_value_f32_c3,
    npp_threshold::threshold_greater_value_f32_c3_in_place,
    npp_threshold::threshold_less_value_f32_c3,
    npp_threshold::threshold_less_value_f32_c3_in_place
);
impl_packed_fixed_value_threshold_image!(
    f32,
    AC4,
    3,
    npp_threshold::threshold_greater_value_f32_ac4,
    npp_threshold::threshold_greater_value_f32_ac4_in_place,
    npp_threshold::threshold_less_value_f32_ac4,
    npp_threshold::threshold_less_value_f32_ac4_in_place
);
