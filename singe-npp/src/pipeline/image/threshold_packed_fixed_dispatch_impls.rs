use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{AC4, C3, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, threshold_dispatch::PackedFixedThresholdImage};

macro_rules! impl_packed_fixed_threshold_image {
    (
        $ty:ty,
        $layout:ty,
        $channels:literal,
        $greater:path,
        $greater_in_place:path,
        $less:path,
        $less_in_place:path
    ) => {
        impl<'a> PackedFixedThresholdImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn threshold_channels_greater_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
            ) -> Result<()> {
                $greater(stream_context, source, destination, thresholds)
            }

            fn threshold_channels_greater_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
            ) -> Result<()> {
                $greater_in_place(stream_context, image, thresholds)
            }

            fn threshold_channels_less_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
            ) -> Result<()> {
                $less(stream_context, source, destination, thresholds)
            }

            fn threshold_channels_less_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
            ) -> Result<()> {
                $less_in_place(stream_context, image, thresholds)
            }
        }
    };
}

impl_packed_fixed_threshold_image!(
    u8,
    C3,
    3,
    npp_threshold::threshold_greater_u8_c3,
    npp_threshold::threshold_greater_u8_c3_in_place,
    npp_threshold::threshold_less_u8_c3,
    npp_threshold::threshold_less_u8_c3_in_place
);
impl_packed_fixed_threshold_image!(
    u8,
    AC4,
    3,
    npp_threshold::threshold_greater_u8_ac4,
    npp_threshold::threshold_greater_u8_ac4_in_place,
    npp_threshold::threshold_less_u8_ac4,
    npp_threshold::threshold_less_u8_ac4_in_place
);
impl_packed_fixed_threshold_image!(
    u16,
    C3,
    3,
    npp_threshold::threshold_greater_u16_c3,
    npp_threshold::threshold_greater_u16_c3_in_place,
    npp_threshold::threshold_less_u16_c3,
    npp_threshold::threshold_less_u16_c3_in_place
);
impl_packed_fixed_threshold_image!(
    u16,
    AC4,
    3,
    npp_threshold::threshold_greater_u16_ac4,
    npp_threshold::threshold_greater_u16_ac4_in_place,
    npp_threshold::threshold_less_u16_ac4,
    npp_threshold::threshold_less_u16_ac4_in_place
);
impl_packed_fixed_threshold_image!(
    i16,
    C3,
    3,
    npp_threshold::threshold_greater_i16_c3,
    npp_threshold::threshold_greater_i16_c3_in_place,
    npp_threshold::threshold_less_i16_c3,
    npp_threshold::threshold_less_i16_c3_in_place
);
impl_packed_fixed_threshold_image!(
    i16,
    AC4,
    3,
    npp_threshold::threshold_greater_i16_ac4,
    npp_threshold::threshold_greater_i16_ac4_in_place,
    npp_threshold::threshold_less_i16_ac4,
    npp_threshold::threshold_less_i16_ac4_in_place
);
impl_packed_fixed_threshold_image!(
    f32,
    C3,
    3,
    npp_threshold::threshold_greater_f32_c3,
    npp_threshold::threshold_greater_f32_c3_in_place,
    npp_threshold::threshold_less_f32_c3,
    npp_threshold::threshold_less_f32_c3_in_place
);
impl_packed_fixed_threshold_image!(
    f32,
    AC4,
    3,
    npp_threshold::threshold_greater_f32_ac4,
    npp_threshold::threshold_greater_f32_ac4_in_place,
    npp_threshold::threshold_less_f32_ac4,
    npp_threshold::threshold_less_f32_ac4_in_place
);
