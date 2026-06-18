use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{C1, ImageView, ImageViewMut},
    },
    types::ComparisonOperation,
};

use super::{ImagePipeline, threshold_dispatch::ValueThresholdImage};

macro_rules! impl_value_threshold_image {
    ($ty:ty, $threshold:path, $threshold_in_place:path) => {
        impl<'a> ValueThresholdImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn threshold_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold(
                    stream_context,
                    source,
                    destination,
                    threshold,
                    value,
                    operation,
                )
            }

            fn threshold_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                value: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold_in_place(stream_context, image, threshold, value, operation)
            }
        }
    };
}

macro_rules! impl_packed_value_threshold_image {
    ($ty:ty, $layout:ty, $channels:literal, $threshold:path, $threshold_in_place:path) => {
        impl<'a> PackedValueThresholdImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn threshold_channels_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold(
                    stream_context,
                    source,
                    destination,
                    thresholds,
                    values,
                    operation,
                )
            }

            fn threshold_channels_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                values: [$ty; $channels],
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold_in_place(stream_context, image, thresholds, values, operation)
            }
        }
    };
}

#[path = "threshold_less_greater_value_dispatch_impls.rs"]
mod less_greater;
#[path = "threshold_packed_value_dispatch_impls.rs"]
mod packed_value;

impl_value_threshold_image!(
    u8,
    npp_threshold::threshold_value_u8_c1,
    npp_threshold::threshold_value_u8_c1_in_place
);
impl_value_threshold_image!(
    u16,
    npp_threshold::threshold_value_u16_c1,
    npp_threshold::threshold_value_u16_c1_in_place
);
impl_value_threshold_image!(
    i16,
    npp_threshold::threshold_value_i16_c1,
    npp_threshold::threshold_value_i16_c1_in_place
);
impl_value_threshold_image!(
    f32,
    npp_threshold::threshold_value_f32_c1,
    npp_threshold::threshold_value_f32_c1_in_place
);
