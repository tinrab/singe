use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{C1, ImageView, ImageViewMut},
    },
    types::ComparisonOperation,
};

use super::{ImagePipeline, threshold_dispatch::ThresholdImage};

macro_rules! impl_threshold_image {
    ($ty:ty, $threshold:path, $threshold_in_place:path) => {
        impl<'a> ThresholdImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn threshold_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold(stream_context, source, destination, threshold, operation)
            }

            fn threshold_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                threshold: $ty,
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold_in_place(stream_context, image, threshold, operation)
            }
        }
    };
}

macro_rules! impl_packed_threshold_image {
    ($ty:ty, $layout:ty, $channels:literal, $threshold:path, $threshold_in_place:path) => {
        impl<'a> PackedThresholdImage<$ty, $layout, $channels> for ImagePipeline<'a, $ty, $layout> {
            fn threshold_channels_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold(stream_context, source, destination, thresholds, operation)
            }

            fn threshold_channels_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                thresholds: [$ty; $channels],
                operation: ComparisonOperation,
            ) -> Result<()> {
                $threshold_in_place(stream_context, image, thresholds, operation)
            }
        }
    };
}

#[path = "threshold_packed_dispatch_impls.rs"]
mod packed_impls;

impl_threshold_image!(
    u8,
    npp_threshold::threshold_u8_c1,
    npp_threshold::threshold_u8_c1_in_place
);
impl_threshold_image!(
    u16,
    npp_threshold::threshold_u16_c1,
    npp_threshold::threshold_u16_c1_in_place
);
impl_threshold_image!(
    i16,
    npp_threshold::threshold_i16_c1,
    npp_threshold::threshold_i16_c1_in_place
);
impl_threshold_image!(
    f32,
    npp_threshold::threshold_f32_c1,
    npp_threshold::threshold_f32_c1_in_place
);
