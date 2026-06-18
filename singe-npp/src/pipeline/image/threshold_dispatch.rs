use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    types::ComparisonOperation,
};

#[path = "threshold_value_traits.rs"]
mod value_traits;

pub use value_traits::*;

pub trait ThresholdImage<T> {
    fn threshold_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        operation: ComparisonOperation,
    ) -> Result<()>;

    fn threshold_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

pub trait PackedThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>;

    fn threshold_channels_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>;
}

pub trait FixedThresholdImage<T> {
    fn threshold_greater_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()>;

    fn threshold_greater_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()>;

    fn threshold_less_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()>;

    fn threshold_less_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()>;
}

pub trait PackedFixedThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_greater_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
    ) -> Result<()>;

    fn threshold_channels_greater_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
    ) -> Result<()>;

    fn threshold_channels_less_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
    ) -> Result<()>;

    fn threshold_channels_less_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
    ) -> Result<()>;
}

pub trait FusedAbsoluteDifferenceThresholdImage<T, L> {
    fn fused_absolute_difference_threshold_greater_value_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()>;

    fn fused_absolute_difference_threshold_greater_value_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        threshold: T,
        value: T,
    ) -> Result<()>;
}
