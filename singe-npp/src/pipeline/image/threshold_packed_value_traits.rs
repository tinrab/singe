use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::ComparisonOperation,
};

pub trait PackedValueThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>;

    fn threshold_channels_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
        operation: ComparisonOperation,
    ) -> Result<()>;
}

pub trait PackedFixedValueThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
    ) -> Result<()>;

    fn threshold_channels_greater_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
    ) -> Result<()>;

    fn threshold_channels_less_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
    ) -> Result<()>;

    fn threshold_channels_less_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        thresholds: [T; CHANNELS],
        values: [T; CHANNELS],
    ) -> Result<()>;
}

pub trait PackedLessGreaterValueThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_less_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        lower_thresholds: [T; CHANNELS],
        lower_values: [T; CHANNELS],
        upper_thresholds: [T; CHANNELS],
        upper_values: [T; CHANNELS],
    ) -> Result<()>;

    fn threshold_channels_less_greater_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        lower_thresholds: [T; CHANNELS],
        lower_values: [T; CHANNELS],
        upper_thresholds: [T; CHANNELS],
        upper_values: [T; CHANNELS],
    ) -> Result<()>;
}
