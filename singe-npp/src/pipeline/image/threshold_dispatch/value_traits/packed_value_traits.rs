use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageViewMut},
    types::ComparisonOperation,
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful out-of-place calls initialize every ROI channel except ignored AC4 alpha.
/// In-place calls preserve valid values in every channel.
pub unsafe trait PackedValueThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
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

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful out-of-place calls initialize every ROI channel except ignored AC4 alpha.
/// In-place calls preserve valid values in every channel.
pub unsafe trait PackedFixedValueThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
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
        destination: &mut ImageOutputView<'_, T, L>,
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

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful out-of-place calls initialize every ROI channel except ignored AC4 alpha.
/// In-place calls preserve valid values in every channel.
pub unsafe trait PackedLessGreaterValueThresholdImage<T, L, const CHANNELS: usize> {
    fn threshold_channels_less_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
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
