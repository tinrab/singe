use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView, ImageViewMut},
    types::ComparisonOperation,
};

mod packed_value_traits;

pub use packed_value_traits::{
    PackedFixedValueThresholdImage, PackedLessGreaterValueThresholdImage, PackedValueThresholdImage,
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful out-of-place calls initialize every ROI channel except ignored AC4 alpha.
/// In-place calls preserve valid values in every channel.
pub unsafe trait ValueThresholdImage<T> {
    fn threshold_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<()>;

    fn threshold_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful out-of-place calls initialize every ROI channel except ignored AC4 alpha.
/// In-place calls preserve valid values in every channel.
pub unsafe trait FixedValueThresholdImage<T> {
    fn threshold_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()>;

    fn threshold_greater_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()>;

    fn threshold_less_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()>;

    fn threshold_less_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native ABI.
/// Successful out-of-place calls initialize every ROI channel except ignored AC4 alpha.
/// In-place calls preserve valid values in every channel.
pub unsafe trait LessGreaterValueThresholdImage<T> {
    fn threshold_less_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<()>;

    fn threshold_less_greater_value_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<()>;
}
