use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    types::ComparisonOperation,
};

#[path = "threshold_packed_value_traits.rs"]
mod packed_value_traits;

pub use packed_value_traits::{
    PackedFixedValueThresholdImage, PackedLessGreaterValueThresholdImage, PackedValueThresholdImage,
};

pub trait ValueThresholdImage<T> {
    fn threshold_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
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

pub trait FixedValueThresholdImage<T> {
    fn threshold_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
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
        destination: &mut ImageViewMut<'_, T, C1>,
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

pub trait LessGreaterValueThresholdImage<T> {
    fn threshold_less_greater_value_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
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
