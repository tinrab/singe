use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    types::ComparisonOperation,
};

use super::super::{
    ImagePipeline,
    threshold_dispatch::{ThresholdImage, ValueThresholdImage},
};

#[path = "thresholding_fixed_into_methods.rs"]
mod fixed_into_methods;
#[path = "thresholding_fixed_value_into_methods.rs"]
mod fixed_value_into_methods;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: ThresholdImage<T>,
{
    pub fn threshold_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdImage<T>>::threshold_image(
            stream_context,
            source,
            destination,
            threshold,
            operation,
        )
    }

    pub fn threshold_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdImage<T>>::threshold_image_in_place(
            stream_context,
            image,
            threshold,
            operation,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: ValueThresholdImage<T>,
{
    pub fn threshold_value_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ValueThresholdImage<T>>::threshold_value_image(
            stream_context,
            source,
            destination,
            threshold,
            value,
            operation,
        )
    }

    pub fn threshold_value_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ValueThresholdImage<T>>::threshold_value_image_in_place(
            stream_context,
            image,
            threshold,
            value,
            operation,
        )
    }
}
