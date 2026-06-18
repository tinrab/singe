use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, threshold_dispatch::FixedValueThresholdImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: FixedValueThresholdImage<T>,
{
    pub fn threshold_greater_value_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FixedValueThresholdImage<T>>::threshold_greater_value_image(
            stream_context,
            source,
            destination,
            threshold,
            value,
        )
    }

    pub fn threshold_greater_value_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FixedValueThresholdImage<T>>::threshold_greater_value_image_in_place(
            stream_context,
            image,
            threshold,
            value,
        )
    }

    pub fn threshold_less_value_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FixedValueThresholdImage<T>>::threshold_less_value_image(
            stream_context,
            source,
            destination,
            threshold,
            value,
        )
    }

    pub fn threshold_less_value_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
    ) -> Result<()> {
        <Self as FixedValueThresholdImage<T>>::threshold_less_value_image_in_place(
            stream_context,
            image,
            threshold,
            value,
        )
    }
}
