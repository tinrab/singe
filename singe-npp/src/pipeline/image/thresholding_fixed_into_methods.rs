use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, threshold_dispatch::FixedThresholdImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: FixedThresholdImage<T>,
{
    pub fn threshold_greater_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_greater_image(
            stream_context,
            source,
            destination,
            threshold,
        )
    }

    pub fn threshold_greater_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_greater_image_in_place(
            stream_context,
            image,
            threshold,
        )
    }

    pub fn threshold_less_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_less_image(
            stream_context,
            source,
            destination,
            threshold,
        )
    }

    pub fn threshold_less_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_less_image_in_place(
            stream_context,
            image,
            threshold,
        )
    }
}
