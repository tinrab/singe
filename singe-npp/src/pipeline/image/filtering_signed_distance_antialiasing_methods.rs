use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::super::{
    ImagePipeline, filtering_distance_traits::SignedDistanceTransformPbaAntialiasingImage,
};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn signed_distance_transform_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: SignedDistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as SignedDistanceTransformPbaAntialiasingImage<T>>::signed_distance_transform_pba_antialiasing_image(
            stream_context,
            source,
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            destination,
        )
    }

    pub fn signed_distance_transform_abs_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: SignedDistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as SignedDistanceTransformPbaAntialiasingImage<T>>::signed_distance_transform_abs_pba_antialiasing_image(
            stream_context,
            source,
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            destination,
        )
    }
}
