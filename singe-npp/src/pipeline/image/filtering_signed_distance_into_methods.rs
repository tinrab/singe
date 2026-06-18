use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, filtering_distance_traits::SignedDistanceTransformPbaImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn signed_distance_transform_pba_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SignedDistanceTransformPbaImage<T, D>,
    {
        <Self as SignedDistanceTransformPbaImage<T, D>>::signed_distance_transform_pba_image(
            stream_context,
            source,
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            destination,
        )
    }

    pub fn signed_distance_transform_abs_pba_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>
    where
        D: Copy,
        Self: SignedDistanceTransformPbaImage<T, D>,
    {
        <Self as SignedDistanceTransformPbaImage<T, D>>::signed_distance_transform_abs_pba_image(
            stream_context,
            source,
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            destination,
        )
    }
}
