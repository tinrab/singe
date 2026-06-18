use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::super::super::{ImagePipeline, filtering_traits::DistanceTransformPbaAntialiasingImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn distance_transform_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: DistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as DistanceTransformPbaAntialiasingImage<
            T,
        >>::distance_transform_pba_antialiasing_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }

    pub fn distance_transform_abs_pba_antialiasing_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>
    where
        Self: DistanceTransformPbaAntialiasingImage<T>,
    {
        <Self as DistanceTransformPbaAntialiasingImage<
            T,
        >>::distance_transform_abs_pba_antialiasing_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }
}
