use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

pub trait DistanceTransformPbaImage<T, D> {
    fn distance_transform_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>;

    fn distance_transform_abs_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>;
}

pub trait DistanceTransformPbaAntialiasingImage<T> {
    fn distance_transform_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>;

    fn distance_transform_abs_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>;
}

pub trait SignedDistanceTransformPbaImage<T, D> {
    fn signed_distance_transform_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>;

    fn signed_distance_transform_abs_pba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>;
}

pub trait SignedDistanceTransformPbaAntialiasingImage<T> {
    fn signed_distance_transform_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>;

    fn signed_distance_transform_abs_pba_antialiasing_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        destination: &mut ImageViewMut<'_, f64, C1>,
    ) -> Result<()>;
}
