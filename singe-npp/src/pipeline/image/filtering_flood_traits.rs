use crate::{
    context::StreamContext,
    error::Result,
    image::view::ImageViewMut,
    types::{ConnectedRegion, ImageNormalization, Point},
};

pub trait FloodFillImage<T, L> {
    type Value;

    fn flood_fill_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        seed: Point,
        new_value: Self::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;

    fn flood_fill_boundary_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        seed: Point,
        new_value: Self::Value,
        boundary_value: Self::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;

    fn flood_fill_range_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        seed: Point,
        min: Self::Value,
        max: Self::Value,
        new_value: Self::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;

    fn flood_fill_range_boundary_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        seed: Point,
        min: Self::Value,
        max: Self::Value,
        new_value: Self::Value,
        boundary_value: Self::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;

    fn flood_fill_gradient_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        seed: Point,
        min: Self::Value,
        max: Self::Value,
        new_value: Self::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;

    fn flood_fill_gradient_boundary_image(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        seed: Point,
        min: Self::Value,
        max: Self::Value,
        new_value: Self::Value,
        boundary_value: Self::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}
