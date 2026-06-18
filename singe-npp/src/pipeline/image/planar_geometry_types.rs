use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{C1, ImageView, PlanarImageView, PlanarImageViewMut},
    },
};

pub(in crate::pipeline::image) type PlanarResize<T, const C: usize> =
    for<'source, 'destination> fn(
        &StreamContext,
        &geometry::Resize,
        &PlanarImageView<'source, T, C>,
        &mut PlanarImageViewMut<'destination, T, C>,
    ) -> Result<()>;

pub(in crate::pipeline::image) type PlanarResizeSqrPixel<T, const C: usize> =
    for<'source, 'destination> fn(
        &StreamContext,
        &geometry::ResizeSqrPixel,
        &PlanarImageView<'source, T, C>,
        &mut PlanarImageViewMut<'destination, T, C>,
    ) -> Result<()>;

pub(in crate::pipeline::image) type PlanarWarpAffine<T, const C: usize> =
    for<'source, 'destination> fn(
        &StreamContext,
        &geometry::WarpAffine,
        &PlanarImageView<'source, T, C>,
        &mut PlanarImageViewMut<'destination, T, C>,
    ) -> Result<()>;

pub(in crate::pipeline::image) type PlanarWarpQuad<T, const C: usize> =
    for<'source, 'destination> fn(
        &StreamContext,
        &geometry::WarpQuad,
        &PlanarImageView<'source, T, C>,
        &mut PlanarImageViewMut<'destination, T, C>,
    ) -> Result<()>;

pub(in crate::pipeline::image) type PlanarWarpPerspective<T, const C: usize> =
    for<'source, 'destination> fn(
        &StreamContext,
        &geometry::WarpPerspective,
        &PlanarImageView<'source, T, C>,
        &mut PlanarImageViewMut<'destination, T, C>,
    ) -> Result<()>;

pub(in crate::pipeline::image) type PlanarRemap<T, M, const C: usize> =
    for<'source, 'x_map, 'y_map, 'destination> fn(
        &StreamContext,
        &geometry::Remap,
        &PlanarImageView<'source, T, C>,
        &ImageView<'x_map, M, C1>,
        &ImageView<'y_map, M, C1>,
        &mut PlanarImageViewMut<'destination, T, C>,
    ) -> Result<()>;
