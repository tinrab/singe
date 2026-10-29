use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{C1, ImageView, PlanarImageOutputView, PlanarImageView},
    },
};

pub type PlanarResize<T, const C: usize> = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &geometry::Resize,
    &PlanarImageView<'source, T, C>,
    &mut PlanarImageOutputView<'destination, T, C>,
) -> Result<()>;

pub type PlanarResizeSqrPixel<T, const C: usize> =
    for<'source, 'destination> unsafe fn(
        &StreamContext,
        &geometry::ResizeSqrPixel,
        &PlanarImageView<'source, T, C>,
        &mut PlanarImageOutputView<'destination, T, C>,
    ) -> Result<()>;

pub type PlanarWarpAffine<T, const C: usize> = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &geometry::WarpAffine,
    &PlanarImageView<'source, T, C>,
    &mut PlanarImageOutputView<'destination, T, C>,
) -> Result<()>;

pub type PlanarWarpQuad<T, const C: usize> = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &geometry::WarpQuad,
    &PlanarImageView<'source, T, C>,
    &mut PlanarImageOutputView<'destination, T, C>,
) -> Result<()>;

pub type PlanarWarpPerspective<T, const C: usize> =
    for<'source, 'destination> unsafe fn(
        &StreamContext,
        &geometry::WarpPerspective,
        &PlanarImageView<'source, T, C>,
        &mut PlanarImageOutputView<'destination, T, C>,
    ) -> Result<()>;

pub type PlanarRemap<T, M, const C: usize> =
    for<'source, 'x_map, 'y_map, 'destination> unsafe fn(
        &StreamContext,
        &geometry::Remap,
        &PlanarImageView<'source, T, C>,
        &ImageView<'x_map, M, C1>,
        &ImageView<'y_map, M, C1>,
        &mut PlanarImageOutputView<'destination, T, C>,
    ) -> Result<()>;
