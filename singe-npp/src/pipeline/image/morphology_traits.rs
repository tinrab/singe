use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

pub use self::composite_traits::{
    CompositeMorphologyBorderImage, CompositeMorphologyBorderScratchImage,
};

#[path = "morphology_composite_traits.rs"]
mod composite_traits;

pub(super) type GrayMaskMorphologyBorderOperation<T, M> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, C1>,
        Point,
        &mut ImageViewMut<'destination, T, C1>,
        &[M],
        Size,
        Point,
        BorderType,
    ) -> Result<()>;

pub(super) type MaskMorphologyBorderOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    Point,
    &mut ImageViewMut<'destination, T, L>,
    &[u8],
    Size,
    Point,
    BorderType,
) -> Result<()>;

pub trait MaskMorphologyImage<T, L> {
    fn dilate_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn erode_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

pub trait Morphology3x3Image<T, L> {
    fn dilate_3x3_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn erode_3x3_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait MaskMorphologyBorderImage<T, L> {
    fn dilate_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn erode_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait Morphology3x3BorderImage<T, L> {
    fn dilate_3x3_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>;

    fn erode_3x3_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait GrayMaskMorphologyBorderImage<T, M> {
    fn gray_dilate_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn gray_erode_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}
