pub use self::composite_traits::{
    CompositeMorphologyBorderImage, CompositeMorphologyBorderScratchImage,
};
use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView, ImageWindow},
    types::{BorderType, Point, Size},
};

mod composite_traits;

pub type GrayMaskMorphologyBorderOperation<T, M> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, C1>,
    Point,
    &mut ImageOutputView<'destination, T, C1>,
    &[M],
    Size,
    Point,
    BorderType,
) -> Result<()>;

pub type MaskMorphologyBorderOperation<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    Point,
    &mut ImageOutputView<'destination, T, L>,
    &[u8],
    Size,
    Point,
    BorderType,
) -> Result<()>;

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait MaskMorphologyImage<T, L> {
    fn dilate_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn erode_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait Morphology3x3Image<T, L> {
    fn dilate_3x3_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn erode_3x3_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait MaskMorphologyBorderImage<T, L> {
    fn dilate_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn erode_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait Morphology3x3BorderImage<T, L> {
    fn dilate_3x3_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>;

    fn erode_3x3_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait GrayMaskMorphologyBorderImage<T, M> {
    fn gray_dilate_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn gray_erode_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}
