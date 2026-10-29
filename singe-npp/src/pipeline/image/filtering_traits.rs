pub use crate::pipeline::image::{
    filtering_distance_traits::*, filtering_edge_traits::*, filtering_edge_typed_traits::*,
    filtering_feature_traits::*, filtering_flood_traits::*, filtering_kernel_traits::*,
    filtering_masked_traits::*, filtering_segmentation_traits::*, filtering_separable_traits::*,
};
use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageWindow},
    types::{BorderType, Point, Size},
};

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait BoxFilterImage<T, L> {
    fn filter_box_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait BoxBorderFilterImage<T, L> {
    fn filter_box_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait AdaptiveBoxThresholdBorderImage<T, L> {
    fn filter_threshold_adaptive_box_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        delta: f32,
        value_greater_than: T,
        value_less_or_equal: T,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait NeighborhoodFilterImage<T, L> {
    fn filter_max_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn filter_min_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait NeighborhoodBorderFilterImage<T, L> {
    fn filter_max_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_min_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait MedianFilterImage<T, L> {
    fn filter_median_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait MedianBorderFilterImage<T, L> {
    fn filter_median_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}
