use crate::{
    ImageViewMut,
    context::StreamContext,
    error::Result,
    types::{ConnectedRegion, ImageNormalization, Point},
};

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait FloodFillImage<T, L> {
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
