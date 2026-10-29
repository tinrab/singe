use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageWindow},
    types::{BorderType, MaskSize, Point},
};

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait TypedEdgeDirectionalFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_vertical_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_scharr_horizontal_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
    ) -> Result<()>;

    fn filter_scharr_vertical_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait TypedEdgeDirectionalBorderFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_vertical_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_scharr_horizontal_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_scharr_vertical_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>;
}
