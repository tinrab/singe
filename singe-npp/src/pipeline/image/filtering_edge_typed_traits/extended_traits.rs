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
pub unsafe trait TypedSobelExtendedFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_second_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_vertical_second_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_cross_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_laplace_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample, including all declared destination channels.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait TypedSobelExtendedBorderFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_second_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_vertical_second_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_cross_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_laplace_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}
