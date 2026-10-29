use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView},
    types::{BorderType, MaskSize, Point, Size},
};

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait MaskedKernelBorderFilterImage<T, L> {
    fn filter_high_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_low_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_gauss_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_laplace_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait HighLowGaussBorderFilterImage<T, L> {
    fn filter_high_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_low_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_gauss_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native ABI and validate the full initialized source footprint.
/// Complete borrowed native work on every path.
/// Successful dispatch writes every active ROI channel with valid values; ignored alpha stays unchanged.
pub unsafe trait UnsharpBorderFilterImage<T, L> {
    fn filter_unsharp_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait GaussPyramidBorderFilterImage<T, L> {
    fn filter_gauss_pyramid_layer_down_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_gauss_pyramid_layer_up_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait BilateralGaussBorderFilterImage<T, L> {
    fn filter_bilateral_gauss_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait WienerBorderFilterImage<T, L, const CHANNELS: usize> {
    fn filter_wiener_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; CHANNELS],
        border_type: BorderType,
    ) -> Result<()>;
}
