use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    types::{
        BorderType, DifferentialKernel, HistogramOfGradientsConfig, ImageNormalization, MaskSize,
        Point, Size,
    },
};

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait CannyBorderFilterImage<T, L> {
    fn filter_canny_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait HarrisCornersBorderFilterImage<T, L> {
    fn filter_harris_corners_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element ABI and complete borrowed work on every path.
/// Successful dispatch initializes every requested output element, with no implicit padding promotion.
pub unsafe trait GradientVectorBorderImage<T, L, G> {
    fn gradient_vector_prewitt_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination_x: &mut ImageOutputView<'_, G, C1>,
        destination_y: &mut ImageOutputView<'_, G, C1>,
        destination_magnitude: &mut ImageOutputView<'_, G, C1>,
        destination_angle: &mut ImageOutputView<'_, f32, C1>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;

    fn gradient_vector_scharr_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination_x: &mut ImageOutputView<'_, G, C1>,
        destination_y: &mut ImageOutputView<'_, G, C1>,
        destination_magnitude: &mut ImageOutputView<'_, G, C1>,
        destination_angle: &mut ImageOutputView<'_, f32, C1>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;

    fn gradient_vector_sobel_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination_x: &mut ImageOutputView<'_, G, C1>,
        destination_y: &mut ImageOutputView<'_, G, C1>,
        destination_magnitude: &mut ImageOutputView<'_, G, C1>,
        destination_angle: &mut ImageOutputView<'_, f32, C1>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element ABI and complete borrowed work on every path.
/// Successful dispatch initializes every requested output element, with no implicit padding promotion.
pub unsafe trait HistogramOfGradientsBorderImage<T, L> {
    fn histogram_of_gradients_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        locations: &[Point],
        descriptors: &mut DeviceOutputView<'_, f32>,
        roi: Size,
        config: HistogramOfGradientsConfig,
        scratch: &mut DeviceOutputView<'_, u8>,
        border_type: BorderType,
    ) -> Result<()>;
}
