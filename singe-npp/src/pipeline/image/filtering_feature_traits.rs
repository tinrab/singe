use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    types::{
        BorderType, DifferentialKernel, HistogramOfGradientsConfig, ImageNormalization, MaskSize,
        Point, Size,
    },
};

pub trait CannyBorderFilterImage<T, L> {
    fn filter_canny_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait HarrisCornersBorderFilterImage<T, L> {
    fn filter_harris_corners_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait GradientVectorBorderImage<T, L, G> {
    fn gradient_vector_prewitt_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination_x: &mut ImageViewMut<'_, G, C1>,
        destination_y: &mut ImageViewMut<'_, G, C1>,
        destination_magnitude: &mut ImageViewMut<'_, G, C1>,
        destination_angle: &mut ImageViewMut<'_, f32, C1>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;

    fn gradient_vector_scharr_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination_x: &mut ImageViewMut<'_, G, C1>,
        destination_y: &mut ImageViewMut<'_, G, C1>,
        destination_magnitude: &mut ImageViewMut<'_, G, C1>,
        destination_angle: &mut ImageViewMut<'_, f32, C1>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;

    fn gradient_vector_sobel_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination_x: &mut ImageViewMut<'_, G, C1>,
        destination_y: &mut ImageViewMut<'_, G, C1>,
        destination_magnitude: &mut ImageViewMut<'_, G, C1>,
        destination_angle: &mut ImageViewMut<'_, f32, C1>,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait HistogramOfGradientsBorderImage<T, L> {
    fn histogram_of_gradients_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        locations: &[Point],
        descriptors: &mut DeviceMemory<f32>,
        roi: Size,
        config: HistogramOfGradientsConfig,
        scratch: &mut DeviceMemory<u8>,
        border_type: BorderType,
    ) -> Result<()>;
}
