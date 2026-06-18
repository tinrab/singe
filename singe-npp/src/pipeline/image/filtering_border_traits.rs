use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, MaskSize, Point, Size},
};

pub trait MaskedKernelBorderFilterImage<T, L> {
    fn filter_high_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_low_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_gauss_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_laplace_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait HighLowGaussBorderFilterImage<T, L> {
    fn filter_high_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_low_pass_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_gauss_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait UnsharpBorderFilterImage<T, L> {
    fn filter_unsharp_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait GaussPyramidBorderFilterImage<T, L> {
    fn filter_gauss_pyramid_layer_down_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_gauss_pyramid_layer_up_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait BilateralGaussBorderFilterImage<T, L> {
    fn filter_bilateral_gauss_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait WienerBorderFilterImage<T, L, const CHANNELS: usize> {
    fn filter_wiener_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; CHANNELS],
        border_type: BorderType,
    ) -> Result<()>;
}
