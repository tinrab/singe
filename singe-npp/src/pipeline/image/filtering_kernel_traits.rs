use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

pub trait IntegerKernelFilterImage<T, L> {
    fn filter_kernel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
    ) -> Result<()>;

    fn filter_kernel_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait FloatKernelFilterImage<T, L> {
    fn filter_kernel32f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn filter_kernel32f_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait DoubleKernelFilterImage<T, L> {
    fn filter_kernel64f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f64],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

pub trait TypedFloatKernelFilterImage<S, SL, D, DL> {
    fn filter_kernel32f_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, S, SL>,
        destination: &mut ImageViewMut<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn filter_kernel32f_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, S, SL>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}
