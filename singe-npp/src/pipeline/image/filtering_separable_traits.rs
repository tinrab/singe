use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, Point},
};

pub trait IntegerSeparableFilterImage<T, L> {
    fn filter_column_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()>;

    fn filter_row_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()>;
}

pub trait FloatSeparableFilterImage<T, L> {
    fn filter_column32f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
    ) -> Result<()>;

    fn filter_row32f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
    ) -> Result<()>;
}

pub trait DoubleSeparableFilterImage<T, L> {
    fn filter_column64f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()>;

    fn filter_row64f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()>;
}

pub trait SumWindowFilterImage<T, L> {
    fn sum_window_column_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()>;

    fn sum_window_row_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()>;
}

pub trait IntegerSeparableBorderFilterImage<T, L> {
    fn filter_column_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_row_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait FloatSeparableBorderFilterImage<T, L> {
    fn filter_column_border32f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_row_border32f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait SumWindowBorderFilterImage<T, L> {
    fn sum_window_column_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;

    fn sum_window_row_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}
