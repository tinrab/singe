use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{
    ImagePipeline,
    filtering_traits::{DoubleSeparableFilterImage, FloatSeparableFilterImage},
};

macro_rules! impl_float_separable_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        impl<'a> FloatSeparableFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_column32f_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[f32],
                anchor: i32,
            ) -> Result<()> {
                $column(stream_context, source, destination, kernel, anchor)
            }

            fn filter_row32f_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[f32],
                anchor: i32,
            ) -> Result<()> {
                $row(stream_context, source, destination, kernel, anchor)
            }
        }
    };
}

macro_rules! impl_double_separable_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        impl<'a> DoubleSeparableFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_column64f_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[f64],
                anchor: i32,
            ) -> Result<()> {
                $column(stream_context, source, destination, kernel, anchor)
            }

            fn filter_row64f_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[f64],
                anchor: i32,
            ) -> Result<()> {
                $row(stream_context, source, destination, kernel, anchor)
            }
        }
    };
}

impl_double_separable_filter_image!(
    f64,
    C1,
    filtering::filter_column_f64_c1,
    filtering::filter_row_f64_c1
);
impl_float_separable_filter_image!(
    f32,
    C1,
    filtering::filter_column_f32_c1,
    filtering::filter_row_f32_c1
);
impl_float_separable_filter_image!(
    f32,
    C3,
    filtering::filter_column_f32_c3,
    filtering::filter_row_f32_c3
);
impl_float_separable_filter_image!(
    f32,
    C4,
    filtering::filter_column_f32_c4,
    filtering::filter_row_f32_c4
);
impl_float_separable_filter_image!(
    f32,
    AC4,
    filtering::filter_column_f32_ac4,
    filtering::filter_row_f32_ac4
);
