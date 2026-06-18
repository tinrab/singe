use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::ImagePipeline;

use super::filtering_traits::*;

macro_rules! impl_integer_separable_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        impl<'a> IntegerSeparableFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_column_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
            ) -> Result<()> {
                $column(stream_context, source, destination, kernel, anchor, divisor)
            }

            fn filter_row_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
            ) -> Result<()> {
                $row(stream_context, source, destination, kernel, anchor, divisor)
            }
        }
    };
}

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

#[path = "filtering_dispatch_float_separable.rs"]
mod float_separable;
#[path = "filtering_dispatch_i16_float_separable.rs"]
mod i16_float_separable;
#[path = "filtering_dispatch_integer_separable.rs"]
mod integer_separable;

impl_float_separable_filter_image!(
    u8,
    C1,
    filtering::filter_column32f_u8_c1,
    filtering::filter_row32f_u8_c1
);
impl_float_separable_filter_image!(
    u8,
    C3,
    filtering::filter_column32f_u8_c3,
    filtering::filter_row32f_u8_c3
);
impl_float_separable_filter_image!(
    u8,
    C4,
    filtering::filter_column32f_u8_c4,
    filtering::filter_row32f_u8_c4
);
impl_float_separable_filter_image!(
    u8,
    AC4,
    filtering::filter_column32f_u8_ac4,
    filtering::filter_row32f_u8_ac4
);
impl_float_separable_filter_image!(
    u16,
    C1,
    filtering::filter_column32f_u16_c1,
    filtering::filter_row32f_u16_c1
);
impl_float_separable_filter_image!(
    u16,
    C3,
    filtering::filter_column32f_u16_c3,
    filtering::filter_row32f_u16_c3
);
impl_float_separable_filter_image!(
    u16,
    C4,
    filtering::filter_column32f_u16_c4,
    filtering::filter_row32f_u16_c4
);
impl_float_separable_filter_image!(
    u16,
    AC4,
    filtering::filter_column32f_u16_ac4,
    filtering::filter_row32f_u16_ac4
);
