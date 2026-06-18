use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point},
};

use super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_integer_separable_border_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        impl<'a> IntegerSeparableBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_column_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
                border_type: BorderType,
            ) -> Result<()> {
                $column(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    kernel,
                    anchor,
                    divisor,
                    border_type,
                )
            }

            fn filter_row_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[i32],
                anchor: i32,
                divisor: i32,
                border_type: BorderType,
            ) -> Result<()> {
                $row(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    kernel,
                    anchor,
                    divisor,
                    border_type,
                )
            }
        }
    };
}

#[path = "filtering_dispatch_separable_float_border.rs"]
mod float_border;

impl_integer_separable_border_filter_image!(
    u8,
    C1,
    filtering::filter_column_border_u8_c1,
    filtering::filter_row_border_u8_c1
);
impl_integer_separable_border_filter_image!(
    u8,
    C3,
    filtering::filter_column_border_u8_c3,
    filtering::filter_row_border_u8_c3
);
impl_integer_separable_border_filter_image!(
    u8,
    C4,
    filtering::filter_column_border_u8_c4,
    filtering::filter_row_border_u8_c4
);
impl_integer_separable_border_filter_image!(
    u8,
    AC4,
    filtering::filter_column_border_u8_ac4,
    filtering::filter_row_border_u8_ac4
);
impl_integer_separable_border_filter_image!(
    u16,
    C1,
    filtering::filter_column_border_u16_c1,
    filtering::filter_row_border_u16_c1
);
impl_integer_separable_border_filter_image!(
    u16,
    C3,
    filtering::filter_column_border_u16_c3,
    filtering::filter_row_border_u16_c3
);
impl_integer_separable_border_filter_image!(
    u16,
    C4,
    filtering::filter_column_border_u16_c4,
    filtering::filter_row_border_u16_c4
);
impl_integer_separable_border_filter_image!(
    u16,
    AC4,
    filtering::filter_column_border_u16_ac4,
    filtering::filter_row_border_u16_ac4
);
impl_integer_separable_border_filter_image!(
    i16,
    C1,
    filtering::filter_column_border_i16_c1,
    filtering::filter_row_border_i16_c1
);
impl_integer_separable_border_filter_image!(
    i16,
    C3,
    filtering::filter_column_border_i16_c3,
    filtering::filter_row_border_i16_c3
);
impl_integer_separable_border_filter_image!(
    i16,
    C4,
    filtering::filter_column_border_i16_c4,
    filtering::filter_row_border_i16_c4
);
impl_integer_separable_border_filter_image!(
    i16,
    AC4,
    filtering::filter_column_border_i16_ac4,
    filtering::filter_row_border_i16_ac4
);
