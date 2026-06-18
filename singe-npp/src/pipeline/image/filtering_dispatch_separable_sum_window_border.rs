use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point},
};

use super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_sum_window_border_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        impl<'a> SumWindowBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn sum_window_column_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                mask_size: i32,
                anchor: i32,
                border_type: BorderType,
            ) -> Result<()> {
                $column(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    anchor,
                    border_type,
                )
            }

            fn sum_window_row_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                mask_size: i32,
                anchor: i32,
                border_type: BorderType,
            ) -> Result<()> {
                $row(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    anchor,
                    border_type,
                )
            }
        }
    };
}

impl_sum_window_border_filter_image!(
    u8,
    C1,
    filtering::sum_window_column_border_u8_c1,
    filtering::sum_window_row_border_u8_c1
);
impl_sum_window_border_filter_image!(
    u8,
    C3,
    filtering::sum_window_column_border_u8_c3,
    filtering::sum_window_row_border_u8_c3
);
impl_sum_window_border_filter_image!(
    u8,
    C4,
    filtering::sum_window_column_border_u8_c4,
    filtering::sum_window_row_border_u8_c4
);
impl_sum_window_border_filter_image!(
    u16,
    C1,
    filtering::sum_window_column_border_u16_c1,
    filtering::sum_window_row_border_u16_c1
);
impl_sum_window_border_filter_image!(
    u16,
    C3,
    filtering::sum_window_column_border_u16_c3,
    filtering::sum_window_row_border_u16_c3
);
impl_sum_window_border_filter_image!(
    u16,
    C4,
    filtering::sum_window_column_border_u16_c4,
    filtering::sum_window_row_border_u16_c4
);
impl_sum_window_border_filter_image!(
    i16,
    C1,
    filtering::sum_window_column_border_i16_c1,
    filtering::sum_window_row_border_i16_c1
);
impl_sum_window_border_filter_image!(
    i16,
    C3,
    filtering::sum_window_column_border_i16_c3,
    filtering::sum_window_row_border_i16_c3
);
impl_sum_window_border_filter_image!(
    i16,
    C4,
    filtering::sum_window_column_border_i16_c4,
    filtering::sum_window_row_border_i16_c4
);
