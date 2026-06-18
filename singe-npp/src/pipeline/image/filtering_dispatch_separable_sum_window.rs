use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_sum_window_filter_image {
    ($ty:ty, $layout:ty, $column:path, $row:path) => {
        impl<'a> SumWindowFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn sum_window_column_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                mask_size: i32,
                anchor: i32,
            ) -> Result<()> {
                $column(stream_context, source, destination, mask_size, anchor)
            }

            fn sum_window_row_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                mask_size: i32,
                anchor: i32,
            ) -> Result<()> {
                $row(stream_context, source, destination, mask_size, anchor)
            }
        }
    };
}

impl_sum_window_filter_image!(
    u8,
    C1,
    filtering::sum_window_column_u8_c1,
    filtering::sum_window_row_u8_c1
);
impl_sum_window_filter_image!(
    u8,
    C3,
    filtering::sum_window_column_u8_c3,
    filtering::sum_window_row_u8_c3
);
impl_sum_window_filter_image!(
    u8,
    C4,
    filtering::sum_window_column_u8_c4,
    filtering::sum_window_row_u8_c4
);
impl_sum_window_filter_image!(
    u16,
    C1,
    filtering::sum_window_column_u16_c1,
    filtering::sum_window_row_u16_c1
);
impl_sum_window_filter_image!(
    u16,
    C3,
    filtering::sum_window_column_u16_c3,
    filtering::sum_window_row_u16_c3
);
impl_sum_window_filter_image!(
    u16,
    C4,
    filtering::sum_window_column_u16_c4,
    filtering::sum_window_row_u16_c4
);
impl_sum_window_filter_image!(
    i16,
    C1,
    filtering::sum_window_column_i16_c1,
    filtering::sum_window_row_i16_c1
);
impl_sum_window_filter_image!(
    i16,
    C3,
    filtering::sum_window_column_i16_c3,
    filtering::sum_window_row_i16_c3
);
impl_sum_window_filter_image!(
    i16,
    C4,
    filtering::sum_window_column_i16_c4,
    filtering::sum_window_row_i16_c4
);
