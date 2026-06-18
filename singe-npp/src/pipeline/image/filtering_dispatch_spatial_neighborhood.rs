use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{Point, Size},
};

use super::super::{ImagePipeline, filtering_traits::NeighborhoodFilterImage};

macro_rules! impl_neighborhood_filter_image {
    ($ty:ty, $layout:ty, $filter_max:path, $filter_min:path) => {
        impl<'a> NeighborhoodFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_max_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                $filter_max(stream_context, source, destination, mask_size, anchor)
            }

            fn filter_min_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                $filter_min(stream_context, source, destination, mask_size, anchor)
            }
        }
    };
}

impl_neighborhood_filter_image!(
    u8,
    C1,
    filtering::filter_max_u8_c1,
    filtering::filter_min_u8_c1
);
impl_neighborhood_filter_image!(
    u8,
    C3,
    filtering::filter_max_u8_c3,
    filtering::filter_min_u8_c3
);
impl_neighborhood_filter_image!(
    u8,
    C4,
    filtering::filter_max_u8_c4,
    filtering::filter_min_u8_c4
);
impl_neighborhood_filter_image!(
    u8,
    AC4,
    filtering::filter_max_u8_ac4,
    filtering::filter_min_u8_ac4
);
impl_neighborhood_filter_image!(
    u16,
    C1,
    filtering::filter_max_u16_c1,
    filtering::filter_min_u16_c1
);
impl_neighborhood_filter_image!(
    u16,
    C3,
    filtering::filter_max_u16_c3,
    filtering::filter_min_u16_c3
);
impl_neighborhood_filter_image!(
    u16,
    C4,
    filtering::filter_max_u16_c4,
    filtering::filter_min_u16_c4
);
impl_neighborhood_filter_image!(
    u16,
    AC4,
    filtering::filter_max_u16_ac4,
    filtering::filter_min_u16_ac4
);
impl_neighborhood_filter_image!(
    i16,
    C1,
    filtering::filter_max_i16_c1,
    filtering::filter_min_i16_c1
);
impl_neighborhood_filter_image!(
    i16,
    C3,
    filtering::filter_max_i16_c3,
    filtering::filter_min_i16_c3
);
impl_neighborhood_filter_image!(
    i16,
    C4,
    filtering::filter_max_i16_c4,
    filtering::filter_min_i16_c4
);
impl_neighborhood_filter_image!(
    i16,
    AC4,
    filtering::filter_max_i16_ac4,
    filtering::filter_min_i16_ac4
);
impl_neighborhood_filter_image!(
    f32,
    C1,
    filtering::filter_max_f32_c1,
    filtering::filter_min_f32_c1
);
impl_neighborhood_filter_image!(
    f32,
    C3,
    filtering::filter_max_f32_c3,
    filtering::filter_min_f32_c3
);
impl_neighborhood_filter_image!(
    f32,
    C4,
    filtering::filter_max_f32_c4,
    filtering::filter_min_f32_c4
);
impl_neighborhood_filter_image!(
    f32,
    AC4,
    filtering::filter_max_f32_ac4,
    filtering::filter_min_f32_ac4
);
