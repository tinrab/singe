use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{Point, Size},
};

use super::ImagePipeline;

use super::filtering_traits::*;

#[path = "filtering_dispatch_spatial_border.rs"]
mod filtering_dispatch_spatial_border;
#[path = "filtering_dispatch_spatial_neighborhood.rs"]
mod filtering_dispatch_spatial_neighborhood;

macro_rules! impl_box_filter_image {
    ($ty:ty, $layout:ty, $filter_box:path) => {
        impl<'a> BoxFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_box_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                $filter_box(stream_context, source, destination, mask_size, anchor)
            }
        }
    };
}

#[path = "filtering_dispatch_median.rs"]
mod filtering_dispatch_median;

impl_box_filter_image!(u8, C1, filtering::filter_box_u8_c1);
impl_box_filter_image!(u8, C3, filtering::filter_box_u8_c3);
impl_box_filter_image!(u8, C4, filtering::filter_box_u8_c4);
impl_box_filter_image!(u8, AC4, filtering::filter_box_u8_ac4);
impl_box_filter_image!(u16, C1, filtering::filter_box_u16_c1);
impl_box_filter_image!(u16, C3, filtering::filter_box_u16_c3);
impl_box_filter_image!(u16, C4, filtering::filter_box_u16_c4);
impl_box_filter_image!(u16, AC4, filtering::filter_box_u16_ac4);
impl_box_filter_image!(i16, C1, filtering::filter_box_i16_c1);
impl_box_filter_image!(i16, C3, filtering::filter_box_i16_c3);
impl_box_filter_image!(i16, C4, filtering::filter_box_i16_c4);
impl_box_filter_image!(i16, AC4, filtering::filter_box_i16_ac4);
impl_box_filter_image!(f32, C1, filtering::filter_box_f32_c1);
impl_box_filter_image!(f32, C3, filtering::filter_box_f32_c3);
impl_box_filter_image!(f32, C4, filtering::filter_box_f32_c4);
impl_box_filter_image!(f32, AC4, filtering::filter_box_f32_ac4);
impl_box_filter_image!(f64, C1, filtering::filter_box_f64_c1);
