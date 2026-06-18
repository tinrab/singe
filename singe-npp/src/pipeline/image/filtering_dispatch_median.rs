use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{Point, Size},
};

use super::super::{ImagePipeline, filtering_traits::MedianFilterImage};

macro_rules! impl_median_filter_image {
    ($ty:ty, $layout:ty, $filter_median:path) => {
        impl<'a> MedianFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_median_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                $filter_median(stream_context, source, destination, mask_size, anchor)
            }
        }
    };
}

impl_median_filter_image!(u8, C1, filtering::filter_median_u8_c1);
impl_median_filter_image!(u8, C3, filtering::filter_median_u8_c3);
impl_median_filter_image!(u8, C4, filtering::filter_median_u8_c4);
impl_median_filter_image!(u8, AC4, filtering::filter_median_u8_ac4);
impl_median_filter_image!(u16, C1, filtering::filter_median_u16_c1);
impl_median_filter_image!(u16, C3, filtering::filter_median_u16_c3);
impl_median_filter_image!(u16, C4, filtering::filter_median_u16_c4);
impl_median_filter_image!(u16, AC4, filtering::filter_median_u16_ac4);
impl_median_filter_image!(i16, C1, filtering::filter_median_i16_c1);
impl_median_filter_image!(i16, C4, filtering::filter_median_i16_c4);
impl_median_filter_image!(i16, AC4, filtering::filter_median_i16_ac4);
impl_median_filter_image!(f32, C1, filtering::filter_median_f32_c1);
impl_median_filter_image!(f32, C3, filtering::filter_median_f32_c3);
impl_median_filter_image!(f32, C4, filtering::filter_median_f32_c4);
impl_median_filter_image!(f32, AC4, filtering::filter_median_f32_ac4);
