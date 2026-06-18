use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::{BorderType, Point, Size},
};

use super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_wiener_border_filter_image {
    ($ty:ty, $layout:ty, $channels:literal, $wiener:path) => {
        impl<'a> WienerBorderFilterImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_wiener_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: Size,
                anchor: Point,
                noise: &mut [f32; $channels],
                border_type: BorderType,
            ) -> Result<()> {
                $wiener(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    anchor,
                    noise,
                    border_type,
                )
            }
        }
    };
}

impl_wiener_border_filter_image!(u8, C1, 1, filtering::filter_wiener_border_u8_c1);
impl_wiener_border_filter_image!(u8, C3, 3, filtering::filter_wiener_border_u8_c3);
impl_wiener_border_filter_image!(u8, C4, 4, filtering::filter_wiener_border_u8_c4);
impl_wiener_border_filter_image!(u8, AC4, 3, filtering::filter_wiener_border_u8_ac4);
impl_wiener_border_filter_image!(i16, C1, 1, filtering::filter_wiener_border_i16_c1);
impl_wiener_border_filter_image!(i16, C3, 3, filtering::filter_wiener_border_i16_c3);
impl_wiener_border_filter_image!(i16, C4, 4, filtering::filter_wiener_border_i16_c4);
impl_wiener_border_filter_image!(i16, AC4, 3, filtering::filter_wiener_border_i16_ac4);
impl_wiener_border_filter_image!(f32, C1, 1, filtering::filter_wiener_border_f32_c1);
impl_wiener_border_filter_image!(f32, C3, 3, filtering::filter_wiener_border_f32_c3);
impl_wiener_border_filter_image!(f32, C4, 4, filtering::filter_wiener_border_f32_c4);
impl_wiener_border_filter_image!(f32, AC4, 3, filtering::filter_wiener_border_f32_ac4);
