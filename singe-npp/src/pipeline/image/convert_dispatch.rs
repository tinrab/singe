use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::ImagePipeline;

pub trait ConvertImage<T, U, L> {
    fn convert_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
    ) -> Result<()>;
}

macro_rules! impl_convert_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $convert:path) => {
        impl<'a> ConvertImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn convert_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                $convert(stream_context, source, destination)
            }
        }
    };
}

impl_convert_image!(i8, u8, C1, exchange::convert_i8_to_u8_c1);
impl_convert_image!(i8, u16, C1, exchange::convert_i8_to_u16_c1);
impl_convert_image!(i8, i16, C1, exchange::convert_i8_to_i16_c1);
impl_convert_image!(i8, u32, C1, exchange::convert_i8_to_u32_c1);
impl_convert_image!(i8, i32, C1, exchange::convert_i8_to_i32_c1);
impl_convert_image!(i8, i32, C3, exchange::convert_i8_to_i32_c3);
impl_convert_image!(i8, i32, C4, exchange::convert_i8_to_i32_c4);
impl_convert_image!(i8, i32, AC4, exchange::convert_i8_to_i32_ac4);
impl_convert_image!(i8, f32, C1, exchange::convert_i8_to_f32_c1);
impl_convert_image!(i8, f32, C3, exchange::convert_i8_to_f32_c3);
impl_convert_image!(i8, f32, C4, exchange::convert_i8_to_f32_c4);
impl_convert_image!(i8, f32, AC4, exchange::convert_i8_to_f32_ac4);
impl_convert_image!(i32, i8, C1, exchange::convert_i32_to_i8_c1);
impl_convert_image!(i32, i8, C3, exchange::convert_i32_to_i8_c3);
impl_convert_image!(i32, i8, C4, exchange::convert_i32_to_i8_c4);
impl_convert_image!(i32, i8, AC4, exchange::convert_i32_to_i8_ac4);
impl_convert_image!(f16, f32, C1, exchange::convert_f16_to_f32_c1);
impl_convert_image!(f16, f32, C3, exchange::convert_f16_to_f32_c3);
impl_convert_image!(f16, f32, C4, exchange::convert_f16_to_f32_c4);
impl_convert_image!(f16, f32, AC4, exchange::convert_f16_to_f32_ac4);
impl_convert_image!(u8, f32, C1, exchange::convert_to_f32_c1);
impl_convert_image!(u8, f32, C3, exchange::convert_to_f32_c3);
impl_convert_image!(u8, f32, C4, exchange::convert_to_f32_c4);
impl_convert_image!(u8, f32, AC4, exchange::convert_to_f32_ac4);
impl_convert_image!(u8, u16, C1, exchange::convert_u8_to_u16_c1);
impl_convert_image!(u8, u16, C3, exchange::convert_u8_to_u16_c3);
impl_convert_image!(u8, u16, C4, exchange::convert_u8_to_u16_c4);
impl_convert_image!(u8, u16, AC4, exchange::convert_u8_to_u16_ac4);
impl_convert_image!(u8, i16, C1, exchange::convert_u8_to_i16_c1);
impl_convert_image!(u8, i16, C3, exchange::convert_u8_to_i16_c3);
impl_convert_image!(u8, i16, C4, exchange::convert_u8_to_i16_c4);
impl_convert_image!(u8, i16, AC4, exchange::convert_u8_to_i16_ac4);
impl_convert_image!(u8, i32, C1, exchange::convert_u8_to_i32_c1);
impl_convert_image!(u8, i32, C3, exchange::convert_u8_to_i32_c3);
impl_convert_image!(u8, i32, C4, exchange::convert_u8_to_i32_c4);
impl_convert_image!(u8, i32, AC4, exchange::convert_u8_to_i32_ac4);
impl_convert_image!(u16, u8, C1, exchange::convert_u16_to_u8_c1);
impl_convert_image!(u16, u8, C3, exchange::convert_u16_to_u8_c3);
impl_convert_image!(u16, u8, C4, exchange::convert_u16_to_u8_c4);
impl_convert_image!(u16, u8, AC4, exchange::convert_u16_to_u8_ac4);
impl_convert_image!(u16, i32, C1, exchange::convert_u16_to_i32_c1);
impl_convert_image!(u16, i32, C3, exchange::convert_u16_to_i32_c3);
impl_convert_image!(u16, i32, C4, exchange::convert_u16_to_i32_c4);
impl_convert_image!(u16, i32, AC4, exchange::convert_u16_to_i32_ac4);
impl_convert_image!(u16, f32, C1, exchange::convert_u16_to_f32_c1);
impl_convert_image!(u16, f32, C3, exchange::convert_u16_to_f32_c3);
impl_convert_image!(u16, f32, C4, exchange::convert_u16_to_f32_c4);
impl_convert_image!(u16, f32, AC4, exchange::convert_u16_to_f32_ac4);
impl_convert_image!(i16, u8, C1, exchange::convert_i16_to_u8_c1);
impl_convert_image!(i16, u8, C3, exchange::convert_i16_to_u8_c3);
impl_convert_image!(i16, u8, C4, exchange::convert_i16_to_u8_c4);
impl_convert_image!(i16, u8, AC4, exchange::convert_i16_to_u8_ac4);
impl_convert_image!(i16, u16, C1, exchange::convert_i16_to_u16_c1);
impl_convert_image!(i16, i32, C1, exchange::convert_i16_to_i32_c1);
impl_convert_image!(i16, i32, C3, exchange::convert_i16_to_i32_c3);
impl_convert_image!(i16, i32, C4, exchange::convert_i16_to_i32_c4);
impl_convert_image!(i16, i32, AC4, exchange::convert_i16_to_i32_ac4);
impl_convert_image!(i16, u32, C1, exchange::convert_i16_to_u32_c1);
impl_convert_image!(i16, f32, C1, exchange::convert_i16_to_f32_c1);
impl_convert_image!(i16, f32, C3, exchange::convert_i16_to_f32_c3);
impl_convert_image!(i16, f32, C4, exchange::convert_i16_to_f32_c4);
impl_convert_image!(i16, f32, AC4, exchange::convert_i16_to_f32_ac4);
impl_convert_image!(i32, u8, C1, exchange::convert_i32_to_u8_c1);
impl_convert_image!(i32, u8, C3, exchange::convert_i32_to_u8_c3);
impl_convert_image!(i32, u8, C4, exchange::convert_i32_to_u8_c4);
impl_convert_image!(i32, u8, AC4, exchange::convert_i32_to_u8_ac4);
impl_convert_image!(i32, u32, C1, exchange::convert_i32_to_u32_c1);
impl_convert_image!(i32, f32, C1, exchange::convert_i32_to_f32_c1);
impl_convert_image!(u16, u32, C1, exchange::convert_u16_to_u32_c1);
impl_convert_image!(u32, f32, C1, exchange::convert_u32_to_f32_c1);
