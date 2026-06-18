use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::RoundMode,
};

use super::ImagePipeline;

pub trait RoundConvertImage<T, U, L> {
    fn round_convert_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
        round_mode: RoundMode,
    ) -> Result<()>;
}

pub trait ScaledRoundConvertImage<T, U, L> {
    fn scaled_round_convert_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>;
}

macro_rules! impl_round_convert_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $convert:path) => {
        impl<'a> RoundConvertImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn round_convert_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
                round_mode: RoundMode,
            ) -> Result<()> {
                $convert(stream_context, source, destination, round_mode)
            }
        }
    };
}

macro_rules! impl_scaled_round_convert_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $convert:path) => {
        impl<'a> ScaledRoundConvertImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn scaled_round_convert_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()> {
                $convert(
                    stream_context,
                    source,
                    destination,
                    round_mode,
                    scale_factor,
                )
            }
        }
    };
}

impl_round_convert_image!(f32, u8, C1, exchange::convert_f32_to_u8_c1);
impl_round_convert_image!(f32, u8, C3, exchange::convert_f32_to_u8_c3);
impl_round_convert_image!(f32, u8, C4, exchange::convert_f32_to_u8_c4);
impl_round_convert_image!(f32, u8, AC4, exchange::convert_f32_to_u8_ac4);
impl_round_convert_image!(f32, i8, C1, exchange::convert_f32_to_i8_c1);
impl_round_convert_image!(f32, i8, C3, exchange::convert_f32_to_i8_c3);
impl_round_convert_image!(f32, i8, C4, exchange::convert_f32_to_i8_c4);
impl_round_convert_image!(f32, i8, AC4, exchange::convert_f32_to_i8_ac4);
impl_round_convert_image!(f32, u16, C1, exchange::convert_f32_to_u16_c1);
impl_round_convert_image!(f32, u16, C3, exchange::convert_f32_to_u16_c3);
impl_round_convert_image!(f32, u16, C4, exchange::convert_f32_to_u16_c4);
impl_round_convert_image!(f32, u16, AC4, exchange::convert_f32_to_u16_ac4);
impl_round_convert_image!(f32, i16, C1, exchange::convert_f32_to_i16_c1);
impl_round_convert_image!(f32, i16, C3, exchange::convert_f32_to_i16_c3);
impl_round_convert_image!(f32, i16, C4, exchange::convert_f32_to_i16_c4);
impl_round_convert_image!(f32, i16, AC4, exchange::convert_f32_to_i16_ac4);
impl_round_convert_image!(f32, f16, C1, exchange::convert_f32_to_f16_c1);
impl_round_convert_image!(f32, f16, C3, exchange::convert_f32_to_f16_c3);
impl_round_convert_image!(f32, f16, C4, exchange::convert_f32_to_f16_c4);
impl_round_convert_image!(f32, f16, AC4, exchange::convert_f32_to_f16_ac4);

impl_scaled_round_convert_image!(u16, i16, C1, exchange::convert_u16_to_i16_scaled_c1);
impl_scaled_round_convert_image!(u16, i8, C1, exchange::convert_u16_to_i8_scaled_c1);
impl_scaled_round_convert_image!(i16, i8, C1, exchange::convert_i16_to_i8_scaled_c1);
impl_scaled_round_convert_image!(u32, u8, C1, exchange::convert_u32_to_u8_scaled_c1);
impl_scaled_round_convert_image!(u32, i8, C1, exchange::convert_u32_to_i8_scaled_c1);
impl_scaled_round_convert_image!(u32, u16, C1, exchange::convert_u32_to_u16_scaled_c1);
impl_scaled_round_convert_image!(u32, i16, C1, exchange::convert_u32_to_i16_scaled_c1);
impl_scaled_round_convert_image!(u32, i32, C1, exchange::convert_u32_to_i32_scaled_c1);
impl_scaled_round_convert_image!(i32, u16, C1, exchange::convert_i32_to_u16_scaled_c1);
impl_scaled_round_convert_image!(i32, i16, C1, exchange::convert_i32_to_i16_scaled_c1);
impl_scaled_round_convert_image!(f32, u8, C1, exchange::convert_f32_to_u8_scaled_c1);
impl_scaled_round_convert_image!(f32, i8, C1, exchange::convert_f32_to_i8_scaled_c1);
impl_scaled_round_convert_image!(f32, u16, C1, exchange::convert_f32_to_u16_scaled_c1);
impl_scaled_round_convert_image!(f32, i16, C1, exchange::convert_f32_to_i16_scaled_c1);
impl_scaled_round_convert_image!(f32, u32, C1, exchange::convert_f32_to_u32_scaled_c1);
impl_scaled_round_convert_image!(f32, i32, C1, exchange::convert_f32_to_i32_scaled_c1);
impl_scaled_round_convert_image!(u8, i8, C1, exchange::convert_u8_to_i8_scaled_c1);
