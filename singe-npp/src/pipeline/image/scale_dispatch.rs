use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::HintAlgorithm,
};

use super::ImagePipeline;

#[path = "scale_conversion_dispatch.rs"]
mod conversion;
pub use conversion::{ScaleToF32Image, ScaleToI16Image, ScaleToI32Image, ScaleToU16Image};

pub trait ScaleImage<T, U, L> {
    fn scale_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
    ) -> Result<()>;
}

pub trait ScaleRangeImage<T, U, L> {
    fn scale_range_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
        min: f32,
        max: f32,
    ) -> Result<()>;
}

pub trait ScaleHintImage<T, U, L> {
    fn scale_hint_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, U, L>,
        hint: HintAlgorithm,
    ) -> Result<()>;
}

macro_rules! impl_scale_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $scale:path) => {
        impl<'a> ScaleImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn scale_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                $scale(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_scale_range_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $scale:path) => {
        impl<'a> ScaleRangeImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn scale_range_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
                min: f32,
                max: f32,
            ) -> Result<()> {
                $scale(stream_context, source, destination, min, max)
            }
        }
    };
}

macro_rules! impl_scale_hint_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $scale:path) => {
        impl<'a> ScaleHintImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn scale_hint_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
                hint: HintAlgorithm,
            ) -> Result<()> {
                $scale(stream_context, source, destination, hint)
            }
        }
    };
}

impl_scale_range_image!(u8, f32, C1, exchange::scale_u8_to_f32_c1);
impl_scale_range_image!(u8, f32, C3, exchange::scale_u8_to_f32_c3);
impl_scale_range_image!(u8, f32, C4, exchange::scale_u8_to_f32_c4);
impl_scale_range_image!(u8, f32, AC4, exchange::scale_u8_to_f32_ac4);
impl_scale_image!(u8, i32, C1, exchange::scale_u8_to_i32_c1);
impl_scale_image!(u8, i32, C3, exchange::scale_u8_to_i32_c3);
impl_scale_image!(u8, i32, C4, exchange::scale_u8_to_i32_c4);
impl_scale_image!(u8, i32, AC4, exchange::scale_u8_to_i32_ac4);
impl_scale_image!(u8, u16, C1, exchange::scale_u8_to_u16_c1);
impl_scale_image!(u8, u16, C3, exchange::scale_u8_to_u16_c3);
impl_scale_image!(u8, u16, C4, exchange::scale_u8_to_u16_c4);
impl_scale_image!(u8, u16, AC4, exchange::scale_u8_to_u16_ac4);
impl_scale_image!(u8, i16, C1, exchange::scale_u8_to_i16_c1);
impl_scale_image!(u8, i16, C3, exchange::scale_u8_to_i16_c3);
impl_scale_image!(u8, i16, C4, exchange::scale_u8_to_i16_c4);
impl_scale_image!(u8, i16, AC4, exchange::scale_u8_to_i16_ac4);
impl_scale_range_image!(f32, u8, C1, exchange::scale_f32_to_u8_c1);
impl_scale_range_image!(f32, u8, C3, exchange::scale_f32_to_u8_c3);
impl_scale_range_image!(f32, u8, C4, exchange::scale_f32_to_u8_c4);
impl_scale_range_image!(f32, u8, AC4, exchange::scale_f32_to_u8_ac4);
impl_scale_hint_image!(i32, u8, C1, exchange::scale_i32_to_u8_c1);
impl_scale_hint_image!(i32, u8, C3, exchange::scale_i32_to_u8_c3);
impl_scale_hint_image!(i32, u8, C4, exchange::scale_i32_to_u8_c4);
impl_scale_hint_image!(i32, u8, AC4, exchange::scale_i32_to_u8_ac4);
impl_scale_hint_image!(u16, u8, C1, exchange::scale_u16_to_u8_c1);
impl_scale_hint_image!(u16, u8, C3, exchange::scale_u16_to_u8_c3);
impl_scale_hint_image!(u16, u8, C4, exchange::scale_u16_to_u8_c4);
impl_scale_hint_image!(u16, u8, AC4, exchange::scale_u16_to_u8_ac4);
impl_scale_hint_image!(i16, u8, C1, exchange::scale_i16_to_u8_c1);
impl_scale_hint_image!(i16, u8, C3, exchange::scale_i16_to_u8_c3);
impl_scale_hint_image!(i16, u8, C4, exchange::scale_i16_to_u8_c4);
impl_scale_hint_image!(i16, u8, AC4, exchange::scale_i16_to_u8_ac4);
