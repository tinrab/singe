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

pub trait ScaleToU8Image<L> {
    fn scale_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
        min: f32,
        max: f32,
    ) -> Result<()>;
}

pub trait ScaleI32ToU8Image<L> {
    fn scale_i32_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, i32, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()>;
}

pub trait ScaleU16ToU8Image<L> {
    fn scale_u16_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u16, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()>;
}

pub trait ScaleI16ToU8Image<L> {
    fn scale_i16_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, i16, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()>;
}

macro_rules! impl_scale_to_u8_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleToU8Image<$layout> for ImagePipeline<'a, f32, $layout> {
            fn scale_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageViewMut<'_, u8, $layout>,
                min: f32,
                max: f32,
            ) -> Result<()> {
                $scale(stream_context, source, destination, min, max)
            }
        }
    };
}

macro_rules! impl_scale_i32_to_u8_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleI32ToU8Image<$layout> for ImagePipeline<'a, i32, $layout> {
            fn scale_i32_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, i32, $layout>,
                destination: &mut ImageViewMut<'_, u8, $layout>,
                hint: HintAlgorithm,
            ) -> Result<()> {
                $scale(stream_context, source, destination, hint)
            }
        }
    };
}

macro_rules! impl_scale_u16_to_u8_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleU16ToU8Image<$layout> for ImagePipeline<'a, u16, $layout> {
            fn scale_u16_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u16, $layout>,
                destination: &mut ImageViewMut<'_, u8, $layout>,
                hint: HintAlgorithm,
            ) -> Result<()> {
                $scale(stream_context, source, destination, hint)
            }
        }
    };
}

macro_rules! impl_scale_i16_to_u8_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleI16ToU8Image<$layout> for ImagePipeline<'a, i16, $layout> {
            fn scale_i16_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, i16, $layout>,
                destination: &mut ImageViewMut<'_, u8, $layout>,
                hint: HintAlgorithm,
            ) -> Result<()> {
                $scale(stream_context, source, destination, hint)
            }
        }
    };
}

impl_scale_to_u8_image!(C1, exchange::scale_f32_to_u8_c1);
impl_scale_to_u8_image!(C3, exchange::scale_f32_to_u8_c3);
impl_scale_to_u8_image!(C4, exchange::scale_f32_to_u8_c4);
impl_scale_to_u8_image!(AC4, exchange::scale_f32_to_u8_ac4);

impl_scale_i32_to_u8_image!(C1, exchange::scale_i32_to_u8_c1);
impl_scale_i32_to_u8_image!(C3, exchange::scale_i32_to_u8_c3);
impl_scale_i32_to_u8_image!(C4, exchange::scale_i32_to_u8_c4);
impl_scale_i32_to_u8_image!(AC4, exchange::scale_i32_to_u8_ac4);

impl_scale_u16_to_u8_image!(C1, exchange::scale_u16_to_u8_c1);
impl_scale_u16_to_u8_image!(C3, exchange::scale_u16_to_u8_c3);
impl_scale_u16_to_u8_image!(C4, exchange::scale_u16_to_u8_c4);
impl_scale_u16_to_u8_image!(AC4, exchange::scale_u16_to_u8_ac4);

impl_scale_i16_to_u8_image!(C1, exchange::scale_i16_to_u8_c1);
impl_scale_i16_to_u8_image!(C3, exchange::scale_i16_to_u8_c3);
impl_scale_i16_to_u8_image!(C4, exchange::scale_i16_to_u8_c4);
impl_scale_i16_to_u8_image!(AC4, exchange::scale_i16_to_u8_ac4);
