use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::ImagePipeline;

pub trait ScaleToF32Image<L> {
    fn scale_to_f32_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, f32, L>,
        min: f32,
        max: f32,
    ) -> Result<()>;
}

pub trait ScaleToI32Image<L> {
    fn scale_to_i32_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, i32, L>,
    ) -> Result<()>;
}

pub trait ScaleToU16Image<L> {
    fn scale_to_u16_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, u16, L>,
    ) -> Result<()>;
}

pub trait ScaleToI16Image<L> {
    fn scale_to_i16_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, i16, L>,
    ) -> Result<()>;
}

macro_rules! impl_scale_to_f32_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleToF32Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_f32_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                min: f32,
                max: f32,
            ) -> Result<()> {
                $scale(stream_context, source, destination, min, max)
            }
        }
    };
}

macro_rules! impl_scale_to_i32_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleToI32Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_i32_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageViewMut<'_, i32, $layout>,
            ) -> Result<()> {
                $scale(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_scale_to_u16_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleToU16Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_u16_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageViewMut<'_, u16, $layout>,
            ) -> Result<()> {
                $scale(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_scale_to_i16_image {
    ($layout:ty, $scale:path) => {
        impl<'a> ScaleToI16Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_i16_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageViewMut<'_, i16, $layout>,
            ) -> Result<()> {
                $scale(stream_context, source, destination)
            }
        }
    };
}

impl_scale_to_f32_image!(C1, exchange::scale_u8_to_f32_c1);
impl_scale_to_f32_image!(C3, exchange::scale_u8_to_f32_c3);
impl_scale_to_f32_image!(C4, exchange::scale_u8_to_f32_c4);
impl_scale_to_f32_image!(AC4, exchange::scale_u8_to_f32_ac4);

impl_scale_to_i32_image!(C1, exchange::scale_u8_to_i32_c1);
impl_scale_to_i32_image!(C3, exchange::scale_u8_to_i32_c3);
impl_scale_to_i32_image!(C4, exchange::scale_u8_to_i32_c4);
impl_scale_to_i32_image!(AC4, exchange::scale_u8_to_i32_ac4);

impl_scale_to_u16_image!(C1, exchange::scale_u8_to_u16_c1);
impl_scale_to_u16_image!(C3, exchange::scale_u8_to_u16_c3);
impl_scale_to_u16_image!(C4, exchange::scale_u8_to_u16_c4);
impl_scale_to_u16_image!(AC4, exchange::scale_u8_to_u16_ac4);

impl_scale_to_i16_image!(C1, exchange::scale_u8_to_i16_c1);
impl_scale_to_i16_image!(C3, exchange::scale_u8_to_i16_c3);
impl_scale_to_i16_image!(C4, exchange::scale_u8_to_i16_c4);
impl_scale_to_i16_image!(AC4, exchange::scale_u8_to_i16_ac4);
