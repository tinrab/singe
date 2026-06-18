use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, ImageView, ImageViewMut},
    },
    pipeline::ImagePipeline,
};

#[path = "color_ac4_yuv_static_methods.rs"]
mod yuv_static_methods;

impl<'a> ImagePipeline<'a, u8, AC4> {
    pub fn rgb_to_xyz_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::rgb_to_xyz_u8_ac4(stream_context, source, destination)
    }

    pub fn xyz_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::xyz_to_rgb_u8_ac4(stream_context, source, destination)
    }

    pub fn rgb_to_luv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::rgb_to_luv_u8_ac4(stream_context, source, destination)
    }

    pub fn luv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::luv_to_rgb_u8_ac4(stream_context, source, destination)
    }

    pub fn rgb_to_hsv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::rgb_to_hsv_u8_ac4(stream_context, source, destination)
    }

    pub fn hsv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::hsv_to_rgb_u8_ac4(stream_context, source, destination)
    }

    pub fn rgb_to_hls_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::rgb_to_hls_u8_ac4(stream_context, source, destination)
    }

    pub fn hls_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        destination: &mut ImageViewMut<'_, u8, AC4>,
    ) -> Result<()> {
        color::hls_to_rgb_u8_ac4(stream_context, source, destination)
    }
}
