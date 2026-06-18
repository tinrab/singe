use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C3, ImageView, ImageViewMut},
    },
    pipeline::ImagePipeline,
};

#[path = "color_c3_yuv_static_methods.rs"]
mod yuv_methods;

impl<'a> ImagePipeline<'a, u8, C3> {
    pub fn rgb_to_xyz_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::rgb_to_xyz_u8_c3(stream_context, source, destination)
    }

    pub fn xyz_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::xyz_to_rgb_u8_c3(stream_context, source, destination)
    }

    pub fn rgb_to_luv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::rgb_to_luv_u8_c3(stream_context, source, destination)
    }

    pub fn luv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::luv_to_rgb_u8_c3(stream_context, source, destination)
    }

    pub fn rgb_to_hsv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::rgb_to_hsv_u8_c3(stream_context, source, destination)
    }

    pub fn hsv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::hsv_to_rgb_u8_c3(stream_context, source, destination)
    }

    pub fn rgb_to_hls_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::rgb_to_hls_u8_c3(stream_context, source, destination)
    }

    pub fn hls_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::hls_to_rgb_u8_c3(stream_context, source, destination)
    }

    pub fn bgr_to_lab_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::bgr_to_lab_u8_c3(stream_context, source, destination)
    }

    pub fn lab_to_bgr_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::lab_to_bgr_u8_c3(stream_context, source, destination)
    }
}
