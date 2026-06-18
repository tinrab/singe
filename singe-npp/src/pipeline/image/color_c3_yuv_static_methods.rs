use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C3, ImageView, ImageViewMut},
    },
    pipeline::ImagePipeline,
};

impl<'a> ImagePipeline<'a, u8, C3> {
    pub fn rgb_to_yuv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::rgb_to_yuv_u8_c3(stream_context, source, destination)
    }

    pub fn yuv_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::yuv_to_rgb_u8_c3(stream_context, source, destination)
    }

    pub fn rgb_to_ycbcr_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::rgb_to_ycbcr_u8_c3(stream_context, source, destination)
    }

    pub fn ycbcr_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::ycbcr_to_rgb_u8_c3(stream_context, source, destination)
    }

    pub fn rgb_to_ycc_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::rgb_to_ycc_u8_c3(stream_context, source, destination)
    }

    pub fn ycc_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::ycc_to_rgb_u8_c3(stream_context, source, destination)
    }

    pub fn bgr_to_yuv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::bgr_to_yuv_u8_c3(stream_context, source, destination)
    }

    pub fn yuv_to_bgr_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::yuv_to_bgr_u8_c3(stream_context, source, destination)
    }
}
