use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, C3, ImageView, ImageViewMut},
    },
    pipeline::ImagePipeline,
};

#[path = "color_c2_to_c4_static_methods.rs"]
mod to_c4_methods;

impl<'a> ImagePipeline<'a, u8, C2> {
    pub fn yuv422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::yuv422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
    }

    pub fn ycbcr422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::ycbcr422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
    }

    pub fn ycrcb422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::ycrcb422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
    }

    pub fn cbycr422_to_rgb_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::cbycr422_to_rgb_u8_c2_to_c3(stream_context, source, destination)
    }

    pub fn ycbcr422_to_bgr_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::ycbcr422_to_bgr_u8_c2_to_c3(stream_context, source, destination)
    }

    pub fn cbycr422_to_bgr_709hdtv_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C3>,
    ) -> Result<()> {
        color::cbycr422_to_bgr_709hdtv_u8_c2_to_c3(stream_context, source, destination)
    }

    pub fn ycbcr422_to_ycrcb422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::ycbcr422_to_ycrcb422_u8_c2(stream_context, source, destination)
    }

    pub fn ycbcr422_to_cbycr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::ycbcr422_to_cbycr422_u8_c2(stream_context, source, destination)
    }

    pub fn cbycr422_to_ycbcr422_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C2>,
    ) -> Result<()> {
        color::cbycr422_to_ycbcr422_u8_c2(stream_context, source, destination)
    }
}
