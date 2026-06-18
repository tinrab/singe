use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, C4, ImageView, ImageViewMut},
    },
    pipeline::ImagePipeline,
};

impl<'a> ImagePipeline<'a, u8, C2> {
    pub fn ycbcr422_to_bgr_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C4>,
        alpha: u8,
    ) -> Result<()> {
        color::ycbcr422_to_bgr_u8_c2_to_c4(stream_context, source, destination, alpha)
    }

    pub fn cbycr422_to_bgr_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C4>,
        alpha: u8,
    ) -> Result<()> {
        color::cbycr422_to_bgr_u8_c2_to_c4(stream_context, source, destination, alpha)
    }

    pub fn cbycr422_to_bgr_709hdtv_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C2>,
        destination: &mut ImageViewMut<'_, u8, C4>,
        alpha: u8,
    ) -> Result<()> {
        color::cbycr422_to_bgr_709hdtv_u8_c2_to_c4(stream_context, source, destination, alpha)
    }
}
