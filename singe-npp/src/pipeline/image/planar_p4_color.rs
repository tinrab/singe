use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        memory::Image,
        view::{AC4, C3},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::PlanarImage;

use self::helpers::PlanarP4ColorConversionExt;

#[path = "planar_p4_color_helpers.rs"]
mod helpers;

impl PlanarImage<u8, 4> {
    pub fn bgr_to_hls(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::bgr_to_hls_u8_p4)
    }

    pub fn hls_to_bgr(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::hls_to_bgr_u8_p4)
    }

    pub fn ycck_to_cmyk_jpeg_601(self, stream_context: &StreamContext) -> Result<Self> {
        self.color_convert(stream_context, color::ycck_to_cmyk_jpeg_601_u8_p4)
    }

    pub fn cmyk_or_ycck_to_rgb_jpeg_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_p3(stream_context, color::cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_p3)
    }

    pub fn cmyk_or_ycck_to_bgr_jpeg_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_p3(stream_context, color::cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_p3)
    }

    pub fn cmyk_or_ycck_to_rgb_jpeg_c3(
        self,
        stream_context: &StreamContext,
    ) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_c3(stream_context, color::cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3)
    }

    pub fn cmyk_or_ycck_to_bgr_jpeg_c3(
        self,
        stream_context: &StreamContext,
    ) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_c3(stream_context, color::cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3)
    }

    pub fn bgr_to_hls_ac4(self, stream_context: &StreamContext) -> Result<Image<u8, AC4>>
    where
        Workspace: ImageAllocator<u8, AC4>,
    {
        self.color_convert_to_ac4(stream_context, color::bgr_to_hls_u8_p4_to_ac4)
    }

    pub fn hls_to_bgr_ac4(self, stream_context: &StreamContext) -> Result<Image<u8, AC4>>
    where
        Workspace: ImageAllocator<u8, AC4>,
    {
        self.color_convert_to_ac4(stream_context, color::hls_to_bgr_u8_p4_to_ac4)
    }
}
