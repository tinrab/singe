use crate::{
    context::StreamContext,
    error::Result,
    image::{color, memory::Image, view::C3},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::planar::SubsampledPlanarImage;

impl SubsampledPlanarImage<u8> {
    pub fn ycbcr420_to_rgb_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::ycbcr420_to_rgb_jpeg_u8_p3_to_c3)
    }

    pub fn ycbcr422_to_rgb_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::ycbcr422_to_rgb_jpeg_u8_p3_to_c3)
    }

    pub fn ycbcr411_to_rgb_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::ycbcr411_to_rgb_jpeg_u8_p3_to_c3)
    }

    pub fn ycbcr420_to_bgr_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::ycbcr420_to_bgr_jpeg_u8_p3_to_c3)
    }

    pub fn ycbcr422_to_bgr_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::ycbcr422_to_bgr_jpeg_u8_p3_to_c3)
    }

    pub fn ycbcr411_to_bgr_jpeg_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::ycbcr411_to_bgr_jpeg_u8_p3_to_c3)
    }
}
