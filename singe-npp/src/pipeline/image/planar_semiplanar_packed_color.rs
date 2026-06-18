use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        memory::Image,
        view::{C3, C4},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::planar::SemiplanarImage;

impl SemiplanarImage<u8> {
    pub fn nv12_to_rgb_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_rgb_u8_p2_to_c3)
    }

    pub fn nv12_to_rgb_709hdtv_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_rgb_709hdtv_u8_p2_to_c3)
    }

    pub fn nv12_to_rgb_709csc_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_rgb_709csc_u8_p2_to_c3)
    }

    pub fn nv12_to_bgr_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_bgr_u8_p2_to_c3)
    }

    pub fn nv12_to_bgr_709hdtv_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_bgr_709hdtv_u8_p2_to_c3)
    }

    pub fn nv12_to_bgr_709csc_c3(self, stream_context: &StreamContext) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_convert_to_packed(stream_context, color::nv12_to_bgr_709csc_u8_p2_to_c3)
    }

    pub fn nv21_to_rgb_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed(stream_context, color::nv21_to_rgb_u8_p2_to_c4)
    }

    pub fn nv21_to_bgr_c4(self, stream_context: &StreamContext) -> Result<Image<u8, C4>>
    where
        Workspace: ImageAllocator<u8, C4>,
    {
        self.color_convert_to_packed(stream_context, color::nv21_to_bgr_u8_p2_to_c4)
    }

    pub fn nv12_to_rgb_color_twist_c3(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        self.color_twist_to_packed(
            stream_context,
            twist,
            color::nv12_to_rgb_u8_color_twist_p2_to_c3,
        )
    }
}

impl SemiplanarImage<u16> {
    pub fn nv12_to_rgb_color_twist_c3(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Image<u16, C3>>
    where
        Workspace: ImageAllocator<u16, C3>,
    {
        self.color_twist_to_packed(
            stream_context,
            twist,
            color::nv12_to_rgb_u16_color_twist_p2_to_c3,
        )
    }
}
