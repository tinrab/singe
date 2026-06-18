use crate::{
    error::Result,
    image::{color, view::C3},
    pipeline::{ImageAllocator, ImageBacking, ImagePipeline, Workspace},
};

use super::SameLayoutColorConvert;

#[path = "color_c3_same_layout_static_methods.rs"]
mod static_methods;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    pub fn rgb_to_xyz(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_xyz_u8_c3)
    }

    pub fn xyz_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::xyz_to_rgb_u8_c3)
    }

    pub fn rgb_to_luv(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_luv_u8_c3)
    }

    pub fn luv_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::luv_to_rgb_u8_c3)
    }

    pub fn rgb_to_hsv(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_hsv_u8_c3)
    }

    pub fn hsv_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::hsv_to_rgb_u8_c3)
    }

    pub fn rgb_to_hls(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_hls_u8_c3)
    }

    pub fn hls_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::hls_to_rgb_u8_c3)
    }

    pub fn bgr_to_lab(self) -> Result<Self> {
        self.color_convert_same_layout(color::bgr_to_lab_u8_c3)
    }

    pub fn lab_to_bgr(self) -> Result<Self> {
        self.color_convert_same_layout(color::lab_to_bgr_u8_c3)
    }

    pub fn rgb_to_yuv(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_yuv_u8_c3)
    }

    pub fn yuv_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::yuv_to_rgb_u8_c3)
    }

    pub fn rgb_to_ycbcr(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_ycbcr_u8_c3)
    }

    pub fn ycbcr_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycbcr_to_rgb_u8_c3)
    }

    pub fn rgb_to_ycc(self) -> Result<Self> {
        self.color_convert_same_layout(color::rgb_to_ycc_u8_c3)
    }

    pub fn ycc_to_rgb(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycc_to_rgb_u8_c3)
    }

    pub fn bgr_to_yuv(self) -> Result<Self> {
        self.color_convert_same_layout(color::bgr_to_yuv_u8_c3)
    }

    pub fn yuv_to_bgr(self) -> Result<Self> {
        self.color_convert_same_layout(color::yuv_to_bgr_u8_c3)
    }

    fn color_convert_same_layout(self, operation: SameLayoutColorConvert<u8, C3>) -> Result<Self> {
        let mut destination = self.workspace.image::<u8, C3>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
