use crate::{
    error::Result,
    image::{
        color,
        view::{C2, C4},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::ImagePipeline;

impl<'a> ImagePipeline<'a, u8, C2>
where
    Workspace: ImageAllocator<u8, C4>,
{
    pub fn ycbcr422_to_bgr_c4(self, alpha: u8) -> Result<ImagePipeline<'a, u8, C4>> {
        self.color_convert_to_c4(alpha, color::ycbcr422_to_bgr_u8_c2_to_c4)
    }

    pub fn cbycr422_to_bgr_c4(self, alpha: u8) -> Result<ImagePipeline<'a, u8, C4>> {
        self.color_convert_to_c4(alpha, color::cbycr422_to_bgr_u8_c2_to_c4)
    }

    pub fn cbycr422_to_bgr_709hdtv_c4(self, alpha: u8) -> Result<ImagePipeline<'a, u8, C4>> {
        self.color_convert_to_c4(alpha, color::cbycr422_to_bgr_709hdtv_u8_c2_to_c4)
    }
}
