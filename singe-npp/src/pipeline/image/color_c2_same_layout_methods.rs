use crate::{
    error::Result,
    image::{color, view::C2},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::ImagePipeline;

impl<'a> ImagePipeline<'a, u8, C2>
where
    Workspace: ImageAllocator<u8, C2>,
{
    pub fn ycbcr422_to_ycrcb422(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycbcr422_to_ycrcb422_u8_c2)
    }

    pub fn ycbcr422_to_cbycr422(self) -> Result<Self> {
        self.color_convert_same_layout(color::ycbcr422_to_cbycr422_u8_c2)
    }

    pub fn cbycr422_to_ycbcr422(self) -> Result<Self> {
        self.color_convert_same_layout(color::cbycr422_to_ycbcr422_u8_c2)
    }
}
