use crate::{
    error::Result,
    image::{
        color,
        view::{C1, C2, C3},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::planar::SemiplanarImage;
use super::super::ImagePipeline;

impl<'a> ImagePipeline<'a, u16, C3>
where
    Workspace: ImageAllocator<u16, C1> + ImageAllocator<u16, C2>,
{
    pub fn rgb_to_nv12(self, twist: color::ColorTwistMatrix) -> Result<SemiplanarImage<u16>> {
        let mut destination = SemiplanarImage::create_nv12(self.size())?;

        {
            let source = self.view()?;
            let mut destination_y = destination.y.view_mut()?;
            let mut destination_uv = destination.uv.view_mut()?;
            color::rgb_to_nv12_u16_color_twist_c3_to_p2(
                self.stream_context,
                &source,
                &mut destination_y,
                &mut destination_uv,
                twist,
            )?;
        }

        Ok(destination)
    }
}
