use crate::{
    error::Result,
    image::{
        color,
        view::{C1, C2, C3},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{super::super::planar::SemiplanarImage, ImagePipeline, Nv12PackedForward};

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C1>,
{
    pub fn rgb_to_nv12(self, twist: color::ColorTwistMatrix) -> Result<SemiplanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        self.color_twist_to_nv12(twist, color::rgb_to_nv12_u8_color_twist_c3_to_p2)
    }

    fn color_twist_to_nv12(
        self,
        twist: color::ColorTwistMatrix,
        operation: Nv12PackedForward<u8, C3>,
    ) -> Result<SemiplanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C2>,
    {
        let mut destination = SemiplanarImage::create_nv12(self.size())?;

        {
            let source = self.view()?;
            let mut destination_y = destination.y.view_mut()?;
            let mut destination_uv = destination.uv.view_mut()?;
            operation(
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
