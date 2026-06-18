use crate::{
    error::Result,
    image::{geometry, view::C1},
    pipeline::{ImageAllocator, Workspace},
    workspace::ScratchBuffer,
};

use super::{ImageBacking, ImagePipeline, ResizeSqrPixelAdvanced};

impl<'a> ImagePipeline<'a, u8, C1>
where
    Workspace: ImageAllocator<u8, C1>,
{
    pub fn resize_sqr_pixel_advanced_buffer_size(resize: &ResizeSqrPixelAdvanced) -> Result<usize> {
        geometry::resize_sqr_pixel_c1_advanced_buffer_size::<u8>(resize)
    }

    pub fn resize_sqr_pixel_advanced_with_scratch(
        self,
        resize: ResizeSqrPixelAdvanced,
        scratch: &mut ScratchBuffer,
    ) -> Result<Self> {
        let mut destination = self
            .workspace
            .image::<u8, C1>(resize.destination_roi.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            geometry::resize_sqr_pixel_c1_advanced_with_scratch(
                self.stream_context,
                &resize,
                &source,
                &mut destination_view,
                scratch,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
