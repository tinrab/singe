use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C3, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::{ColorSpace, Point},
};

use super::{ImageBacking, ImagePipeline};

#[path = "color_uyvp_u16_ac4_packed_methods.rs"]
mod ac4_methods;

impl<'a> ImagePipeline<'a, u16, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    pub fn rgb_to_uyvp_10u_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u16, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C3>,
        color_space: ColorSpace,
    ) -> Result<()> {
        color::rgb_to_uyvp_10u_u16_c3(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }

    pub fn rgb_to_uyvp_10u(
        self,
        source_offset: Point,
        color_space: ColorSpace,
    ) -> Result<ImagePipeline<'a, u8, C3>> {
        let mut destination = self.workspace.image::<u8, C3>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            color::rgb_to_uyvp_10u_u16_c3(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                color_space,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u16, C3>,
{
    pub fn uyvp_10u_to_rgb_u16_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u16, C3>,
        color_space: ColorSpace,
    ) -> Result<()> {
        color::uyvp_10u_to_rgb_u16_c3(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }

    pub fn uyvp_10u_to_rgb_u16(
        self,
        source_offset: Point,
        color_space: ColorSpace,
    ) -> Result<ImagePipeline<'a, u16, C3>> {
        let mut destination = self.workspace.image::<u16, C3>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            color::uyvp_10u_to_rgb_u16_c3(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                color_space,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
