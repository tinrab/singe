use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C3, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::{ColorSpace, Point},
};

use super::super::{ImageBacking, ImagePipeline};

impl<'a> ImagePipeline<'a, u16, AC4>
where
    Workspace: ImageAllocator<u8, C3>,
{
    pub fn rgba_to_uyvp_10u_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u16, AC4>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C3>,
        color_space: ColorSpace,
    ) -> Result<()> {
        color::rgba_to_uyvp_10u_u16_ac4(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }

    pub fn rgba_to_uyvp_10u(
        self,
        source_offset: Point,
        color_space: ColorSpace,
    ) -> Result<ImagePipeline<'a, u8, C3>> {
        let mut destination = self.workspace.image::<u8, C3>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            color::rgba_to_uyvp_10u_u16_ac4(
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
    Workspace: ImageAllocator<u16, AC4>,
{
    pub fn uyvp_10u_to_rgba_u16_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u16, AC4>,
        color_space: ColorSpace,
        alpha: u16,
    ) -> Result<()> {
        color::uyvp_10u_to_rgb_u16_ac4(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
            alpha,
        )
    }

    pub fn uyvp_10u_to_rgba_u16(
        self,
        source_offset: Point,
        color_space: ColorSpace,
        alpha: u16,
    ) -> Result<ImagePipeline<'a, u16, AC4>> {
        let mut destination = self.workspace.image::<u16, AC4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            color::uyvp_10u_to_rgb_u16_ac4(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                color_space,
                alpha,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
