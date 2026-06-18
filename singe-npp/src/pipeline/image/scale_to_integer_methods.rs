use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::scale_dispatch::{ScaleToI32Image, ScaleToU16Image};
use super::super::{ImageBacking, ImagePipeline};

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Self: ScaleToI32Image<L>,
{
    pub fn scale_to_i32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, i32, L>,
    ) -> Result<()> {
        <Self as ScaleToI32Image<L>>::scale_to_i32_image(stream_context, source, destination)
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<i32, L>,
    Self: ScaleToI32Image<L>,
{
    pub fn scale_to_i32(self) -> Result<ImagePipeline<'a, i32, L>> {
        let mut destination = self.workspace.image::<i32, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleToI32Image<L>>::scale_to_i32_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Self: ScaleToU16Image<L>,
{
    pub fn scale_to_u16_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, u16, L>,
    ) -> Result<()> {
        <Self as ScaleToU16Image<L>>::scale_to_u16_image(stream_context, source, destination)
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u16, L>,
    Self: ScaleToU16Image<L>,
{
    pub fn scale_to_u16(self) -> Result<ImagePipeline<'a, u16, L>> {
        let mut destination = self.workspace.image::<u16, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ScaleToU16Image<L>>::scale_to_u16_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
