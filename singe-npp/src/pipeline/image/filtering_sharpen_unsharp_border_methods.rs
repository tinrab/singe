use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::super::{
    ImageBacking, ImagePipeline,
    filtering::{DirectionalBorderFilterImage, UnsharpBorderFilterImage},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: DirectionalBorderFilterImage<T, L>,
{
    pub fn filter_sharpen_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as DirectionalBorderFilterImage<T, L>>::filter_sharpen_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: UnsharpBorderFilterImage<T, L>,
{
    pub fn filter_unsharp_border(
        self,
        source_offset: Point,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as UnsharpBorderFilterImage<T, L>>::filter_unsharp_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                radius,
                sigma,
                weight,
                threshold,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
