use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::morphology_traits::Morphology3x3Image;
use super::{ImageBacking, ImagePipeline};

#[path = "morphology_3x3_border_methods.rs"]
mod border_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn dilate_3x3_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: Morphology3x3Image<T, L>,
    {
        <Self as Morphology3x3Image<T, L>>::dilate_3x3_image(stream_context, source, destination)
    }

    pub fn erode_3x3_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: Morphology3x3Image<T, L>,
    {
        <Self as Morphology3x3Image<T, L>>::erode_3x3_image(stream_context, source, destination)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: Morphology3x3Image<T, L>,
{
    pub fn dilate_3x3(self) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as Morphology3x3Image<T, L>>::dilate_3x3_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub fn erode_3x3(self) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as Morphology3x3Image<T, L>>::erode_3x3_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
