use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::super::super::morphology_traits::Morphology3x3BorderImage;
use super::{ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn dilate_3x3_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: Morphology3x3BorderImage<T, L>,
    {
        <Self as Morphology3x3BorderImage<T, L>>::dilate_3x3_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    pub fn erode_3x3_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: Morphology3x3BorderImage<T, L>,
    {
        <Self as Morphology3x3BorderImage<T, L>>::erode_3x3_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: Morphology3x3BorderImage<T, L>,
{
    pub fn dilate_3x3_border(self, source_offset: Point, border_type: BorderType) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as Morphology3x3BorderImage<T, L>>::dilate_3x3_border_image(
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

    pub fn erode_3x3_border(self, source_offset: Point, border_type: BorderType) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as Morphology3x3BorderImage<T, L>>::erode_3x3_border_image(
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
