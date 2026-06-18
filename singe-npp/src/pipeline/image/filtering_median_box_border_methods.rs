use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::super::super::filtering::{BoxBorderFilterImage, MedianBorderFilterImage};
use super::{ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MedianBorderFilterImage<T, L>,
{
    pub fn filter_median_border(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as MedianBorderFilterImage<T, L>>::filter_median_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                anchor,
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
    Self: BoxBorderFilterImage<T, L>,
{
    pub fn filter_box_border(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as BoxBorderFilterImage<T, L>>::filter_box_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                anchor,
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
