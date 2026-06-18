use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::super::{ImageBacking, ImagePipeline, morphology_traits::MaskMorphologyBorderImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MaskMorphologyBorderImage<T, L>,
{
    pub fn dilate_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as MaskMorphologyBorderImage<T, L>>::dilate_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask,
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

    pub fn erode_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as MaskMorphologyBorderImage<T, L>>::erode_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask,
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
