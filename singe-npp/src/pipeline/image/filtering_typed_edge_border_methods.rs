use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, MaskSize, Point},
};

use super::super::{ImageBacking, ImagePipeline};

#[path = "filtering_typed_edge_extended_border_methods.rs"]
mod extended_methods;
#[path = "filtering_typed_edge_scharr_border_methods.rs"]
mod scharr_methods;
#[path = "filtering_typed_edge_sobel_border_methods.rs"]
mod sobel_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub(super) fn typed_edge_directional_border_filter<D, M>(
        self,
        source_offset: Point,
        border_type: BorderType,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, D, M>,
            BorderType,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
    {
        let mut destination = self.workspace.image::<D, M>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                border_type,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub(super) fn typed_masked_edge_directional_border_filter<D, M>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, D, M>,
            MaskSize,
            BorderType,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
    {
        let mut destination = self.workspace.image::<D, M>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                border_type,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
