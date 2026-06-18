use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::MaskSize,
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_typed_edge_extended_methods.rs"]
mod extended_methods;
#[path = "filtering_typed_edge_border_methods.rs"]
mod filtering_typed_edge_border_methods;
#[path = "filtering_typed_edge_scharr_methods.rs"]
mod scharr_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn filter_sobel_horizontal_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_horizontal_to_image,
        )
    }

    pub fn filter_sobel_vertical_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_vertical_to_image,
        )
    }

    pub(super) fn typed_edge_directional_filter<D, M>(
        self,
        filter: fn(&StreamContext, &ImageView<'_, T, L>, &mut ImageViewMut<'_, D, M>) -> Result<()>,
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
            filter(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub(super) fn typed_masked_edge_directional_filter<D, M>(
        self,
        mask_size: MaskSize,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, D, M>,
            MaskSize,
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
                &mut destination_view,
                mask_size,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
