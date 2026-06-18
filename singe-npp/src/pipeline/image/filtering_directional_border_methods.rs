use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_directional_border_into_methods.rs"]
mod into_methods;
#[path = "filtering_roberts_border_methods.rs"]
mod roberts_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: EdgeDirectionalBorderFilterImage<T, L>,
{
    pub fn filter_prewitt_horizontal_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_horizontal_border_image,
        )
    }

    pub fn filter_prewitt_vertical_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_vertical_border_image,
        )
    }

    pub fn filter_sobel_horizontal_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_horizontal_border_image,
        )
    }

    pub fn filter_sobel_vertical_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_vertical_border_image,
        )
    }

    pub(super) fn edge_directional_border_filter(
        self,
        source_offset: Point,
        border_type: BorderType,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, T, L>,
            BorderType,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
