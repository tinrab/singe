use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::super::filtering::NeighborhoodBorderFilterImage;
use super::{ImageBacking, ImagePipeline};

#[path = "filtering_median_box_border_methods.rs"]
mod median_box_border_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: NeighborhoodBorderFilterImage<T, L>,
{
    pub fn filter_max_border(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.neighborhood_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as NeighborhoodBorderFilterImage<T, L>>::filter_max_border_image,
        )
    }

    pub fn filter_min_border(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.neighborhood_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as NeighborhoodBorderFilterImage<T, L>>::filter_min_border_image,
        )
    }

    fn neighborhood_border_filter(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, T, L>,
            Size,
            Point,
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
