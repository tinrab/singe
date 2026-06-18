use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{Point, Size},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_neighborhood_border_methods.rs"]
mod border_methods;
#[path = "filtering_neighborhood_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MedianFilterImage<T, L>,
{
    pub fn filter_median(self, mask_size: Size, anchor: Point) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as MedianFilterImage<T, L>>::filter_median_image(
                self.stream_context,
                &source,
                &mut destination_view,
                mask_size,
                anchor,
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
    Self: NeighborhoodFilterImage<T, L>,
{
    pub fn filter_max(self, mask_size: Size, anchor: Point) -> Result<Self> {
        self.neighborhood_filter(
            mask_size,
            anchor,
            <Self as NeighborhoodFilterImage<T, L>>::filter_max_image,
        )
    }

    pub fn filter_min(self, mask_size: Size, anchor: Point) -> Result<Self> {
        self.neighborhood_filter(
            mask_size,
            anchor,
            <Self as NeighborhoodFilterImage<T, L>>::filter_min_image,
        )
    }

    fn neighborhood_filter(
        self,
        mask_size: Size,
        anchor: Point,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            Size,
            Point,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(
                self.stream_context,
                &source,
                &mut destination_view,
                mask_size,
                anchor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
