use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::super::{ImageBacking, ImagePipeline, filtering::*};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ScharrBorderFilterImage<T, L>,
{
    pub fn filter_scharr_horizontal_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.scharr_border_filter(
            source_offset,
            border_type,
            <Self as ScharrBorderFilterImage<T, L>>::filter_scharr_horizontal_border_image,
        )
    }

    pub fn filter_scharr_vertical_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.scharr_border_filter(
            source_offset,
            border_type,
            <Self as ScharrBorderFilterImage<T, L>>::filter_scharr_vertical_border_image,
        )
    }

    fn scharr_border_filter(
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

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ScharrFilterImage<T, L>,
{
    pub fn filter_scharr_horizontal(self) -> Result<Self> {
        self.scharr_filter(<Self as ScharrFilterImage<T, L>>::filter_scharr_horizontal_image)
    }

    pub fn filter_scharr_vertical(self) -> Result<Self> {
        self.scharr_filter(<Self as ScharrFilterImage<T, L>>::filter_scharr_vertical_image)
    }

    fn scharr_filter(
        self,
        filter: fn(&StreamContext, &ImageView<'_, T, L>, &mut ImageViewMut<'_, T, L>) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
