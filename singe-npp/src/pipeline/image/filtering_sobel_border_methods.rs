use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, MaskSize, Point},
};

use super::super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_sobel_border_second_methods.rs"]
mod second_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: SobelExtendedBorderFilterImage<T, L>,
{
    pub fn filter_sobel_horizontal_mask_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<
                T,
                L,
            >>::filter_sobel_horizontal_mask_border_image,
        )
    }

    pub fn filter_sobel_vertical_mask_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<T, L>>::filter_sobel_vertical_mask_border_image,
        )
    }

    pub(super) fn sobel_extended_border_filter(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, T, L>,
            MaskSize,
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
