#[path = "filtering_basic_border_into_methods.rs"]
mod into_methods;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, MaskSize, Point},
};

use super::{ImageBacking, ImagePipeline, filtering::HighLowGaussBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: HighLowGaussBorderFilterImage<T, L>,
{
    pub fn filter_high_pass_border_partial(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.high_low_gauss_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as HighLowGaussBorderFilterImage<T, L>>::filter_high_pass_border_image,
        )
    }

    pub fn filter_low_pass_border_partial(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.high_low_gauss_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as HighLowGaussBorderFilterImage<T, L>>::filter_low_pass_border_image,
        )
    }

    pub fn filter_gauss_border_partial(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.high_low_gauss_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as HighLowGaussBorderFilterImage<T, L>>::filter_gauss_border_image,
        )
    }

    fn high_low_gauss_border_filter(
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
