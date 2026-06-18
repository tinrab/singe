use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{MaskSize, Point, Size},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_basic_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: HighLowGaussFilterImage<T, L>,
{
    pub fn filter_high_pass_partial(self, mask_size: MaskSize) -> Result<Self> {
        self.high_low_gauss_filter(
            mask_size,
            <Self as HighLowGaussFilterImage<T, L>>::filter_high_pass_image,
        )
    }

    pub fn filter_low_pass_partial(self, mask_size: MaskSize) -> Result<Self> {
        self.high_low_gauss_filter(
            mask_size,
            <Self as HighLowGaussFilterImage<T, L>>::filter_low_pass_image,
        )
    }

    pub fn filter_gauss_partial(self, mask_size: MaskSize) -> Result<Self> {
        self.high_low_gauss_filter(
            mask_size,
            <Self as HighLowGaussFilterImage<T, L>>::filter_gauss_image,
        )
    }

    fn high_low_gauss_filter(
        self,
        mask_size: MaskSize,
        filter: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut ImageViewMut<'_, T, L>,
            MaskSize,
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
    Self: BoxFilterImage<T, L>,
{
    pub fn filter_box(self, mask_size: Size, anchor: Point) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as BoxFilterImage<T, L>>::filter_box_image(
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
