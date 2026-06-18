use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::MaskSize,
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_scharr_methods.rs"]
mod scharr_methods;
#[path = "filtering_sobel_border_methods.rs"]
mod sobel_border_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: SobelExtendedFilterImage<T, L>,
{
    pub fn filter_sobel_horizontal_mask(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_horizontal_mask_image,
        )
    }

    pub fn filter_sobel_vertical_mask(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_vertical_mask_image,
        )
    }

    pub fn filter_sobel_horizontal_second(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_horizontal_second_image,
        )
    }

    pub fn filter_sobel_vertical_second(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_vertical_second_image,
        )
    }

    pub fn filter_sobel_cross(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_cross_image,
        )
    }

    fn sobel_extended_filter(
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
