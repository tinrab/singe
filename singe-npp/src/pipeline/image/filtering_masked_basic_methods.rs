use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::MaskSize,
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_sharpen_methods.rs"]
mod sharpen_methods;

#[path = "filtering_masked_basic_into_methods.rs"]
mod into_methods;
impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MaskedKernelFilterImage<T, L>,
{
    pub fn filter_high_pass(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_high_pass_image,
        )
    }

    pub fn filter_low_pass(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_low_pass_image,
        )
    }

    pub fn filter_gauss(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_gauss_image,
        )
    }

    pub fn filter_laplace(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_laplace_image,
        )
    }

    fn masked_kernel_filter(
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
