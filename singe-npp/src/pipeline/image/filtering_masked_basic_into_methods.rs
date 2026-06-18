use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::MaskSize,
};

use super::super::{ImagePipeline, filtering::MaskedKernelFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedKernelFilterImage<T, L>,
{
    pub fn filter_high_pass_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_high_pass_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    pub fn filter_low_pass_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_low_pass_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    pub fn filter_gauss_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_gauss_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    pub fn filter_laplace_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_laplace_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }
}
