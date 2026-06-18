use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
};

use super::super::super::super::{ImagePipeline, statistics::MaskedChannelNormRelativeImage};

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: MaskedChannelNormRelativeImage<T, C3>,
{
    pub fn norm_relative_inf_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormRelativeImage<T, C3>>::norm_relative_inf_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }

    pub fn norm_relative_l1_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormRelativeImage<T, C3>>::norm_relative_l1_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }

    pub fn norm_relative_l2_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormRelativeImage<T, C3>>::norm_relative_l2_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }
}
