use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
};

use super::super::super::{ImagePipeline, statistics::MaskedChannelNormDiffImage};

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: MaskedChannelNormDiffImage<T, C3>,
{
    pub fn norm_diff_inf_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_inf_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }

    pub fn norm_diff_l1_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l1_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }

    pub fn norm_diff_l2_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l2_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }
}
