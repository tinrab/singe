use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
};

use super::super::super::super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedChannelScalarStatisticImage},
};

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: MaskedChannelScalarStatisticImage<T, C3>,
{
    pub fn norm_l2_channel_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelScalarStatisticImage<T, C3>>::norm_l2_channel_masked(
            stream_context,
            source,
            mask,
            channel,
            output,
        )
    }

    pub fn norm_l2_channel_masked(
        self,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_scalar_norm(
            mask,
            channel,
            <Self as MaskedChannelScalarStatisticImage<T, C3>>::norm_l2_channel_masked,
        )
    }
}
