use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
};

use super::super::super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedScalarStatisticImage},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedScalarStatisticImage<T, L>,
{
    pub fn norm_inf_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::norm_inf_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

    pub fn norm_l1_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::norm_l1_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

    pub fn norm_l2_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::norm_l2_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

    pub fn norm_inf_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::norm_inf_masked,
        )
    }

    pub fn norm_l1_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::norm_l1_masked,
        )
    }

    pub fn norm_l2_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::norm_l2_masked,
        )
    }
}
