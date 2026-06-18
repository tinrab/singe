use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
};

use super::super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedNormDiffImage},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedNormDiffImage<T, L>,
{
    pub fn norm_diff_l2_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedNormDiffImage<T, L>>::norm_diff_l2_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    pub fn norm_diff_l2_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_diff(
            other,
            mask,
            <Self as MaskedNormDiffImage<T, L>>::norm_diff_l2_masked,
        )
    }
}
