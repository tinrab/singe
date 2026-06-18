use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
};

use super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedNormDiffImage},
};

#[path = "statistics_masked_norm_diff_l2_methods.rs"]
mod l2_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedNormDiffImage<T, L>,
{
    pub fn norm_diff_inf_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedNormDiffImage<T, L>>::norm_diff_inf_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    pub fn norm_diff_l1_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedNormDiffImage<T, L>>::norm_diff_l1_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    pub fn norm_diff_inf_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_diff(
            other,
            mask,
            <Self as MaskedNormDiffImage<T, L>>::norm_diff_inf_masked,
        )
    }

    pub fn norm_diff_l1_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_diff(
            other,
            mask,
            <Self as MaskedNormDiffImage<T, L>>::norm_diff_l1_masked,
        )
    }

    pub(super) fn masked_norm_diff(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        metric: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &MaskView<'_>,
            &mut DeviceMemory<f64>,
        ) -> Result<()>,
    ) -> Result<ImageStatistic<f64>> {
        let mut output = DeviceMemory::<f64>::create(1)?;
        {
            let source = self.view()?;
            metric(self.stream_context, &source, other, mask, &mut output)?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}
