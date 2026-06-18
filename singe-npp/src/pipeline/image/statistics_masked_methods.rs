use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
};

use super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedNormRelativeImage},
};

#[path = "statistics_masked_mean_standard_deviation_methods.rs"]
mod mean_standard_deviation_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedNormRelativeImage<T, L>,
{
    pub fn norm_relative_inf_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedNormRelativeImage<T, L>>::norm_relative_inf_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    pub fn norm_relative_l1_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l1_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    pub fn norm_relative_l2_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l2_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    pub fn norm_relative_inf_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_relative(
            other,
            mask,
            <Self as MaskedNormRelativeImage<T, L>>::norm_relative_inf_masked,
        )
    }

    pub fn norm_relative_l1_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_relative(
            other,
            mask,
            <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l1_masked,
        )
    }

    pub fn norm_relative_l2_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_relative(
            other,
            mask,
            <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l2_masked,
        )
    }

    fn masked_norm_relative(
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
