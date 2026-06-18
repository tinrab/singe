use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
};

use super::super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedChannelNormDiffImage},
};

#[path = "statistics_masked_channel_norm_diff_into_methods.rs"]
mod norm_diff_into_methods;
#[path = "statistics_masked_channel_relative_norm_methods.rs"]
mod relative_norm_methods;
#[path = "statistics_masked_channel_scalar_norm_methods.rs"]
mod scalar_norm_methods;

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: MaskedChannelNormDiffImage<T, C3>,
{
    pub fn norm_diff_inf_channel_masked(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_norm_diff(
            other,
            mask,
            channel,
            <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_inf_channel_masked,
        )
    }

    pub fn norm_diff_l1_channel_masked(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_norm_diff(
            other,
            mask,
            channel,
            <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l1_channel_masked,
        )
    }

    pub fn norm_diff_l2_channel_masked(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_norm_diff(
            other,
            mask,
            channel,
            <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l2_channel_masked,
        )
    }

    fn masked_channel_norm_diff(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        metric: fn(
            &StreamContext,
            &ImageView<'_, T, C3>,
            &ImageView<'_, T, C3>,
            &MaskView<'_>,
            usize,
            &mut DeviceMemory<f64>,
        ) -> Result<()>,
    ) -> Result<ImageStatistic<f64>> {
        let mut output = DeviceMemory::<f64>::create(1)?;
        {
            let source = self.view()?;
            metric(
                self.stream_context,
                &source,
                other,
                mask,
                channel,
                &mut output,
            )?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}
