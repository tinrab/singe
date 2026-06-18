use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
};

use super::{
    super::statistics::{
        ImageMeanStandardDeviation, ImageStatistic, MaskedMeanStandardDeviationImage,
    },
    ImagePipeline,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedMeanStandardDeviationImage<T, L>,
{
    pub fn mean_standard_deviation_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedMeanStandardDeviationImage<T, L>>::mean_standard_deviation_masked(
            stream_context,
            source,
            mask,
            mean,
            standard_deviation,
        )
    }

    pub fn mean_standard_deviation_masked(
        self,
        mask: &MaskView<'_>,
    ) -> Result<ImageMeanStandardDeviation<f64>> {
        let mut mean = DeviceMemory::<f64>::create(
            <Self as MaskedMeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        let mut standard_deviation = DeviceMemory::<f64>::create(
            <Self as MaskedMeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as MaskedMeanStandardDeviationImage<T, L>>::mean_standard_deviation_masked(
                self.stream_context,
                &source,
                mask,
                &mut mean,
                &mut standard_deviation,
            )?;
        }
        Ok(ImageMeanStandardDeviation {
            mean: ImageStatistic::from_values(mean),
            standard_deviation: ImageStatistic::from_values(standard_deviation),
        })
    }
}
