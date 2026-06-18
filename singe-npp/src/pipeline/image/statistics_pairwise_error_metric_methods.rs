use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::super::{
    ImagePipeline,
    statistics::{ErrorMetricImage, ImageStatistic},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ErrorMetricImage<T, L>,
{
    pub fn maximum_error_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ErrorMetricImage<T, L>>::maximum_error(stream_context, source_1, source_2, output)
    }

    pub fn average_error_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ErrorMetricImage<T, L>>::average_error(stream_context, source_1, source_2, output)
    }

    pub fn maximum_error(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f64>> {
        self.error_metric(other, <Self as ErrorMetricImage<T, L>>::maximum_error)
    }

    pub fn average_error(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f64>> {
        self.error_metric(other, <Self as ErrorMetricImage<T, L>>::average_error)
    }

    fn error_metric(
        self,
        other: &ImageView<'_, T, L>,
        metric: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut DeviceMemory<f64>,
        ) -> Result<()>,
    ) -> Result<ImageStatistic<f64>> {
        let mut output = DeviceMemory::<f64>::create(1)?;
        {
            let source = self.view()?;
            metric(self.stream_context, &source, other, &mut output)?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}
