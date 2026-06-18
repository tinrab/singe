use singe_cuda::memory::DeviceMemory;

#[path = "statistics_pairwise_error_metric_methods.rs"]
mod error_metric_methods;
#[path = "statistics_pairwise_quality_metric_methods.rs"]
mod quality_metric_methods;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{DotProductImage, ImageStatistic},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DotProductImage<T, L>,
{
    pub fn dot_product_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as DotProductImage<T, L>>::dot_product(stream_context, source_1, source_2, output)
    }

    pub fn dot_product(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f64>> {
        let mut output =
            DeviceMemory::<f64>::create(<Self as DotProductImage<T, L>>::OUTPUT_CHANNELS)?;
        {
            let source = self.view()?;
            <Self as DotProductImage<T, L>>::dot_product(
                self.stream_context,
                &source,
                other,
                &mut output,
            )?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}
