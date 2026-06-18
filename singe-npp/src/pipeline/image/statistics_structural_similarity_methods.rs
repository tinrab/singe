use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{ImageStatistic, StructuralSimilarityImage},
};

#[path = "statistics_multiscale_structural_similarity_methods.rs"]
mod multiscale_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: StructuralSimilarityImage<T, L>,
{
    pub fn structural_similarity_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()> {
        <Self as StructuralSimilarityImage<T, L>>::structural_similarity(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    pub fn structural_similarity(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f32>> {
        let mut output = DeviceMemory::<f32>::create(
            <Self as StructuralSimilarityImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as StructuralSimilarityImage<T, L>>::structural_similarity(
                self.stream_context,
                &source,
                other,
                &mut output,
            )?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}
