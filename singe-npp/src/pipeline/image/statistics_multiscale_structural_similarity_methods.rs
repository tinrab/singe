use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::super::{
    ImagePipeline,
    statistics::{
        ImageStatistic, MultiscaleStructuralSimilarityImage,
        WeightedMultiscaleStructuralSimilarityImage,
    },
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MultiscaleStructuralSimilarityImage<T, L>,
{
    pub fn multiscale_structural_similarity_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()> {
        <Self as MultiscaleStructuralSimilarityImage<T, L>>::multiscale_structural_similarity(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    pub fn multiscale_structural_similarity(
        self,
        other: &ImageView<'_, T, L>,
    ) -> Result<ImageStatistic<f32>> {
        let mut output = DeviceMemory::<f32>::create(
            <Self as MultiscaleStructuralSimilarityImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as MultiscaleStructuralSimilarityImage<T, L>>::multiscale_structural_similarity(
                self.stream_context,
                &source,
                other,
                &mut output,
            )?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WeightedMultiscaleStructuralSimilarityImage<T, L>,
{
    pub fn weighted_multiscale_structural_similarity_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()> {
        <Self as WeightedMultiscaleStructuralSimilarityImage<
            T,
            L,
        >>::weighted_multiscale_structural_similarity(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    pub fn weighted_multiscale_structural_similarity(
        self,
        other: &ImageView<'_, T, L>,
    ) -> Result<ImageStatistic<f32>> {
        let mut output = DeviceMemory::<f32>::create(
            <Self as WeightedMultiscaleStructuralSimilarityImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as WeightedMultiscaleStructuralSimilarityImage<
                T,
                L,
            >>::weighted_multiscale_structural_similarity(
                self.stream_context,
                &source,
                other,
                &mut output,
            )?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}
