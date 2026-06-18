use singe_cuda::memory::DeviceMemory;

use crate::{context::StreamContext, error::Result, image::view::ImageView};

pub trait ErrorMetricImage<T, L> {
    fn maximum_error(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn average_error(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

#[macro_use]
#[path = "statistics_quality_norm_dispatch.rs"]
mod norm_dispatch;

pub use norm_dispatch::{
    MaskedChannelNormDiffImage, MaskedChannelNormRelativeImage, MaskedNormDiffImage,
    MaskedNormRelativeImage,
};

pub trait QualityMetricImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn mean_squared_error(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()>;

    fn peak_signal_to_noise_ratio(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()>;
}

pub trait StructuralSimilarityImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn structural_similarity(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()>;
}

pub trait MultiscaleStructuralSimilarityImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn multiscale_structural_similarity(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()>;
}

pub trait WeightedMultiscaleStructuralSimilarityImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn weighted_multiscale_structural_similarity(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()>;
}

#[macro_use]
#[path = "statistics_quality_dispatch_macros.rs"]
mod macros;
