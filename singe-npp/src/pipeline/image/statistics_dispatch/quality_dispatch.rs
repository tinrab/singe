use singe_cuda::view::DeviceOutputView;

use crate::{context::StreamContext, error::Result, image::view::ImageView};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait ErrorMetricImage<T, L> {
    fn maximum_error(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn average_error(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

#[macro_use]
mod norm_dispatch;

pub use norm_dispatch::{
    MaskedChannelNormDiffImage, MaskedChannelNormRelativeImage, MaskedNormDiffImage,
    MaskedNormRelativeImage,
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait QualityMetricImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn mean_squared_error(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()>;

    fn peak_signal_to_noise_ratio(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait StructuralSimilarityImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn structural_similarity(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MultiscaleStructuralSimilarityImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn multiscale_structural_similarity(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait WeightedMultiscaleStructuralSimilarityImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn weighted_multiscale_structural_similarity(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()>;
}

#[macro_use]
mod macros;
