use singe_cuda::memory::DeviceMemory;

use crate::{context::StreamContext, error::Result, image::view::ImageView};

pub(super) use paired_indexed_extremum::{
    ChannelPairIndexedExtremumStatisticImage, MaskedChannelPairIndexedExtremumStatisticImage,
    MaskedPairIndexedExtremumStatisticImage, PairIndexedExtremumStatisticImage,
};

#[macro_use]
#[path = "statistics_scalar_dispatch.rs"]
mod scalar_dispatch;

#[macro_use]
#[path = "statistics_quality_dispatch.rs"]
mod quality_dispatch;

#[path = "statistics_histogram_dispatch.rs"]
mod histogram_dispatch;

#[macro_use]
#[path = "statistics_mean_dispatch.rs"]
mod mean_dispatch;

pub(super) use histogram_dispatch::{
    HistogramEvenChannelsImage, HistogramEvenImage, HistogramRangeChannelsImage,
    HistogramRangeImage,
};
pub(super) use mean_dispatch::{
    ChannelMeanStandardDeviationImage, MaskedChannelMeanStandardDeviationImage,
    MaskedMeanStandardDeviationImage, MeanStandardDeviationImage,
};
pub(super) use quality_dispatch::{
    ErrorMetricImage, MaskedChannelNormDiffImage, MaskedChannelNormRelativeImage,
    MaskedNormDiffImage, MaskedNormRelativeImage, MultiscaleStructuralSimilarityImage,
    QualityMetricImage, StructuralSimilarityImage, WeightedMultiscaleStructuralSimilarityImage,
};
pub(super) use scalar_dispatch::{
    MaskedChannelScalarStatisticImage, MaskedScalarStatisticImage, ScalarStatisticImage,
};

#[macro_use]
#[path = "statistics_metric_dispatch.rs"]
mod metric_dispatch;

pub(super) use metric_dispatch::{CountInRangeImage, DotProductImage, EveryStatisticImage};

#[path = "statistics_paired_indexed_extremum_dispatch.rs"]
mod paired_indexed_extremum;

pub trait ExtremumStatisticImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn min(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
    ) -> Result<()>;

    fn max(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
    ) -> Result<()>;

    fn min_max(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
    ) -> Result<()>;
}

pub trait IndexedExtremumStatisticImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn min_index(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
        index_x: &mut DeviceMemory<i32>,
        index_y: &mut DeviceMemory<i32>,
    ) -> Result<()>;

    fn max_index(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
        index_x: &mut DeviceMemory<i32>,
        index_y: &mut DeviceMemory<i32>,
    ) -> Result<()>;
}

#[macro_use]
#[path = "statistics_histogram_dispatch_macros.rs"]
mod histogram_dispatch_macros;

#[macro_use]
#[path = "statistics_indexed_extremum_dispatch_macros.rs"]
mod indexed_extremum_dispatch_macros;

macro_rules! impl_extremum_statistic_image {
    (
        $ty:ty,
        $layout:ty,
        $min:path,
        $max:path,
        $min_max:path
    ) => {
        impl<'a> ExtremumStatisticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn min(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<$ty>,
            ) -> Result<()> {
                $min(stream_context, source, output)
            }

            fn max(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<$ty>,
            ) -> Result<()> {
                $max(stream_context, source, output)
            }

            fn min_max(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                min: &mut DeviceMemory<$ty>,
                max: &mut DeviceMemory<$ty>,
            ) -> Result<()> {
                $min_max(stream_context, source, min, max)
            }
        }
    };
}

#[path = "statistics_histogram_dot_dispatch_impls.rs"]
mod histogram_dot_dispatch_impls;

#[path = "statistics_quality_dispatch_impls.rs"]
mod quality_dispatch_impls;

#[path = "statistics_extremum_dispatch_impls.rs"]
mod extremum_dispatch_impls;

#[path = "statistics_indexed_extremum_dispatch_impls.rs"]
mod indexed_extremum_dispatch_impls;

#[path = "statistics_every_dispatch_impls.rs"]
mod every_dispatch_impls;

#[path = "statistics_dispatch_impls.rs"]
mod impls;
#[path = "statistics_mean_dispatch_impls.rs"]
mod mean_dispatch_impls;
