pub use paired_indexed_extremum::{
    ChannelPairIndexedExtremumStatisticImage, MaskedChannelPairIndexedExtremumStatisticImage,
    MaskedPairIndexedExtremumStatisticImage, PairIndexedExtremumStatisticImage,
};
use singe_cuda::view::{DeviceOutputView, DeviceRepr};

use crate::{context::StreamContext, error::Result, image::view::ImageView};

#[macro_use]
mod scalar_dispatch;

#[macro_use]
mod quality_dispatch;

mod histogram_dispatch;

#[macro_use]
mod mean_dispatch;

pub use histogram_dispatch::{
    HistogramEvenChannelsImage, HistogramEvenImage, HistogramRangeChannelsImage,
    HistogramRangeImage,
};
pub use mean_dispatch::{
    ChannelMeanStandardDeviationImage, MaskedChannelMeanStandardDeviationImage,
    MaskedMeanStandardDeviationImage, MeanStandardDeviationImage,
};
pub use quality_dispatch::{
    ErrorMetricImage, MaskedChannelNormDiffImage, MaskedChannelNormRelativeImage,
    MaskedNormDiffImage, MaskedNormRelativeImage, MultiscaleStructuralSimilarityImage,
    QualityMetricImage, StructuralSimilarityImage, WeightedMultiscaleStructuralSimilarityImage,
};
pub use scalar_dispatch::{
    MaskedChannelScalarStatisticImage, MaskedScalarStatisticImage, ScalarStatisticImage,
};

#[macro_use]
mod metric_dispatch;

pub use metric_dispatch::{CountInRangeImage, DotProductImage, EveryStatisticImage};

mod paired_indexed_extremum;

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait ExtremumStatisticImage<T: DeviceRepr, L> {
    const OUTPUT_CHANNELS: usize;

    fn min(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
    ) -> Result<()>;

    fn max(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
    ) -> Result<()>;

    fn min_max(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait IndexedExtremumStatisticImage<T: DeviceRepr, L> {
    const OUTPUT_CHANNELS: usize;

    fn min_index(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
        index_x: &mut DeviceOutputView<'_, i32>,
        index_y: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()>;

    fn max_index(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
        index_x: &mut DeviceOutputView<'_, i32>,
        index_y: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()>;
}

#[macro_use]
mod histogram_dispatch_macros;

#[macro_use]
mod indexed_extremum_dispatch_macros;

macro_rules! impl_extremum_statistic_image {
    (
        $ty:ty,
        $layout:ty,
        $min:path,
        $max:path,
        $min_max:path
    ) => {
        unsafe impl<'a> ExtremumStatisticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn min(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $min(stream_context, source, output) }
            }

            fn max(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $max(stream_context, source, output) }
            }

            fn min_max(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                min: &mut DeviceOutputView<'_, $ty>,
                max: &mut DeviceOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(min)?;
                stream_context.prepare_region(max)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $min_max(stream_context, source, min, max) }
            }
        }
    };
}

mod histogram_dot_dispatch_impls;

mod quality_dispatch_impls;

mod extremum_dispatch_impls;

mod indexed_extremum_dispatch_impls;

mod every_dispatch_impls;

mod impls;
mod mean_dispatch_impls;
