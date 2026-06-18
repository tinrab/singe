use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{statistics, view::ImageView},
};

use super::ImagePipeline;

pub use self::outputs::{
    ContiguousImage, ImageHistograms, ImageIndexedMinMax, ImageIndexedStatistic,
    ImageMeanStandardDeviation, ImageMinMax, ImageSquaredIntegral, ImageStatistic,
};
pub(super) use super::statistics_dispatch::*;

#[path = "statistics_outputs.rs"]
mod outputs;

impl<'a, T, L> ImagePipeline<'a, T, L> {
    pub fn even_levels_i32(levels: &mut [i32], lower_level: i32, upper_level: i32) -> Result<()> {
        statistics::even_levels_i32(levels, lower_level, upper_level)
    }

    pub fn even_levels<U: statistics::EvenLevels>(
        levels: &mut [U],
        lower_level: U,
        upper_level: U,
    ) -> Result<()> {
        statistics::even_levels(levels, lower_level, upper_level)
    }
}

pub(super) type BatchQualityMetric<T, L> = for<'source_0, 'source_1> fn(
    &StreamContext,
    &[ImageView<'source_0, T, L>],
    &[ImageView<'source_1, T, L>],
    &mut DeviceMemory<f32>,
) -> Result<()>;
