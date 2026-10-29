use singe_cuda::view::DeviceOutputView;

pub use self::outputs::{
    ContiguousImage, ImageHistograms, ImageIndexedMinMax, ImageIndexedStatistic,
    ImageMeanStandardDeviation, ImageMinMax, ImageSquaredIntegral, ImageStatistic,
};
pub use crate::pipeline::image::statistics_dispatch::*;
use crate::{
    context::StreamContext,
    error::Result,
    image::{statistics, view::ImageView},
    pipeline::image::ImagePipeline,
};

mod outputs;

impl<T, L> ImagePipeline<'_, T, L> {
    /// Fills a host slice with evenly distributed integer histogram boundaries.
    ///
    /// `lower_level` and `upper_level` specify the lowest and greatest boundaries. The
    /// provided implementation uses `i32` and calls the native host routine directly,
    /// without a CUDA stream. NPP requires at least two entries; the wrapper forwards
    /// the count and bounds without additional validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice length cannot fit `i32` or NPP rejects the level
    /// count or boundary parameters.
    pub fn even_levels_i32(levels: &mut [i32], lower_level: i32, upper_level: i32) -> Result<()> {
        statistics::even_levels_i32(levels, lower_level, upper_level)
    }

    /// Fills a host slice with evenly distributed integer histogram boundaries.
    ///
    /// `lower_level` and `upper_level` specify the lowest and greatest boundaries. The
    /// provided implementation uses `i32` and calls the native host routine directly,
    /// without a CUDA stream. NPP requires at least two entries; the wrapper forwards
    /// the count and bounds without additional validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice length cannot fit `i32` or NPP rejects the level
    /// count or boundary parameters.
    pub fn even_levels<U: statistics::EvenLevels>(
        levels: &mut [U],
        lower_level: U,
        upper_level: U,
    ) -> Result<()> {
        statistics::even_levels(levels, lower_level, upper_level)
    }
}

pub type BatchQualityMetric<T, L> = for<'source_0, 'source_1> unsafe fn(
    &StreamContext,
    &[ImageView<'source_0, T, L>],
    &[ImageView<'source_1, T, L>],
    &mut DeviceOutputView<'_, f32>,
) -> Result<()>;
