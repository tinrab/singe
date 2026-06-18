use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

pub trait CountInRangeImage<T, L> {
    type Bounds: Copy;

    const OUTPUT_CHANNELS: usize;

    fn count_in_range(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        counts: &mut DeviceMemory<i32>,
        lower_bound: Self::Bounds,
        upper_bound: Self::Bounds,
    ) -> Result<()>;
}

pub trait DotProductImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn dot_product(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

pub trait EveryStatisticImage<T, L> {
    fn max_every(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn min_every(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_every_statistic_image {
    ($ty:ty, $layout:ty, $max_every:path, $min_every:path) => {
        impl<'a> EveryStatisticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn max_every(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $max_every(stream_context, source, source_destination)
            }

            fn min_every(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $min_every(stream_context, source, source_destination)
            }
        }
    };
}

macro_rules! impl_dot_product_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $dot_product:path) => {
        impl<'a> DotProductImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn dot_product(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $dot_product(stream_context, source_1, source_2, output)
            }
        }
    };
}

macro_rules! impl_count_in_range_image {
    (
        $ty:ty,
        $layout:ty,
        $bounds:ty,
        $output_channels:expr,
        $count_in_range:path
    ) => {
        impl<'a> CountInRangeImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Bounds = $bounds;

            const OUTPUT_CHANNELS: usize = $output_channels;

            fn count_in_range(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                counts: &mut DeviceMemory<i32>,
                lower_bound: Self::Bounds,
                upper_bound: Self::Bounds,
            ) -> Result<()> {
                $count_in_range(stream_context, source, counts, lower_bound, upper_bound)
            }
        }
    };
}
