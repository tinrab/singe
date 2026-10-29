use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait CountInRangeImage<T, L> {
    type Bounds: Copy;

    const OUTPUT_CHANNELS: usize;

    fn count_in_range(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        counts: &mut DeviceOutputView<'_, i32>,
        lower_bound: Self::Bounds,
        upper_bound: Self::Bounds,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait DotProductImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn dot_product(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait EveryStatisticImage<T, L> {
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
        unsafe impl<'a> EveryStatisticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn max_every(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $max_every(stream_context, source, source_destination)
                })
            }

            fn min_every(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $min_every(stream_context, source, source_destination)
                })
            }
        }
    };
}

macro_rules! impl_dot_product_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $dot_product:path) => {
        unsafe impl<'a> DotProductImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn dot_product(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $dot_product(stream_context, source_1, source_2, output) }
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
        unsafe impl<'a> CountInRangeImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Bounds = $bounds;

            const OUTPUT_CHANNELS: usize = $output_channels;

            fn count_in_range(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                counts: &mut DeviceOutputView<'_, i32>,
                lower_bound: Self::Bounds,
                upper_bound: Self::Bounds,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(counts)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $count_in_range(stream_context, source, counts, lower_bound, upper_bound) }
            }
        }
    };
}
