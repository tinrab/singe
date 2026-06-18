use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

pub use self::relative_norm::{MaskedChannelNormRelativeImage, MaskedNormRelativeImage};

#[path = "statistics_quality_relative_norm_dispatch.rs"]
mod relative_norm;

pub trait MaskedNormDiffImage<T, L> {
    fn norm_diff_inf_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_diff_l1_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_diff_l2_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

pub trait MaskedChannelNormDiffImage<T, L> {
    fn norm_diff_inf_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_diff_l1_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_diff_l2_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

macro_rules! impl_masked_norm_diff_image {
    ($ty:ty, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        impl<'a> MaskedNormDiffImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            fn norm_diff_inf_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_inf(stream_context, source_1, source_2, mask, output)
            }

            fn norm_diff_l1_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l1(stream_context, source_1, source_2, mask, output)
            }

            fn norm_diff_l2_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l2(stream_context, source_1, source_2, mask, output)
            }
        }
    };
}

#[macro_use]
#[path = "statistics_quality_channel_norm_dispatch_macros.rs"]
mod channel_norm_dispatch_macros;

#[macro_use]
#[path = "statistics_quality_relative_norm_dispatch_macros.rs"]
mod relative_norm_dispatch_macros;
