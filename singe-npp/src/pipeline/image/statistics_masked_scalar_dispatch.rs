use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

pub trait MaskedScalarStatisticImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn mean_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_inf_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_l1_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_l2_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

macro_rules! impl_masked_scalar_statistic_image {
    ($ty:ty, $mean:path, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        impl<'a> MaskedScalarStatisticImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            const OUTPUT_CHANNELS: usize = 1;

            fn mean_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $mean(stream_context, source, mask, output)
            }

            fn norm_inf_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_inf(stream_context, source, mask, output)
            }

            fn norm_l1_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l1(stream_context, source, mask, output)
            }

            fn norm_l2_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l2(stream_context, source, mask, output)
            }
        }
    };
}
