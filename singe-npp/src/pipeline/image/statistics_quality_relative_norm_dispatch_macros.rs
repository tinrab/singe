macro_rules! impl_masked_norm_relative_image {
    ($ty:ty, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        impl<'a> MaskedNormRelativeImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            fn norm_relative_inf_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_inf(stream_context, source_1, source_2, mask, output)
            }

            fn norm_relative_l1_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l1(stream_context, source_1, source_2, mask, output)
            }

            fn norm_relative_l2_masked(
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

macro_rules! impl_masked_channel_norm_relative_image {
    ($ty:ty, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        impl<'a> MaskedChannelNormRelativeImage<$ty, C3> for ImagePipeline<'a, $ty, C3> {
            fn norm_relative_inf_channel_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C3>,
                source_2: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_inf(stream_context, source_1, source_2, mask, channel, output)
            }

            fn norm_relative_l1_channel_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C3>,
                source_2: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l1(stream_context, source_1, source_2, mask, channel, output)
            }

            fn norm_relative_l2_channel_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C3>,
                source_2: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l2(stream_context, source_1, source_2, mask, channel, output)
            }
        }
    };
}
