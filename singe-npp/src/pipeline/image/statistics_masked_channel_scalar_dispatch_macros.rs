macro_rules! impl_masked_channel_scalar_statistic_image {
    ($ty:ty, $mean:path, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        impl<'a> MaskedChannelScalarStatisticImage<$ty, C3> for ImagePipeline<'a, $ty, C3> {
            fn mean_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $mean(stream_context, source, mask, channel, output)
            }

            fn norm_inf_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_inf(stream_context, source, mask, channel, output)
            }

            fn norm_l1_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l1(stream_context, source, mask, channel, output)
            }

            fn norm_l2_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l2(stream_context, source, mask, channel, output)
            }
        }
    };
}
