macro_rules! impl_masked_channel_norm_diff_image {
    ($ty:ty, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        unsafe impl<'a> MaskedChannelNormDiffImage<$ty, C3> for ImagePipeline<'a, $ty, C3> {
            fn norm_diff_inf_channel_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C3>,
                source_2: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_inf(stream_context, source_1, source_2, mask, channel, output) }
            }

            fn norm_diff_l1_channel_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C3>,
                source_2: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l1(stream_context, source_1, source_2, mask, channel, output) }
            }

            fn norm_diff_l2_channel_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C3>,
                source_2: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l2(stream_context, source_1, source_2, mask, channel, output) }
            }
        }
    };
}
