macro_rules! impl_error_metric_image {
    ($ty:ty, $layout:ty, $maximum_error:path, $average_error:path) => {
        unsafe impl<'a> ErrorMetricImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn maximum_error(
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
                unsafe { $maximum_error(stream_context, source_1, source_2, output) }
            }

            fn average_error(
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
                unsafe { $average_error(stream_context, source_1, source_2, output) }
            }
        }
    };
}

macro_rules! impl_quality_metric_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $mse:path, $psnr:path) => {
        unsafe impl<'a> QualityMetricImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn mean_squared_error(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f32>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $mse(stream_context, source_1, source_2, output) }
            }

            fn peak_signal_to_noise_ratio(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f32>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $psnr(stream_context, source_1, source_2, output) }
            }
        }
    };
}

macro_rules! impl_structural_similarity_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $metric:path) => {
        unsafe impl<'a> StructuralSimilarityImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn structural_similarity(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f32>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $metric(stream_context, source_1, source_2, output) }
            }
        }
    };
}

macro_rules! impl_multiscale_structural_similarity_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $metric:path) => {
        unsafe impl<'a> MultiscaleStructuralSimilarityImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn multiscale_structural_similarity(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f32>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $metric(stream_context, source_1, source_2, output) }
            }
        }
    };
}

macro_rules! impl_weighted_multiscale_structural_similarity_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $metric:path) => {
        unsafe impl<'a> WeightedMultiscaleStructuralSimilarityImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn weighted_multiscale_structural_similarity(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f32>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $metric(stream_context, source_1, source_2, output) }
            }
        }
    };
}
