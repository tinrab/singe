macro_rules! impl_error_metric_image {
    ($ty:ty, $layout:ty, $maximum_error:path, $average_error:path) => {
        impl<'a> ErrorMetricImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn maximum_error(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $maximum_error(stream_context, source_1, source_2, output)
            }

            fn average_error(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $average_error(stream_context, source_1, source_2, output)
            }
        }
    };
}

macro_rules! impl_quality_metric_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $mse:path, $psnr:path) => {
        impl<'a> QualityMetricImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn mean_squared_error(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f32>,
            ) -> Result<()> {
                $mse(stream_context, source_1, source_2, output)
            }

            fn peak_signal_to_noise_ratio(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f32>,
            ) -> Result<()> {
                $psnr(stream_context, source_1, source_2, output)
            }
        }
    };
}

macro_rules! impl_structural_similarity_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $metric:path) => {
        impl<'a> StructuralSimilarityImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn structural_similarity(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f32>,
            ) -> Result<()> {
                $metric(stream_context, source_1, source_2, output)
            }
        }
    };
}

macro_rules! impl_multiscale_structural_similarity_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $metric:path) => {
        impl<'a> MultiscaleStructuralSimilarityImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn multiscale_structural_similarity(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f32>,
            ) -> Result<()> {
                $metric(stream_context, source_1, source_2, output)
            }
        }
    };
}

macro_rules! impl_weighted_multiscale_structural_similarity_image {
    ($ty:ty, $layout:ty, $output_channels:expr, $metric:path) => {
        impl<'a> WeightedMultiscaleStructuralSimilarityImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            const OUTPUT_CHANNELS: usize = $output_channels;

            fn weighted_multiscale_structural_similarity(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, $layout>,
                source_2: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f32>,
            ) -> Result<()> {
                $metric(stream_context, source_1, source_2, output)
            }
        }
    };
}
