macro_rules! impl_less_greater_value_threshold_image {
    ($ty:ty, $threshold:path, $threshold_in_place:path) => {
        impl<'a> LessGreaterValueThresholdImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn threshold_less_greater_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                lower_threshold: $ty,
                lower_value: $ty,
                upper_threshold: $ty,
                upper_value: $ty,
            ) -> Result<()> {
                $threshold(
                    stream_context,
                    source,
                    destination,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )
            }

            fn threshold_less_greater_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                lower_threshold: $ty,
                lower_value: $ty,
                upper_threshold: $ty,
                upper_value: $ty,
            ) -> Result<()> {
                $threshold_in_place(
                    stream_context,
                    image,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )
            }
        }
    };
}

macro_rules! impl_packed_less_greater_value_threshold_image {
    ($ty:ty, $layout:ty, $channels:literal, $threshold:path, $threshold_in_place:path) => {
        impl<'a> PackedLessGreaterValueThresholdImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn threshold_channels_less_greater_value_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                lower_thresholds: [$ty; $channels],
                lower_values: [$ty; $channels],
                upper_thresholds: [$ty; $channels],
                upper_values: [$ty; $channels],
            ) -> Result<()> {
                $threshold(
                    stream_context,
                    source,
                    destination,
                    lower_thresholds,
                    lower_values,
                    upper_thresholds,
                    upper_values,
                )
            }

            fn threshold_channels_less_greater_value_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                lower_thresholds: [$ty; $channels],
                lower_values: [$ty; $channels],
                upper_thresholds: [$ty; $channels],
                upper_values: [$ty; $channels],
            ) -> Result<()> {
                $threshold_in_place(
                    stream_context,
                    image,
                    lower_thresholds,
                    lower_values,
                    upper_thresholds,
                    upper_values,
                )
            }
        }
    };
}
