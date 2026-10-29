macro_rules! impl_cross_correlation_norm_level_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> CrossCorrelationNormLevelImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $full(stream_context, source, template, destination) }
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $same(stream_context, source, template, destination) }
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $valid(stream_context, source, template, destination) }
            }
        }
    };
}

macro_rules! impl_cross_correlation_norm_level_scaled_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> CrossCorrelationNormLevelScaledImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $full(stream_context, source, template, destination, scale_factor) }
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $same(stream_context, source, template, destination, scale_factor) }
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $valid(stream_context, source, template, destination, scale_factor) }
            }
        }
    };
}

macro_rules! impl_cross_correlation_norm_level_to_image {
    ($ty:ty, $destination_ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> CrossCorrelationNormLevelToImage<$ty, $destination_ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $full(stream_context, source, template, destination) }
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $same(stream_context, source, template, destination) }
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $valid(stream_context, source, template, destination) }
            }
        }
    };
}

macro_rules! impl_cross_correlation_norm_level_advanced_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> CrossCorrelationNormLevelAdvancedImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $full(stream_context, source, template, destination) }
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $same(stream_context, source, template, destination) }
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $valid(stream_context, source, template, destination) }
            }
        }
    };
}

macro_rules! impl_cross_correlation_norm_level_advanced_to_image {
    ($ty:ty, $destination_ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> CrossCorrelationNormLevelAdvancedToImage<$ty, $destination_ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $full(stream_context, source, template, destination) }
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $same(stream_context, source, template, destination) }
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe { $valid(stream_context, source, template, destination) }
            }
        }
    };
}
