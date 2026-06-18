#[macro_use]
#[path = "template_square_distance_macros.rs"]
mod square_distance_macros;
#[macro_use]
#[path = "template_cross_correlation_norm_level_macros.rs"]
mod cross_correlation_norm_level_macros;

macro_rules! impl_cross_correlation_norm_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        impl<'a> CrossCorrelationNormImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $full(stream_context, source, template, destination)
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $same(stream_context, source, template, destination)
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $valid(stream_context, source, template, destination)
            }
        }
    };
}

macro_rules! impl_cross_correlation_norm_to_image {
    ($ty:ty, $destination_ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        impl<'a> CrossCorrelationNormToImage<$ty, $destination_ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                $full(stream_context, source, template, destination)
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                $same(stream_context, source, template, destination)
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                $valid(stream_context, source, template, destination)
            }
        }
    };
}

macro_rules! impl_cross_correlation_norm_scaled_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        impl<'a> CrossCorrelationNormScaledImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $full(stream_context, source, template, destination, scale_factor)
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $same(stream_context, source, template, destination, scale_factor)
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $valid(stream_context, source, template, destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_cross_correlation_image {
    ($ty:ty, $layout:ty, $valid:path) => {
        impl<'a> CrossCorrelationImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $valid(stream_context, source, template, destination)
            }
        }
    };
}

macro_rules! impl_cross_correlation_to_image {
    ($ty:ty, $destination_ty:ty, $layout:ty, $valid:path) => {
        impl<'a> CrossCorrelationToImage<$ty, $destination_ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                $valid(stream_context, source, template, destination)
            }
        }
    };
}
