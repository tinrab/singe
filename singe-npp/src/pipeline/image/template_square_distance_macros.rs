macro_rules! impl_square_distance_norm_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        impl<'a> SquareDistanceNormImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
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

macro_rules! impl_square_distance_norm_to_image {
    ($ty:ty, $destination_ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        impl<'a> SquareDistanceNormToImage<$ty, $destination_ty, $layout>
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

macro_rules! impl_square_distance_norm_scaled_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        impl<'a> SquareDistanceNormScaledImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
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
