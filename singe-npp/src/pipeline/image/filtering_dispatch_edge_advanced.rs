macro_rules! impl_unsharp_border_filter_image {
    ($ty:ty, $layout:ty, $unsharp:path) => {
        impl<'a> UnsharpBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_unsharp_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                radius: f32,
                sigma: f32,
                weight: f32,
                threshold: f32,
                border_type: BorderType,
            ) -> Result<()> {
                $unsharp(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    radius,
                    sigma,
                    weight,
                    threshold,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_advanced_gauss_filter_image {
    ($ty:ty, $layout:ty, $gauss:path, $gauss_border:path) => {
        impl<'a> AdvancedGaussFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_gauss_advanced_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[f32],
            ) -> Result<()> {
                $gauss(stream_context, source, destination, kernel)
            }

            fn filter_gauss_advanced_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                $gauss_border(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    kernel,
                    border_type,
                )
            }
        }
    };
}

#[path = "filtering_dispatch_gauss_advanced_impls.rs"]
mod gauss_advanced_impls;
#[path = "filtering_dispatch_gauss_pyramid_bilateral.rs"]
mod gauss_pyramid_bilateral;
#[path = "filtering_dispatch_unsharp_border_impls.rs"]
mod unsharp_border_impls;
