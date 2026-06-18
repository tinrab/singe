macro_rules! impl_edge_directional_filter_image {
    (
        $ty:ty,
        $layout:ty,
        $prewitt_horizontal:path,
        $prewitt_vertical:path,
        $roberts_down:path,
        $roberts_up:path,
        $sobel_horizontal:path,
        $sobel_vertical:path
    ) => {
        impl<'a> EdgeDirectionalFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_prewitt_horizontal_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $prewitt_horizontal(stream_context, source, destination)
            }

            fn filter_prewitt_vertical_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $prewitt_vertical(stream_context, source, destination)
            }

            fn filter_roberts_down_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $roberts_down(stream_context, source, destination)
            }

            fn filter_roberts_up_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $roberts_up(stream_context, source, destination)
            }

            fn filter_sobel_horizontal_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $sobel_horizontal(stream_context, source, destination)
            }

            fn filter_sobel_vertical_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $sobel_vertical(stream_context, source, destination)
            }
        }
    };
}

#[path = "filtering_dispatch_float_edge_directional.rs"]
mod float_impls;
#[path = "filtering_dispatch_integer_edge_directional.rs"]
mod integer_impls;
