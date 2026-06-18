#[path = "filtering_dispatch_edge_typed_border.rs"]
mod border;
#[path = "filtering_dispatch_edge_typed_directional.rs"]
mod directional;

macro_rules! impl_sobel_extended_filter_image {
    (
        $ty:ty,
        $layout:ty,
        $horizontal_mask:path,
        $vertical_mask:path,
        $horizontal_second:path,
        $vertical_second:path,
        $cross:path
    ) => {
        impl<'a> SobelExtendedFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_sobel_horizontal_mask_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $horizontal_mask(stream_context, source, destination, mask_size)
            }

            fn filter_sobel_vertical_mask_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $vertical_mask(stream_context, source, destination, mask_size)
            }

            fn filter_sobel_horizontal_second_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $horizontal_second(stream_context, source, destination, mask_size)
            }

            fn filter_sobel_vertical_second_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $vertical_second(stream_context, source, destination, mask_size)
            }

            fn filter_sobel_cross_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $cross(stream_context, source, destination, mask_size)
            }
        }
    };
}

macro_rules! impl_scharr_filter_image {
    ($ty:ty, $layout:ty, $horizontal:path, $vertical:path) => {
        impl<'a> ScharrFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_scharr_horizontal_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $horizontal(stream_context, source, destination)
            }

            fn filter_scharr_vertical_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $vertical(stream_context, source, destination)
            }
        }
    };
}

#[path = "filtering_dispatch_edge_typed_f32.rs"]
mod f32_impls;
