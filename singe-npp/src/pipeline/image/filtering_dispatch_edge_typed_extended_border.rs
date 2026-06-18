macro_rules! impl_sobel_extended_border_filter_image {
    (
        $ty:ty,
        $layout:ty,
        $horizontal_mask:path,
        $vertical_mask:path,
        $horizontal_second:path,
        $vertical_second:path,
        $cross:path
    ) => {
        impl<'a> SobelExtendedBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_sobel_horizontal_mask_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $horizontal_mask(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_sobel_vertical_mask_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $vertical_mask(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_sobel_horizontal_second_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $horizontal_second(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_sobel_vertical_second_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $vertical_second(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_sobel_cross_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $cross(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_scharr_border_filter_image {
    ($ty:ty, $layout:ty, $horizontal:path, $vertical:path) => {
        impl<'a> ScharrBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_scharr_horizontal_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $horizontal(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn filter_scharr_vertical_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $vertical(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }
        }
    };
}

#[path = "filtering_dispatch_edge_typed_extended_border_f32.rs"]
mod f32_impls;
