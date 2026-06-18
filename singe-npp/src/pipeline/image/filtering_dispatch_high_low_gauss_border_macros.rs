macro_rules! impl_high_low_gauss_border_filter_image {
    ($ty:ty, $layout:ty, $high_pass:path, $low_pass:path, $gauss:path) => {
        impl<'a> HighLowGaussBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_high_pass_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $high_pass(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_low_pass_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $low_pass(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_gauss_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $gauss(
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
