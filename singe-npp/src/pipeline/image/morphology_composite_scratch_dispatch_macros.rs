macro_rules! impl_composite_morphology_border_scratch_image {
    (
        $ty:ty,
        $layout:ty,
        $close_with_scratch:path,
        $open_with_scratch:path,
        $top_hat_with_scratch:path,
        $black_hat_with_scratch:path,
        $gradient_with_scratch:path
    ) => {
        impl<'a> CompositeMorphologyBorderScratchImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn morph_close_border_image_with_scratch(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                scratch: &mut ScratchBuffer,
                border_type: BorderType,
            ) -> Result<()> {
                $close_with_scratch(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    scratch,
                    border_type,
                )
            }

            fn morph_open_border_image_with_scratch(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                scratch: &mut ScratchBuffer,
                border_type: BorderType,
            ) -> Result<()> {
                $open_with_scratch(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    scratch,
                    border_type,
                )
            }

            fn morph_top_hat_border_image_with_scratch(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                scratch: &mut ScratchBuffer,
                border_type: BorderType,
            ) -> Result<()> {
                $top_hat_with_scratch(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    scratch,
                    border_type,
                )
            }

            fn morph_black_hat_border_image_with_scratch(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                scratch: &mut ScratchBuffer,
                border_type: BorderType,
            ) -> Result<()> {
                $black_hat_with_scratch(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    scratch,
                    border_type,
                )
            }

            fn morph_gradient_border_image_with_scratch(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                scratch: &mut ScratchBuffer,
                border_type: BorderType,
            ) -> Result<()> {
                $gradient_with_scratch(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    scratch,
                    border_type,
                )
            }
        }
    };
}
