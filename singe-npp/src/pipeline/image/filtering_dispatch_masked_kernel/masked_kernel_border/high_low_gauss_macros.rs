macro_rules! impl_high_low_gauss_border_filter_image {
    ($ty:ty, $layout:ty, $high_pass:path, $low_pass:path, $gauss:path) => {
        unsafe impl<'a> HighLowGaussBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_high_pass_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $high_pass(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        border_type,
                    )
                })
            }

            fn filter_low_pass_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $low_pass(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        border_type,
                    )
                })
            }

            fn filter_gauss_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $gauss(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        border_type,
                    )
                })
            }
        }
    };
}
