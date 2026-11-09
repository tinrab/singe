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
        unsafe impl<'a> SobelExtendedBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_sobel_horizontal_mask_border_image(
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
                    $horizontal_mask(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        border_type,
                    )
                })
            }

            fn filter_sobel_vertical_mask_border_image(
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
                    $vertical_mask(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        border_type,
                    )
                })
            }

            fn filter_sobel_horizontal_second_border_image(
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
                    $horizontal_second(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        border_type,
                    )
                })
            }

            fn filter_sobel_vertical_second_border_image(
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
                    $vertical_second(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        border_type,
                    )
                })
            }

            fn filter_sobel_cross_border_image(
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
                    $cross(
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

macro_rules! impl_scharr_border_filter_image {
    ($ty:ty, $layout:ty, $horizontal:path, $vertical:path) => {
        unsafe impl<'a> ScharrBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_scharr_horizontal_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $horizontal(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }

            fn filter_scharr_vertical_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $vertical(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }
        }
    };
}

mod f32_impls;
