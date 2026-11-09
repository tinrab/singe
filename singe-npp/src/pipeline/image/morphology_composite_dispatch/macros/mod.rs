#[macro_use]
mod scratch_macros;

macro_rules! impl_morph_buffer_size {
    ($ty:ty, $layout:ty, $buffer_size:path) => {
        impl<'a> ImagePipeline<'a, $ty, $layout> {
            /// Returns the scratch size in bytes for composite morphology on `roi`.
            ///
            /// # Errors
            ///
            /// Returns an error if either ROI dimension is nonpositive, NPP rejects the query, or its returned size cannot be represented as `usize`.
            pub fn morph_buffer_size(roi: Size) -> Result<usize> {
                $buffer_size(roi)
            }
        }
    };
}

macro_rules! impl_composite_morphology_border_image {
    (
        $ty:ty,
        $layout:ty,
        $close_with_scratch:path,
        $close:path,
        $open_with_scratch:path,
        $open:path,
        $top_hat_with_scratch:path,
        $top_hat:path,
        $black_hat_with_scratch:path,
        $black_hat:path,
        $gradient_with_scratch:path,
        $gradient:path
    ) => {
        unsafe impl<'a> CompositeMorphologyBorderImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn morph_close_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $close(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                })
            }

            fn morph_open_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $open(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                })
            }

            fn morph_top_hat_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $top_hat(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                })
            }

            fn morph_black_hat_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $black_hat(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                })
            }

            fn morph_gradient_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $gradient(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask,
                        mask_size,
                        anchor,
                        border_type,
                    )
                })
            }
        }

        impl_composite_morphology_border_scratch_image!(
            $ty,
            $layout,
            $close_with_scratch,
            $open_with_scratch,
            $top_hat_with_scratch,
            $black_hat_with_scratch,
            $gradient_with_scratch
        );
    };
}
