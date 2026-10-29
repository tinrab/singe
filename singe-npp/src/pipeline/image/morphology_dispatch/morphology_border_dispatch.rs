macro_rules! impl_mask_morphology_border_image {
    ($ty:ty, $layout:ty, $dilate:path, $erode:path) => {
        unsafe impl<'a> MaskMorphologyBorderImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn dilate_border_image(
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
                    $dilate(
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

            fn erode_border_image(
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
                    $erode(
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
    };
}

macro_rules! impl_morphology_3x3_border_image {
    ($ty:ty, $layout:ty, $dilate:path, $erode:path) => {
        unsafe impl<'a> Morphology3x3BorderImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn dilate_3x3_border_image(
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
                    $dilate(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }

            fn erode_3x3_border_image(
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
                    $erode(
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

macro_rules! impl_gray_mask_morphology_border_image {
    ($ty:ty, $mask_ty:ty, $dilate:path, $erode:path) => {
        unsafe impl<'a> GrayMaskMorphologyBorderImage<$ty, $mask_ty>
            for ImagePipeline<'a, $ty, C1>
        {
            fn gray_dilate_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                mask: &[$mask_ty],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $dilate(
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

            fn gray_erode_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                mask: &[$mask_ty],
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $erode(
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
    };
}

mod morphology_3x3_border_dispatch_impls;

mod morphology_mask_border_dispatch_impls;

mod morphology_gray_mask_border_dispatch_impls;
