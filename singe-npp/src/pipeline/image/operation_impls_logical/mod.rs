macro_rules! impl_logical_not_image {
    ($layout:ty, $logical_not:path, $logical_not_in_place:path) => {
        unsafe impl<'a> LogicalNotImage<u8, $layout> for ImagePipeline<'a, u8, $layout> {
            fn logical_not_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageOutputView<'_, u8, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $logical_not(stream_context, source, destination) })
            }

            fn logical_not_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $logical_not_in_place(stream_context, source_destination)
                })
            }
        }
    };
}

macro_rules! impl_logical_binary_image {
    (
        $ty:ty,
        $layout:ty,
        $and:path,
        $and_in_place:path,
        $or:path,
        $or_in_place:path,
        $xor:path,
        $xor_in_place:path
    ) => {
        unsafe impl<'a> LogicalBinaryImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn logical_and_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $and(stream_context, left, right, destination) })
            }

            fn logical_and_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $and_in_place(stream_context, source, source_destination)
                })
            }

            fn logical_or_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $or(stream_context, left, right, destination) })
            }

            fn logical_or_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $or_in_place(stream_context, source, source_destination)
                })
            }

            fn logical_xor_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $xor(stream_context, left, right, destination) })
            }

            fn logical_xor_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $xor_in_place(stream_context, source, source_destination)
                })
            }
        }
    };
}

#[macro_use]
mod logical_constant_macros;

mod logical_constant;

mod logical_binary;

mod logical_not;
