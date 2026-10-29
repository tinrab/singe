macro_rules! impl_square_distance_norm_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> SquareDistanceNormImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $full(stream_context, source, template, destination) })
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $same(stream_context, source, template, destination) })
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $valid(stream_context, source, template, destination) })
            }
        }
    };
}

macro_rules! impl_square_distance_norm_to_image {
    ($ty:ty, $destination_ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> SquareDistanceNormToImage<$ty, $destination_ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $full(stream_context, source, template, destination) })
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $same(stream_context, source, template, destination) })
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $valid(stream_context, source, template, destination) })
            }
        }
    };
}

macro_rules! impl_square_distance_norm_scaled_image {
    ($ty:ty, $layout:ty, $full:path, $same:path, $valid:path) => {
        unsafe impl<'a> SquareDistanceNormScaledImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn full(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $full(stream_context, source, template, destination, scale_factor)
                })
            }

            fn same(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $same(stream_context, source, template, destination, scale_factor)
                })
            }

            fn valid(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                template: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(template)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $valid(stream_context, source, template, destination, scale_factor)
                })
            }
        }
    };
}
