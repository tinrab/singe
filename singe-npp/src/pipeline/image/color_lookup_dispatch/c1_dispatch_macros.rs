macro_rules! impl_lookup_table_image {
    (
        $ty:ty,
        $table_ty:ty,
        $lookup:path,
        $lookup_in_place:path,
        $linear:path,
        $linear_in_place:path,
        $cubic:path,
        $cubic_in_place:path
    ) => {
        unsafe impl<'a> LookupTableImage<$ty> for ImagePipeline<'a, $ty, C1> {
            type TableValue = $table_ty;

            fn lookup_table_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(values)?;
                stream_context.prepare_region(levels)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $lookup(stream_context, source, destination, values, levels)
                })
            }

            fn lookup_table_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context.prepare_region(values)?;
                stream_context.prepare_region(levels)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $lookup_in_place(stream_context, image, values, levels)
                })
            }

            fn lookup_table_linear_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(values)?;
                stream_context.prepare_region(levels)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $linear(stream_context, source, destination, values, levels)
                })
            }

            fn lookup_table_linear_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context.prepare_region(values)?;
                stream_context.prepare_region(levels)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $linear_in_place(stream_context, image, values, levels)
                })
            }

            fn lookup_table_cubic_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(values)?;
                stream_context.prepare_region(levels)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $cubic(stream_context, source, destination, values, levels)
                })
            }

            fn lookup_table_cubic_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context.prepare_region(values)?;
                stream_context.prepare_region(levels)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $cubic_in_place(stream_context, image, values, levels)
                })
            }
        }
    };
}
