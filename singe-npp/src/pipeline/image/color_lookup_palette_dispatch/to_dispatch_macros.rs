macro_rules! impl_palette_lookup_table_to_image {
    ($ty:ty, $destination_ty:ty, $layout:ty, $table_ty:ty, $lookup:path) => {
        unsafe impl<'a> PaletteLookupTableToImage<$ty, $destination_ty, $layout>
            for ImagePipeline<'a, $ty, C1>
        {
            type TableValue = $table_ty;

            fn lookup_table_palette_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
                table: &DeviceMemory<Self::TableValue>,
                bit_size: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(table)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $lookup(stream_context, source, destination, table, bit_size)
                })
            }
        }
    };
}

macro_rules! impl_palette_lookup_table_swap_image {
    ($ty:ty, $lookup:path) => {
        unsafe impl<'a> PaletteLookupTableSwapImage<$ty> for ImagePipeline<'a, $ty, C3> {
            fn lookup_table_palette_swap_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                destination: &mut ImageOutputView<'_, $ty, C4>,
                alpha: i32,
                tables: &[&DeviceMemory<$ty>; 3],
                bit_size: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                for table in tables {
                    stream_context.prepare_region(*table)?;
                }
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $lookup(stream_context, source, destination, alpha, tables, bit_size)
                })
            }
        }
    };
}
