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
        impl<'a> LookupTableImage<$ty> for ImagePipeline<'a, $ty, C1> {
            type TableValue = $table_ty;

            fn lookup_table_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                $lookup(stream_context, source, destination, values, levels)
            }

            fn lookup_table_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                $lookup_in_place(stream_context, image, values, levels)
            }

            fn lookup_table_linear_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                $linear(stream_context, source, destination, values, levels)
            }

            fn lookup_table_linear_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                $linear_in_place(stream_context, image, values, levels)
            }

            fn lookup_table_cubic_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                $cubic(stream_context, source, destination, values, levels)
            }

            fn lookup_table_cubic_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, C1>,
                values: &DeviceMemory<Self::TableValue>,
                levels: &DeviceMemory<Self::TableValue>,
            ) -> Result<()> {
                $cubic_in_place(stream_context, image, values, levels)
            }
        }
    };
}
