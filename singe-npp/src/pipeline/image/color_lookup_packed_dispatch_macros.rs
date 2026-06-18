macro_rules! impl_packed_lookup_table_image {
    (
        $ty:ty,
        $layout:ty,
        $channels:literal,
        $table_ty:ty,
        $lookup:path,
        $lookup_in_place:path,
        $linear:path,
        $linear_in_place:path,
        $cubic:path,
        $cubic_in_place:path
    ) => {
        impl<'a> PackedLookupTableImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            type TableValue = $table_ty;

            fn lookup_table_channels_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                values: &[&DeviceMemory<Self::TableValue>; $channels],
                levels: &[&DeviceMemory<Self::TableValue>; $channels],
            ) -> Result<()> {
                $lookup(stream_context, source, destination, values, levels)
            }

            fn lookup_table_channels_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                values: &[&DeviceMemory<Self::TableValue>; $channels],
                levels: &[&DeviceMemory<Self::TableValue>; $channels],
            ) -> Result<()> {
                $lookup_in_place(stream_context, image, values, levels)
            }

            fn lookup_table_channels_linear_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                values: &[&DeviceMemory<Self::TableValue>; $channels],
                levels: &[&DeviceMemory<Self::TableValue>; $channels],
            ) -> Result<()> {
                $linear(stream_context, source, destination, values, levels)
            }

            fn lookup_table_channels_linear_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                values: &[&DeviceMemory<Self::TableValue>; $channels],
                levels: &[&DeviceMemory<Self::TableValue>; $channels],
            ) -> Result<()> {
                $linear_in_place(stream_context, image, values, levels)
            }

            fn lookup_table_channels_cubic_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                values: &[&DeviceMemory<Self::TableValue>; $channels],
                levels: &[&DeviceMemory<Self::TableValue>; $channels],
            ) -> Result<()> {
                $cubic(stream_context, source, destination, values, levels)
            }

            fn lookup_table_channels_cubic_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                values: &[&DeviceMemory<Self::TableValue>; $channels],
                levels: &[&DeviceMemory<Self::TableValue>; $channels],
            ) -> Result<()> {
                $cubic_in_place(stream_context, image, values, levels)
            }
        }
    };
}
