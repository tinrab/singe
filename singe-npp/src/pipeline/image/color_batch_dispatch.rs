macro_rules! impl_color_convert_batch_same_layout_image {
    ($name:ident, $advanced_name:ident, $convert:path, $convert_advanced:path) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, u8, C3>],
            destinations: &mut [ImageViewMut<'_, u8, C3>],
        ) -> Result<()> {
            $convert(stream_context, sources, destinations)
        }

        pub fn $advanced_name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, u8, C3>],
            destinations: &mut [ImageViewMut<'_, u8, C3>],
        ) -> Result<()> {
            $convert_advanced(stream_context, sources, destinations)
        }
    };
}

macro_rules! impl_color_convert_batch_planar_to_packed_image {
    ($name:ident, $advanced_name:ident, $convert:path, $convert_advanced:path) => {
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, u8, 3>],
            destinations: &mut [ImageViewMut<'_, u8, C3>],
        ) -> Result<()> {
            $convert(stream_context, sources, destinations)
        }

        pub fn $advanced_name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, u8, 3>],
            destinations: &mut [ImageViewMut<'_, u8, C3>],
        ) -> Result<()> {
            $convert_advanced(stream_context, sources, destinations)
        }
    };
}

macro_rules! impl_subsampled_color_convert_batch_planar_to_packed_image {
    ($name:ident, $advanced_name:ident, $convert:path, $convert_advanced:path) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, u8, C1>],
            source_plane_1: &[ImageView<'_, u8, C1>],
            source_plane_2: &[ImageView<'_, u8, C1>],
            destinations: &mut [ImageViewMut<'_, u8, C3>],
        ) -> Result<()> {
            $convert(
                stream_context,
                source_plane_0,
                source_plane_1,
                source_plane_2,
                destinations,
            )
        }

        pub fn $advanced_name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, u8, C1>],
            source_plane_1: &[ImageView<'_, u8, C1>],
            source_plane_2: &[ImageView<'_, u8, C1>],
            destinations: &mut [ImageViewMut<'_, u8, C3>],
        ) -> Result<()> {
            $convert_advanced(
                stream_context,
                source_plane_0,
                source_plane_1,
                source_plane_2,
                destinations,
            )
        }
    };
}
