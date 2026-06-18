macro_rules! impl_static_rgb_to_uyvp_planar {
    ($method:ident, $source_ty:ty, $source_layout:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination_0: &mut ImageViewMut<'_, u8, C1>,
            destination_1: &mut ImageViewMut<'_, u8, C1>,
            destination_2: &mut ImageViewMut<'_, u8, C1>,
            color_space: ColorSpace,
        ) -> Result<()> {
            $convert(
                stream_context,
                source,
                source_offset,
                destination_0,
                destination_1,
                destination_2,
                color_space,
            )
        }
    };
}

macro_rules! impl_static_rgb_planar_to_uyvp {
    ($method:ident, $source_ty:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $source_ty, C1>,
            source_1: &ImageView<'_, $source_ty, C1>,
            source_2: &ImageView<'_, $source_ty, C1>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, u8, C3>,
            color_space: ColorSpace,
        ) -> Result<()> {
            $convert(
                stream_context,
                source_0,
                source_1,
                source_2,
                source_offset,
                destination,
                color_space,
            )
        }
    };
}

macro_rules! impl_static_uyvp_to_rgb_planar {
    ($method:ident, $destination_ty:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C3>,
            source_offset: Point,
            destination_0: &mut ImageViewMut<'_, $destination_ty, C1>,
            destination_1: &mut ImageViewMut<'_, $destination_ty, C1>,
            destination_2: &mut ImageViewMut<'_, $destination_ty, C1>,
            color_space: ColorSpace,
        ) -> Result<()> {
            $convert(
                stream_context,
                source,
                source_offset,
                destination_0,
                destination_1,
                destination_2,
                color_space,
            )
        }
    };
}
