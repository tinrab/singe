macro_rules! impl_static_planar_to_c2_convert {
    ($method:ident, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, u8, C1>,
            source_1: &ImageView<'_, u8, C1>,
            source_2: &ImageView<'_, u8, C1>,
            destination: &mut ImageViewMut<'_, u8, C2>,
        ) -> Result<()> {
            $convert(stream_context, source_0, source_1, source_2, destination)
        }
    };
}

macro_rules! impl_static_c2_to_planar_convert {
    ($method:ident, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_0: &mut ImageViewMut<'_, u8, C1>,
            destination_1: &mut ImageViewMut<'_, u8, C1>,
            destination_2: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            $convert(
                stream_context,
                source,
                destination_0,
                destination_1,
                destination_2,
            )
        }
    };
}

macro_rules! impl_static_c2_to_planar3_convert {
    ($method:ident, $convert:path) => {
        impl_static_c2_to_planar_convert!($method, $convert);
    };
}

macro_rules! impl_static_planar3_to_planar3_convert {
    ($method:ident, $ty:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination_0: &mut ImageViewMut<'_, $ty, C1>,
            destination_1: &mut ImageViewMut<'_, $ty, C1>,
            destination_2: &mut ImageViewMut<'_, $ty, C1>,
        ) -> Result<()> {
            $convert(
                stream_context,
                source_0,
                source_1,
                source_2,
                destination_0,
                destination_1,
                destination_2,
            )
        }
    };
}

macro_rules! impl_static_planar3_to_planar2_convert {
    ($method:ident, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, u8, C1>,
            source_1: &ImageView<'_, u8, C1>,
            source_2: &ImageView<'_, u8, C1>,
            destination_0: &mut ImageViewMut<'_, u8, C1>,
            destination_1: &mut ImageViewMut<'_, u8, C2>,
        ) -> Result<()> {
            $convert(
                stream_context,
                source_0,
                source_1,
                source_2,
                destination_0,
                destination_1,
            )
        }
    };
}

macro_rules! impl_static_c2_to_planar2_convert {
    ($method:ident, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_0: &mut ImageViewMut<'_, u8, C1>,
            destination_1: &mut ImageViewMut<'_, u8, C2>,
        ) -> Result<()> {
            $convert(stream_context, source, destination_0, destination_1)
        }
    };
}
