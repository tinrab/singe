macro_rules! impl_static_planar3_to_packed_twist {
    ($method:ident, $ty:ty, $destination_layout:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, $destination_layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            $convert(
                stream_context,
                source_0,
                source_1,
                source_2,
                destination,
                twist,
            )
        }
    };
}

macro_rules! impl_static_planar3_to_ac4_twist {
    ($method:ident, $ty:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination: &mut ImageViewMut<'_, $ty, AC4>,
            twist: ColorTwistMatrix,
            alpha: $ty,
        ) -> Result<()> {
            $convert(
                stream_context,
                source_0,
                source_1,
                source_2,
                destination,
                twist,
                alpha,
            )
        }
    };
}

macro_rules! impl_static_planar3_to_planar3_twist {
    ($method:ident, $ty:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination_0: &mut ImageViewMut<'_, $ty, C1>,
            destination_1: &mut ImageViewMut<'_, $ty, C1>,
            destination_2: &mut ImageViewMut<'_, $ty, C1>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            $convert(
                stream_context,
                source_0,
                source_1,
                source_2,
                destination_0,
                destination_1,
                destination_2,
                twist,
            )
        }
    };
}

macro_rules! impl_static_packed_to_planar3_twist {
    ($method:ident, $ty:ty, $source_layout:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination_0: &mut ImageViewMut<'_, $ty, C1>,
            destination_1: &mut ImageViewMut<'_, $ty, C1>,
            destination_2: &mut ImageViewMut<'_, $ty, C1>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            $convert(
                stream_context,
                source,
                destination_0,
                destination_1,
                destination_2,
                twist,
            )
        }
    };
}

macro_rules! impl_static_different_layout_twist {
    ($method:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $ty, $destination_layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            $convert(stream_context, source, destination, twist)
        }
    };
}
