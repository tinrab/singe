#[macro_use]
#[path = "color_static_uyvp_dispatch.rs"]
mod uyvp_dispatch;
#[macro_use]
#[path = "color_static_planar_dispatch_macros.rs"]
mod planar_dispatch_macros;

macro_rules! impl_static_different_layout_convert {
    ($method:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $ty, $destination_layout>,
        ) -> Result<()> {
            $convert(stream_context, source, destination)
        }
    };
}

macro_rules! impl_static_packed_to_planar_convert {
    ($method:ident, $ty:ty, $source_layout:ty, $planes:literal, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
        ) -> Result<()> {
            $convert(stream_context, source, destination)
        }
    };
}

macro_rules! impl_static_planar_to_packed_convert {
    ($method:ident, $ty:ty, $planes:literal, $destination_layout:ty, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageViewMut<'_, $ty, $destination_layout>,
        ) -> Result<()> {
            $convert(stream_context, source, destination)
        }
    };
}

macro_rules! impl_static_planar_to_packed_constant_alpha_convert {
    ($method:ident, $ty:ty, $planes:literal, $convert:path) => {
        pub fn $method(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageViewMut<'_, $ty, C4>,
            alpha: u8,
        ) -> Result<()> {
            $convert(stream_context, source, destination, alpha)
        }
    };
}
