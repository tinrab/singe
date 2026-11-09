#[macro_use]
mod uyvp_dispatch;
#[macro_use]
mod planar_dispatch_macros;

macro_rules! impl_static_different_layout_convert {
    ($doc:literal, $method:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $ty, $destination_layout>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe { $convert(stream_context, source, destination) })
        }
    };
}

macro_rules! impl_static_packed_to_planar_convert {
    ($doc:literal, $method:ident, $ty:ty, $source_layout:ty, $planes:literal, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            for plane in destination.planes() {
                stream_context.prepare_region(plane)?;
            }
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe { $convert(stream_context, source, destination) })
        }
    };
}

macro_rules! impl_static_planar_to_packed_convert {
    ($doc:literal, $method:ident, $ty:ty, $planes:literal, $destination_layout:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageOutputView<'_, $ty, $destination_layout>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(destination)?;
            for plane in source.planes() {
                stream_context.prepare_region(plane)?;
            }
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe { $convert(stream_context, source, destination) })
        }
    };
}

macro_rules! impl_static_planar_to_packed_constant_alpha_convert {
    ($doc:literal, $method:ident, $ty:ty, $planes:literal, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, $ty, $planes>,
            destination: &mut ImageOutputView<'_, $ty, C4>,
            alpha: u8,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(destination)?;
            for plane in source.planes() {
                stream_context.prepare_region(plane)?;
            }
            stream_context
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                .sync_scope(|| unsafe { $convert(stream_context, source, destination, alpha) })
        }
    };
}
