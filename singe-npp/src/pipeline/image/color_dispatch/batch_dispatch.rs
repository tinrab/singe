macro_rules! impl_color_convert_batch_same_layout_image {
    ($doc:literal, $advanced_doc:literal, $name:ident, $advanced_name:ident, $convert:path, $convert_advanced:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, u8, C3>],
            destinations: &mut [ImageOutputView<'_, u8, C3>],
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            for image in sources.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.prepare_region(image)?;
            }
            // SAFETY: the native helper completes borrowed use on every path.
            unsafe { $convert(stream_context, sources, destinations) }
        }

        #[doc = $advanced_doc]
        #[doc = concat!("Uses [`", stringify!($convert_advanced), "`].")]
        pub fn $advanced_name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, u8, C3>],
            destinations: &mut [ImageOutputView<'_, u8, C3>],
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            for image in sources.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.prepare_region(image)?;
            }
            // SAFETY: the native helper completes borrowed use on every path.
            unsafe { $convert_advanced(stream_context, sources, destinations) }
        }
    };
}

macro_rules! impl_color_convert_batch_planar_to_packed_image {
    ($doc:literal, $advanced_doc:literal, $name:ident, $advanced_name:ident, $convert:path, $convert_advanced:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, u8, 3>],
            destinations: &mut [ImageOutputView<'_, u8, C3>],
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            for image in destinations.iter() {
                stream_context.prepare_region(image)?;
            }
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe { $convert(stream_context, sources, destinations) })
        }

        #[doc = $advanced_doc]
        #[doc = concat!("Uses [`", stringify!($convert_advanced), "`].")]
        pub fn $advanced_name(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, u8, 3>],
            destinations: &mut [ImageOutputView<'_, u8, C3>],
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            for image in destinations.iter() {
                stream_context.prepare_region(image)?;
            }
            stream_context
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                .sync_scope(|| unsafe { $convert_advanced(stream_context, sources, destinations) })
        }
    };
}

macro_rules! impl_subsampled_color_convert_batch_planar_to_packed_image {
    ($doc:literal, $advanced_doc:literal, $name:ident, $advanced_name:ident, $convert:path, $convert_advanced:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, u8, C1>],
            source_plane_1: &[ImageView<'_, u8, C1>],
            source_plane_2: &[ImageView<'_, u8, C1>],
            destinations: &mut [ImageOutputView<'_, u8, C3>],
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            for image in source_plane_0.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in source_plane_1.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in source_plane_2.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.prepare_region(image)?;
            }
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source_plane_0,
                    source_plane_1,
                    source_plane_2,
                    destinations,
                )
            })
        }

        #[doc = $advanced_doc]
        #[doc = concat!("Uses [`", stringify!($convert_advanced), "`].")]
        pub fn $advanced_name(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, u8, C1>],
            source_plane_1: &[ImageView<'_, u8, C1>],
            source_plane_2: &[ImageView<'_, u8, C1>],
            destinations: &mut [ImageOutputView<'_, u8, C3>],
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            for image in source_plane_0.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in source_plane_1.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in source_plane_2.iter() {
                stream_context.prepare_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.prepare_region(image)?;
            }
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert_advanced(
                    stream_context,
                    source_plane_0,
                    source_plane_1,
                    source_plane_2,
                    destinations,
                )
            })
        }
    };
}
