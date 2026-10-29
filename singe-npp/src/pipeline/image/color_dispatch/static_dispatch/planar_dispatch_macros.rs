macro_rules! impl_static_planar_to_c2_convert {
    ($doc:literal, $method:ident, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, u8, C1>,
            source_1: &ImageView<'_, u8, C1>,
            source_2: &ImageView<'_, u8, C1>,
            destination: &mut ImageOutputView<'_, u8, C2>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source_0)?;
            stream_context.prepare_region(source_1)?;
            stream_context.prepare_region(source_2)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(stream_context, source_0, source_1, source_2, destination)
            })
        }
    };
}

macro_rules! impl_static_c2_to_planar_convert {
    ($doc:literal, $method:ident, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_0: &mut ImageOutputView<'_, u8, C1>,
            destination_1: &mut ImageOutputView<'_, u8, C1>,
            destination_2: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination_0)?;
            stream_context.prepare_region(destination_1)?;
            stream_context.prepare_region(destination_2)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source,
                    destination_0,
                    destination_1,
                    destination_2,
                )
            })
        }
    };
}

macro_rules! impl_static_c2_to_planar3_convert {
    ($doc:literal, $method:ident, $convert:path) => {
        impl_static_c2_to_planar_convert!($doc, $method, $convert);
    };
}

macro_rules! impl_static_planar3_to_planar3_convert {
    ($doc:literal, $method:ident, $ty:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination_0: &mut ImageOutputView<'_, $ty, C1>,
            destination_1: &mut ImageOutputView<'_, $ty, C1>,
            destination_2: &mut ImageOutputView<'_, $ty, C1>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source_0)?;
            stream_context.prepare_region(source_1)?;
            stream_context.prepare_region(source_2)?;
            stream_context.prepare_region(destination_0)?;
            stream_context.prepare_region(destination_1)?;
            stream_context.prepare_region(destination_2)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source_0,
                    source_1,
                    source_2,
                    destination_0,
                    destination_1,
                    destination_2,
                )
            })
        }
    };
}

macro_rules! impl_static_planar3_to_planar2_convert {
    ($doc:literal, $method:ident, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, u8, C1>,
            source_1: &ImageView<'_, u8, C1>,
            source_2: &ImageView<'_, u8, C1>,
            destination_0: &mut ImageOutputView<'_, u8, C1>,
            destination_1: &mut ImageOutputView<'_, u8, C2>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source_0)?;
            stream_context.prepare_region(source_1)?;
            stream_context.prepare_region(source_2)?;
            stream_context.prepare_region(destination_0)?;
            stream_context.prepare_region(destination_1)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source_0,
                    source_1,
                    source_2,
                    destination_0,
                    destination_1,
                )
            })
        }
    };
}

macro_rules! impl_static_c2_to_planar2_convert {
    ($doc:literal, $method:ident, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_0: &mut ImageOutputView<'_, u8, C1>,
            destination_1: &mut ImageOutputView<'_, u8, C2>,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination_0)?;
            stream_context.prepare_region(destination_1)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(stream_context, source, destination_0, destination_1)
            })
        }
    };
}
