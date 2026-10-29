macro_rules! impl_static_planar3_to_packed_twist {
    ($doc:literal, $method:ident, $ty:ty, $destination_layout:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, $destination_layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source_0)?;
            stream_context.prepare_region(source_1)?;
            stream_context.prepare_region(source_2)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source_0,
                    source_1,
                    source_2,
                    destination,
                    twist,
                )
            })
        }
    };
}

macro_rules! impl_static_planar3_to_ac4_twist {
    ($doc:literal, $method:ident, $ty:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination: &mut ImageOutputView<'_, $ty, AC4>,
            twist: ColorTwistMatrix,
            alpha: $ty,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source_0)?;
            stream_context.prepare_region(source_1)?;
            stream_context.prepare_region(source_2)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source_0,
                    source_1,
                    source_2,
                    destination,
                    twist,
                    alpha,
                )
            })
        }
    };
}

macro_rules! impl_static_planar3_to_planar3_twist {
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
            twist: ColorTwistMatrix,
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
                    twist,
                )
            })
        }
    };
}

macro_rules! impl_static_packed_to_planar3_twist {
    ($doc:literal, $method:ident, $ty:ty, $source_layout:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination_0: &mut ImageOutputView<'_, $ty, C1>,
            destination_1: &mut ImageOutputView<'_, $ty, C1>,
            destination_2: &mut ImageOutputView<'_, $ty, C1>,
            twist: ColorTwistMatrix,
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
                    twist,
                )
            })
        }
    };
}

macro_rules! impl_static_different_layout_twist {
    ($doc:literal, $method:ident, $ty:ty, $source_layout:ty, $destination_layout:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $source_layout>,
            destination: &mut ImageOutputView<'_, $ty, $destination_layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination)?;
            stream_context
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                .sync_scope(|| unsafe { $convert(stream_context, source, destination, twist) })
        }
    };
}
