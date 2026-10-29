macro_rules! impl_static_rgb_to_uyvp_planar {
    ($doc:literal, $method:ident, $source_ty:ty, $source_layout:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub unsafe fn $method(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination_0: *mut u8,
            destination_1: *mut u8,
            destination_2: *mut u8,
            destination_step: i32,
            color_space: ColorSpace,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source,
                    source_offset,
                    destination_0,
                    destination_1,
                    destination_2,
                    destination_step,
                    color_space,
                )
            })
        }
    };
}

macro_rules! impl_static_rgb_planar_to_uyvp {
    ($doc:literal, $method:ident, $source_ty:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $source_ty, C1>,
            source_1: &ImageView<'_, $source_ty, C1>,
            source_2: &ImageView<'_, $source_ty, C1>,
            source_offset: Point,
            destination: &mut $crate::image::video::VideoOutputView<'_, u8>,
            color_space: ColorSpace,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source_0)?;
            stream_context.prepare_region(source_1)?;
            stream_context.prepare_region(source_2)?;
            for plane in destination.planes() {
                stream_context.prepare_region(plane)?;
            }
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source_0,
                    source_1,
                    source_2,
                    source_offset,
                    destination,
                    color_space,
                )
            })
        }
    };
}

macro_rules! impl_static_uyvp_to_rgb_planar {
    ($doc:literal, $method:ident, $destination_ty:ty, $convert:path) => {
        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($convert), "`].")]
        pub fn $method(
            stream_context: &StreamContext,
            source: &$crate::image::video::VideoView<'_, u8>,
            source_offset: Point,
            destination_0: &mut ImageOutputView<'_, $destination_ty, C1>,
            destination_1: &mut ImageOutputView<'_, $destination_ty, C1>,
            destination_2: &mut ImageOutputView<'_, $destination_ty, C1>,
            color_space: ColorSpace,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            for plane in source.planes() {
                stream_context.prepare_region(plane)?;
            }
            stream_context.prepare_region(destination_0)?;
            stream_context.prepare_region(destination_1)?;
            stream_context.prepare_region(destination_2)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $convert(
                    stream_context,
                    source,
                    source_offset,
                    destination_0,
                    destination_1,
                    destination_2,
                    color_space,
                )
            })
        }
    };
}
