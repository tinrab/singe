macro_rules! impl_rgb_to_uyvp_packed {
    ($name:ident, $source_ty:ty, $source_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, u8, C3>,
            color_space: ColorSpace,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_rgb_to_uyvp_planar {
    ($name:ident, $source_ty:ty, $source_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination_0: &mut ImageViewMut<'_, u8, C1>,
            destination_1: &mut ImageViewMut<'_, u8, C1>,
            destination_2: &mut ImageViewMut<'_, u8, C1>,
            color_space: ColorSpace,
        ) -> Result<()> {
            validate_rgb_destination_planes(
                source.size(),
                destination_0,
                destination_1,
                destination_2,
            )?;

            let mut destination_planes = [
                destination_0.as_mut_ptr().cast(),
                destination_1.as_mut_ptr().cast(),
                destination_2.as_mut_ptr().cast(),
            ];

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination_planes.as_mut_ptr(),
                    destination_0.step(),
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_rgb_planar_to_uyvp_packed {
    ($name:ident, $source_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $source_ty, C1>,
            source_1: &ImageView<'_, $source_ty, C1>,
            source_2: &ImageView<'_, $source_ty, C1>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, u8, C3>,
            color_space: ColorSpace,
        ) -> Result<()> {
            let roi = validate_rgb_source_planes(source_0, source_1, source_2)?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [
                source_0.as_ptr().cast(),
                source_1.as_ptr().cast(),
                source_2.as_ptr(),
            ];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_0.step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    roi.into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_uyvp_to_rgb_planar {
    ($name:ident, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C3>,
            source_offset: Point,
            destination_0: &mut ImageViewMut<'_, $destination_ty, C1>,
            destination_1: &mut ImageViewMut<'_, $destination_ty, C1>,
            destination_2: &mut ImageViewMut<'_, $destination_ty, C1>,
            color_space: ColorSpace,
        ) -> Result<()> {
            validate_rgb_destination_planes(
                source.size(),
                destination_0,
                destination_1,
                destination_2,
            )?;

            let mut destination_planes = [
                destination_0.as_mut_ptr().cast(),
                destination_1.as_mut_ptr().cast(),
                destination_2.as_mut_ptr().cast(),
            ];

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination_planes.as_mut_ptr(),
                    destination_0.step(),
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_uyvp_to_rgb_packed {
    ($name:ident, $destination_ty:ty, $destination_layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C3>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            color_space: ColorSpace,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    color_space.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_uyvp_to_rgb_packed_constant_alpha {
    ($name:ident, $destination_ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C3>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $destination_ty, AC4>,
            color_space: ColorSpace,
            alpha: $destination_ty,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    color_space.into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
