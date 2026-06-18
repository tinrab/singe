macro_rules! impl_nv12_color_twist_planar_forward {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, $ty, C1>,
            source_1: &ImageView<'_, $ty, C1>,
            source_2: &ImageView<'_, $ty, C1>,
            destination_y: &mut ImageViewMut<'_, $ty, C1>,
            destination_uv: &mut ImageViewMut<'_, $ty, C2>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            let roi = validate_rgb_source_planes(source_0, source_1, source_2)?;
            validate_nv12_destination_planes(roi, destination_y, destination_uv)?;

            let source_planes = [
                source_0.as_ptr().cast(),
                source_1.as_ptr().cast(),
                source_2.as_ptr(),
            ];
            let mut source_steps = [source_0.step(), source_1.step(), source_2.step()];
            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_uv.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [destination_y.step(), destination_uv.step()];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    roi.into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_nv12_color_twist_packed_forward {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination_y: &mut ImageViewMut<'_, $ty, C1>,
            destination_uv: &mut ImageViewMut<'_, $ty, C2>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            validate_nv12_destination_planes(source.size(), destination_y, destination_uv)?;

            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_uv.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [destination_y.step(), destination_uv.step()];

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    source.size().into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_nv12_color_twist_packed_inverse {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, $ty, C1>,
            source_uv: &ImageView<'_, $ty, C2>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            let roi = validate_nv12_source_planes(source_y, source_uv)?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [source_y.as_ptr().cast(), source_uv.as_ptr().cast()];
            let mut source_steps = [source_y.step(), source_uv.step()];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    roi.into(),
                    twist.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_nv12_packed_inverse {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, $ty, C1>,
            source_uv: &ImageView<'_, $ty, C2>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            let roi = validate_nv12_source_planes(source_y, source_uv)?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [source_y.as_ptr().cast(), source_uv.as_ptr().cast()];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_y.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_nv12_to_planar {
    ($name:ident, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_uv: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_u: &mut ImageViewMut<'_, u8, C1>,
            destination_v: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            let roi = validate_nv12_source_planes(source_y, source_uv)?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_u,
                destination_v,
                2,
                2,
            )?;

            let source_planes = [source_y.as_ptr().cast(), source_uv.as_ptr().cast()];
            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_u.as_mut_ptr().cast(),
                destination_v.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_u.step(),
                destination_v.step(),
            ];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_y.step(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
