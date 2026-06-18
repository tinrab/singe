macro_rules! impl_jpeg_subsampled_planar_forward {
    ($name:ident, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, u8, C1>,
            source_1: &ImageView<'_, u8, C1>,
            source_2: &ImageView<'_, u8, C1>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_cb: &mut ImageViewMut<'_, u8, C1>,
            destination_cr: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            let roi = validate_rgb_source_planes(source_0, source_1, source_2)?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_cb,
                destination_cr,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;

            let source_planes = [
                source_0.as_ptr().cast(),
                source_1.as_ptr().cast(),
                source_2.as_ptr(),
            ];
            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_cb.as_mut_ptr().cast(),
                destination_cr.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_cb.step(),
                destination_cr.step(),
            ];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_0.step(),
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

macro_rules! impl_jpeg_subsampled_packed_forward {
    ($name:ident, $layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, $layout>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_cb: &mut ImageViewMut<'_, u8, C1>,
            destination_cr: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            validate_jpeg_subsampled_destination_planes(
                source.size(),
                destination_y,
                destination_cb,
                destination_cr,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;

            let mut destination_planes = [
                destination_y.as_mut_ptr().cast(),
                destination_cb.as_mut_ptr().cast(),
                destination_cr.as_mut_ptr().cast(),
            ];
            let mut destination_steps = [
                destination_y.step(),
                destination_cb.step(),
                destination_cr.step(),
            ];

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_planes.as_mut_ptr(),
                    destination_steps.as_mut_ptr().cast(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_jpeg_subsampled_planar_inverse {
    ($name:ident, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination_0: &mut ImageViewMut<'_, u8, C1>,
            destination_1: &mut ImageViewMut<'_, u8, C1>,
            destination_2: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;
            validate_rgb_destination_planes(roi, destination_0, destination_1, destination_2)?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];
            let mut destination_planes = [
                destination_0.as_mut_ptr().cast(),
                destination_1.as_mut_ptr().cast(),
                destination_2.as_mut_ptr().cast(),
            ];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
                    destination_planes.as_mut_ptr(),
                    destination_0.step(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_jpeg_subsampled_packed_inverse {
    ($name:ident, $layout:ty, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination: &mut ImageViewMut<'_, u8, $layout>,
        ) -> Result<()> {
            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
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

macro_rules! impl_jpeg_subsampled_packed_inverse_constant_alpha {
    ($name:ident, $ffi:ident, $horizontal_subsampling:literal, $vertical_subsampling:literal) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination: &mut ImageViewMut<'_, u8, C4>,
            alpha: u8,
        ) -> Result<()> {
            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $horizontal_subsampling,
                $vertical_subsampling,
            )?;
            validate_same_size(roi, destination.size())?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr(),
                    source_steps.as_mut_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    roi.into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
