macro_rules! impl_subsampled_p2_to_p3 {
    (
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cbcr: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_cb: &mut ImageViewMut<'_, u8, C1>,
            destination_cr: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            let roi = validate_subsampled_p2_source_planes(
                source_y,
                source_cbcr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_cb,
                destination_cr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
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
                    source_y.as_ptr().cast(),
                    source_y.step(),
                    source_cbcr.as_ptr().cast(),
                    source_cbcr.step(),
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

macro_rules! impl_subsampled_c2_to_p3 {
    (
        $name:ident,
        $ffi:ident,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_cb: &mut ImageViewMut<'_, u8, C1>,
            destination_cr: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            validate_jpeg_subsampled_destination_planes(
                source.size(),
                destination_y,
                destination_cb,
                destination_cr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
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

macro_rules! impl_subsampled_c2_to_p2 {
    (
        $name:ident,
        $ffi:ident,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, u8, C2>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_cbcr: &mut ImageViewMut<'_, u8, C2>,
        ) -> Result<()> {
            validate_subsampled_p2_destination_planes(
                source.size(),
                destination_y,
                destination_cbcr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination_y.as_mut_ptr().cast(),
                    destination_y.step(),
                    destination_cbcr.as_mut_ptr().cast(),
                    destination_cbcr.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_p3_to_p2 {
    (
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_cbcr: &mut ImageViewMut<'_, u8, C2>,
        ) -> Result<()> {
            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_subsampled_p2_destination_planes(
                roi,
                destination_y,
                destination_cbcr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];

            unsafe {
                try_ffi!(sys::$ffi(
                    source_planes.as_ptr().cast_mut(),
                    source_steps.as_mut_ptr().cast(),
                    destination_y.as_mut_ptr().cast(),
                    destination_y.step(),
                    destination_cbcr.as_mut_ptr().cast(),
                    destination_cbcr.step(),
                    roi.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_subsampled_p3_to_c2 {
    (
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination: &mut ImageViewMut<'_, u8, C2>,
        ) -> Result<()> {
            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
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
                    source_planes.as_ptr().cast_mut(),
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

macro_rules! impl_subsampled_p3_to_p3 {
    (
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cb: &ImageView<'_, u8, C1>,
            source_cr: &ImageView<'_, u8, C1>,
            destination_y: &mut ImageViewMut<'_, u8, C1>,
            destination_cb: &mut ImageViewMut<'_, u8, C1>,
            destination_cr: &mut ImageViewMut<'_, u8, C1>,
        ) -> Result<()> {
            let roi = validate_jpeg_subsampled_source_planes(
                source_y,
                source_cb,
                source_cr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_jpeg_subsampled_destination_planes(
                roi,
                destination_y,
                destination_cb,
                destination_cr,
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
            )?;

            let source_planes = [
                source_y.as_ptr().cast(),
                source_cb.as_ptr().cast(),
                source_cr.as_ptr(),
            ];
            let mut source_steps = [source_y.step(), source_cb.step(), source_cr.step()];
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
                    source_planes.as_ptr().cast_mut(),
                    source_steps.as_mut_ptr().cast(),
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

macro_rules! impl_subsampled_p2_to_c2 {
    (
        $name:ident,
        $ffi:ident,
        $source_horizontal_subsampling:literal,
        $source_vertical_subsampling:literal
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, u8, C1>,
            source_cbcr: &ImageView<'_, u8, C2>,
            destination: &mut ImageViewMut<'_, u8, C2>,
        ) -> Result<()> {
            let roi = validate_subsampled_p2_source_planes(
                source_y,
                source_cbcr,
                $source_horizontal_subsampling,
                $source_vertical_subsampling,
            )?;
            validate_same_size(roi, destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_y.as_ptr().cast(),
                    source_y.step(),
                    source_cbcr.as_ptr().cast(),
                    source_cbcr.step(),
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

macro_rules! impl_subsampled_packed_to_p3 {
    (
        $name:ident,
        $layout:ty,
        $ffi:ident,
        $destination_horizontal_subsampling:literal,
        $destination_vertical_subsampling:literal
    ) => {
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
                $destination_horizontal_subsampling,
                $destination_vertical_subsampling,
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
