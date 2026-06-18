macro_rules! impl_filter_box_border_advanced {
    ($buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident, $buffer_size_ffi:ident, $channels:expr) => {
        pub fn $buffer_size_name(roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(roi.into(), $channels, &raw mut bytes,))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_mask(mask_size, anchor)?;
            let required_bytes = $buffer_size_name(destination.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size.into(),
                    anchor.into(),
                    border_type.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_advanced {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[f32],
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_advanced_filter_kernel(kernel)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    kernel.as_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_advanced_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[f32],
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_advanced_filter_kernel(kernel)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    kernel.as_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_gauss_pyramid_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            rate: f32,
            kernel: &[f32],
            border_type: BorderType,
        ) -> Result<()> {
            validate_gauss_pyramid_rate(rate)?;
            validate_advanced_filter_kernel(kernel)?;
            validate_source_offset(source.size(), source_offset)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    rate,
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    kernel.as_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_bilateral_gauss_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            radius: i32,
            step_between_source_pixels: i32,
            value_square_sigma: f32,
            position_square_sigma: f32,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_bilateral_gauss_parameters(
                radius,
                step_between_source_pixels,
                value_square_sigma,
                position_square_sigma,
                border_type,
            )?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    radius,
                    step_between_source_pixels,
                    value_square_sigma,
                    position_square_sigma,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
