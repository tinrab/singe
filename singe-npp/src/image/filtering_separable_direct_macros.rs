macro_rules! impl_filter_separable {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[i32],
            anchor: i32,
            divisor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_any(kernel, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    anchor,
                    divisor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_float {
    ($name:ident, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, f32, $layout>,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            kernel: &[f32],
            anchor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_f32(kernel, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    anchor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_separable_float_typed {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            kernel: &[$pixel_ty],
            anchor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_kernel_any(kernel, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    kernel.as_ptr().cast(),
                    i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
                        name: "kernel length".into(),
                    })?,
                    anchor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_sum_window {
    ($name:ident, $source_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_axis_mask(mask_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size,
                    anchor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_sum_window_border {
    ($name:ident, $source_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, f32, $layout>,
            mask_size: i32,
            anchor: i32,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_filter_axis_mask(mask_size, anchor)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    mask_size,
                    anchor,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
