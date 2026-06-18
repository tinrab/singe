macro_rules! impl_filter_median {
    ($buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi_name:ident) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            mask_size: Size,
        ) -> Result<usize> {
            validate_filter_mask(mask_size, Point { x: 0, y: 0 })?;
            let mut bytes = 0_u32;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    mask_size.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            mask_size: Size,
            anchor: Point,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            validate_filter_mask(mask_size, anchor)?;
            let required_bytes = $buffer_size_name(stream_context, source.size(), mask_size)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask_size.into(),
                    anchor.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_median_border {
    ($buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi_name:ident) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            roi: Size,
            mask_size: Size,
            border_type: BorderType,
        ) -> Result<usize> {
            validate_filter_mask(mask_size, Point { x: 0, y: 0 })?;
            let mut bytes = 0_u32;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    mask_size.into(),
                    &raw mut bytes,
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
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
            let required_bytes =
                $buffer_size_name(stream_context, destination.size(), mask_size, border_type)?;
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
                    scratch.as_mut_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_unsharp_border {
    ($buffer_size_name:ident, $name:ident, $pixel_ty:ty, $layout:ty, $buffer_size_ffi:ident, $ffi_name:ident) => {
        pub fn $buffer_size_name(radius: f32, sigma: f32) -> Result<usize> {
            validate_unsharp_parameters(radius, sigma)?;
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(radius, sigma, &raw mut bytes,))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            radius: f32,
            sigma: f32,
            weight: f32,
            threshold: f32,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            validate_unsharp_parameters(radius, sigma)?;
            let required_bytes = $buffer_size_name(radius, sigma)?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    radius,
                    sigma,
                    weight,
                    threshold,
                    border_type.into(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
