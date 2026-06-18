macro_rules! impl_filter_directional {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_directional_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_directional_typed {
    ($name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_filter_directional_typed_border {
    ($name:ident, $source_ty:ty, $source_layout:ty, $destination_ty:ty, $destination_layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, $source_layout>,
            source_offset: Point,
            destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
