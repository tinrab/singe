macro_rules! impl_convert {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageViewMut<'_, $dst_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_convert_round {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageViewMut<'_, $dst_ty, $layout>,
            round_mode: RoundMode,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    round_mode.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_convert_scaled_round {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $src_ty, $layout>,
            destination: &mut ImageViewMut<'_, $dst_ty, $layout>,
            round_mode: RoundMode,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
