macro_rules! impl_add_constant_scaled_scalar {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constant: $ty,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constant.into_npp(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scaled_scalar_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            constant: $ty,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    constant.into_npp(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scaled_array {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constants: [$ty; $channels],
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.as_ptr().cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scaled_array_in_place {
    ($name:ident, $ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            constants: [$ty; $channels],
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.as_ptr().cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_scalar {
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident) => {
        impl_add_constant_scalar!($name, $pixel_ty, $constant_ty, $layout, $ffi, |x| x);
    };
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident, $into:expr) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            constant: $constant_ty,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    $into(constant).into_npp(),
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

macro_rules! impl_add_constant_scalar_in_place {
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident) => {
        impl_add_constant_scalar_in_place!($name, $pixel_ty, $constant_ty, $layout, $ffi, |x| x);
    };
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $ffi:ident, $into:expr) => {
        pub fn $name(
            stream_context: &StreamContext,
            constant: $constant_ty,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    $into(constant).into_npp(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_constant_array {
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            constants: [$constant_ty; $channels],
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.as_ptr().cast(),
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

macro_rules! impl_add_constant_array_in_place {
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $channels:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            constants: [$constant_ty; $channels],
            source_destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.as_ptr().cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_device_constant_scaled {
    ($name:ident, $ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            constants: &DeviceMemory<$ty>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let constants = device_constant_ptr(constants, $constant_len)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.cast(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_device_constant_scaled_in_place {
    ($name:ident, $ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            constants: &DeviceMemory<$ty>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            let constants = device_constant_ptr(constants, $constant_len)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_add_device_constant {
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            constants: &DeviceMemory<$constant_ty>,
            destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let constants = device_constant_ptr(constants, $constant_len)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    constants.cast(),
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

macro_rules! impl_add_device_constant_in_place {
    ($name:ident, $pixel_ty:ty, $constant_ty:ty, $layout:ty, $constant_len:literal, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            constants: &DeviceMemory<$constant_ty>,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, $layout>,
        ) -> Result<()> {
            let constants = device_constant_ptr(constants, $constant_len)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    constants.cast(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_scaled {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_scaled_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_round_scaled {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_round_scaled_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
            round_mode: RoundMode,
        ) -> Result<()> {
            validate_same_size(source.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    round_mode.into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_binary_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), source_destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
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

macro_rules! impl_unary_operation_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation_scaled {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_unary_operation_scaled_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            scale_factor: i32,
        ) -> Result<()> {
            unsafe {
                try_ffi!(sys::$ffi(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    scale_factor,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}
