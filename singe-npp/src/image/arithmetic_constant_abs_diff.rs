use super::*;

macro_rules! impl_absolute_difference_constant {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            constant: $ty,
            destination: &mut ImageViewMut<'_, $ty, C1>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    constant.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_absolute_difference_device_constant {
    ($name:ident, $ty:ty, $ffi:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            constant: &DeviceMemory<$ty>,
            destination: &mut ImageViewMut<'_, $ty, C1>,
        ) -> Result<()> {
            validate_same_size(source.size(), destination.size())?;
            let constant = device_constant_mut_ptr(constant, 1)?;
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    constant.cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_absolute_difference_constant!(
    absolute_difference_constant_u8_c1,
    u8,
    nppiAbsDiffC_8u_C1R_Ctx
);
impl_absolute_difference_constant!(
    absolute_difference_constant_u16_c1,
    u16,
    nppiAbsDiffC_16u_C1R_Ctx
);
impl_absolute_difference_constant!(
    absolute_difference_constant_f32_c1,
    f32,
    nppiAbsDiffC_32f_C1R_Ctx
);
impl_absolute_difference_device_constant!(
    absolute_difference_device_constant_u8_c1,
    u8,
    nppiAbsDiffDeviceC_8u_C1R_Ctx
);
impl_absolute_difference_device_constant!(
    absolute_difference_device_constant_u16_c1,
    u16,
    nppiAbsDiffDeviceC_16u_C1R_Ctx
);
impl_absolute_difference_device_constant!(
    absolute_difference_device_constant_f32_c1,
    f32,
    nppiAbsDiffDeviceC_32f_C1R_Ctx
);
