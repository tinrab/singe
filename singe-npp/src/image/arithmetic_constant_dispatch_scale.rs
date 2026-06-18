use super::*;

impl_generic_constant_scalar_operation!(
    MultiplyConstantScaleC1,
    multiply_constant_scale,
    multiply_constant_scale_c1,
    C1,
    [
        u8, u8 => multiply_constant_scale_u8_c1,
        u16, u16 => multiply_constant_scale_u16_c1,
    ]
);
impl_generic_constant_scalar_operation_in_place!(
    MultiplyConstantScaleC1InPlace,
    multiply_constant_scale_in_place,
    multiply_constant_scale_c1_in_place,
    C1,
    [
        u8, u8 => multiply_constant_scale_u8_c1_in_place,
        u16, u16 => multiply_constant_scale_u16_c1_in_place,
    ]
);
impl_generic_constant_array_operation!(
    MultiplyConstantScaleC3,
    multiply_constant_scale,
    multiply_constant_scale_c3,
    C3,
    3,
    [
        u8, u8 => multiply_constant_scale_u8_c3,
        u16, u16 => multiply_constant_scale_u16_c3,
    ]
);
impl_generic_constant_array_operation_in_place!(
    MultiplyConstantScaleC3InPlace,
    multiply_constant_scale_in_place,
    multiply_constant_scale_c3_in_place,
    C3,
    3,
    [
        u8, u8 => multiply_constant_scale_u8_c3_in_place,
        u16, u16 => multiply_constant_scale_u16_c3_in_place,
    ]
);
impl_generic_constant_array_operation!(
    MultiplyConstantScaleC4,
    multiply_constant_scale,
    multiply_constant_scale_c4,
    C4,
    4,
    [
        u8, u8 => multiply_constant_scale_u8_c4,
        u16, u16 => multiply_constant_scale_u16_c4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    MultiplyConstantScaleC4InPlace,
    multiply_constant_scale_in_place,
    multiply_constant_scale_c4_in_place,
    C4,
    4,
    [
        u8, u8 => multiply_constant_scale_u8_c4_in_place,
        u16, u16 => multiply_constant_scale_u16_c4_in_place,
    ]
);
impl_generic_constant_array_operation!(
    MultiplyConstantScaleAc4,
    multiply_constant_scale,
    multiply_constant_scale_ac4,
    AC4,
    3,
    [
        u8, u8 => multiply_constant_scale_u8_ac4,
        u16, u16 => multiply_constant_scale_u16_ac4,
    ]
);
impl_generic_constant_array_operation_in_place!(
    MultiplyConstantScaleAc4InPlace,
    multiply_constant_scale_in_place,
    multiply_constant_scale_ac4_in_place,
    AC4,
    3,
    [
        u8, u8 => multiply_constant_scale_u8_ac4_in_place,
        u16, u16 => multiply_constant_scale_u16_ac4_in_place,
    ]
);

impl_generic_device_constant_operation!(
    MultiplyDeviceConstantScaleC1,
    multiply_device_constant_scale,
    multiply_device_constant_scale_c1,
    C1,
    [
        u8, u8 => multiply_device_constant_scale_u8_c1,
        u16, u16 => multiply_device_constant_scale_u16_c1,
    ]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantScaleC1InPlace,
    multiply_device_constant_scale_in_place,
    multiply_device_constant_scale_c1_in_place,
    C1,
    [
        u8, u8 => multiply_device_constant_scale_u8_c1_in_place,
        u16, u16 => multiply_device_constant_scale_u16_c1_in_place,
    ]
);
impl_generic_device_constant_operation!(
    MultiplyDeviceConstantScaleC3,
    multiply_device_constant_scale,
    multiply_device_constant_scale_c3,
    C3,
    [
        u8, u8 => multiply_device_constant_scale_u8_c3,
        u16, u16 => multiply_device_constant_scale_u16_c3,
    ]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantScaleC3InPlace,
    multiply_device_constant_scale_in_place,
    multiply_device_constant_scale_c3_in_place,
    C3,
    [
        u8, u8 => multiply_device_constant_scale_u8_c3_in_place,
        u16, u16 => multiply_device_constant_scale_u16_c3_in_place,
    ]
);
impl_generic_device_constant_operation!(
    MultiplyDeviceConstantScaleC4,
    multiply_device_constant_scale,
    multiply_device_constant_scale_c4,
    C4,
    [
        u8, u8 => multiply_device_constant_scale_u8_c4,
        u16, u16 => multiply_device_constant_scale_u16_c4,
    ]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantScaleC4InPlace,
    multiply_device_constant_scale_in_place,
    multiply_device_constant_scale_c4_in_place,
    C4,
    [
        u8, u8 => multiply_device_constant_scale_u8_c4_in_place,
        u16, u16 => multiply_device_constant_scale_u16_c4_in_place,
    ]
);
impl_generic_device_constant_operation!(
    MultiplyDeviceConstantScaleAc4,
    multiply_device_constant_scale,
    multiply_device_constant_scale_ac4,
    AC4,
    [
        u8, u8 => multiply_device_constant_scale_u8_ac4,
        u16, u16 => multiply_device_constant_scale_u16_ac4,
    ]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantScaleAc4InPlace,
    multiply_device_constant_scale_in_place,
    multiply_device_constant_scale_ac4_in_place,
    AC4,
    [
        u8, u8 => multiply_device_constant_scale_u8_ac4_in_place,
        u16, u16 => multiply_device_constant_scale_u16_ac4_in_place,
    ]
);

impl_generic_constant_scalar_operation!(
    AbsoluteDifferenceConstantC1,
    absolute_difference_constant,
    absolute_difference_constant_c1,
    C1,
    [
        u8, u8 => absolute_difference_constant_u8_c1,
        u16, u16 => absolute_difference_constant_u16_c1,
        f32, f32 => absolute_difference_constant_f32_c1,
    ]
);
impl_generic_device_constant_operation!(
    AbsoluteDifferenceDeviceConstantC1,
    absolute_difference_device_constant,
    absolute_difference_device_constant_c1,
    C1,
    [
        u8, u8 => absolute_difference_device_constant_u8_c1,
        u16, u16 => absolute_difference_device_constant_u16_c1,
        f32, f32 => absolute_difference_device_constant_f32_c1,
    ]
);
