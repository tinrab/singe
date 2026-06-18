use super::*;

impl_generic_device_constant_operation!(
    AddDeviceConstantC1,
    add_device_constant,
    add_device_constant_c1,
    C1,
    [
        f16, f32 => add_device_constant_f16_c1,
        f32, f32 => add_device_constant_f32_c1,
    ]
);
impl_generic_device_constant_operation_in_place!(
    AddDeviceConstantC1InPlace,
    add_device_constant_in_place,
    add_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => add_device_constant_f16_c1_in_place,
        f32, f32 => add_device_constant_f32_c1_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledAddDeviceConstantC1,
    add_device_constant_scaled,
    add_device_constant_scaled_c1,
    C1,
    [
        u8 => add_device_constant_u8_c1,
        u16 => add_device_constant_u16_c1,
        i16 => add_device_constant_i16_c1,
        i32 => add_device_constant_i32_c1,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledAddDeviceConstantC1InPlace,
    add_device_constant_scaled_in_place,
    add_device_constant_scaled_c1_in_place,
    C1,
    [
        u8 => add_device_constant_u8_c1_in_place,
        u16 => add_device_constant_u16_c1_in_place,
        i16 => add_device_constant_i16_c1_in_place,
        i32 => add_device_constant_i32_c1_in_place,
    ]
);
impl_generic_device_constant_operation!(
    AddDeviceConstantC3,
    add_device_constant,
    add_device_constant_c3,
    C3,
    [
        f16, f32 => add_device_constant_f16_c3,
        f32, f32 => add_device_constant_f32_c3,
    ]
);
impl_generic_device_constant_operation_in_place!(
    AddDeviceConstantC3InPlace,
    add_device_constant_in_place,
    add_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => add_device_constant_f16_c3_in_place,
        f32, f32 => add_device_constant_f32_c3_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledAddDeviceConstantC3,
    add_device_constant_scaled,
    add_device_constant_scaled_c3,
    C3,
    [
        u8 => add_device_constant_u8_c3,
        u16 => add_device_constant_u16_c3,
        i16 => add_device_constant_i16_c3,
        i32 => add_device_constant_i32_c3,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledAddDeviceConstantC3InPlace,
    add_device_constant_scaled_in_place,
    add_device_constant_scaled_c3_in_place,
    C3,
    [
        u8 => add_device_constant_u8_c3_in_place,
        u16 => add_device_constant_u16_c3_in_place,
        i16 => add_device_constant_i16_c3_in_place,
        i32 => add_device_constant_i32_c3_in_place,
    ]
);
impl_generic_device_constant_operation!(
    AddDeviceConstantC4,
    add_device_constant,
    add_device_constant_c4,
    C4,
    [
        f16, f32 => add_device_constant_f16_c4,
        f32, f32 => add_device_constant_f32_c4,
    ]
);
impl_generic_device_constant_operation_in_place!(
    AddDeviceConstantC4InPlace,
    add_device_constant_in_place,
    add_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => add_device_constant_f16_c4_in_place,
        f32, f32 => add_device_constant_f32_c4_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledAddDeviceConstantC4,
    add_device_constant_scaled,
    add_device_constant_scaled_c4,
    C4,
    [
        u8 => add_device_constant_u8_c4,
        u16 => add_device_constant_u16_c4,
        i16 => add_device_constant_i16_c4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledAddDeviceConstantC4InPlace,
    add_device_constant_scaled_in_place,
    add_device_constant_scaled_c4_in_place,
    C4,
    [
        u8 => add_device_constant_u8_c4_in_place,
        u16 => add_device_constant_u16_c4_in_place,
        i16 => add_device_constant_i16_c4_in_place,
    ]
);
impl_generic_device_constant_operation!(
    AddDeviceConstantAc4,
    add_device_constant,
    add_device_constant_ac4,
    AC4,
    [f32, f32 => add_device_constant_f32_ac4]
);
impl_generic_device_constant_operation_in_place!(
    AddDeviceConstantAc4InPlace,
    add_device_constant_in_place,
    add_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => add_device_constant_f32_ac4_in_place]
);
impl_generic_scaled_device_constant_operation!(
    ScaledAddDeviceConstantAc4,
    add_device_constant_scaled,
    add_device_constant_scaled_ac4,
    AC4,
    [
        u8 => add_device_constant_u8_ac4,
        u16 => add_device_constant_u16_ac4,
        i16 => add_device_constant_i16_ac4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledAddDeviceConstantAc4InPlace,
    add_device_constant_scaled_in_place,
    add_device_constant_scaled_ac4_in_place,
    AC4,
    [
        u8 => add_device_constant_u8_ac4_in_place,
        u16 => add_device_constant_u16_ac4_in_place,
        i16 => add_device_constant_i16_ac4_in_place,
    ]
);

impl_generic_device_constant_operation!(
    MultiplyDeviceConstantC1,
    multiply_device_constant,
    multiply_device_constant_c1,
    C1,
    [
        f16, f32 => multiply_device_constant_f16_c1,
        f32, f32 => multiply_device_constant_f32_c1,
    ]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantC1InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => multiply_device_constant_f16_c1_in_place,
        f32, f32 => multiply_device_constant_f32_c1_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledMultiplyDeviceConstantC1,
    multiply_device_constant_scaled,
    multiply_device_constant_scaled_c1,
    C1,
    [
        u8 => multiply_device_constant_u8_c1,
        u16 => multiply_device_constant_u16_c1,
        i16 => multiply_device_constant_i16_c1,
        i32 => multiply_device_constant_i32_c1,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledMultiplyDeviceConstantC1InPlace,
    multiply_device_constant_scaled_in_place,
    multiply_device_constant_scaled_c1_in_place,
    C1,
    [
        u8 => multiply_device_constant_u8_c1_in_place,
        u16 => multiply_device_constant_u16_c1_in_place,
        i16 => multiply_device_constant_i16_c1_in_place,
        i32 => multiply_device_constant_i32_c1_in_place,
    ]
);
impl_generic_device_constant_operation!(
    MultiplyDeviceConstantC3,
    multiply_device_constant,
    multiply_device_constant_c3,
    C3,
    [
        f16, f32 => multiply_device_constant_f16_c3,
        f32, f32 => multiply_device_constant_f32_c3,
    ]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantC3InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => multiply_device_constant_f16_c3_in_place,
        f32, f32 => multiply_device_constant_f32_c3_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledMultiplyDeviceConstantC3,
    multiply_device_constant_scaled,
    multiply_device_constant_scaled_c3,
    C3,
    [
        u8 => multiply_device_constant_u8_c3,
        u16 => multiply_device_constant_u16_c3,
        i16 => multiply_device_constant_i16_c3,
        i32 => multiply_device_constant_i32_c3,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledMultiplyDeviceConstantC3InPlace,
    multiply_device_constant_scaled_in_place,
    multiply_device_constant_scaled_c3_in_place,
    C3,
    [
        u8 => multiply_device_constant_u8_c3_in_place,
        u16 => multiply_device_constant_u16_c3_in_place,
        i16 => multiply_device_constant_i16_c3_in_place,
        i32 => multiply_device_constant_i32_c3_in_place,
    ]
);
impl_generic_device_constant_operation!(
    MultiplyDeviceConstantC4,
    multiply_device_constant,
    multiply_device_constant_c4,
    C4,
    [
        f16, f32 => multiply_device_constant_f16_c4,
        f32, f32 => multiply_device_constant_f32_c4,
    ]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantC4InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => multiply_device_constant_f16_c4_in_place,
        f32, f32 => multiply_device_constant_f32_c4_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledMultiplyDeviceConstantC4,
    multiply_device_constant_scaled,
    multiply_device_constant_scaled_c4,
    C4,
    [
        u8 => multiply_device_constant_u8_c4,
        u16 => multiply_device_constant_u16_c4,
        i16 => multiply_device_constant_i16_c4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledMultiplyDeviceConstantC4InPlace,
    multiply_device_constant_scaled_in_place,
    multiply_device_constant_scaled_c4_in_place,
    C4,
    [
        u8 => multiply_device_constant_u8_c4_in_place,
        u16 => multiply_device_constant_u16_c4_in_place,
        i16 => multiply_device_constant_i16_c4_in_place,
    ]
);
impl_generic_device_constant_operation!(
    MultiplyDeviceConstantAc4,
    multiply_device_constant,
    multiply_device_constant_ac4,
    AC4,
    [f32, f32 => multiply_device_constant_f32_ac4]
);
impl_generic_device_constant_operation_in_place!(
    MultiplyDeviceConstantAc4InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => multiply_device_constant_f32_ac4_in_place]
);
impl_generic_scaled_device_constant_operation!(
    ScaledMultiplyDeviceConstantAc4,
    multiply_device_constant_scaled,
    multiply_device_constant_scaled_ac4,
    AC4,
    [
        u8 => multiply_device_constant_u8_ac4,
        u16 => multiply_device_constant_u16_ac4,
        i16 => multiply_device_constant_i16_ac4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledMultiplyDeviceConstantAc4InPlace,
    multiply_device_constant_scaled_in_place,
    multiply_device_constant_scaled_ac4_in_place,
    AC4,
    [
        u8 => multiply_device_constant_u8_ac4_in_place,
        u16 => multiply_device_constant_u16_ac4_in_place,
        i16 => multiply_device_constant_i16_ac4_in_place,
    ]
);

impl_generic_device_constant_operation!(
    SubtractDeviceConstantC1,
    subtract_device_constant,
    subtract_device_constant_c1,
    C1,
    [
        f16, f32 => subtract_device_constant_f16_c1,
        f32, f32 => subtract_device_constant_f32_c1,
    ]
);
impl_generic_device_constant_operation_in_place!(
    SubtractDeviceConstantC1InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => subtract_device_constant_f16_c1_in_place,
        f32, f32 => subtract_device_constant_f32_c1_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledSubtractDeviceConstantC1,
    subtract_device_constant_scaled,
    subtract_device_constant_scaled_c1,
    C1,
    [
        u8 => subtract_device_constant_u8_c1,
        u16 => subtract_device_constant_u16_c1,
        i16 => subtract_device_constant_i16_c1,
        i32 => subtract_device_constant_i32_c1,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledSubtractDeviceConstantC1InPlace,
    subtract_device_constant_scaled_in_place,
    subtract_device_constant_scaled_c1_in_place,
    C1,
    [
        u8 => subtract_device_constant_u8_c1_in_place,
        u16 => subtract_device_constant_u16_c1_in_place,
        i16 => subtract_device_constant_i16_c1_in_place,
        i32 => subtract_device_constant_i32_c1_in_place,
    ]
);
impl_generic_device_constant_operation!(
    SubtractDeviceConstantC3,
    subtract_device_constant,
    subtract_device_constant_c3,
    C3,
    [
        f16, f32 => subtract_device_constant_f16_c3,
        f32, f32 => subtract_device_constant_f32_c3,
    ]
);
impl_generic_device_constant_operation_in_place!(
    SubtractDeviceConstantC3InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => subtract_device_constant_f16_c3_in_place,
        f32, f32 => subtract_device_constant_f32_c3_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledSubtractDeviceConstantC3,
    subtract_device_constant_scaled,
    subtract_device_constant_scaled_c3,
    C3,
    [
        u8 => subtract_device_constant_u8_c3,
        u16 => subtract_device_constant_u16_c3,
        i16 => subtract_device_constant_i16_c3,
        i32 => subtract_device_constant_i32_c3,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledSubtractDeviceConstantC3InPlace,
    subtract_device_constant_scaled_in_place,
    subtract_device_constant_scaled_c3_in_place,
    C3,
    [
        u8 => subtract_device_constant_u8_c3_in_place,
        u16 => subtract_device_constant_u16_c3_in_place,
        i16 => subtract_device_constant_i16_c3_in_place,
        i32 => subtract_device_constant_i32_c3_in_place,
    ]
);
impl_generic_device_constant_operation!(
    SubtractDeviceConstantC4,
    subtract_device_constant,
    subtract_device_constant_c4,
    C4,
    [
        f16, f32 => subtract_device_constant_f16_c4,
        f32, f32 => subtract_device_constant_f32_c4,
    ]
);
impl_generic_device_constant_operation_in_place!(
    SubtractDeviceConstantC4InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => subtract_device_constant_f16_c4_in_place,
        f32, f32 => subtract_device_constant_f32_c4_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledSubtractDeviceConstantC4,
    subtract_device_constant_scaled,
    subtract_device_constant_scaled_c4,
    C4,
    [
        u8 => subtract_device_constant_u8_c4,
        u16 => subtract_device_constant_u16_c4,
        i16 => subtract_device_constant_i16_c4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledSubtractDeviceConstantC4InPlace,
    subtract_device_constant_scaled_in_place,
    subtract_device_constant_scaled_c4_in_place,
    C4,
    [
        u8 => subtract_device_constant_u8_c4_in_place,
        u16 => subtract_device_constant_u16_c4_in_place,
        i16 => subtract_device_constant_i16_c4_in_place,
    ]
);
impl_generic_device_constant_operation!(
    SubtractDeviceConstantAc4,
    subtract_device_constant,
    subtract_device_constant_ac4,
    AC4,
    [f32, f32 => subtract_device_constant_f32_ac4]
);
impl_generic_device_constant_operation_in_place!(
    SubtractDeviceConstantAc4InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => subtract_device_constant_f32_ac4_in_place]
);
impl_generic_scaled_device_constant_operation!(
    ScaledSubtractDeviceConstantAc4,
    subtract_device_constant_scaled,
    subtract_device_constant_scaled_ac4,
    AC4,
    [
        u8 => subtract_device_constant_u8_ac4,
        u16 => subtract_device_constant_u16_ac4,
        i16 => subtract_device_constant_i16_ac4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledSubtractDeviceConstantAc4InPlace,
    subtract_device_constant_scaled_in_place,
    subtract_device_constant_scaled_ac4_in_place,
    AC4,
    [
        u8 => subtract_device_constant_u8_ac4_in_place,
        u16 => subtract_device_constant_u16_ac4_in_place,
        i16 => subtract_device_constant_i16_ac4_in_place,
    ]
);

impl_generic_device_constant_operation!(
    DivideDeviceConstantC1,
    divide_device_constant,
    divide_device_constant_c1,
    C1,
    [
        f16, f32 => divide_device_constant_f16_c1,
        f32, f32 => divide_device_constant_f32_c1,
    ]
);
impl_generic_device_constant_operation_in_place!(
    DivideDeviceConstantC1InPlace,
    divide_device_constant_in_place,
    divide_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => divide_device_constant_f16_c1_in_place,
        f32, f32 => divide_device_constant_f32_c1_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledDivideDeviceConstantC1,
    divide_device_constant_scaled,
    divide_device_constant_scaled_c1,
    C1,
    [
        u8 => divide_device_constant_u8_c1,
        u16 => divide_device_constant_u16_c1,
        i16 => divide_device_constant_i16_c1,
        i32 => divide_device_constant_i32_c1,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledDivideDeviceConstantC1InPlace,
    divide_device_constant_scaled_in_place,
    divide_device_constant_scaled_c1_in_place,
    C1,
    [
        u8 => divide_device_constant_u8_c1_in_place,
        u16 => divide_device_constant_u16_c1_in_place,
        i16 => divide_device_constant_i16_c1_in_place,
        i32 => divide_device_constant_i32_c1_in_place,
    ]
);
impl_generic_device_constant_operation!(
    DivideDeviceConstantC3,
    divide_device_constant,
    divide_device_constant_c3,
    C3,
    [
        f16, f32 => divide_device_constant_f16_c3,
        f32, f32 => divide_device_constant_f32_c3,
    ]
);
impl_generic_device_constant_operation_in_place!(
    DivideDeviceConstantC3InPlace,
    divide_device_constant_in_place,
    divide_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => divide_device_constant_f16_c3_in_place,
        f32, f32 => divide_device_constant_f32_c3_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledDivideDeviceConstantC3,
    divide_device_constant_scaled,
    divide_device_constant_scaled_c3,
    C3,
    [
        u8 => divide_device_constant_u8_c3,
        u16 => divide_device_constant_u16_c3,
        i16 => divide_device_constant_i16_c3,
        i32 => divide_device_constant_i32_c3,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledDivideDeviceConstantC3InPlace,
    divide_device_constant_scaled_in_place,
    divide_device_constant_scaled_c3_in_place,
    C3,
    [
        u8 => divide_device_constant_u8_c3_in_place,
        u16 => divide_device_constant_u16_c3_in_place,
        i16 => divide_device_constant_i16_c3_in_place,
        i32 => divide_device_constant_i32_c3_in_place,
    ]
);
impl_generic_device_constant_operation!(
    DivideDeviceConstantC4,
    divide_device_constant,
    divide_device_constant_c4,
    C4,
    [
        f16, f32 => divide_device_constant_f16_c4,
        f32, f32 => divide_device_constant_f32_c4,
    ]
);
impl_generic_device_constant_operation_in_place!(
    DivideDeviceConstantC4InPlace,
    divide_device_constant_in_place,
    divide_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => divide_device_constant_f16_c4_in_place,
        f32, f32 => divide_device_constant_f32_c4_in_place,
    ]
);
impl_generic_scaled_device_constant_operation!(
    ScaledDivideDeviceConstantC4,
    divide_device_constant_scaled,
    divide_device_constant_scaled_c4,
    C4,
    [
        u8 => divide_device_constant_u8_c4,
        u16 => divide_device_constant_u16_c4,
        i16 => divide_device_constant_i16_c4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledDivideDeviceConstantC4InPlace,
    divide_device_constant_scaled_in_place,
    divide_device_constant_scaled_c4_in_place,
    C4,
    [
        u8 => divide_device_constant_u8_c4_in_place,
        u16 => divide_device_constant_u16_c4_in_place,
        i16 => divide_device_constant_i16_c4_in_place,
    ]
);
impl_generic_device_constant_operation!(
    DivideDeviceConstantAc4,
    divide_device_constant,
    divide_device_constant_ac4,
    AC4,
    [f32, f32 => divide_device_constant_f32_ac4]
);
impl_generic_device_constant_operation_in_place!(
    DivideDeviceConstantAc4InPlace,
    divide_device_constant_in_place,
    divide_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => divide_device_constant_f32_ac4_in_place]
);
impl_generic_scaled_device_constant_operation!(
    ScaledDivideDeviceConstantAc4,
    divide_device_constant_scaled,
    divide_device_constant_scaled_ac4,
    AC4,
    [
        u8 => divide_device_constant_u8_ac4,
        u16 => divide_device_constant_u16_ac4,
        i16 => divide_device_constant_i16_ac4,
    ]
);
impl_generic_scaled_device_constant_operation_in_place!(
    ScaledDivideDeviceConstantAc4InPlace,
    divide_device_constant_scaled_in_place,
    divide_device_constant_scaled_ac4_in_place,
    AC4,
    [
        u8 => divide_device_constant_u8_ac4_in_place,
        u16 => divide_device_constant_u16_ac4_in_place,
        i16 => divide_device_constant_i16_ac4_in_place,
    ]
);
