use singe_cuda::{memory::DeviceMemory, types::f16};

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        raw::arithmetic::{
            add_device_constant_f16_c1, add_device_constant_f16_c1_in_place,
            add_device_constant_f16_c3, add_device_constant_f16_c3_in_place,
            add_device_constant_f16_c4, add_device_constant_f16_c4_in_place,
            add_device_constant_f32_ac4, add_device_constant_f32_ac4_in_place,
            add_device_constant_f32_c1, add_device_constant_f32_c1_in_place,
            add_device_constant_f32_c3, add_device_constant_f32_c3_in_place,
            add_device_constant_f32_c4, add_device_constant_f32_c4_in_place,
            add_device_constant_i16_ac4, add_device_constant_i16_ac4_in_place,
            add_device_constant_i16_c1, add_device_constant_i16_c1_in_place,
            add_device_constant_i16_c3, add_device_constant_i16_c3_in_place,
            add_device_constant_i16_c4, add_device_constant_i16_c4_in_place,
            add_device_constant_i32_c1, add_device_constant_i32_c1_in_place,
            add_device_constant_i32_c3, add_device_constant_i32_c3_in_place,
            add_device_constant_u8_ac4, add_device_constant_u8_ac4_in_place,
            add_device_constant_u8_c1, add_device_constant_u8_c1_in_place,
            add_device_constant_u8_c3, add_device_constant_u8_c3_in_place,
            add_device_constant_u8_c4, add_device_constant_u8_c4_in_place,
            add_device_constant_u16_ac4, add_device_constant_u16_ac4_in_place,
            add_device_constant_u16_c1, add_device_constant_u16_c1_in_place,
            add_device_constant_u16_c3, add_device_constant_u16_c3_in_place,
            add_device_constant_u16_c4, add_device_constant_u16_c4_in_place,
            divide_device_constant_f16_c1, divide_device_constant_f16_c1_in_place,
            divide_device_constant_f16_c3, divide_device_constant_f16_c3_in_place,
            divide_device_constant_f16_c4, divide_device_constant_f16_c4_in_place,
            divide_device_constant_f32_ac4, divide_device_constant_f32_ac4_in_place,
            divide_device_constant_f32_c1, divide_device_constant_f32_c1_in_place,
            divide_device_constant_f32_c3, divide_device_constant_f32_c3_in_place,
            divide_device_constant_f32_c4, divide_device_constant_f32_c4_in_place,
            divide_device_constant_i16_ac4, divide_device_constant_i16_ac4_in_place,
            divide_device_constant_i16_c1, divide_device_constant_i16_c1_in_place,
            divide_device_constant_i16_c3, divide_device_constant_i16_c3_in_place,
            divide_device_constant_i16_c4, divide_device_constant_i16_c4_in_place,
            divide_device_constant_i32_c1, divide_device_constant_i32_c1_in_place,
            divide_device_constant_i32_c3, divide_device_constant_i32_c3_in_place,
            divide_device_constant_u8_ac4, divide_device_constant_u8_ac4_in_place,
            divide_device_constant_u8_c1, divide_device_constant_u8_c1_in_place,
            divide_device_constant_u8_c3, divide_device_constant_u8_c3_in_place,
            divide_device_constant_u8_c4, divide_device_constant_u8_c4_in_place,
            divide_device_constant_u16_ac4, divide_device_constant_u16_ac4_in_place,
            divide_device_constant_u16_c1, divide_device_constant_u16_c1_in_place,
            divide_device_constant_u16_c3, divide_device_constant_u16_c3_in_place,
            divide_device_constant_u16_c4, divide_device_constant_u16_c4_in_place,
            multiply_device_constant_f16_c1, multiply_device_constant_f16_c1_in_place,
            multiply_device_constant_f16_c3, multiply_device_constant_f16_c3_in_place,
            multiply_device_constant_f16_c4, multiply_device_constant_f16_c4_in_place,
            multiply_device_constant_f32_ac4, multiply_device_constant_f32_ac4_in_place,
            multiply_device_constant_f32_c1, multiply_device_constant_f32_c1_in_place,
            multiply_device_constant_f32_c3, multiply_device_constant_f32_c3_in_place,
            multiply_device_constant_f32_c4, multiply_device_constant_f32_c4_in_place,
            multiply_device_constant_i16_ac4, multiply_device_constant_i16_ac4_in_place,
            multiply_device_constant_i16_c1, multiply_device_constant_i16_c1_in_place,
            multiply_device_constant_i16_c3, multiply_device_constant_i16_c3_in_place,
            multiply_device_constant_i16_c4, multiply_device_constant_i16_c4_in_place,
            multiply_device_constant_i32_c1, multiply_device_constant_i32_c1_in_place,
            multiply_device_constant_i32_c3, multiply_device_constant_i32_c3_in_place,
            multiply_device_constant_u8_ac4, multiply_device_constant_u8_ac4_in_place,
            multiply_device_constant_u8_c1, multiply_device_constant_u8_c1_in_place,
            multiply_device_constant_u8_c3, multiply_device_constant_u8_c3_in_place,
            multiply_device_constant_u8_c4, multiply_device_constant_u8_c4_in_place,
            multiply_device_constant_u16_ac4, multiply_device_constant_u16_ac4_in_place,
            multiply_device_constant_u16_c1, multiply_device_constant_u16_c1_in_place,
            multiply_device_constant_u16_c3, multiply_device_constant_u16_c3_in_place,
            multiply_device_constant_u16_c4, multiply_device_constant_u16_c4_in_place,
            subtract_device_constant_f16_c1, subtract_device_constant_f16_c1_in_place,
            subtract_device_constant_f16_c3, subtract_device_constant_f16_c3_in_place,
            subtract_device_constant_f16_c4, subtract_device_constant_f16_c4_in_place,
            subtract_device_constant_f32_ac4, subtract_device_constant_f32_ac4_in_place,
            subtract_device_constant_f32_c1, subtract_device_constant_f32_c1_in_place,
            subtract_device_constant_f32_c3, subtract_device_constant_f32_c3_in_place,
            subtract_device_constant_f32_c4, subtract_device_constant_f32_c4_in_place,
            subtract_device_constant_i16_ac4, subtract_device_constant_i16_ac4_in_place,
            subtract_device_constant_i16_c1, subtract_device_constant_i16_c1_in_place,
            subtract_device_constant_i16_c3, subtract_device_constant_i16_c3_in_place,
            subtract_device_constant_i16_c4, subtract_device_constant_i16_c4_in_place,
            subtract_device_constant_i32_c1, subtract_device_constant_i32_c1_in_place,
            subtract_device_constant_i32_c3, subtract_device_constant_i32_c3_in_place,
            subtract_device_constant_u8_ac4, subtract_device_constant_u8_ac4_in_place,
            subtract_device_constant_u8_c1, subtract_device_constant_u8_c1_in_place,
            subtract_device_constant_u8_c3, subtract_device_constant_u8_c3_in_place,
            subtract_device_constant_u8_c4, subtract_device_constant_u8_c4_in_place,
            subtract_device_constant_u16_ac4, subtract_device_constant_u16_ac4_in_place,
            subtract_device_constant_u16_c1, subtract_device_constant_u16_c1_in_place,
            subtract_device_constant_u16_c3, subtract_device_constant_u16_c3_in_place,
            subtract_device_constant_u16_c4, subtract_device_constant_u16_c4_in_place,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    types::DataTypeLike,
};

impl_generic_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`.",

    AddDeviceConstantC1,
    add_device_constant,
    add_device_constant_c1,
    C1,
    [
        f16, f32 => add_device_constant_f16_c1,
        f32, f32 => add_device_constant_f32_c1,
    ]
);

impl_generic_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    AddDeviceConstantC1InPlace,
    add_device_constant_in_place,
    add_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => add_device_constant_f16_c1_in_place,
        f32, f32 => add_device_constant_f32_c1_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`.",

    AddDeviceConstantC3,
    add_device_constant,
    add_device_constant_c3,
    C3,
    [
        f16, f32 => add_device_constant_f16_c3,
        f32, f32 => add_device_constant_f32_c3,
    ]
);

impl_generic_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    AddDeviceConstantC3InPlace,
    add_device_constant_in_place,
    add_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => add_device_constant_f16_c3_in_place,
        f32, f32 => add_device_constant_f32_c3_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`.",

    AddDeviceConstantC4,
    add_device_constant,
    add_device_constant_c4,
    C4,
    [
        f16, f32 => add_device_constant_f16_c4,
        f32, f32 => add_device_constant_f32_c4,
    ]
);

impl_generic_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    AddDeviceConstantC4InPlace,
    add_device_constant_in_place,
    add_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => add_device_constant_f16_c4_in_place,
        f32, f32 => add_device_constant_f32_c4_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    AddDeviceConstantAc4,
    add_device_constant,
    add_device_constant_ac4,
    AC4,
    [f32, f32 => add_device_constant_f32_ac4]
);

impl_generic_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    AddDeviceConstantAc4InPlace,
    add_device_constant_in_place,
    add_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => add_device_constant_f32_ac4_in_place]
);

impl_generic_scaled_device_constant_operation!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_device_constant_operation_in_place!("Adds a constant to each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",

    MultiplyDeviceConstantC1,
    multiply_device_constant,
    multiply_device_constant_c1,
    C1,
    [
        f16, f32 => multiply_device_constant_f16_c1,
        f32, f32 => multiply_device_constant_f32_c1,
    ]
);

impl_generic_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    MultiplyDeviceConstantC1InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => multiply_device_constant_f16_c1_in_place,
        f32, f32 => multiply_device_constant_f32_c1_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",

    MultiplyDeviceConstantC3,
    multiply_device_constant,
    multiply_device_constant_c3,
    C3,
    [
        f16, f32 => multiply_device_constant_f16_c3,
        f32, f32 => multiply_device_constant_f32_c3,
    ]
);

impl_generic_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    MultiplyDeviceConstantC3InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => multiply_device_constant_f16_c3_in_place,
        f32, f32 => multiply_device_constant_f32_c3_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",

    MultiplyDeviceConstantC4,
    multiply_device_constant,
    multiply_device_constant_c4,
    C4,
    [
        f16, f32 => multiply_device_constant_f16_c4,
        f32, f32 => multiply_device_constant_f32_c4,
    ]
);

impl_generic_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    MultiplyDeviceConstantC4InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => multiply_device_constant_f16_c4_in_place,
        f32, f32 => multiply_device_constant_f32_c4_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    MultiplyDeviceConstantAc4,
    multiply_device_constant,
    multiply_device_constant_ac4,
    AC4,
    [f32, f32 => multiply_device_constant_f32_ac4]
);

impl_generic_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    MultiplyDeviceConstantAc4InPlace,
    multiply_device_constant_in_place,
    multiply_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => multiply_device_constant_f32_ac4_in_place]
);

impl_generic_scaled_device_constant_operation!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_device_constant_operation_in_place!("Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`.",

    SubtractDeviceConstantC1,
    subtract_device_constant,
    subtract_device_constant_c1,
    C1,
    [
        f16, f32 => subtract_device_constant_f16_c1,
        f32, f32 => subtract_device_constant_f32_c1,
    ]
);

impl_generic_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    SubtractDeviceConstantC1InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => subtract_device_constant_f16_c1_in_place,
        f32, f32 => subtract_device_constant_f32_c1_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`.",

    SubtractDeviceConstantC3,
    subtract_device_constant,
    subtract_device_constant_c3,
    C3,
    [
        f16, f32 => subtract_device_constant_f16_c3,
        f32, f32 => subtract_device_constant_f32_c3,
    ]
);

impl_generic_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    SubtractDeviceConstantC3InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => subtract_device_constant_f16_c3_in_place,
        f32, f32 => subtract_device_constant_f32_c3_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`.",

    SubtractDeviceConstantC4,
    subtract_device_constant,
    subtract_device_constant_c4,
    C4,
    [
        f16, f32 => subtract_device_constant_f16_c4,
        f32, f32 => subtract_device_constant_f32_c4,
    ]
);

impl_generic_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    SubtractDeviceConstantC4InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => subtract_device_constant_f16_c4_in_place,
        f32, f32 => subtract_device_constant_f32_c4_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    SubtractDeviceConstantAc4,
    subtract_device_constant,
    subtract_device_constant_ac4,
    AC4,
    [f32, f32 => subtract_device_constant_f32_ac4]
);

impl_generic_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    SubtractDeviceConstantAc4InPlace,
    subtract_device_constant_in_place,
    subtract_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => subtract_device_constant_f32_ac4_in_place]
);

impl_generic_scaled_device_constant_operation!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_device_constant_operation_in_place!("Subtracts a constant from each processed channel. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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

impl_generic_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",

    DivideDeviceConstantC1,
    divide_device_constant,
    divide_device_constant_c1,
    C1,
    [
        f16, f32 => divide_device_constant_f16_c1,
        f32, f32 => divide_device_constant_f32_c1,
    ]
);

impl_generic_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    DivideDeviceConstantC1InPlace,
    divide_device_constant_in_place,
    divide_device_constant_c1_in_place,
    C1,
    [
        f16, f32 => divide_device_constant_f16_c1_in_place,
        f32, f32 => divide_device_constant_f32_c1_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",

    DivideDeviceConstantC3,
    divide_device_constant,
    divide_device_constant_c3,
    C3,
    [
        f16, f32 => divide_device_constant_f16_c3,
        f32, f32 => divide_device_constant_f32_c3,
    ]
);

impl_generic_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    DivideDeviceConstantC3InPlace,
    divide_device_constant_in_place,
    divide_device_constant_c3_in_place,
    C3,
    [
        f16, f32 => divide_device_constant_f16_c3_in_place,
        f32, f32 => divide_device_constant_f32_c3_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",

    DivideDeviceConstantC4,
    divide_device_constant,
    divide_device_constant_c4,
    C4,
    [
        f16, f32 => divide_device_constant_f16_c4,
        f32, f32 => divide_device_constant_f32_c4,
    ]
);

impl_generic_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",

    DivideDeviceConstantC4InPlace,
    divide_device_constant_in_place,
    divide_device_constant_c4_in_place,
    C4,
    [
        f16, f32 => divide_device_constant_f16_c4_in_place,
        f32, f32 => divide_device_constant_f32_c4_in_place,
    ]
);

impl_generic_scaled_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",

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

impl_generic_scaled_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",

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

impl_generic_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`. Leaves the alpha channel unchanged.",

    DivideDeviceConstantAc4,
    divide_device_constant,
    divide_device_constant_ac4,
    AC4,
    [f32, f32 => divide_device_constant_f32_ac4]
);

impl_generic_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

    DivideDeviceConstantAc4InPlace,
    divide_device_constant_in_place,
    divide_device_constant_ac4_in_place,
    AC4,
    [f32, f32 => divide_device_constant_f32_ac4_in_place]
);

impl_generic_scaled_device_constant_operation!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",

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

impl_generic_scaled_device_constant_operation_in_place!("Divides each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",

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
