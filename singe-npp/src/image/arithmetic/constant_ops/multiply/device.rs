use singe_cuda::{memory::DeviceMemory, types::f16};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic_validation::{device_constant_ptr, validate_same_size},
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
};

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_u8_c1,
    u8,
    C1,
    1,
    nppiMulDeviceC_8u_C1RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_u8_c1_in_place,
    u8,
    C1,
    1,
    nppiMulDeviceC_8u_C1IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_u8_c3,
    u8,
    C3,
    3,
    nppiMulDeviceC_8u_C3RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_u8_c3_in_place,
    u8,
    C3,
    3,
    nppiMulDeviceC_8u_C3IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_device_constant_u8_ac4,
    u8,
    AC4,
    3,
    nppiMulDeviceC_8u_AC4RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_device_constant_u8_ac4_in_place,
    u8,
    AC4,
    3,
    nppiMulDeviceC_8u_AC4IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_u8_c4,
    u8,
    C4,
    4,
    nppiMulDeviceC_8u_C4RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_u8_c4_in_place,
    u8,
    C4,
    4,
    nppiMulDeviceC_8u_C4IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_u16_c1,
    u16,
    C1,
    1,
    nppiMulDeviceC_16u_C1RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_u16_c1_in_place,
    u16,
    C1,
    1,
    nppiMulDeviceC_16u_C1IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_u16_c3,
    u16,
    C3,
    3,
    nppiMulDeviceC_16u_C3RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_u16_c3_in_place,
    u16,
    C3,
    3,
    nppiMulDeviceC_16u_C3IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_device_constant_u16_ac4,
    u16,
    AC4,
    3,
    nppiMulDeviceC_16u_AC4RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_device_constant_u16_ac4_in_place,
    u16,
    AC4,
    3,
    nppiMulDeviceC_16u_AC4IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_u16_c4,
    u16,
    C4,
    4,
    nppiMulDeviceC_16u_C4RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_u16_c4_in_place,
    u16,
    C4,
    4,
    nppiMulDeviceC_16u_C4IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_i16_c1,
    i16,
    C1,
    1,
    nppiMulDeviceC_16s_C1RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_i16_c1_in_place,
    i16,
    C1,
    1,
    nppiMulDeviceC_16s_C1IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_i16_c3,
    i16,
    C3,
    3,
    nppiMulDeviceC_16s_C3RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_i16_c3_in_place,
    i16,
    C3,
    3,
    nppiMulDeviceC_16s_C3IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_device_constant_i16_ac4,
    i16,
    AC4,
    3,
    nppiMulDeviceC_16s_AC4RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_device_constant_i16_ac4_in_place,
    i16,
    AC4,
    3,
    nppiMulDeviceC_16s_AC4IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_i16_c4,
    i16,
    C4,
    4,
    nppiMulDeviceC_16s_C4RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_i16_c4_in_place,
    i16,
    C4,
    4,
    nppiMulDeviceC_16s_C4IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_i32_c1,
    i32,
    C1,
    1,
    nppiMulDeviceC_32s_C1RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_i32_c1_in_place,
    i32,
    C1,
    1,
    nppiMulDeviceC_32s_C1IRSfs_Ctx
);

impl_add_device_constant_scaled!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Writes the result to `destination`.",
    multiply_device_constant_i32_c3,
    i32,
    C3,
    3,
    nppiMulDeviceC_32s_C3RSfs_Ctx
);

impl_add_device_constant_scaled_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Applies native integer scaling and saturation using `scale_factor`. Updates `source_destination` in place.",
    multiply_device_constant_i32_c3_in_place,
    i32,
    C3,
    3,
    nppiMulDeviceC_32s_C3IRSfs_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_f16_c1,
    f16,
    f32,
    C1,
    1,
    nppiMulDeviceC_16f_C1R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_f16_c1_in_place,
    f16,
    f32,
    C1,
    1,
    nppiMulDeviceC_16f_C1IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_f16_c3,
    f16,
    f32,
    C3,
    3,
    nppiMulDeviceC_16f_C3R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_f16_c3_in_place,
    f16,
    f32,
    C3,
    3,
    nppiMulDeviceC_16f_C3IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_f16_c4,
    f16,
    f32,
    C4,
    4,
    nppiMulDeviceC_16f_C4R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_f16_c4_in_place,
    f16,
    f32,
    C4,
    4,
    nppiMulDeviceC_16f_C4IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_f32_c1,
    f32,
    f32,
    C1,
    1,
    nppiMulDeviceC_32f_C1R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_f32_c1_in_place,
    f32,
    f32,
    C1,
    1,
    nppiMulDeviceC_32f_C1IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_f32_c3,
    f32,
    f32,
    C3,
    3,
    nppiMulDeviceC_32f_C3R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_f32_c3_in_place,
    f32,
    f32,
    C3,
    3,
    nppiMulDeviceC_32f_C3IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_device_constant_f32_ac4,
    f32,
    f32,
    AC4,
    3,
    nppiMulDeviceC_32f_AC4R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_device_constant_f32_ac4_in_place,
    f32,
    f32,
    AC4,
    3,
    nppiMulDeviceC_32f_AC4IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_f32_c4,
    f32,
    f32,
    C4,
    4,
    nppiMulDeviceC_32f_C4R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_f32_c4_in_place,
    f32,
    f32,
    C4,
    4,
    nppiMulDeviceC_32f_C4IR_Ctx
);
