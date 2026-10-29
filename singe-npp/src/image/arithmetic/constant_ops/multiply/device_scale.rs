use singe_cuda::memory::DeviceMemory;
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

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_scale_u8_c1,
    u8,
    u8,
    C1,
    1,
    nppiMulDeviceCScale_8u_C1R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_scale_u8_c1_in_place,
    u8,
    u8,
    C1,
    1,
    nppiMulDeviceCScale_8u_C1IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_scale_u8_c3,
    u8,
    u8,
    C3,
    3,
    nppiMulDeviceCScale_8u_C3R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_scale_u8_c3_in_place,
    u8,
    u8,
    C3,
    3,
    nppiMulDeviceCScale_8u_C3IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_device_constant_scale_u8_ac4,
    u8,
    u8,
    AC4,
    3,
    nppiMulDeviceCScale_8u_AC4R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_device_constant_scale_u8_ac4_in_place,
    u8,
    u8,
    AC4,
    3,
    nppiMulDeviceCScale_8u_AC4IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_scale_u8_c4,
    u8,
    u8,
    C4,
    4,
    nppiMulDeviceCScale_8u_C4R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_scale_u8_c4_in_place,
    u8,
    u8,
    C4,
    4,
    nppiMulDeviceCScale_8u_C4IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_scale_u16_c1,
    u16,
    u16,
    C1,
    1,
    nppiMulDeviceCScale_16u_C1R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_scale_u16_c1_in_place,
    u16,
    u16,
    C1,
    1,
    nppiMulDeviceCScale_16u_C1IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_scale_u16_c3,
    u16,
    u16,
    C3,
    3,
    nppiMulDeviceCScale_16u_C3R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_scale_u16_c3_in_place,
    u16,
    u16,
    C3,
    3,
    nppiMulDeviceCScale_16u_C3IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`. Leaves the alpha channel unchanged.",
    multiply_device_constant_scale_u16_ac4,
    u16,
    u16,
    AC4,
    3,
    nppiMulDeviceCScale_16u_AC4R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place. Leaves the alpha channel unchanged.",
    multiply_device_constant_scale_u16_ac4_in_place,
    u16,
    u16,
    AC4,
    3,
    nppiMulDeviceCScale_16u_AC4IR_Ctx
);

impl_add_device_constant!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Writes the result to `destination`.",
    multiply_device_constant_scale_u16_c4,
    u16,
    u16,
    C4,
    4,
    nppiMulDeviceCScale_16u_C4R_Ctx
);

impl_add_device_constant_in_place!(
    "Multiplies each processed channel by a constant with NPP's fixed bit-width scaling. Reads per-channel constants from device memory. Updates `source_destination` in place.",
    multiply_device_constant_scale_u16_c4_in_place,
    u16,
    u16,
    C4,
    4,
    nppiMulDeviceCScale_16u_C4IR_Ctx
);
