use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold::validate_same_size,
        view::{AC4, C1, C3, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::{ComparisonOperation, DataTypeLike},
};

impl_threshold_value_c1!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged.",
    threshold_value_u8_c1,
    u8,
    nppiThreshold_Val_8u_C1R_Ctx
);

impl_threshold_value_c1!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged.",
    threshold_value_u16_c1,
    u16,
    nppiThreshold_Val_16u_C1R_Ctx
);

impl_threshold_value_c1!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged.",
    threshold_value_i16_c1,
    i16,
    nppiThreshold_Val_16s_C1R_Ctx
);

impl_threshold_value_c1!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged.",
    threshold_value_f32_c1,
    f32,
    nppiThreshold_Val_32f_C1R_Ctx
);

impl_threshold_value_c1_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Updates `image` in place.",
    threshold_value_u8_c1_in_place,
    u8,
    nppiThreshold_Val_8u_C1IR_Ctx
);

impl_threshold_value_c1_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Updates `image` in place.",
    threshold_value_u16_c1_in_place,
    u16,
    nppiThreshold_Val_16u_C1IR_Ctx
);

impl_threshold_value_c1_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Updates `image` in place.",
    threshold_value_i16_c1_in_place,
    i16,
    nppiThreshold_Val_16s_C1IR_Ctx
);

impl_threshold_value_c1_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Updates `image` in place.",
    threshold_value_f32_c1_in_place,
    f32,
    nppiThreshold_Val_32f_C1IR_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel.",
    threshold_value_u8_c3,
    u8,
    C3,
    3,
    nppiThreshold_Val_8u_C3R_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. The destination alpha channel is not processed.",
    threshold_value_u8_ac4,
    u8,
    AC4,
    3,
    nppiThreshold_Val_8u_AC4R_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel.",
    threshold_value_u16_c3,
    u16,
    C3,
    3,
    nppiThreshold_Val_16u_C3R_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. The destination alpha channel is not processed.",
    threshold_value_u16_ac4,
    u16,
    AC4,
    3,
    nppiThreshold_Val_16u_AC4R_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel.",
    threshold_value_i16_c3,
    i16,
    C3,
    3,
    nppiThreshold_Val_16s_C3R_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. The destination alpha channel is not processed.",
    threshold_value_i16_ac4,
    i16,
    AC4,
    3,
    nppiThreshold_Val_16s_AC4R_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel.",
    threshold_value_f32_c3,
    f32,
    C3,
    3,
    nppiThreshold_Val_32f_C3R_Ctx
);

impl_threshold_value_packed!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. The destination alpha channel is not processed.",
    threshold_value_f32_ac4,
    f32,
    AC4,
    3,
    nppiThreshold_Val_32f_AC4R_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place.",
    threshold_value_u8_c3_in_place,
    u8,
    C3,
    3,
    nppiThreshold_Val_8u_C3IR_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place. The alpha channel is unchanged.",
    threshold_value_u8_ac4_in_place,
    u8,
    AC4,
    3,
    nppiThreshold_Val_8u_AC4IR_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place.",
    threshold_value_u16_c3_in_place,
    u16,
    C3,
    3,
    nppiThreshold_Val_16u_C3IR_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place. The alpha channel is unchanged.",
    threshold_value_u16_ac4_in_place,
    u16,
    AC4,
    3,
    nppiThreshold_Val_16u_AC4IR_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place.",
    threshold_value_i16_c3_in_place,
    i16,
    C3,
    3,
    nppiThreshold_Val_16s_C3IR_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place. The alpha channel is unchanged.",
    threshold_value_i16_ac4_in_place,
    i16,
    AC4,
    3,
    nppiThreshold_Val_16s_AC4IR_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place.",
    threshold_value_f32_c3_in_place,
    f32,
    C3,
    3,
    nppiThreshold_Val_32f_C3IR_Ctx
);

impl_threshold_value_packed_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place. The alpha channel is unchanged.",
    threshold_value_f32_ac4_in_place,
    f32,
    AC4,
    3,
    nppiThreshold_Val_32f_AC4IR_Ctx
);

impl_generic_threshold_value_scalar!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged.",
    ThresholdValueC1, threshold_value, threshold_value_c1, [
    u8 => threshold_value_u8_c1,
    u16 => threshold_value_u16_c1,
    i16 => threshold_value_i16_c1,
    f32 => threshold_value_f32_c1,
    ]
);

impl_generic_threshold_value_scalar_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Updates `image` in place.",
    ThresholdValueC1InPlace,
    threshold_value_in_place,
    threshold_value_c1_in_place,
    [
        u8 => threshold_value_u8_c1_in_place,
        u16 => threshold_value_u16_c1_in_place,
        i16 => threshold_value_i16_c1_in_place,
        f32 => threshold_value_f32_c1_in_place,
    ]
);

impl_generic_threshold_value_array!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel.",
    ThresholdValueC3,
    threshold_value,
    threshold_value_c3,
    C3,
    3,
    [
        u8 => threshold_value_u8_c3,
        u16 => threshold_value_u16_c3,
        i16 => threshold_value_i16_c3,
        f32 => threshold_value_f32_c3,
    ]
);

impl_generic_threshold_value_array_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place.",
    ThresholdValueC3InPlace,
    threshold_value_in_place,
    threshold_value_c3_in_place,
    C3,
    3,
    [
        u8 => threshold_value_u8_c3_in_place,
        u16 => threshold_value_u16_c3_in_place,
        i16 => threshold_value_i16_c3_in_place,
        f32 => threshold_value_f32_c3_in_place,
    ]
);

impl_generic_threshold_value_array!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are copied unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. The destination alpha channel is not processed.",
    ThresholdValueAc4,
    threshold_value,
    threshold_value_ac4,
    AC4,
    3,
    [
        u8 => threshold_value_u8_ac4,
        u16 => threshold_value_u16_ac4,
        i16 => threshold_value_i16_ac4,
        f32 => threshold_value_f32_ac4,
    ]
);

impl_generic_threshold_value_array_in_place!(
    "Replaces samples that satisfy the comparison selected by `operation` against the threshold with the replacement value; other samples are left unchanged. Thresholds and replacement values, when supplied, apply independently to each processed channel. Updates `image` in place. The alpha channel is unchanged.",
    ThresholdValueAc4InPlace,
    threshold_value_in_place,
    threshold_value_ac4_in_place,
    AC4,
    3,
    [
        u8 => threshold_value_u8_ac4_in_place,
        u16 => threshold_value_u16_ac4_in_place,
        i16 => threshold_value_i16_ac4_in_place,
        f32 => threshold_value_f32_ac4_in_place,
    ]
);
