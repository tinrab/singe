use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{validate_same_size, validate_statistic_output},
        view::{AC4, C1, C3, C4, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_u8_c1_buffer_size,
    norm_rel_inf_u8_c1,
    u8,
    C1,
    1,
    nppiNormRelInfGetBufferHostSize_8u_C1R_Ctx,
    nppiNormRel_Inf_8u_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_u8_c3_buffer_size,
    norm_rel_inf_u8_c3,
    u8,
    C3,
    3,
    nppiNormRelInfGetBufferHostSize_8u_C3R_Ctx,
    nppiNormRel_Inf_8u_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_u8_c4_buffer_size,
    norm_rel_inf_u8_c4,
    u8,
    C4,
    4,
    nppiNormRelInfGetBufferHostSize_8u_C4R_Ctx,
    nppiNormRel_Inf_8u_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_inf_u8_ac4_buffer_size,
    norm_rel_inf_u8_ac4,
    u8,
    AC4,
    3,
    nppiNormRelInfGetBufferHostSize_8u_AC4R_Ctx,
    nppiNormRel_Inf_8u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_u16_c1_buffer_size,
    norm_rel_inf_u16_c1,
    u16,
    C1,
    1,
    nppiNormRelInfGetBufferHostSize_16u_C1R_Ctx,
    nppiNormRel_Inf_16u_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_u16_c3_buffer_size,
    norm_rel_inf_u16_c3,
    u16,
    C3,
    3,
    nppiNormRelInfGetBufferHostSize_16u_C3R_Ctx,
    nppiNormRel_Inf_16u_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_u16_c4_buffer_size,
    norm_rel_inf_u16_c4,
    u16,
    C4,
    4,
    nppiNormRelInfGetBufferHostSize_16u_C4R_Ctx,
    nppiNormRel_Inf_16u_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_inf_u16_ac4_buffer_size,
    norm_rel_inf_u16_ac4,
    u16,
    AC4,
    3,
    nppiNormRelInfGetBufferHostSize_16u_AC4R_Ctx,
    nppiNormRel_Inf_16u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_i16_c1_buffer_size,
    norm_rel_inf_i16_c1,
    i16,
    C1,
    1,
    nppiNormRelInfGetBufferHostSize_16s_C1R_Ctx,
    nppiNormRel_Inf_16s_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_i16_c3_buffer_size,
    norm_rel_inf_i16_c3,
    i16,
    C3,
    3,
    nppiNormRelInfGetBufferHostSize_16s_C3R_Ctx,
    nppiNormRel_Inf_16s_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_i16_c4_buffer_size,
    norm_rel_inf_i16_c4,
    i16,
    C4,
    4,
    nppiNormRelInfGetBufferHostSize_16s_C4R_Ctx,
    nppiNormRel_Inf_16s_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_inf_i16_ac4_buffer_size,
    norm_rel_inf_i16_ac4,
    i16,
    AC4,
    3,
    nppiNormRelInfGetBufferHostSize_16s_AC4R_Ctx,
    nppiNormRel_Inf_16s_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_f32_c1_buffer_size,
    norm_rel_inf_f32_c1,
    f32,
    C1,
    1,
    nppiNormRelInfGetBufferHostSize_32f_C1R_Ctx,
    nppiNormRel_Inf_32f_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_f32_c3_buffer_size,
    norm_rel_inf_f32_c3,
    f32,
    C3,
    3,
    nppiNormRelInfGetBufferHostSize_32f_C3R_Ctx,
    nppiNormRel_Inf_32f_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs.",
    norm_rel_inf_f32_c4_buffer_size,
    norm_rel_inf_f32_c4,
    f32,
    C4,
    4,
    nppiNormRelInfGetBufferHostSize_32f_C4R_Ctx,
    nppiNormRel_Inf_32f_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative infinity-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_inf_f32_ac4_buffer_size,
    norm_rel_inf_f32_ac4,
    f32,
    AC4,
    3,
    nppiNormRelInfGetBufferHostSize_32f_AC4R_Ctx,
    nppiNormRel_Inf_32f_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_u8_c1_buffer_size,
    norm_rel_l1_u8_c1,
    u8,
    C1,
    1,
    nppiNormRelL1GetBufferHostSize_8u_C1R_Ctx,
    nppiNormRel_L1_8u_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_u8_c3_buffer_size,
    norm_rel_l1_u8_c3,
    u8,
    C3,
    3,
    nppiNormRelL1GetBufferHostSize_8u_C3R_Ctx,
    nppiNormRel_L1_8u_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_u8_c4_buffer_size,
    norm_rel_l1_u8_c4,
    u8,
    C4,
    4,
    nppiNormRelL1GetBufferHostSize_8u_C4R_Ctx,
    nppiNormRel_L1_8u_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l1_u8_ac4_buffer_size,
    norm_rel_l1_u8_ac4,
    u8,
    AC4,
    3,
    nppiNormRelL1GetBufferHostSize_8u_AC4R_Ctx,
    nppiNormRel_L1_8u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_u16_c1_buffer_size,
    norm_rel_l1_u16_c1,
    u16,
    C1,
    1,
    nppiNormRelL1GetBufferHostSize_16u_C1R_Ctx,
    nppiNormRel_L1_16u_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_u16_c3_buffer_size,
    norm_rel_l1_u16_c3,
    u16,
    C3,
    3,
    nppiNormRelL1GetBufferHostSize_16u_C3R_Ctx,
    nppiNormRel_L1_16u_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_u16_c4_buffer_size,
    norm_rel_l1_u16_c4,
    u16,
    C4,
    4,
    nppiNormRelL1GetBufferHostSize_16u_C4R_Ctx,
    nppiNormRel_L1_16u_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l1_u16_ac4_buffer_size,
    norm_rel_l1_u16_ac4,
    u16,
    AC4,
    3,
    nppiNormRelL1GetBufferHostSize_16u_AC4R_Ctx,
    nppiNormRel_L1_16u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_i16_c1_buffer_size,
    norm_rel_l1_i16_c1,
    i16,
    C1,
    1,
    nppiNormRelL1GetBufferHostSize_16s_C1R_Ctx,
    nppiNormRel_L1_16s_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_i16_c3_buffer_size,
    norm_rel_l1_i16_c3,
    i16,
    C3,
    3,
    nppiNormRelL1GetBufferHostSize_16s_C3R_Ctx,
    nppiNormRel_L1_16s_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_i16_c4_buffer_size,
    norm_rel_l1_i16_c4,
    i16,
    C4,
    4,
    nppiNormRelL1GetBufferHostSize_16s_C4R_Ctx,
    nppiNormRel_L1_16s_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l1_i16_ac4_buffer_size,
    norm_rel_l1_i16_ac4,
    i16,
    AC4,
    3,
    nppiNormRelL1GetBufferHostSize_16s_AC4R_Ctx,
    nppiNormRel_L1_16s_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_f32_c1_buffer_size,
    norm_rel_l1_f32_c1,
    f32,
    C1,
    1,
    nppiNormRelL1GetBufferHostSize_32f_C1R_Ctx,
    nppiNormRel_L1_32f_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_f32_c3_buffer_size,
    norm_rel_l1_f32_c3,
    f32,
    C3,
    3,
    nppiNormRelL1GetBufferHostSize_32f_C3R_Ctx,
    nppiNormRel_L1_32f_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs.",
    norm_rel_l1_f32_c4_buffer_size,
    norm_rel_l1_f32_c4,
    f32,
    C4,
    4,
    nppiNormRelL1GetBufferHostSize_32f_C4R_Ctx,
    nppiNormRel_L1_32f_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L1-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l1_f32_ac4_buffer_size,
    norm_rel_l1_f32_ac4,
    f32,
    AC4,
    3,
    nppiNormRelL1GetBufferHostSize_32f_AC4R_Ctx,
    nppiNormRel_L1_32f_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_u8_c1_buffer_size,
    norm_rel_l2_u8_c1,
    u8,
    C1,
    1,
    nppiNormRelL2GetBufferHostSize_8u_C1R_Ctx,
    nppiNormRel_L2_8u_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_u8_c3_buffer_size,
    norm_rel_l2_u8_c3,
    u8,
    C3,
    3,
    nppiNormRelL2GetBufferHostSize_8u_C3R_Ctx,
    nppiNormRel_L2_8u_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_u8_c4_buffer_size,
    norm_rel_l2_u8_c4,
    u8,
    C4,
    4,
    nppiNormRelL2GetBufferHostSize_8u_C4R_Ctx,
    nppiNormRel_L2_8u_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l2_u8_ac4_buffer_size,
    norm_rel_l2_u8_ac4,
    u8,
    AC4,
    3,
    nppiNormRelL2GetBufferHostSize_8u_AC4R_Ctx,
    nppiNormRel_L2_8u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_u16_c1_buffer_size,
    norm_rel_l2_u16_c1,
    u16,
    C1,
    1,
    nppiNormRelL2GetBufferHostSize_16u_C1R_Ctx,
    nppiNormRel_L2_16u_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_u16_c3_buffer_size,
    norm_rel_l2_u16_c3,
    u16,
    C3,
    3,
    nppiNormRelL2GetBufferHostSize_16u_C3R_Ctx,
    nppiNormRel_L2_16u_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_u16_c4_buffer_size,
    norm_rel_l2_u16_c4,
    u16,
    C4,
    4,
    nppiNormRelL2GetBufferHostSize_16u_C4R_Ctx,
    nppiNormRel_L2_16u_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l2_u16_ac4_buffer_size,
    norm_rel_l2_u16_ac4,
    u16,
    AC4,
    3,
    nppiNormRelL2GetBufferHostSize_16u_AC4R_Ctx,
    nppiNormRel_L2_16u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_i16_c1_buffer_size,
    norm_rel_l2_i16_c1,
    i16,
    C1,
    1,
    nppiNormRelL2GetBufferHostSize_16s_C1R_Ctx,
    nppiNormRel_L2_16s_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_i16_c3_buffer_size,
    norm_rel_l2_i16_c3,
    i16,
    C3,
    3,
    nppiNormRelL2GetBufferHostSize_16s_C3R_Ctx,
    nppiNormRel_L2_16s_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_i16_c4_buffer_size,
    norm_rel_l2_i16_c4,
    i16,
    C4,
    4,
    nppiNormRelL2GetBufferHostSize_16s_C4R_Ctx,
    nppiNormRel_L2_16s_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l2_i16_ac4_buffer_size,
    norm_rel_l2_i16_ac4,
    i16,
    AC4,
    3,
    nppiNormRelL2GetBufferHostSize_16s_AC4R_Ctx,
    nppiNormRel_L2_16s_AC4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_f32_c1_buffer_size,
    norm_rel_l2_f32_c1,
    f32,
    C1,
    1,
    nppiNormRelL2GetBufferHostSize_32f_C1R_Ctx,
    nppiNormRel_L2_32f_C1R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_f32_c3_buffer_size,
    norm_rel_l2_f32_c3,
    f32,
    C3,
    3,
    nppiNormRelL2GetBufferHostSize_32f_C3R_Ctx,
    nppiNormRel_L2_32f_C3R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs.",
    norm_rel_l2_f32_c4_buffer_size,
    norm_rel_l2_f32_c4,
    f32,
    C4,
    4,
    nppiNormRelL2GetBufferHostSize_32f_C4R_Ctx,
    nppiNormRel_L2_32f_C4R_Ctx
);

impl_error_metric!(
    "Computes the relative L2-norm error between two image ROIs. The alpha channel is excluded.",
    norm_rel_l2_f32_ac4_buffer_size,
    norm_rel_l2_f32_ac4,
    f32,
    AC4,
    3,
    nppiNormRelL2GetBufferHostSize_32f_AC4R_Ctx,
    nppiNormRel_L2_32f_AC4R_Ctx
);

impl_generic_dot_prod!(
    "Computes the relative infinity-norm error between two image ROIs.",
    NormRelInfC1,
    norm_rel_inf_c1,
    norm_rel_inf_c1_buffer_size,
    C1,
    [
        (u8, norm_rel_inf_u8_c1, norm_rel_inf_u8_c1_buffer_size),
        (u16, norm_rel_inf_u16_c1, norm_rel_inf_u16_c1_buffer_size),
        (i16, norm_rel_inf_i16_c1, norm_rel_inf_i16_c1_buffer_size),
        (f32, norm_rel_inf_f32_c1, norm_rel_inf_f32_c1_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative infinity-norm error between two image ROIs.",
    NormRelInfC3,
    norm_rel_inf_c3,
    norm_rel_inf_c3_buffer_size,
    C3,
    [
        (u8, norm_rel_inf_u8_c3, norm_rel_inf_u8_c3_buffer_size),
        (u16, norm_rel_inf_u16_c3, norm_rel_inf_u16_c3_buffer_size),
        (i16, norm_rel_inf_i16_c3, norm_rel_inf_i16_c3_buffer_size),
        (f32, norm_rel_inf_f32_c3, norm_rel_inf_f32_c3_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative infinity-norm error between two image ROIs.",
    NormRelInfC4,
    norm_rel_inf_c4,
    norm_rel_inf_c4_buffer_size,
    C4,
    [
        (u8, norm_rel_inf_u8_c4, norm_rel_inf_u8_c4_buffer_size),
        (u16, norm_rel_inf_u16_c4, norm_rel_inf_u16_c4_buffer_size),
        (i16, norm_rel_inf_i16_c4, norm_rel_inf_i16_c4_buffer_size),
        (f32, norm_rel_inf_f32_c4, norm_rel_inf_f32_c4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative infinity-norm error between two image ROIs. The alpha channel is excluded.",
    NormRelInfAC4,
    norm_rel_inf_ac4,
    norm_rel_inf_ac4_buffer_size,
    AC4,
    [
        (u8, norm_rel_inf_u8_ac4, norm_rel_inf_u8_ac4_buffer_size),
        (u16, norm_rel_inf_u16_ac4, norm_rel_inf_u16_ac4_buffer_size),
        (i16, norm_rel_inf_i16_ac4, norm_rel_inf_i16_ac4_buffer_size),
        (f32, norm_rel_inf_f32_ac4, norm_rel_inf_f32_ac4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L1-norm error between two image ROIs.",
    NormRelL1C1,
    norm_rel_l1_c1,
    norm_rel_l1_c1_buffer_size,
    C1,
    [
        (u8, norm_rel_l1_u8_c1, norm_rel_l1_u8_c1_buffer_size),
        (u16, norm_rel_l1_u16_c1, norm_rel_l1_u16_c1_buffer_size),
        (i16, norm_rel_l1_i16_c1, norm_rel_l1_i16_c1_buffer_size),
        (f32, norm_rel_l1_f32_c1, norm_rel_l1_f32_c1_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L1-norm error between two image ROIs.",
    NormRelL1C3,
    norm_rel_l1_c3,
    norm_rel_l1_c3_buffer_size,
    C3,
    [
        (u8, norm_rel_l1_u8_c3, norm_rel_l1_u8_c3_buffer_size),
        (u16, norm_rel_l1_u16_c3, norm_rel_l1_u16_c3_buffer_size),
        (i16, norm_rel_l1_i16_c3, norm_rel_l1_i16_c3_buffer_size),
        (f32, norm_rel_l1_f32_c3, norm_rel_l1_f32_c3_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L1-norm error between two image ROIs.",
    NormRelL1C4,
    norm_rel_l1_c4,
    norm_rel_l1_c4_buffer_size,
    C4,
    [
        (u8, norm_rel_l1_u8_c4, norm_rel_l1_u8_c4_buffer_size),
        (u16, norm_rel_l1_u16_c4, norm_rel_l1_u16_c4_buffer_size),
        (i16, norm_rel_l1_i16_c4, norm_rel_l1_i16_c4_buffer_size),
        (f32, norm_rel_l1_f32_c4, norm_rel_l1_f32_c4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L1-norm error between two image ROIs. The alpha channel is excluded.",
    NormRelL1AC4,
    norm_rel_l1_ac4,
    norm_rel_l1_ac4_buffer_size,
    AC4,
    [
        (u8, norm_rel_l1_u8_ac4, norm_rel_l1_u8_ac4_buffer_size),
        (u16, norm_rel_l1_u16_ac4, norm_rel_l1_u16_ac4_buffer_size),
        (i16, norm_rel_l1_i16_ac4, norm_rel_l1_i16_ac4_buffer_size),
        (f32, norm_rel_l1_f32_ac4, norm_rel_l1_f32_ac4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L2-norm error between two image ROIs.",
    NormRelL2C1,
    norm_rel_l2_c1,
    norm_rel_l2_c1_buffer_size,
    C1,
    [
        (u8, norm_rel_l2_u8_c1, norm_rel_l2_u8_c1_buffer_size),
        (u16, norm_rel_l2_u16_c1, norm_rel_l2_u16_c1_buffer_size),
        (i16, norm_rel_l2_i16_c1, norm_rel_l2_i16_c1_buffer_size),
        (f32, norm_rel_l2_f32_c1, norm_rel_l2_f32_c1_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L2-norm error between two image ROIs.",
    NormRelL2C3,
    norm_rel_l2_c3,
    norm_rel_l2_c3_buffer_size,
    C3,
    [
        (u8, norm_rel_l2_u8_c3, norm_rel_l2_u8_c3_buffer_size),
        (u16, norm_rel_l2_u16_c3, norm_rel_l2_u16_c3_buffer_size),
        (i16, norm_rel_l2_i16_c3, norm_rel_l2_i16_c3_buffer_size),
        (f32, norm_rel_l2_f32_c3, norm_rel_l2_f32_c3_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L2-norm error between two image ROIs.",
    NormRelL2C4,
    norm_rel_l2_c4,
    norm_rel_l2_c4_buffer_size,
    C4,
    [
        (u8, norm_rel_l2_u8_c4, norm_rel_l2_u8_c4_buffer_size),
        (u16, norm_rel_l2_u16_c4, norm_rel_l2_u16_c4_buffer_size),
        (i16, norm_rel_l2_i16_c4, norm_rel_l2_i16_c4_buffer_size),
        (f32, norm_rel_l2_f32_c4, norm_rel_l2_f32_c4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the relative L2-norm error between two image ROIs. The alpha channel is excluded.",
    NormRelL2AC4,
    norm_rel_l2_ac4,
    norm_rel_l2_ac4_buffer_size,
    AC4,
    [
        (u8, norm_rel_l2_u8_ac4, norm_rel_l2_u8_ac4_buffer_size),
        (u16, norm_rel_l2_u16_ac4, norm_rel_l2_u16_ac4_buffer_size),
        (i16, norm_rel_l2_i16_ac4, norm_rel_l2_i16_ac4_buffer_size),
        (f32, norm_rel_l2_f32_ac4, norm_rel_l2_f32_ac4_buffer_size),
    ]
);
