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
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_u8_c1_buffer_size,
    norm_diff_inf_u8_c1,
    u8,
    C1,
    1,
    nppiNormDiffInfGetBufferHostSize_8u_C1R_Ctx,
    nppiNormDiff_Inf_8u_C1R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_u16_c1_buffer_size,
    norm_diff_inf_u16_c1,
    u16,
    C1,
    1,
    nppiNormDiffInfGetBufferHostSize_16u_C1R_Ctx,
    nppiNormDiff_Inf_16u_C1R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_i16_c1_buffer_size,
    norm_diff_inf_i16_c1,
    i16,
    C1,
    1,
    nppiNormDiffInfGetBufferHostSize_16s_C1R_Ctx,
    nppiNormDiff_Inf_16s_C1R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_f32_c1_buffer_size,
    norm_diff_inf_f32_c1,
    f32,
    C1,
    1,
    nppiNormDiffInfGetBufferHostSize_32f_C1R_Ctx,
    nppiNormDiff_Inf_32f_C1R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_u8_c3_buffer_size,
    norm_diff_inf_u8_c3,
    u8,
    C3,
    3,
    nppiNormDiffInfGetBufferHostSize_8u_C3R_Ctx,
    nppiNormDiff_Inf_8u_C3R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_u16_c3_buffer_size,
    norm_diff_inf_u16_c3,
    u16,
    C3,
    3,
    nppiNormDiffInfGetBufferHostSize_16u_C3R_Ctx,
    nppiNormDiff_Inf_16u_C3R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_i16_c3_buffer_size,
    norm_diff_inf_i16_c3,
    i16,
    C3,
    3,
    nppiNormDiffInfGetBufferHostSize_16s_C3R_Ctx,
    nppiNormDiff_Inf_16s_C3R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_f32_c3_buffer_size,
    norm_diff_inf_f32_c3,
    f32,
    C3,
    3,
    nppiNormDiffInfGetBufferHostSize_32f_C3R_Ctx,
    nppiNormDiff_Inf_32f_C3R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_u8_c4_buffer_size,
    norm_diff_inf_u8_c4,
    u8,
    C4,
    4,
    nppiNormDiffInfGetBufferHostSize_8u_C4R_Ctx,
    nppiNormDiff_Inf_8u_C4R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_u16_c4_buffer_size,
    norm_diff_inf_u16_c4,
    u16,
    C4,
    4,
    nppiNormDiffInfGetBufferHostSize_16u_C4R_Ctx,
    nppiNormDiff_Inf_16u_C4R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_i16_c4_buffer_size,
    norm_diff_inf_i16_c4,
    i16,
    C4,
    4,
    nppiNormDiffInfGetBufferHostSize_16s_C4R_Ctx,
    nppiNormDiff_Inf_16s_C4R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs.",
    norm_diff_inf_f32_c4_buffer_size,
    norm_diff_inf_f32_c4,
    f32,
    C4,
    4,
    nppiNormDiffInfGetBufferHostSize_32f_C4R_Ctx,
    nppiNormDiff_Inf_32f_C4R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_inf_u8_ac4_buffer_size,
    norm_diff_inf_u8_ac4,
    u8,
    AC4,
    3,
    nppiNormDiffInfGetBufferHostSize_8u_AC4R_Ctx,
    nppiNormDiff_Inf_8u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_inf_u16_ac4_buffer_size,
    norm_diff_inf_u16_ac4,
    u16,
    AC4,
    3,
    nppiNormDiffInfGetBufferHostSize_16u_AC4R_Ctx,
    nppiNormDiff_Inf_16u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_inf_i16_ac4_buffer_size,
    norm_diff_inf_i16_ac4,
    i16,
    AC4,
    3,
    nppiNormDiffInfGetBufferHostSize_16s_AC4R_Ctx,
    nppiNormDiff_Inf_16s_AC4R_Ctx
);

impl_error_metric!(
    "Computes the infinity norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_inf_f32_ac4_buffer_size,
    norm_diff_inf_f32_ac4,
    f32,
    AC4,
    3,
    nppiNormDiffInfGetBufferHostSize_32f_AC4R_Ctx,
    nppiNormDiff_Inf_32f_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_u8_c1_buffer_size,
    norm_diff_l1_u8_c1,
    u8,
    C1,
    1,
    nppiNormDiffL1GetBufferHostSize_8u_C1R_Ctx,
    nppiNormDiff_L1_8u_C1R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_u16_c1_buffer_size,
    norm_diff_l1_u16_c1,
    u16,
    C1,
    1,
    nppiNormDiffL1GetBufferHostSize_16u_C1R_Ctx,
    nppiNormDiff_L1_16u_C1R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_i16_c1_buffer_size,
    norm_diff_l1_i16_c1,
    i16,
    C1,
    1,
    nppiNormDiffL1GetBufferHostSize_16s_C1R_Ctx,
    nppiNormDiff_L1_16s_C1R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_f32_c1_buffer_size,
    norm_diff_l1_f32_c1,
    f32,
    C1,
    1,
    nppiNormDiffL1GetBufferHostSize_32f_C1R_Ctx,
    nppiNormDiff_L1_32f_C1R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_u8_c3_buffer_size,
    norm_diff_l1_u8_c3,
    u8,
    C3,
    3,
    nppiNormDiffL1GetBufferHostSize_8u_C3R_Ctx,
    nppiNormDiff_L1_8u_C3R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_u16_c3_buffer_size,
    norm_diff_l1_u16_c3,
    u16,
    C3,
    3,
    nppiNormDiffL1GetBufferHostSize_16u_C3R_Ctx,
    nppiNormDiff_L1_16u_C3R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_i16_c3_buffer_size,
    norm_diff_l1_i16_c3,
    i16,
    C3,
    3,
    nppiNormDiffL1GetBufferHostSize_16s_C3R_Ctx,
    nppiNormDiff_L1_16s_C3R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_f32_c3_buffer_size,
    norm_diff_l1_f32_c3,
    f32,
    C3,
    3,
    nppiNormDiffL1GetBufferHostSize_32f_C3R_Ctx,
    nppiNormDiff_L1_32f_C3R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_u8_c4_buffer_size,
    norm_diff_l1_u8_c4,
    u8,
    C4,
    4,
    nppiNormDiffL1GetBufferHostSize_8u_C4R_Ctx,
    nppiNormDiff_L1_8u_C4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_u16_c4_buffer_size,
    norm_diff_l1_u16_c4,
    u16,
    C4,
    4,
    nppiNormDiffL1GetBufferHostSize_16u_C4R_Ctx,
    nppiNormDiff_L1_16u_C4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_i16_c4_buffer_size,
    norm_diff_l1_i16_c4,
    i16,
    C4,
    4,
    nppiNormDiffL1GetBufferHostSize_16s_C4R_Ctx,
    nppiNormDiff_L1_16s_C4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs.",
    norm_diff_l1_f32_c4_buffer_size,
    norm_diff_l1_f32_c4,
    f32,
    C4,
    4,
    nppiNormDiffL1GetBufferHostSize_32f_C4R_Ctx,
    nppiNormDiff_L1_32f_C4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l1_u8_ac4_buffer_size,
    norm_diff_l1_u8_ac4,
    u8,
    AC4,
    3,
    nppiNormDiffL1GetBufferHostSize_8u_AC4R_Ctx,
    nppiNormDiff_L1_8u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l1_u16_ac4_buffer_size,
    norm_diff_l1_u16_ac4,
    u16,
    AC4,
    3,
    nppiNormDiffL1GetBufferHostSize_16u_AC4R_Ctx,
    nppiNormDiff_L1_16u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l1_i16_ac4_buffer_size,
    norm_diff_l1_i16_ac4,
    i16,
    AC4,
    3,
    nppiNormDiffL1GetBufferHostSize_16s_AC4R_Ctx,
    nppiNormDiff_L1_16s_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L1 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l1_f32_ac4_buffer_size,
    norm_diff_l1_f32_ac4,
    f32,
    AC4,
    3,
    nppiNormDiffL1GetBufferHostSize_32f_AC4R_Ctx,
    nppiNormDiff_L1_32f_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_u8_c1_buffer_size,
    norm_diff_l2_u8_c1,
    u8,
    C1,
    1,
    nppiNormDiffL2GetBufferHostSize_8u_C1R_Ctx,
    nppiNormDiff_L2_8u_C1R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_u16_c1_buffer_size,
    norm_diff_l2_u16_c1,
    u16,
    C1,
    1,
    nppiNormDiffL2GetBufferHostSize_16u_C1R_Ctx,
    nppiNormDiff_L2_16u_C1R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_i16_c1_buffer_size,
    norm_diff_l2_i16_c1,
    i16,
    C1,
    1,
    nppiNormDiffL2GetBufferHostSize_16s_C1R_Ctx,
    nppiNormDiff_L2_16s_C1R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_f32_c1_buffer_size,
    norm_diff_l2_f32_c1,
    f32,
    C1,
    1,
    nppiNormDiffL2GetBufferHostSize_32f_C1R_Ctx,
    nppiNormDiff_L2_32f_C1R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_u8_c3_buffer_size,
    norm_diff_l2_u8_c3,
    u8,
    C3,
    3,
    nppiNormDiffL2GetBufferHostSize_8u_C3R_Ctx,
    nppiNormDiff_L2_8u_C3R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_u16_c3_buffer_size,
    norm_diff_l2_u16_c3,
    u16,
    C3,
    3,
    nppiNormDiffL2GetBufferHostSize_16u_C3R_Ctx,
    nppiNormDiff_L2_16u_C3R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_i16_c3_buffer_size,
    norm_diff_l2_i16_c3,
    i16,
    C3,
    3,
    nppiNormDiffL2GetBufferHostSize_16s_C3R_Ctx,
    nppiNormDiff_L2_16s_C3R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_f32_c3_buffer_size,
    norm_diff_l2_f32_c3,
    f32,
    C3,
    3,
    nppiNormDiffL2GetBufferHostSize_32f_C3R_Ctx,
    nppiNormDiff_L2_32f_C3R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_u8_c4_buffer_size,
    norm_diff_l2_u8_c4,
    u8,
    C4,
    4,
    nppiNormDiffL2GetBufferHostSize_8u_C4R_Ctx,
    nppiNormDiff_L2_8u_C4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_u16_c4_buffer_size,
    norm_diff_l2_u16_c4,
    u16,
    C4,
    4,
    nppiNormDiffL2GetBufferHostSize_16u_C4R_Ctx,
    nppiNormDiff_L2_16u_C4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_i16_c4_buffer_size,
    norm_diff_l2_i16_c4,
    i16,
    C4,
    4,
    nppiNormDiffL2GetBufferHostSize_16s_C4R_Ctx,
    nppiNormDiff_L2_16s_C4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs.",
    norm_diff_l2_f32_c4_buffer_size,
    norm_diff_l2_f32_c4,
    f32,
    C4,
    4,
    nppiNormDiffL2GetBufferHostSize_32f_C4R_Ctx,
    nppiNormDiff_L2_32f_C4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l2_u8_ac4_buffer_size,
    norm_diff_l2_u8_ac4,
    u8,
    AC4,
    3,
    nppiNormDiffL2GetBufferHostSize_8u_AC4R_Ctx,
    nppiNormDiff_L2_8u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l2_u16_ac4_buffer_size,
    norm_diff_l2_u16_ac4,
    u16,
    AC4,
    3,
    nppiNormDiffL2GetBufferHostSize_16u_AC4R_Ctx,
    nppiNormDiff_L2_16u_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l2_i16_ac4_buffer_size,
    norm_diff_l2_i16_ac4,
    i16,
    AC4,
    3,
    nppiNormDiffL2GetBufferHostSize_16s_AC4R_Ctx,
    nppiNormDiff_L2_16s_AC4R_Ctx
);

impl_error_metric!(
    "Computes the L2 norm of the difference between two image ROIs. The alpha channel is excluded.",
    norm_diff_l2_f32_ac4_buffer_size,
    norm_diff_l2_f32_ac4,
    f32,
    AC4,
    3,
    nppiNormDiffL2GetBufferHostSize_32f_AC4R_Ctx,
    nppiNormDiff_L2_32f_AC4R_Ctx
);

impl_generic_dot_prod!(
    "Computes the infinity norm of the difference between two image ROIs.",
    NormDiffInfC1,
    norm_diff_inf_c1,
    norm_diff_inf_c1_buffer_size,
    C1,
    [
        (u8, norm_diff_inf_u8_c1, norm_diff_inf_u8_c1_buffer_size),
        (u16, norm_diff_inf_u16_c1, norm_diff_inf_u16_c1_buffer_size),
        (i16, norm_diff_inf_i16_c1, norm_diff_inf_i16_c1_buffer_size),
        (f32, norm_diff_inf_f32_c1, norm_diff_inf_f32_c1_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the infinity norm of the difference between two image ROIs.",
    NormDiffInfC3,
    norm_diff_inf_c3,
    norm_diff_inf_c3_buffer_size,
    C3,
    [
        (u8, norm_diff_inf_u8_c3, norm_diff_inf_u8_c3_buffer_size),
        (u16, norm_diff_inf_u16_c3, norm_diff_inf_u16_c3_buffer_size),
        (i16, norm_diff_inf_i16_c3, norm_diff_inf_i16_c3_buffer_size),
        (f32, norm_diff_inf_f32_c3, norm_diff_inf_f32_c3_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the infinity norm of the difference between two image ROIs.",
    NormDiffInfC4,
    norm_diff_inf_c4,
    norm_diff_inf_c4_buffer_size,
    C4,
    [
        (u8, norm_diff_inf_u8_c4, norm_diff_inf_u8_c4_buffer_size),
        (u16, norm_diff_inf_u16_c4, norm_diff_inf_u16_c4_buffer_size),
        (i16, norm_diff_inf_i16_c4, norm_diff_inf_i16_c4_buffer_size),
        (f32, norm_diff_inf_f32_c4, norm_diff_inf_f32_c4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the infinity norm of the difference between two image ROIs. The alpha channel is excluded.",
    NormDiffInfAC4,
    norm_diff_inf_ac4,
    norm_diff_inf_ac4_buffer_size,
    AC4,
    [
        (u8, norm_diff_inf_u8_ac4, norm_diff_inf_u8_ac4_buffer_size),
        (
            u16,
            norm_diff_inf_u16_ac4,
            norm_diff_inf_u16_ac4_buffer_size
        ),
        (
            i16,
            norm_diff_inf_i16_ac4,
            norm_diff_inf_i16_ac4_buffer_size
        ),
        (
            f32,
            norm_diff_inf_f32_ac4,
            norm_diff_inf_f32_ac4_buffer_size
        ),
    ]
);

impl_generic_dot_prod!(
    "Computes the L1 norm of the difference between two image ROIs.",
    NormDiffL1C1,
    norm_diff_l1_c1,
    norm_diff_l1_c1_buffer_size,
    C1,
    [
        (u8, norm_diff_l1_u8_c1, norm_diff_l1_u8_c1_buffer_size),
        (u16, norm_diff_l1_u16_c1, norm_diff_l1_u16_c1_buffer_size),
        (i16, norm_diff_l1_i16_c1, norm_diff_l1_i16_c1_buffer_size),
        (f32, norm_diff_l1_f32_c1, norm_diff_l1_f32_c1_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the L1 norm of the difference between two image ROIs.",
    NormDiffL1C3,
    norm_diff_l1_c3,
    norm_diff_l1_c3_buffer_size,
    C3,
    [
        (u8, norm_diff_l1_u8_c3, norm_diff_l1_u8_c3_buffer_size),
        (u16, norm_diff_l1_u16_c3, norm_diff_l1_u16_c3_buffer_size),
        (i16, norm_diff_l1_i16_c3, norm_diff_l1_i16_c3_buffer_size),
        (f32, norm_diff_l1_f32_c3, norm_diff_l1_f32_c3_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the L1 norm of the difference between two image ROIs.",
    NormDiffL1C4,
    norm_diff_l1_c4,
    norm_diff_l1_c4_buffer_size,
    C4,
    [
        (u8, norm_diff_l1_u8_c4, norm_diff_l1_u8_c4_buffer_size),
        (u16, norm_diff_l1_u16_c4, norm_diff_l1_u16_c4_buffer_size),
        (i16, norm_diff_l1_i16_c4, norm_diff_l1_i16_c4_buffer_size),
        (f32, norm_diff_l1_f32_c4, norm_diff_l1_f32_c4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the L1 norm of the difference between two image ROIs. The alpha channel is excluded.",
    NormDiffL1AC4,
    norm_diff_l1_ac4,
    norm_diff_l1_ac4_buffer_size,
    AC4,
    [
        (u8, norm_diff_l1_u8_ac4, norm_diff_l1_u8_ac4_buffer_size),
        (u16, norm_diff_l1_u16_ac4, norm_diff_l1_u16_ac4_buffer_size),
        (i16, norm_diff_l1_i16_ac4, norm_diff_l1_i16_ac4_buffer_size),
        (f32, norm_diff_l1_f32_ac4, norm_diff_l1_f32_ac4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the L2 norm of the difference between two image ROIs.",
    NormDiffL2C1,
    norm_diff_l2_c1,
    norm_diff_l2_c1_buffer_size,
    C1,
    [
        (u8, norm_diff_l2_u8_c1, norm_diff_l2_u8_c1_buffer_size),
        (u16, norm_diff_l2_u16_c1, norm_diff_l2_u16_c1_buffer_size),
        (i16, norm_diff_l2_i16_c1, norm_diff_l2_i16_c1_buffer_size),
        (f32, norm_diff_l2_f32_c1, norm_diff_l2_f32_c1_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the L2 norm of the difference between two image ROIs.",
    NormDiffL2C3,
    norm_diff_l2_c3,
    norm_diff_l2_c3_buffer_size,
    C3,
    [
        (u8, norm_diff_l2_u8_c3, norm_diff_l2_u8_c3_buffer_size),
        (u16, norm_diff_l2_u16_c3, norm_diff_l2_u16_c3_buffer_size),
        (i16, norm_diff_l2_i16_c3, norm_diff_l2_i16_c3_buffer_size),
        (f32, norm_diff_l2_f32_c3, norm_diff_l2_f32_c3_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the L2 norm of the difference between two image ROIs.",
    NormDiffL2C4,
    norm_diff_l2_c4,
    norm_diff_l2_c4_buffer_size,
    C4,
    [
        (u8, norm_diff_l2_u8_c4, norm_diff_l2_u8_c4_buffer_size),
        (u16, norm_diff_l2_u16_c4, norm_diff_l2_u16_c4_buffer_size),
        (i16, norm_diff_l2_i16_c4, norm_diff_l2_i16_c4_buffer_size),
        (f32, norm_diff_l2_f32_c4, norm_diff_l2_f32_c4_buffer_size),
    ]
);

impl_generic_dot_prod!(
    "Computes the L2 norm of the difference between two image ROIs. The alpha channel is excluded.",
    NormDiffL2AC4,
    norm_diff_l2_ac4,
    norm_diff_l2_ac4_buffer_size,
    AC4,
    [
        (u8, norm_diff_l2_u8_ac4, norm_diff_l2_u8_ac4_buffer_size),
        (u16, norm_diff_l2_u16_ac4, norm_diff_l2_u16_ac4_buffer_size),
        (i16, norm_diff_l2_i16_ac4, norm_diff_l2_i16_ac4_buffer_size),
        (f32, norm_diff_l2_f32_ac4, norm_diff_l2_f32_ac4_buffer_size),
    ]
);
