use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{validate_coi, validate_mask_size, validate_statistic_output},
        view::{AC4, C1, C3, C4, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_u8_c1_buffer_size,
    norm_l2_u8_c1,
    u8,
    C1,
    1,
    nppiNormL2GetBufferHostSize_8u_C1R_Ctx,
    nppiNorm_L2_8u_C1R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_u16_c1_buffer_size,
    norm_l2_u16_c1,
    u16,
    C1,
    1,
    nppiNormL2GetBufferHostSize_16u_C1R_Ctx,
    nppiNorm_L2_16u_C1R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_i16_c1_buffer_size,
    norm_l2_i16_c1,
    i16,
    C1,
    1,
    nppiNormL2GetBufferHostSize_16s_C1R_Ctx,
    nppiNorm_L2_16s_C1R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_f32_c1_buffer_size,
    norm_l2_f32_c1,
    f32,
    C1,
    1,
    nppiNormL2GetBufferHostSize_32f_C1R_Ctx,
    nppiNorm_L2_32f_C1R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_u8_c3_buffer_size,
    norm_l2_u8_c3,
    u8,
    C3,
    3,
    nppiNormL2GetBufferHostSize_8u_C3R_Ctx,
    nppiNorm_L2_8u_C3R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_u16_c3_buffer_size,
    norm_l2_u16_c3,
    u16,
    C3,
    3,
    nppiNormL2GetBufferHostSize_16u_C3R_Ctx,
    nppiNorm_L2_16u_C3R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_i16_c3_buffer_size,
    norm_l2_i16_c3,
    i16,
    C3,
    3,
    nppiNormL2GetBufferHostSize_16s_C3R_Ctx,
    nppiNorm_L2_16s_C3R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_f32_c3_buffer_size,
    norm_l2_f32_c3,
    f32,
    C3,
    3,
    nppiNormL2GetBufferHostSize_32f_C3R_Ctx,
    nppiNorm_L2_32f_C3R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_u8_c4_buffer_size,
    norm_l2_u8_c4,
    u8,
    C4,
    4,
    nppiNormL2GetBufferHostSize_8u_C4R_Ctx,
    nppiNorm_L2_8u_C4R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_u16_c4_buffer_size,
    norm_l2_u16_c4,
    u16,
    C4,
    4,
    nppiNormL2GetBufferHostSize_16u_C4R_Ctx,
    nppiNorm_L2_16u_C4R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_i16_c4_buffer_size,
    norm_l2_i16_c4,
    i16,
    C4,
    4,
    nppiNormL2GetBufferHostSize_16s_C4R_Ctx,
    nppiNorm_L2_16s_C4R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI.",
    norm_l2_f32_c4_buffer_size,
    norm_l2_f32_c4,
    f32,
    C4,
    4,
    nppiNormL2GetBufferHostSize_32f_C4R_Ctx,
    nppiNorm_L2_32f_C4R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI. The alpha channel is excluded.",
    norm_l2_u8_ac4_buffer_size,
    norm_l2_u8_ac4,
    u8,
    AC4,
    3,
    nppiNormL2GetBufferHostSize_8u_AC4R_Ctx,
    nppiNorm_L2_8u_AC4R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI. The alpha channel is excluded.",
    norm_l2_u16_ac4_buffer_size,
    norm_l2_u16_ac4,
    u16,
    AC4,
    3,
    nppiNormL2GetBufferHostSize_16u_AC4R_Ctx,
    nppiNorm_L2_16u_AC4R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI. The alpha channel is excluded.",
    norm_l2_i16_ac4_buffer_size,
    norm_l2_i16_ac4,
    i16,
    AC4,
    3,
    nppiNormL2GetBufferHostSize_16s_AC4R_Ctx,
    nppiNorm_L2_16s_AC4R_Ctx
);

impl_norm_statistic!(
    "Computes the L2 norm of the source image ROI. The alpha channel is excluded.",
    norm_l2_f32_ac4_buffer_size,
    norm_l2_f32_ac4,
    f32,
    AC4,
    3,
    nppiNormL2GetBufferHostSize_32f_AC4R_Ctx,
    nppiNorm_L2_32f_AC4R_Ctx
);

impl_generic_statistic!(
    "Computes the L2 norm of the source image ROI.",
    NormL2C1,
    norm_l2_c1,
    norm_l2_c1_buffer_size,
    C1,
    [
        (u8, norm_l2_u8_c1, norm_l2_u8_c1_buffer_size),
        (u16, norm_l2_u16_c1, norm_l2_u16_c1_buffer_size),
        (i16, norm_l2_i16_c1, norm_l2_i16_c1_buffer_size),
        (f32, norm_l2_f32_c1, norm_l2_f32_c1_buffer_size),
    ]
);

impl_generic_statistic!(
    "Computes the L2 norm of the source image ROI.",
    NormL2C3,
    norm_l2_c3,
    norm_l2_c3_buffer_size,
    C3,
    [
        (u8, norm_l2_u8_c3, norm_l2_u8_c3_buffer_size),
        (u16, norm_l2_u16_c3, norm_l2_u16_c3_buffer_size),
        (i16, norm_l2_i16_c3, norm_l2_i16_c3_buffer_size),
        (f32, norm_l2_f32_c3, norm_l2_f32_c3_buffer_size),
    ]
);

impl_generic_statistic!(
    "Computes the L2 norm of the source image ROI.",
    NormL2C4,
    norm_l2_c4,
    norm_l2_c4_buffer_size,
    C4,
    [
        (u8, norm_l2_u8_c4, norm_l2_u8_c4_buffer_size),
        (u16, norm_l2_u16_c4, norm_l2_u16_c4_buffer_size),
        (i16, norm_l2_i16_c4, norm_l2_i16_c4_buffer_size),
        (f32, norm_l2_f32_c4, norm_l2_f32_c4_buffer_size),
    ]
);

impl_generic_statistic!(
    "Computes the L2 norm of the source image ROI. The alpha channel is excluded.",
    NormL2AC4,
    norm_l2_ac4,
    norm_l2_ac4_buffer_size,
    AC4,
    [
        (u8, norm_l2_u8_ac4, norm_l2_u8_ac4_buffer_size),
        (u16, norm_l2_u16_ac4, norm_l2_u16_ac4_buffer_size),
        (i16, norm_l2_i16_ac4, norm_l2_i16_ac4_buffer_size),
        (f32, norm_l2_f32_ac4, norm_l2_f32_ac4_buffer_size),
    ]
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_u8_c1_masked_buffer_size,
    norm_l2_u8_c1_masked,
    u8,
    C1,
    nppiNormL2GetBufferHostSize_8u_C1MR_Ctx,
    nppiNorm_L2_8u_C1MR_Ctx
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_i8_c1_masked_buffer_size,
    norm_l2_i8_c1_masked,
    i8,
    C1,
    nppiNormL2GetBufferHostSize_8s_C1MR_Ctx,
    nppiNorm_L2_8s_C1MR_Ctx
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_u16_c1_masked_buffer_size,
    norm_l2_u16_c1_masked,
    u16,
    C1,
    nppiNormL2GetBufferHostSize_16u_C1MR_Ctx,
    nppiNorm_L2_16u_C1MR_Ctx
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_f32_c1_masked_buffer_size,
    norm_l2_f32_c1_masked,
    f32,
    C1,
    nppiNormL2GetBufferHostSize_32f_C1MR_Ctx,
    nppiNorm_L2_32f_C1MR_Ctx
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_u8_c3_masked_buffer_size,
    norm_l2_u8_c3_masked,
    u8,
    C3,
    nppiNormL2GetBufferHostSize_8u_C3CMR_Ctx,
    nppiNorm_L2_8u_C3CMR_Ctx
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_i8_c3_masked_buffer_size,
    norm_l2_i8_c3_masked,
    i8,
    C3,
    nppiNormL2GetBufferHostSize_8s_C3CMR_Ctx,
    nppiNorm_L2_8s_C3CMR_Ctx
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_u16_c3_masked_buffer_size,
    norm_l2_u16_c3_masked,
    u16,
    C3,
    nppiNormL2GetBufferHostSize_16u_C3CMR_Ctx,
    nppiNorm_L2_16u_C3CMR_Ctx
);

impl_norm_statistic_masked!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    norm_l2_f32_c3_masked_buffer_size,
    norm_l2_f32_c3_masked,
    f32,
    C3,
    nppiNormL2GetBufferHostSize_32f_C3CMR_Ctx,
    nppiNorm_L2_32f_C3CMR_Ctx
);

impl_generic_statistic_masked_c1!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    NormL2MaskedC1,
    norm_l2_c1_masked,
    norm_l2_c1_masked_buffer_size,
    [
        (u8, norm_l2_u8_c1_masked, norm_l2_u8_c1_masked_buffer_size),
        (i8, norm_l2_i8_c1_masked, norm_l2_i8_c1_masked_buffer_size),
        (
            u16,
            norm_l2_u16_c1_masked,
            norm_l2_u16_c1_masked_buffer_size
        ),
        (
            f32,
            norm_l2_f32_c1_masked,
            norm_l2_f32_c1_masked_buffer_size
        ),
    ]
);

impl_generic_statistic_masked_c3!(
    "Computes the L2 norm of the source image ROI. Only pixels selected by `mask` contribute.",
    NormL2MaskedC3,
    norm_l2_c3_masked,
    norm_l2_c3_masked_buffer_size,
    [
        (u8, norm_l2_u8_c3_masked, norm_l2_u8_c3_masked_buffer_size),
        (i8, norm_l2_i8_c3_masked, norm_l2_i8_c3_masked_buffer_size),
        (
            u16,
            norm_l2_u16_c3_masked,
            norm_l2_u16_c3_masked_buffer_size
        ),
        (
            f32,
            norm_l2_f32_c3_masked,
            norm_l2_f32_c3_masked_buffer_size
        ),
    ]
);
