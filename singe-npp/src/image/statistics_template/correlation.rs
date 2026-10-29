use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{
            validate_same_size, validate_template_full_destination_size,
            validate_template_valid_destination_size,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_template_match_full_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u8_c1,
    u8,
    C1,
    nppiCrossCorrFull_Norm_8u_C1RSfs_Ctx
);

impl_template_match_full_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u8_c3,
    u8,
    C3,
    nppiCrossCorrFull_Norm_8u_C3RSfs_Ctx
);

impl_template_match_full_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u8_c4,
    u8,
    C4,
    nppiCrossCorrFull_Norm_8u_C4RSfs_Ctx
);

impl_template_match_full_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    cross_correlation_full_norm_u8_ac4,
    u8,
    AC4,
    nppiCrossCorrFull_Norm_8u_AC4RSfs_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_f32_c1,
    f32,
    f32,
    C1,
    nppiCrossCorrFull_Norm_32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_f32_c3,
    f32,
    f32,
    C3,
    nppiCrossCorrFull_Norm_32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_f32_c4,
    f32,
    f32,
    C4,
    nppiCrossCorrFull_Norm_32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    cross_correlation_full_norm_f32_ac4,
    f32,
    f32,
    AC4,
    nppiCrossCorrFull_Norm_32f_AC4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_f64_c1,
    f64,
    f64,
    C1,
    nppiCrossCorrFull_Norm_64f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_f64_c3,
    f64,
    f64,
    C3,
    nppiCrossCorrFull_Norm_64f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_f64_c4,
    f64,
    f64,
    C4,
    nppiCrossCorrFull_Norm_64f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    cross_correlation_full_norm_f64_ac4,
    f64,
    f64,
    AC4,
    nppiCrossCorrFull_Norm_64f_AC4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u8_to_f32_c1,
    u8,
    f32,
    C1,
    nppiCrossCorrFull_Norm_8u32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u8_to_f32_c3,
    u8,
    f32,
    C3,
    nppiCrossCorrFull_Norm_8u32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u8_to_f32_c4,
    u8,
    f32,
    C4,
    nppiCrossCorrFull_Norm_8u32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    cross_correlation_full_norm_u8_to_f32_ac4,
    u8,
    f32,
    AC4,
    nppiCrossCorrFull_Norm_8u32f_AC4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_i8_to_f32_c1,
    i8,
    f32,
    C1,
    nppiCrossCorrFull_Norm_8s32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_i8_to_f32_c3,
    i8,
    f32,
    C3,
    nppiCrossCorrFull_Norm_8s32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_i8_to_f32_c4,
    i8,
    f32,
    C4,
    nppiCrossCorrFull_Norm_8s32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    cross_correlation_full_norm_i8_to_f32_ac4,
    i8,
    f32,
    AC4,
    nppiCrossCorrFull_Norm_8s32f_AC4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiCrossCorrFull_Norm_16u32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u16_to_f32_c3,
    u16,
    f32,
    C3,
    nppiCrossCorrFull_Norm_16u32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    cross_correlation_full_norm_u16_to_f32_c4,
    u16,
    f32,
    C4,
    nppiCrossCorrFull_Norm_16u32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    cross_correlation_full_norm_u16_to_f32_ac4,
    u16,
    f32,
    AC4,
    nppiCrossCorrFull_Norm_16u32f_AC4R_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u8_c1,
    u8,
    C1,
    nppiCrossCorrSame_Norm_8u_C1RSfs_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u8_c3,
    u8,
    C3,
    nppiCrossCorrSame_Norm_8u_C3RSfs_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u8_c4,
    u8,
    C4,
    nppiCrossCorrSame_Norm_8u_C4RSfs_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    cross_correlation_same_norm_u8_ac4,
    u8,
    AC4,
    nppiCrossCorrSame_Norm_8u_AC4RSfs_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_f32_c1,
    f32,
    f32,
    C1,
    nppiCrossCorrSame_Norm_32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_f32_c3,
    f32,
    f32,
    C3,
    nppiCrossCorrSame_Norm_32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_f32_c4,
    f32,
    f32,
    C4,
    nppiCrossCorrSame_Norm_32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    cross_correlation_same_norm_f32_ac4,
    f32,
    f32,
    AC4,
    nppiCrossCorrSame_Norm_32f_AC4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_f64_c1,
    f64,
    f64,
    C1,
    nppiCrossCorrSame_Norm_64f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_f64_c3,
    f64,
    f64,
    C3,
    nppiCrossCorrSame_Norm_64f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_f64_c4,
    f64,
    f64,
    C4,
    nppiCrossCorrSame_Norm_64f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    cross_correlation_same_norm_f64_ac4,
    f64,
    f64,
    AC4,
    nppiCrossCorrSame_Norm_64f_AC4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u8_to_f32_c1,
    u8,
    f32,
    C1,
    nppiCrossCorrSame_Norm_8u32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u8_to_f32_c3,
    u8,
    f32,
    C3,
    nppiCrossCorrSame_Norm_8u32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u8_to_f32_c4,
    u8,
    f32,
    C4,
    nppiCrossCorrSame_Norm_8u32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    cross_correlation_same_norm_u8_to_f32_ac4,
    u8,
    f32,
    AC4,
    nppiCrossCorrSame_Norm_8u32f_AC4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_i8_to_f32_c1,
    i8,
    f32,
    C1,
    nppiCrossCorrSame_Norm_8s32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_i8_to_f32_c3,
    i8,
    f32,
    C3,
    nppiCrossCorrSame_Norm_8s32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_i8_to_f32_c4,
    i8,
    f32,
    C4,
    nppiCrossCorrSame_Norm_8s32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    cross_correlation_same_norm_i8_to_f32_ac4,
    i8,
    f32,
    AC4,
    nppiCrossCorrSame_Norm_8s32f_AC4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiCrossCorrSame_Norm_16u32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u16_to_f32_c3,
    u16,
    f32,
    C3,
    nppiCrossCorrSame_Norm_16u32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    cross_correlation_same_norm_u16_to_f32_c4,
    u16,
    f32,
    C4,
    nppiCrossCorrSame_Norm_16u32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    cross_correlation_same_norm_u16_to_f32_ac4,
    u16,
    f32,
    AC4,
    nppiCrossCorrSame_Norm_16u32f_AC4R_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u8_c1,
    u8,
    C1,
    nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u8_c3,
    u8,
    C3,
    nppiCrossCorrValid_Norm_8u_C3RSfs_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u8_c4,
    u8,
    C4,
    nppiCrossCorrValid_Norm_8u_C4RSfs_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    cross_correlation_valid_norm_u8_ac4,
    u8,
    AC4,
    nppiCrossCorrValid_Norm_8u_AC4RSfs_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_f32_c1,
    f32,
    f32,
    C1,
    nppiCrossCorrValid_Norm_32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_f32_c3,
    f32,
    f32,
    C3,
    nppiCrossCorrValid_Norm_32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_f32_c4,
    f32,
    f32,
    C4,
    nppiCrossCorrValid_Norm_32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    cross_correlation_valid_norm_f32_ac4,
    f32,
    f32,
    AC4,
    nppiCrossCorrValid_Norm_32f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_f64_c1,
    f64,
    f64,
    C1,
    nppiCrossCorrValid_Norm_64f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_f64_c3,
    f64,
    f64,
    C3,
    nppiCrossCorrValid_Norm_64f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_f64_c4,
    f64,
    f64,
    C4,
    nppiCrossCorrValid_Norm_64f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    cross_correlation_valid_norm_f64_ac4,
    f64,
    f64,
    AC4,
    nppiCrossCorrValid_Norm_64f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u8_to_f32_c1,
    u8,
    f32,
    C1,
    nppiCrossCorrValid_Norm_8u32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u8_to_f32_c3,
    u8,
    f32,
    C3,
    nppiCrossCorrValid_Norm_8u32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u8_to_f32_c4,
    u8,
    f32,
    C4,
    nppiCrossCorrValid_Norm_8u32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    cross_correlation_valid_norm_u8_to_f32_ac4,
    u8,
    f32,
    AC4,
    nppiCrossCorrValid_Norm_8u32f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_i8_to_f32_c1,
    i8,
    f32,
    C1,
    nppiCrossCorrValid_Norm_8s32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_i8_to_f32_c3,
    i8,
    f32,
    C3,
    nppiCrossCorrValid_Norm_8s32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_i8_to_f32_c4,
    i8,
    f32,
    C4,
    nppiCrossCorrValid_Norm_8s32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    cross_correlation_valid_norm_i8_to_f32_ac4,
    i8,
    f32,
    AC4,
    nppiCrossCorrValid_Norm_8s32f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiCrossCorrValid_Norm_16u32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u16_to_f32_c3,
    u16,
    f32,
    C3,
    nppiCrossCorrValid_Norm_16u32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_norm_u16_to_f32_c4,
    u16,
    f32,
    C4,
    nppiCrossCorrValid_Norm_16u32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    cross_correlation_valid_norm_u16_to_f32_ac4,
    u16,
    f32,
    AC4,
    nppiCrossCorrValid_Norm_16u32f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_f32_c1,
    f32,
    f32,
    C1,
    nppiCrossCorrValid_32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_f64_c1,
    f64,
    f64,
    C1,
    nppiCrossCorrValid_64f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_u8_to_f32_c1,
    u8,
    f32,
    C1,
    nppiCrossCorrValid_8u32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_i8_to_f32_c1,
    i8,
    f32,
    C1,
    nppiCrossCorrValid_8s32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    cross_correlation_valid_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiCrossCorrValid_16u32f_C1R_Ctx
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    CrossCorrelationFullNormScaledC1,
    cross_correlation_full_norm_c1_scaled,
    C1,
    [(u8, cross_correlation_full_norm_u8_c1)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    CrossCorrelationFullNormScaledC3,
    cross_correlation_full_norm_c3_scaled,
    C3,
    [(u8, cross_correlation_full_norm_u8_c3)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    CrossCorrelationFullNormScaledC4,
    cross_correlation_full_norm_c4_scaled,
    C4,
    [(u8, cross_correlation_full_norm_u8_c4)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    CrossCorrelationFullNormScaledAC4,
    cross_correlation_full_norm_ac4_scaled,
    AC4,
    [(u8, cross_correlation_full_norm_u8_ac4)]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    CrossCorrelationFullNormC1,
    cross_correlation_full_norm_c1,
    C1,
    [
        (f32, f32, cross_correlation_full_norm_f32_c1),
        (f64, f64, cross_correlation_full_norm_f64_c1),
        (u8, f32, cross_correlation_full_norm_u8_to_f32_c1),
        (i8, f32, cross_correlation_full_norm_i8_to_f32_c1),
        (u16, f32, cross_correlation_full_norm_u16_to_f32_c1),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    CrossCorrelationFullNormC3,
    cross_correlation_full_norm_c3,
    C3,
    [
        (f32, f32, cross_correlation_full_norm_f32_c3),
        (f64, f64, cross_correlation_full_norm_f64_c3),
        (u8, f32, cross_correlation_full_norm_u8_to_f32_c3),
        (i8, f32, cross_correlation_full_norm_i8_to_f32_c3),
        (u16, f32, cross_correlation_full_norm_u16_to_f32_c3),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    CrossCorrelationFullNormC4,
    cross_correlation_full_norm_c4,
    C4,
    [
        (f32, f32, cross_correlation_full_norm_f32_c4),
        (f64, f64, cross_correlation_full_norm_f64_c4),
        (u8, f32, cross_correlation_full_norm_u8_to_f32_c4),
        (i8, f32, cross_correlation_full_norm_i8_to_f32_c4),
        (u16, f32, cross_correlation_full_norm_u16_to_f32_c4),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    CrossCorrelationFullNormAC4,
    cross_correlation_full_norm_ac4,
    AC4,
    [
        (f32, f32, cross_correlation_full_norm_f32_ac4),
        (f64, f64, cross_correlation_full_norm_f64_ac4),
        (u8, f32, cross_correlation_full_norm_u8_to_f32_ac4),
        (i8, f32, cross_correlation_full_norm_i8_to_f32_ac4),
        (u16, f32, cross_correlation_full_norm_u16_to_f32_ac4),
    ]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    CrossCorrelationSameNormScaledC1,
    cross_correlation_same_norm_c1_scaled,
    C1,
    [(u8, cross_correlation_same_norm_u8_c1)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    CrossCorrelationSameNormScaledC3,
    cross_correlation_same_norm_c3_scaled,
    C3,
    [(u8, cross_correlation_same_norm_u8_c3)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    CrossCorrelationSameNormScaledC4,
    cross_correlation_same_norm_c4_scaled,
    C4,
    [(u8, cross_correlation_same_norm_u8_c4)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    CrossCorrelationSameNormScaledAC4,
    cross_correlation_same_norm_ac4_scaled,
    AC4,
    [(u8, cross_correlation_same_norm_u8_ac4)]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    CrossCorrelationSameNormC1,
    cross_correlation_same_norm_c1,
    C1,
    [
        (f32, f32, cross_correlation_same_norm_f32_c1),
        (f64, f64, cross_correlation_same_norm_f64_c1),
        (u8, f32, cross_correlation_same_norm_u8_to_f32_c1),
        (i8, f32, cross_correlation_same_norm_i8_to_f32_c1),
        (u16, f32, cross_correlation_same_norm_u16_to_f32_c1),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    CrossCorrelationSameNormC3,
    cross_correlation_same_norm_c3,
    C3,
    [
        (f32, f32, cross_correlation_same_norm_f32_c3),
        (f64, f64, cross_correlation_same_norm_f64_c3),
        (u8, f32, cross_correlation_same_norm_u8_to_f32_c3),
        (i8, f32, cross_correlation_same_norm_i8_to_f32_c3),
        (u16, f32, cross_correlation_same_norm_u16_to_f32_c3),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI.",
    CrossCorrelationSameNormC4,
    cross_correlation_same_norm_c4,
    C4,
    [
        (f32, f32, cross_correlation_same_norm_f32_c4),
        (f64, f64, cross_correlation_same_norm_f64_c4),
        (u8, f32, cross_correlation_same_norm_u8_to_f32_c4),
        (i8, f32, cross_correlation_same_norm_i8_to_f32_c4),
        (u16, f32, cross_correlation_same_norm_u16_to_f32_c4),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    CrossCorrelationSameNormAC4,
    cross_correlation_same_norm_ac4,
    AC4,
    [
        (f32, f32, cross_correlation_same_norm_f32_ac4),
        (f64, f64, cross_correlation_same_norm_f64_ac4),
        (u8, f32, cross_correlation_same_norm_u8_to_f32_ac4),
        (i8, f32, cross_correlation_same_norm_i8_to_f32_ac4),
        (u16, f32, cross_correlation_same_norm_u16_to_f32_ac4),
    ]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    CrossCorrelationValidNormScaledC1,
    cross_correlation_valid_norm_c1_scaled,
    C1,
    [(u8, cross_correlation_valid_norm_u8_c1)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    CrossCorrelationValidNormScaledC3,
    cross_correlation_valid_norm_c3_scaled,
    C3,
    [(u8, cross_correlation_valid_norm_u8_c3)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    CrossCorrelationValidNormScaledC4,
    cross_correlation_valid_norm_c4_scaled,
    C4,
    [(u8, cross_correlation_valid_norm_u8_c4)]
);

impl_generic_template_match_scaled!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    CrossCorrelationValidNormScaledAC4,
    cross_correlation_valid_norm_ac4_scaled,
    AC4,
    [(u8, cross_correlation_valid_norm_u8_ac4)]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    CrossCorrelationValidNormC1,
    cross_correlation_valid_norm_c1,
    C1,
    [
        (f32, f32, cross_correlation_valid_norm_f32_c1),
        (f64, f64, cross_correlation_valid_norm_f64_c1),
        (u8, f32, cross_correlation_valid_norm_u8_to_f32_c1),
        (i8, f32, cross_correlation_valid_norm_i8_to_f32_c1),
        (u16, f32, cross_correlation_valid_norm_u16_to_f32_c1),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    CrossCorrelationValidNormC3,
    cross_correlation_valid_norm_c3,
    C3,
    [
        (f32, f32, cross_correlation_valid_norm_f32_c3),
        (f64, f64, cross_correlation_valid_norm_f64_c3),
        (u8, f32, cross_correlation_valid_norm_u8_to_f32_c3),
        (i8, f32, cross_correlation_valid_norm_i8_to_f32_c3),
        (u16, f32, cross_correlation_valid_norm_u16_to_f32_c3),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    CrossCorrelationValidNormC4,
    cross_correlation_valid_norm_c4,
    C4,
    [
        (f32, f32, cross_correlation_valid_norm_f32_c4),
        (f64, f64, cross_correlation_valid_norm_f64_c4),
        (u8, f32, cross_correlation_valid_norm_u8_to_f32_c4),
        (i8, f32, cross_correlation_valid_norm_i8_to_f32_c4),
        (u16, f32, cross_correlation_valid_norm_u16_to_f32_c4),
    ]
);

impl_generic_template_match!(
    "Computes normalized cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    CrossCorrelationValidNormAC4,
    cross_correlation_valid_norm_ac4,
    AC4,
    [
        (f32, f32, cross_correlation_valid_norm_f32_ac4),
        (f64, f64, cross_correlation_valid_norm_f64_ac4),
        (u8, f32, cross_correlation_valid_norm_u8_to_f32_ac4),
        (i8, f32, cross_correlation_valid_norm_i8_to_f32_ac4),
        (u16, f32, cross_correlation_valid_norm_u16_to_f32_ac4),
    ]
);

impl_generic_template_match!(
    "Computes cross-correlation between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    CrossCorrelationValidC1,
    cross_correlation_valid_c1,
    C1,
    [
        (f32, f32, cross_correlation_valid_f32_c1),
        (f64, f64, cross_correlation_valid_f64_c1),
        (u8, f32, cross_correlation_valid_u8_to_f32_c1),
        (i8, f32, cross_correlation_valid_i8_to_f32_c1),
        (u16, f32, cross_correlation_valid_u16_to_f32_c1),
    ]
);
