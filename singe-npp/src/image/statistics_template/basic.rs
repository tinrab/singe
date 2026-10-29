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
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u8_c1,
    u8,
    C1,
    nppiSqrDistanceFull_Norm_8u_C1RSfs_Ctx
);

impl_template_match_full_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u8_c3,
    u8,
    C3,
    nppiSqrDistanceFull_Norm_8u_C3RSfs_Ctx
);

impl_template_match_full_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u8_c4,
    u8,
    C4,
    nppiSqrDistanceFull_Norm_8u_C4RSfs_Ctx
);

impl_template_match_full_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    square_distance_full_norm_u8_ac4,
    u8,
    AC4,
    nppiSqrDistanceFull_Norm_8u_AC4RSfs_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_f32_c1,
    f32,
    f32,
    C1,
    nppiSqrDistanceFull_Norm_32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_f32_c3,
    f32,
    f32,
    C3,
    nppiSqrDistanceFull_Norm_32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_f32_c4,
    f32,
    f32,
    C4,
    nppiSqrDistanceFull_Norm_32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    square_distance_full_norm_f32_ac4,
    f32,
    f32,
    AC4,
    nppiSqrDistanceFull_Norm_32f_AC4R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u8_to_f32_c1,
    u8,
    f32,
    C1,
    nppiSqrDistanceFull_Norm_8u32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u8_to_f32_c3,
    u8,
    f32,
    C3,
    nppiSqrDistanceFull_Norm_8u32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u8_to_f32_c4,
    u8,
    f32,
    C4,
    nppiSqrDistanceFull_Norm_8u32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    square_distance_full_norm_u8_to_f32_ac4,
    u8,
    f32,
    AC4,
    nppiSqrDistanceFull_Norm_8u32f_AC4R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_i8_to_f32_c1,
    i8,
    f32,
    C1,
    nppiSqrDistanceFull_Norm_8s32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_i8_to_f32_c3,
    i8,
    f32,
    C3,
    nppiSqrDistanceFull_Norm_8s32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_i8_to_f32_c4,
    i8,
    f32,
    C4,
    nppiSqrDistanceFull_Norm_8s32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    square_distance_full_norm_i8_to_f32_ac4,
    i8,
    f32,
    AC4,
    nppiSqrDistanceFull_Norm_8s32f_AC4R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiSqrDistanceFull_Norm_16u32f_C1R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u16_to_f32_c3,
    u16,
    f32,
    C3,
    nppiSqrDistanceFull_Norm_16u32f_C3R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    square_distance_full_norm_u16_to_f32_c4,
    u16,
    f32,
    C4,
    nppiSqrDistanceFull_Norm_16u32f_C4R_Ctx
);

impl_template_match_full!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    square_distance_full_norm_u16_to_f32_ac4,
    u16,
    f32,
    AC4,
    nppiSqrDistanceFull_Norm_16u32f_AC4R_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u8_c1,
    u8,
    C1,
    nppiSqrDistanceSame_Norm_8u_C1RSfs_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u8_c3,
    u8,
    C3,
    nppiSqrDistanceSame_Norm_8u_C3RSfs_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u8_c4,
    u8,
    C4,
    nppiSqrDistanceSame_Norm_8u_C4RSfs_Ctx
);

impl_template_match_same_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    square_distance_same_norm_u8_ac4,
    u8,
    AC4,
    nppiSqrDistanceSame_Norm_8u_AC4RSfs_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_f32_c1,
    f32,
    f32,
    C1,
    nppiSqrDistanceSame_Norm_32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_f32_c3,
    f32,
    f32,
    C3,
    nppiSqrDistanceSame_Norm_32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_f32_c4,
    f32,
    f32,
    C4,
    nppiSqrDistanceSame_Norm_32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    square_distance_same_norm_f32_ac4,
    f32,
    f32,
    AC4,
    nppiSqrDistanceSame_Norm_32f_AC4R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u8_to_f32_c1,
    u8,
    f32,
    C1,
    nppiSqrDistanceSame_Norm_8u32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u8_to_f32_c3,
    u8,
    f32,
    C3,
    nppiSqrDistanceSame_Norm_8u32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u8_to_f32_c4,
    u8,
    f32,
    C4,
    nppiSqrDistanceSame_Norm_8u32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    square_distance_same_norm_u8_to_f32_ac4,
    u8,
    f32,
    AC4,
    nppiSqrDistanceSame_Norm_8u32f_AC4R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_i8_to_f32_c1,
    i8,
    f32,
    C1,
    nppiSqrDistanceSame_Norm_8s32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_i8_to_f32_c3,
    i8,
    f32,
    C3,
    nppiSqrDistanceSame_Norm_8s32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_i8_to_f32_c4,
    i8,
    f32,
    C4,
    nppiSqrDistanceSame_Norm_8s32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    square_distance_same_norm_i8_to_f32_ac4,
    i8,
    f32,
    AC4,
    nppiSqrDistanceSame_Norm_8s32f_AC4R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiSqrDistanceSame_Norm_16u32f_C1R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u16_to_f32_c3,
    u16,
    f32,
    C3,
    nppiSqrDistanceSame_Norm_16u32f_C3R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    square_distance_same_norm_u16_to_f32_c4,
    u16,
    f32,
    C4,
    nppiSqrDistanceSame_Norm_16u32f_C4R_Ctx
);

impl_template_match_same!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    square_distance_same_norm_u16_to_f32_ac4,
    u16,
    f32,
    AC4,
    nppiSqrDistanceSame_Norm_16u32f_AC4R_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u8_c1,
    u8,
    C1,
    nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u8_c3,
    u8,
    C3,
    nppiSqrDistanceValid_Norm_8u_C3RSfs_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u8_c4,
    u8,
    C4,
    nppiSqrDistanceValid_Norm_8u_C4RSfs_Ctx
);

impl_template_match_valid_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    square_distance_valid_norm_u8_ac4,
    u8,
    AC4,
    nppiSqrDistanceValid_Norm_8u_AC4RSfs_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_f32_c1,
    f32,
    f32,
    C1,
    nppiSqrDistanceValid_Norm_32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_f32_c3,
    f32,
    f32,
    C3,
    nppiSqrDistanceValid_Norm_32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_f32_c4,
    f32,
    f32,
    C4,
    nppiSqrDistanceValid_Norm_32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    square_distance_valid_norm_f32_ac4,
    f32,
    f32,
    AC4,
    nppiSqrDistanceValid_Norm_32f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u8_to_f32_c1,
    u8,
    f32,
    C1,
    nppiSqrDistanceValid_Norm_8u32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u8_to_f32_c3,
    u8,
    f32,
    C3,
    nppiSqrDistanceValid_Norm_8u32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u8_to_f32_c4,
    u8,
    f32,
    C4,
    nppiSqrDistanceValid_Norm_8u32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    square_distance_valid_norm_u8_to_f32_ac4,
    u8,
    f32,
    AC4,
    nppiSqrDistanceValid_Norm_8u32f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_i8_to_f32_c1,
    i8,
    f32,
    C1,
    nppiSqrDistanceValid_Norm_8s32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_i8_to_f32_c3,
    i8,
    f32,
    C3,
    nppiSqrDistanceValid_Norm_8s32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_i8_to_f32_c4,
    i8,
    f32,
    C4,
    nppiSqrDistanceValid_Norm_8s32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    square_distance_valid_norm_i8_to_f32_ac4,
    i8,
    f32,
    AC4,
    nppiSqrDistanceValid_Norm_8s32f_AC4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u16_to_f32_c1,
    u16,
    f32,
    C1,
    nppiSqrDistanceValid_Norm_16u32f_C1R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u16_to_f32_c3,
    u16,
    f32,
    C3,
    nppiSqrDistanceValid_Norm_16u32f_C3R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    square_distance_valid_norm_u16_to_f32_c4,
    u16,
    f32,
    C4,
    nppiSqrDistanceValid_Norm_16u32f_C4R_Ctx
);

impl_template_match_valid!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    square_distance_valid_norm_u16_to_f32_ac4,
    u16,
    f32,
    AC4,
    nppiSqrDistanceValid_Norm_16u32f_AC4R_Ctx
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    SquareDistanceFullNormScaledC1,
    square_distance_full_norm_c1_scaled,
    C1,
    [(u8, square_distance_full_norm_u8_c1)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    SquareDistanceFullNormScaledC3,
    square_distance_full_norm_c3_scaled,
    C3,
    [(u8, square_distance_full_norm_u8_c3)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    SquareDistanceFullNormScaledC4,
    square_distance_full_norm_c4_scaled,
    C4,
    [(u8, square_distance_full_norm_u8_c4)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    SquareDistanceFullNormScaledAC4,
    square_distance_full_norm_ac4_scaled,
    AC4,
    [(u8, square_distance_full_norm_u8_ac4)]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    SquareDistanceFullNormC1,
    square_distance_full_norm_c1,
    C1,
    [
        (f32, f32, square_distance_full_norm_f32_c1),
        (u8, f32, square_distance_full_norm_u8_to_f32_c1),
        (i8, f32, square_distance_full_norm_i8_to_f32_c1),
        (u16, f32, square_distance_full_norm_u16_to_f32_c1),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    SquareDistanceFullNormC3,
    square_distance_full_norm_c3,
    C3,
    [
        (f32, f32, square_distance_full_norm_f32_c3),
        (u8, f32, square_distance_full_norm_u8_to_f32_c3),
        (i8, f32, square_distance_full_norm_i8_to_f32_c3),
        (u16, f32, square_distance_full_norm_u16_to_f32_c3),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one.",
    SquareDistanceFullNormC4,
    square_distance_full_norm_c4,
    C4,
    [
        (f32, f32, square_distance_full_norm_f32_c4),
        (u8, f32, square_distance_full_norm_u8_to_f32_c4),
        (i8, f32, square_distance_full_norm_i8_to_f32_c4),
        (u16, f32, square_distance_full_norm_u16_to_f32_c4),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension plus the template dimension minus one. The alpha channel is excluded.",
    SquareDistanceFullNormAC4,
    square_distance_full_norm_ac4,
    AC4,
    [
        (f32, f32, square_distance_full_norm_f32_ac4),
        (u8, f32, square_distance_full_norm_u8_to_f32_ac4),
        (i8, f32, square_distance_full_norm_i8_to_f32_ac4),
        (u16, f32, square_distance_full_norm_u16_to_f32_ac4),
    ]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    SquareDistanceSameNormScaledC1,
    square_distance_same_norm_c1_scaled,
    C1,
    [(u8, square_distance_same_norm_u8_c1)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    SquareDistanceSameNormScaledC3,
    square_distance_same_norm_c3_scaled,
    C3,
    [(u8, square_distance_same_norm_u8_c3)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    SquareDistanceSameNormScaledC4,
    square_distance_same_norm_c4_scaled,
    C4,
    [(u8, square_distance_same_norm_u8_c4)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    SquareDistanceSameNormScaledAC4,
    square_distance_same_norm_ac4_scaled,
    AC4,
    [(u8, square_distance_same_norm_u8_ac4)]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    SquareDistanceSameNormC1,
    square_distance_same_norm_c1,
    C1,
    [
        (f32, f32, square_distance_same_norm_f32_c1),
        (u8, f32, square_distance_same_norm_u8_to_f32_c1),
        (i8, f32, square_distance_same_norm_i8_to_f32_c1),
        (u16, f32, square_distance_same_norm_u16_to_f32_c1),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    SquareDistanceSameNormC3,
    square_distance_same_norm_c3,
    C3,
    [
        (f32, f32, square_distance_same_norm_f32_c3),
        (u8, f32, square_distance_same_norm_u8_to_f32_c3),
        (i8, f32, square_distance_same_norm_i8_to_f32_c3),
        (u16, f32, square_distance_same_norm_u16_to_f32_c3),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI.",
    SquareDistanceSameNormC4,
    square_distance_same_norm_c4,
    C4,
    [
        (f32, f32, square_distance_same_norm_f32_c4),
        (u8, f32, square_distance_same_norm_u8_to_f32_c4),
        (i8, f32, square_distance_same_norm_i8_to_f32_c4),
        (u16, f32, square_distance_same_norm_u16_to_f32_c4),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. The destination ROI must have the same size as the source ROI. The alpha channel is excluded.",
    SquareDistanceSameNormAC4,
    square_distance_same_norm_ac4,
    AC4,
    [
        (f32, f32, square_distance_same_norm_f32_ac4),
        (u8, f32, square_distance_same_norm_u8_to_f32_ac4),
        (i8, f32, square_distance_same_norm_i8_to_f32_ac4),
        (u16, f32, square_distance_same_norm_u16_to_f32_ac4),
    ]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    SquareDistanceValidNormScaledC1,
    square_distance_valid_norm_c1_scaled,
    C1,
    [(u8, square_distance_valid_norm_u8_c1)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    SquareDistanceValidNormScaledC3,
    square_distance_valid_norm_c3_scaled,
    C3,
    [(u8, square_distance_valid_norm_u8_c3)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    SquareDistanceValidNormScaledC4,
    square_distance_valid_norm_c4_scaled,
    C4,
    [(u8, square_distance_valid_norm_u8_c4)]
);

impl_generic_template_match_scaled!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    SquareDistanceValidNormScaledAC4,
    square_distance_valid_norm_ac4_scaled,
    AC4,
    [(u8, square_distance_valid_norm_u8_ac4)]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    SquareDistanceValidNormC1,
    square_distance_valid_norm_c1,
    C1,
    [
        (f32, f32, square_distance_valid_norm_f32_c1),
        (u8, f32, square_distance_valid_norm_u8_to_f32_c1),
        (i8, f32, square_distance_valid_norm_i8_to_f32_c1),
        (u16, f32, square_distance_valid_norm_u16_to_f32_c1),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    SquareDistanceValidNormC3,
    square_distance_valid_norm_c3,
    C3,
    [
        (f32, f32, square_distance_valid_norm_f32_c3),
        (u8, f32, square_distance_valid_norm_u8_to_f32_c3),
        (i8, f32, square_distance_valid_norm_i8_to_f32_c3),
        (u16, f32, square_distance_valid_norm_u16_to_f32_c3),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one.",
    SquareDistanceValidNormC4,
    square_distance_valid_norm_c4,
    C4,
    [
        (f32, f32, square_distance_valid_norm_f32_c4),
        (u8, f32, square_distance_valid_norm_u8_to_f32_c4),
        (i8, f32, square_distance_valid_norm_i8_to_f32_c4),
        (u16, f32, square_distance_valid_norm_u16_to_f32_c4),
    ]
);

impl_generic_template_match!(
    "Computes normalized squared distances between the source image and the template. Each destination dimension must equal the corresponding source dimension minus the template dimension plus one. The alpha channel is excluded.",
    SquareDistanceValidNormAC4,
    square_distance_valid_norm_ac4,
    AC4,
    [
        (f32, f32, square_distance_valid_norm_f32_ac4),
        (u8, f32, square_distance_valid_norm_u8_to_f32_ac4),
        (i8, f32, square_distance_valid_norm_i8_to_f32_ac4),
        (u16, f32, square_distance_valid_norm_u16_to_f32_ac4),
    ]
);
