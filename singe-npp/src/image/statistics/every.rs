use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::validate_same_size,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_u8_c1,
    u8,
    C1,
    nppiMaxEvery_8u_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_u16_c1,
    u16,
    C1,
    nppiMaxEvery_16u_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_i16_c1,
    i16,
    C1,
    nppiMaxEvery_16s_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_f32_c1,
    f32,
    C1,
    nppiMaxEvery_32f_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_u8_c3,
    u8,
    C3,
    nppiMaxEvery_8u_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_u16_c3,
    u16,
    C3,
    nppiMaxEvery_16u_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_i16_c3,
    i16,
    C3,
    nppiMaxEvery_16s_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_f32_c3,
    f32,
    C3,
    nppiMaxEvery_32f_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_u8_c4,
    u8,
    C4,
    nppiMaxEvery_8u_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_u16_c4,
    u16,
    C4,
    nppiMaxEvery_16u_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_i16_c4,
    i16,
    C4,
    nppiMaxEvery_16s_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    max_every_f32_c4,
    f32,
    C4,
    nppiMaxEvery_32f_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample. The alpha channel is excluded.",
    max_every_u8_ac4,
    u8,
    AC4,
    nppiMaxEvery_8u_AC4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample. The alpha channel is excluded.",
    max_every_u16_ac4,
    u16,
    AC4,
    nppiMaxEvery_16u_AC4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample. The alpha channel is excluded.",
    max_every_i16_ac4,
    i16,
    AC4,
    nppiMaxEvery_16s_AC4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample. The alpha channel is excluded.",
    max_every_f32_ac4,
    f32,
    AC4,
    nppiMaxEvery_32f_AC4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_u8_c1,
    u8,
    C1,
    nppiMinEvery_8u_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_u16_c1,
    u16,
    C1,
    nppiMinEvery_16u_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_i16_c1,
    i16,
    C1,
    nppiMinEvery_16s_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_f32_c1,
    f32,
    C1,
    nppiMinEvery_32f_C1IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_u8_c3,
    u8,
    C3,
    nppiMinEvery_8u_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_u16_c3,
    u16,
    C3,
    nppiMinEvery_16u_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_i16_c3,
    i16,
    C3,
    nppiMinEvery_16s_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_f32_c3,
    f32,
    C3,
    nppiMinEvery_32f_C3IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_u8_c4,
    u8,
    C4,
    nppiMinEvery_8u_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_u16_c4,
    u16,
    C4,
    nppiMinEvery_16u_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_i16_c4,
    i16,
    C4,
    nppiMinEvery_16s_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    min_every_f32_c4,
    f32,
    C4,
    nppiMinEvery_32f_C4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample. The alpha channel is excluded.",
    min_every_u8_ac4,
    u8,
    AC4,
    nppiMinEvery_8u_AC4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample. The alpha channel is excluded.",
    min_every_u16_ac4,
    u16,
    AC4,
    nppiMinEvery_16u_AC4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample. The alpha channel is excluded.",
    min_every_i16_ac4,
    i16,
    AC4,
    nppiMinEvery_16s_AC4IR_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample. The alpha channel is excluded.",
    min_every_f32_ac4,
    f32,
    AC4,
    nppiMinEvery_32f_AC4IR_Ctx
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    MaxEveryC1,
    max_every_c1,
    C1,
    [
        (u8, max_every_u8_c1),
        (u16, max_every_u16_c1),
        (i16, max_every_i16_c1),
        (f32, max_every_f32_c1),
    ]
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    MaxEveryC3,
    max_every_c3,
    C3,
    [
        (u8, max_every_u8_c3),
        (u16, max_every_u16_c3),
        (i16, max_every_i16_c3),
        (f32, max_every_f32_c3),
    ]
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample.",
    MaxEveryC4,
    max_every_c4,
    C4,
    [
        (u8, max_every_u8_c4),
        (u16, max_every_u16_c4),
        (i16, max_every_i16_c4),
        (f32, max_every_f32_c4),
    ]
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the maximum of its current value and the corresponding source sample. The alpha channel is excluded.",
    MaxEveryAC4,
    max_every_ac4,
    AC4,
    [
        (u8, max_every_u8_ac4),
        (u16, max_every_u16_ac4),
        (i16, max_every_i16_ac4),
        (f32, max_every_f32_ac4),
    ]
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    MinEveryC1,
    min_every_c1,
    C1,
    [
        (u8, min_every_u8_c1),
        (u16, min_every_u16_c1),
        (i16, min_every_i16_c1),
        (f32, min_every_f32_c1),
    ]
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    MinEveryC3,
    min_every_c3,
    C3,
    [
        (u8, min_every_u8_c3),
        (u16, min_every_u16_c3),
        (i16, min_every_i16_c3),
        (f32, min_every_f32_c3),
    ]
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample.",
    MinEveryC4,
    min_every_c4,
    C4,
    [
        (u8, min_every_u8_c4),
        (u16, min_every_u16_c4),
        (i16, min_every_i16_c4),
        (f32, min_every_f32_c4),
    ]
);

impl_generic_every_statistic_in_place!(
    "Replaces each destination sample with the minimum of its current value and the corresponding source sample. The alpha channel is excluded.",
    MinEveryAC4,
    min_every_ac4,
    AC4,
    [
        (u8, min_every_u8_ac4),
        (u16, min_every_u16_ac4),
        (i16, min_every_i16_ac4),
        (f32, min_every_f32_ac4),
    ]
);
