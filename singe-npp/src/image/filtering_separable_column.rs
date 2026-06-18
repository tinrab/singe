use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::*,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_filter_separable!(filter_column_u8_c1, u8, C1, nppiFilterColumn_8u_C1R_Ctx);
impl_filter_separable!(filter_column_u8_c3, u8, C3, nppiFilterColumn_8u_C3R_Ctx);
impl_filter_separable!(filter_column_u8_c4, u8, C4, nppiFilterColumn_8u_C4R_Ctx);
impl_filter_separable!(filter_column_u8_ac4, u8, AC4, nppiFilterColumn_8u_AC4R_Ctx);
impl_filter_separable!(filter_column_u16_c1, u16, C1, nppiFilterColumn_16u_C1R_Ctx);
impl_filter_separable!(filter_column_u16_c3, u16, C3, nppiFilterColumn_16u_C3R_Ctx);
impl_filter_separable!(filter_column_u16_c4, u16, C4, nppiFilterColumn_16u_C4R_Ctx);
impl_filter_separable!(
    filter_column_u16_ac4,
    u16,
    AC4,
    nppiFilterColumn_16u_AC4R_Ctx
);
impl_filter_separable!(filter_column_i16_c1, i16, C1, nppiFilterColumn_16s_C1R_Ctx);
impl_filter_separable!(filter_column_i16_c3, i16, C3, nppiFilterColumn_16s_C3R_Ctx);
impl_filter_separable!(filter_column_i16_c4, i16, C4, nppiFilterColumn_16s_C4R_Ctx);
impl_filter_separable!(
    filter_column_i16_ac4,
    i16,
    AC4,
    nppiFilterColumn_16s_AC4R_Ctx
);
impl_filter_separable_float!(filter_column_f32_c1, C1, nppiFilterColumn_32f_C1R_Ctx);
impl_filter_separable_float!(filter_column_f32_c3, C3, nppiFilterColumn_32f_C3R_Ctx);
impl_filter_separable_float!(filter_column_f32_c4, C4, nppiFilterColumn_32f_C4R_Ctx);
impl_filter_separable_float!(filter_column_f32_ac4, AC4, nppiFilterColumn_32f_AC4R_Ctx);
impl_filter_separable_float_typed!(filter_column_f64_c1, f64, C1, nppiFilterColumn_64f_C1R_Ctx);
impl_sum_window!(
    sum_window_column_u8_c1,
    u8,
    C1,
    nppiSumWindowColumn_8u32f_C1R_Ctx
);
impl_sum_window!(
    sum_window_column_u8_c3,
    u8,
    C3,
    nppiSumWindowColumn_8u32f_C3R_Ctx
);
impl_sum_window!(
    sum_window_column_u8_c4,
    u8,
    C4,
    nppiSumWindowColumn_8u32f_C4R_Ctx
);
impl_sum_window!(
    sum_window_column_u16_c1,
    u16,
    C1,
    nppiSumWindowColumn_16u32f_C1R_Ctx
);
impl_sum_window!(
    sum_window_column_u16_c3,
    u16,
    C3,
    nppiSumWindowColumn_16u32f_C3R_Ctx
);
impl_sum_window!(
    sum_window_column_u16_c4,
    u16,
    C4,
    nppiSumWindowColumn_16u32f_C4R_Ctx
);
impl_sum_window!(
    sum_window_column_i16_c1,
    i16,
    C1,
    nppiSumWindowColumn_16s32f_C1R_Ctx
);
impl_sum_window!(
    sum_window_column_i16_c3,
    i16,
    C3,
    nppiSumWindowColumn_16s32f_C3R_Ctx
);
impl_sum_window!(
    sum_window_column_i16_c4,
    i16,
    C4,
    nppiSumWindowColumn_16s32f_C4R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u8_c1,
    u8,
    C1,
    nppiFilterColumn32f_8u_C1R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u8_c3,
    u8,
    C3,
    nppiFilterColumn32f_8u_C3R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u8_c4,
    u8,
    C4,
    nppiFilterColumn32f_8u_C4R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u8_ac4,
    u8,
    AC4,
    nppiFilterColumn32f_8u_AC4R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u16_c1,
    u16,
    C1,
    nppiFilterColumn32f_16u_C1R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u16_c3,
    u16,
    C3,
    nppiFilterColumn32f_16u_C3R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u16_c4,
    u16,
    C4,
    nppiFilterColumn32f_16u_C4R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_u16_ac4,
    u16,
    AC4,
    nppiFilterColumn32f_16u_AC4R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_i16_c1,
    i16,
    C1,
    nppiFilterColumn32f_16s_C1R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_i16_c3,
    i16,
    C3,
    nppiFilterColumn32f_16s_C3R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_i16_c4,
    i16,
    C4,
    nppiFilterColumn32f_16s_C4R_Ctx
);
impl_filter_separable_kernel_f32!(
    filter_column32f_i16_ac4,
    i16,
    AC4,
    nppiFilterColumn32f_16s_AC4R_Ctx
);

impl_generic_separable_integer_filter!(
    FilterColumnIntegerC1,
    filter_column_integer,
    filter_column_integer_c1,
    C1,
    [
        u8 => filter_column_u8_c1,
        u16 => filter_column_u16_c1,
        i16 => filter_column_i16_c1,
    ]
);
impl_generic_separable_integer_filter!(
    FilterColumnIntegerC3,
    filter_column_integer,
    filter_column_integer_c3,
    C3,
    [
        u8 => filter_column_u8_c3,
        u16 => filter_column_u16_c3,
        i16 => filter_column_i16_c3,
    ]
);
impl_generic_separable_integer_filter!(
    FilterColumnIntegerC4,
    filter_column_integer,
    filter_column_integer_c4,
    C4,
    [
        u8 => filter_column_u8_c4,
        u16 => filter_column_u16_c4,
        i16 => filter_column_i16_c4,
    ]
);
impl_generic_separable_integer_filter!(
    FilterColumnIntegerAc4,
    filter_column_integer,
    filter_column_integer_ac4,
    AC4,
    [
        u8 => filter_column_u8_ac4,
        u16 => filter_column_u16_ac4,
        i16 => filter_column_i16_ac4,
    ]
);
impl_generic_separable_typed_filter!(
    FilterColumnTypedC1,
    filter_column_typed,
    filter_column_typed_c1,
    C1,
    [
        f32 => filter_column_f32_c1,
        f64 => filter_column_f64_c1,
    ]
);
impl_generic_separable_typed_filter!(
    FilterColumnTypedC3,
    filter_column_typed,
    filter_column_typed_c3,
    C3,
    [f32 => filter_column_f32_c3]
);
impl_generic_separable_typed_filter!(
    FilterColumnTypedC4,
    filter_column_typed,
    filter_column_typed_c4,
    C4,
    [f32 => filter_column_f32_c4]
);
impl_generic_separable_typed_filter!(
    FilterColumnTypedAc4,
    filter_column_typed,
    filter_column_typed_ac4,
    AC4,
    [f32 => filter_column_f32_ac4]
);
impl_generic_sum_window!(SumWindowColumnC1, sum_window_column, sum_window_column_c1, C1, [
    u8 => sum_window_column_u8_c1,
    u16 => sum_window_column_u16_c1,
    i16 => sum_window_column_i16_c1,
]);
impl_generic_sum_window!(SumWindowColumnC3, sum_window_column, sum_window_column_c3, C3, [
    u8 => sum_window_column_u8_c3,
    u16 => sum_window_column_u16_c3,
    i16 => sum_window_column_i16_c3,
]);
impl_generic_sum_window!(SumWindowColumnC4, sum_window_column, sum_window_column_c4, C4, [
    u8 => sum_window_column_u8_c4,
    u16 => sum_window_column_u16_c4,
    i16 => sum_window_column_i16_c4,
]);
