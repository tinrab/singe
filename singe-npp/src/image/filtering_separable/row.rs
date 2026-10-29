use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{
            validate_filter_axis_mask, validate_filter_kernel_any, validate_filter_kernel_f32,
            validate_same_size,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_u8_c1,
    u8,
    C1,
    nppiFilterRow_8u_C1R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_u8_c3,
    u8,
    C3,
    nppiFilterRow_8u_C3R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_u8_c4,
    u8,
    C4,
    nppiFilterRow_8u_C4R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    filter_row_u8_ac4,
    u8,
    AC4,
    nppiFilterRow_8u_AC4R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_u16_c1,
    u16,
    C1,
    nppiFilterRow_16u_C1R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_u16_c3,
    u16,
    C3,
    nppiFilterRow_16u_C3R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_u16_c4,
    u16,
    C4,
    nppiFilterRow_16u_C4R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    filter_row_u16_ac4,
    u16,
    AC4,
    nppiFilterRow_16u_AC4R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_i16_c1,
    i16,
    C1,
    nppiFilterRow_16s_C1R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_i16_c3,
    i16,
    C3,
    nppiFilterRow_16s_C3R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_i16_c4,
    i16,
    C4,
    nppiFilterRow_16s_C4R_Ctx
);

impl_filter_separable!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    filter_row_i16_ac4,
    i16,
    AC4,
    nppiFilterRow_16s_AC4R_Ctx
);

impl_filter_separable_float!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_f32_c1,
    C1,
    nppiFilterRow_32f_C1R_Ctx
);

impl_filter_separable_float!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_f32_c3,
    C3,
    nppiFilterRow_32f_C3R_Ctx
);

impl_filter_separable_float!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_f32_c4,
    C4,
    nppiFilterRow_32f_C4R_Ctx
);

impl_filter_separable_float!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    filter_row_f32_ac4,
    AC4,
    nppiFilterRow_32f_AC4R_Ctx
);

impl_filter_separable_float_typed!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row_f64_c1,
    f64,
    C1,
    nppiFilterRow_64f_C1R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_u8_c1,
    u8,
    C1,
    nppiSumWindowRow_8u32f_C1R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_u8_c3,
    u8,
    C3,
    nppiSumWindowRow_8u32f_C3R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_u8_c4,
    u8,
    C4,
    nppiSumWindowRow_8u32f_C4R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_u16_c1,
    u16,
    C1,
    nppiSumWindowRow_16u32f_C1R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_u16_c3,
    u16,
    C3,
    nppiSumWindowRow_16u32f_C3R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_u16_c4,
    u16,
    C4,
    nppiSumWindowRow_16u32f_C4R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_i16_c1,
    i16,
    C1,
    nppiSumWindowRow_16s32f_C1R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_i16_c3,
    i16,
    C3,
    nppiSumWindowRow_16s32f_C3R_Ctx
);

impl_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    sum_window_row_i16_c4,
    i16,
    C4,
    nppiSumWindowRow_16s32f_C4R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_u8_c1,
    u8,
    C1,
    nppiFilterRow32f_8u_C1R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_u8_c3,
    u8,
    C3,
    nppiFilterRow32f_8u_C3R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_u8_c4,
    u8,
    C4,
    nppiFilterRow32f_8u_C4R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    filter_row32f_u8_ac4,
    u8,
    AC4,
    nppiFilterRow32f_8u_AC4R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_u16_c1,
    u16,
    C1,
    nppiFilterRow32f_16u_C1R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_u16_c3,
    u16,
    C3,
    nppiFilterRow32f_16u_C3R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_u16_c4,
    u16,
    C4,
    nppiFilterRow32f_16u_C4R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    filter_row32f_u16_ac4,
    u16,
    AC4,
    nppiFilterRow32f_16u_AC4R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_i16_c1,
    i16,
    C1,
    nppiFilterRow32f_16s_C1R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_i16_c3,
    i16,
    C3,
    nppiFilterRow32f_16s_C3R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`.",
    filter_row32f_i16_c4,
    i16,
    C4,
    nppiFilterRow32f_16s_C4R_Ctx
);

impl_filter_separable_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    filter_row32f_i16_ac4,
    i16,
    AC4,
    nppiFilterRow32f_16s_AC4R_Ctx
);

impl_generic_separable_integer_filter!(
    "Convolves the source with the one-dimensional row `kernel`.",
    FilterRowIntegerC1, filter_row_integer, filter_row_integer_c1, C1, [
    u8 => filter_row_u8_c1,
    u16 => filter_row_u16_c1,
    i16 => filter_row_i16_c1,
    ]
);

impl_generic_separable_integer_filter!(
    "Convolves the source with the one-dimensional row `kernel`.",
    FilterRowIntegerC3, filter_row_integer, filter_row_integer_c3, C3, [
    u8 => filter_row_u8_c3,
    u16 => filter_row_u16_c3,
    i16 => filter_row_i16_c3,
    ]
);

impl_generic_separable_integer_filter!(
    "Convolves the source with the one-dimensional row `kernel`.",
    FilterRowIntegerC4, filter_row_integer, filter_row_integer_c4, C4, [
    u8 => filter_row_u8_c4,
    u16 => filter_row_u16_c4,
    i16 => filter_row_i16_c4,
    ]
);

impl_generic_separable_integer_filter!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    FilterRowIntegerAc4, filter_row_integer, filter_row_integer_ac4, AC4, [
    u8 => filter_row_u8_ac4,
    u16 => filter_row_u16_ac4,
    i16 => filter_row_i16_ac4,
    ]
);

impl_generic_separable_typed_filter!(
    "Convolves the source with the one-dimensional row `kernel`.",
    FilterRowTypedC1, filter_row_typed, filter_row_typed_c1, C1, [
    f32 => filter_row_f32_c1,
    f64 => filter_row_f64_c1,
    ]
);

impl_generic_separable_typed_filter!(
    "Convolves the source with the one-dimensional row `kernel`.",
    FilterRowTypedC3, filter_row_typed, filter_row_typed_c3, C3, [
    f32 => filter_row_f32_c3,
    ]
);

impl_generic_separable_typed_filter!(
    "Convolves the source with the one-dimensional row `kernel`.",
    FilterRowTypedC4, filter_row_typed, filter_row_typed_c4, C4, [
    f32 => filter_row_f32_c4,
    ]
);

impl_generic_separable_typed_filter!(
    "Convolves the source with the one-dimensional row `kernel`. The destination alpha channel is not processed.",
    FilterRowTypedAc4, filter_row_typed, filter_row_typed_ac4, AC4, [
    f32 => filter_row_f32_ac4,
    ]
);

impl_generic_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    SumWindowRowC1, sum_window_row, sum_window_row_c1, C1, [
    u8 => sum_window_row_u8_c1,
    u16 => sum_window_row_u16_c1,
    i16 => sum_window_row_i16_c1,
    ]
);

impl_generic_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    SumWindowRowC3, sum_window_row, sum_window_row_c3, C3, [
    u8 => sum_window_row_u8_c3,
    u16 => sum_window_row_u16_c3,
    i16 => sum_window_row_i16_c3,
    ]
);

impl_generic_sum_window!(
    "Sums source pixels in a sliding row window into floating-point destination pixels.",
    SumWindowRowC4, sum_window_row, sum_window_row_c4, C4, [
    u8 => sum_window_row_u8_c4,
    u16 => sum_window_row_u16_c4,
    i16 => sum_window_row_i16_c4,
    ]
);
