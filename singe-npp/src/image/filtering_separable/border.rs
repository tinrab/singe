use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{
            validate_border_roi, validate_filter_axis_mask, validate_filter_kernel,
            validate_filter_kernel_f32,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point},
};

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_u8_c1,
    u8,
    C1,
    nppiFilterColumnBorder_8u_C1R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_u8_c3,
    u8,
    C3,
    nppiFilterColumnBorder_8u_C3R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_u8_c4,
    u8,
    C4,
    nppiFilterColumnBorder_8u_C4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_column_border_u8_ac4,
    u8,
    AC4,
    nppiFilterColumnBorder_8u_AC4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_u16_c1,
    u16,
    C1,
    nppiFilterColumnBorder_16u_C1R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_u16_c3,
    u16,
    C3,
    nppiFilterColumnBorder_16u_C3R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_u16_c4,
    u16,
    C4,
    nppiFilterColumnBorder_16u_C4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_column_border_u16_ac4,
    u16,
    AC4,
    nppiFilterColumnBorder_16u_AC4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_i16_c1,
    i16,
    C1,
    nppiFilterColumnBorder_16s_C1R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_i16_c3,
    i16,
    C3,
    nppiFilterColumnBorder_16s_C3R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_i16_c4,
    i16,
    C4,
    nppiFilterColumnBorder_16s_C4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_column_border_i16_ac4,
    i16,
    AC4,
    nppiFilterColumnBorder_16s_AC4R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_f32_c1,
    C1,
    nppiFilterColumnBorder_32f_C1R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_f32_c3,
    C3,
    nppiFilterColumnBorder_32f_C3R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border_f32_c4,
    C4,
    nppiFilterColumnBorder_32f_C4R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_column_border_f32_ac4,
    AC4,
    nppiFilterColumnBorder_32f_AC4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_u8_c1,
    u8,
    C1,
    nppiFilterColumnBorder32f_8u_C1R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_u8_c3,
    u8,
    C3,
    nppiFilterColumnBorder32f_8u_C3R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_u8_c4,
    u8,
    C4,
    nppiFilterColumnBorder32f_8u_C4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_column_border32f_u8_ac4,
    u8,
    AC4,
    nppiFilterColumnBorder32f_8u_AC4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_u16_c1,
    u16,
    C1,
    nppiFilterColumnBorder32f_16u_C1R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_u16_c3,
    u16,
    C3,
    nppiFilterColumnBorder32f_16u_C3R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_u16_c4,
    u16,
    C4,
    nppiFilterColumnBorder32f_16u_C4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_column_border32f_u16_ac4,
    u16,
    AC4,
    nppiFilterColumnBorder32f_16u_AC4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_i16_c1,
    i16,
    C1,
    nppiFilterColumnBorder32f_16s_C1R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_i16_c3,
    i16,
    C3,
    nppiFilterColumnBorder32f_16s_C3R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_column_border32f_i16_c4,
    i16,
    C4,
    nppiFilterColumnBorder32f_16s_C4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional column `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_column_border32f_i16_ac4,
    i16,
    AC4,
    nppiFilterColumnBorder32f_16s_AC4R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_u8_c1,
    u8,
    C1,
    nppiSumWindowColumnBorder_8u32f_C1R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_u8_c3,
    u8,
    C3,
    nppiSumWindowColumnBorder_8u32f_C3R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_u8_c4,
    u8,
    C4,
    nppiSumWindowColumnBorder_8u32f_C4R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_u16_c1,
    u16,
    C1,
    nppiSumWindowColumnBorder_16u32f_C1R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_u16_c3,
    u16,
    C3,
    nppiSumWindowColumnBorder_16u32f_C3R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_u16_c4,
    u16,
    C4,
    nppiSumWindowColumnBorder_16u32f_C4R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_i16_c1,
    i16,
    C1,
    nppiSumWindowColumnBorder_16s32f_C1R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_i16_c3,
    i16,
    C3,
    nppiSumWindowColumnBorder_16s32f_C3R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_column_border_i16_c4,
    i16,
    C4,
    nppiSumWindowColumnBorder_16s32f_C4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_u8_c1,
    u8,
    C1,
    nppiFilterRowBorder_8u_C1R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_u8_c3,
    u8,
    C3,
    nppiFilterRowBorder_8u_C3R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_u8_c4,
    u8,
    C4,
    nppiFilterRowBorder_8u_C4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_row_border_u8_ac4,
    u8,
    AC4,
    nppiFilterRowBorder_8u_AC4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_u16_c1,
    u16,
    C1,
    nppiFilterRowBorder_16u_C1R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_u16_c3,
    u16,
    C3,
    nppiFilterRowBorder_16u_C3R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_u16_c4,
    u16,
    C4,
    nppiFilterRowBorder_16u_C4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_row_border_u16_ac4,
    u16,
    AC4,
    nppiFilterRowBorder_16u_AC4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_i16_c1,
    i16,
    C1,
    nppiFilterRowBorder_16s_C1R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_i16_c3,
    i16,
    C3,
    nppiFilterRowBorder_16s_C3R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_i16_c4,
    i16,
    C4,
    nppiFilterRowBorder_16s_C4R_Ctx
);

impl_filter_separable_border!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_row_border_i16_ac4,
    i16,
    AC4,
    nppiFilterRowBorder_16s_AC4R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_f32_c1,
    C1,
    nppiFilterRowBorder_32f_C1R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_f32_c3,
    C3,
    nppiFilterRowBorder_32f_C3R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border_f32_c4,
    C4,
    nppiFilterRowBorder_32f_C4R_Ctx
);

impl_filter_separable_border_float!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_row_border_f32_ac4,
    AC4,
    nppiFilterRowBorder_32f_AC4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_u8_c1,
    u8,
    C1,
    nppiFilterRowBorder32f_8u_C1R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_u8_c3,
    u8,
    C3,
    nppiFilterRowBorder32f_8u_C3R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_u8_c4,
    u8,
    C4,
    nppiFilterRowBorder32f_8u_C4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_row_border32f_u8_ac4,
    u8,
    AC4,
    nppiFilterRowBorder32f_8u_AC4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_u16_c1,
    u16,
    C1,
    nppiFilterRowBorder32f_16u_C1R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_u16_c3,
    u16,
    C3,
    nppiFilterRowBorder32f_16u_C3R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_u16_c4,
    u16,
    C4,
    nppiFilterRowBorder32f_16u_C4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_row_border32f_u16_ac4,
    u16,
    AC4,
    nppiFilterRowBorder32f_16u_AC4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_i16_c1,
    i16,
    C1,
    nppiFilterRowBorder32f_16s_C1R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_i16_c3,
    i16,
    C3,
    nppiFilterRowBorder32f_16s_C3R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_row_border32f_i16_c4,
    i16,
    C4,
    nppiFilterRowBorder32f_16s_C4R_Ctx
);

impl_filter_separable_border_kernel_f32!(
    "Convolves the source with the one-dimensional row `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_row_border32f_i16_ac4,
    i16,
    AC4,
    nppiFilterRowBorder32f_16s_AC4R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_u8_c1,
    u8,
    C1,
    nppiSumWindowRowBorder_8u32f_C1R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_u8_c3,
    u8,
    C3,
    nppiSumWindowRowBorder_8u32f_C3R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_u8_c4,
    u8,
    C4,
    nppiSumWindowRowBorder_8u32f_C4R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_u16_c1,
    u16,
    C1,
    nppiSumWindowRowBorder_16u32f_C1R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_u16_c3,
    u16,
    C3,
    nppiSumWindowRowBorder_16u32f_C3R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_u16_c4,
    u16,
    C4,
    nppiSumWindowRowBorder_16u32f_C4R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_i16_c1,
    i16,
    C1,
    nppiSumWindowRowBorder_16s32f_C1R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_i16_c3,
    i16,
    C3,
    nppiSumWindowRowBorder_16s32f_C3R_Ctx
);

impl_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    sum_window_row_border_i16_c4,
    i16,
    C4,
    nppiSumWindowRowBorder_16s32f_C4R_Ctx
);

impl_generic_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    SumWindowColumnBorderC1,
    sum_window_column_border,
    sum_window_column_border_c1,
    C1,
    [
        u8 => sum_window_column_border_u8_c1,
        u16 => sum_window_column_border_u16_c1,
        i16 => sum_window_column_border_i16_c1,
    ]
);

impl_generic_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    SumWindowColumnBorderC3,
    sum_window_column_border,
    sum_window_column_border_c3,
    C3,
    [
        u8 => sum_window_column_border_u8_c3,
        u16 => sum_window_column_border_u16_c3,
        i16 => sum_window_column_border_i16_c3,
    ]
);

impl_generic_sum_window_border!(
    "Sums source pixels in a sliding column window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    SumWindowColumnBorderC4,
    sum_window_column_border,
    sum_window_column_border_c4,
    C4,
    [
        u8 => sum_window_column_border_u8_c4,
        u16 => sum_window_column_border_u16_c4,
        i16 => sum_window_column_border_i16_c4,
    ]
);

impl_generic_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    SumWindowRowBorderC1,
    sum_window_row_border,
    sum_window_row_border_c1,
    C1,
    [
        u8 => sum_window_row_border_u8_c1,
        u16 => sum_window_row_border_u16_c1,
        i16 => sum_window_row_border_i16_c1,
    ]
);

impl_generic_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    SumWindowRowBorderC3,
    sum_window_row_border,
    sum_window_row_border_c3,
    C3,
    [
        u8 => sum_window_row_border_u8_c3,
        u16 => sum_window_row_border_u16_c3,
        i16 => sum_window_row_border_i16_c3,
    ]
);

impl_generic_sum_window_border!(
    "Sums source pixels in a sliding row window into floating-point destination pixels. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    SumWindowRowBorderC4,
    sum_window_row_border,
    sum_window_row_border_c4,
    C4,
    [
        u8 => sum_window_row_border_u8_c4,
        u16 => sum_window_row_border_u16_c4,
        i16 => sum_window_row_border_i16_c4,
    ]
);
