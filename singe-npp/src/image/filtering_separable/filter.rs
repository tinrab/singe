use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{
            validate_border_roi, validate_filter_kernel_2d, validate_same_size,
        },
        view::{AC4, C1, C2, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
};

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_u8_c1,
    u8,
    C1,
    nppiFilter_8u_C1R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_u8_c3,
    u8,
    C3,
    nppiFilter_8u_C3R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_u8_c4,
    u8,
    C4,
    nppiFilter_8u_C4R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter_u8_ac4,
    u8,
    AC4,
    nppiFilter_8u_AC4R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_u16_c1,
    u16,
    C1,
    nppiFilter_16u_C1R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_u16_c3,
    u16,
    C3,
    nppiFilter_16u_C3R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_u16_c4,
    u16,
    C4,
    nppiFilter_16u_C4R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter_u16_ac4,
    u16,
    AC4,
    nppiFilter_16u_AC4R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_i16_c1,
    i16,
    C1,
    nppiFilter_16s_C1R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_i16_c3,
    i16,
    C3,
    nppiFilter_16s_C3R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_i16_c4,
    i16,
    C4,
    nppiFilter_16s_C4R_Ctx
);

impl_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter_i16_ac4,
    i16,
    AC4,
    nppiFilter_16s_AC4R_Ctx
);

impl_filter_float_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_f32_c1,
    f32,
    C1,
    nppiFilter_32f_C1R_Ctx
);

impl_filter_float_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_f32_c2,
    f32,
    C2,
    nppiFilter_32f_C2R_Ctx
);

impl_filter_float_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_f32_c3,
    f32,
    C3,
    nppiFilter_32f_C3R_Ctx
);

impl_filter_float_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_f32_c4,
    f32,
    C4,
    nppiFilter_32f_C4R_Ctx
);

impl_filter_float_typed!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter_f32_ac4,
    f32,
    AC4,
    nppiFilter_32f_AC4R_Ctx
);

impl_filter_float_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter_f64_c1,
    f64,
    C1,
    nppiFilter_64f_C1R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_u8_c1,
    u8,
    C1,
    nppiFilterBorder_8u_C1R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_u8_c3,
    u8,
    C3,
    nppiFilterBorder_8u_C3R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_u8_c4,
    u8,
    C4,
    nppiFilterBorder_8u_C4R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border_u8_ac4,
    u8,
    AC4,
    nppiFilterBorder_8u_AC4R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_u16_c1,
    u16,
    C1,
    nppiFilterBorder_16u_C1R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_u16_c3,
    u16,
    C3,
    nppiFilterBorder_16u_C3R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_u16_c4,
    u16,
    C4,
    nppiFilterBorder_16u_C4R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border_u16_ac4,
    u16,
    AC4,
    nppiFilterBorder_16u_AC4R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_i16_c1,
    i16,
    C1,
    nppiFilterBorder_16s_C1R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_i16_c3,
    i16,
    C3,
    nppiFilterBorder_16s_C3R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_i16_c4,
    i16,
    C4,
    nppiFilterBorder_16s_C4R_Ctx
);

impl_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border_i16_ac4,
    i16,
    AC4,
    nppiFilterBorder_16s_AC4R_Ctx
);

impl_filter_float_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_f32_c1,
    f32,
    C1,
    nppiFilterBorder_32f_C1R_Ctx
);

impl_filter_float_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_f32_c2,
    f32,
    C2,
    nppiFilterBorder_32f_C2R_Ctx
);

impl_filter_float_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_f32_c3,
    f32,
    C3,
    nppiFilterBorder_32f_C3R_Ctx
);

impl_filter_float_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border_f32_c4,
    f32,
    C4,
    nppiFilterBorder_32f_C4R_Ctx
);

impl_filter_float_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border_f32_ac4,
    f32,
    AC4,
    nppiFilterBorder_32f_AC4R_Ctx
);

impl_generic_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    FilterIntegerC1, filter_integer, filter_integer_c1, C1, [
    u8 => filter_u8_c1,
    u16 => filter_u16_c1,
    i16 => filter_i16_c1,
    ]
);

impl_generic_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    FilterIntegerC3, filter_integer, filter_integer_c3, C3, [
    u8 => filter_u8_c3,
    u16 => filter_u16_c3,
    i16 => filter_i16_c3,
    ]
);

impl_generic_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`.",
    FilterIntegerC4, filter_integer, filter_integer_c4, C4, [
    u8 => filter_u8_c4,
    u16 => filter_u16_c4,
    i16 => filter_i16_c4,
    ]
);

impl_generic_filter_integer!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    FilterIntegerAc4, filter_integer, filter_integer_ac4, AC4, [
    u8 => filter_u8_ac4,
    u16 => filter_u16_ac4,
    i16 => filter_i16_ac4,
    ]
);

impl_generic_filter_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    FilterTypedC1, filter_typed, filter_typed_c1, C1, [
    f32 => filter_f32_c1,
    f64 => filter_f64_c1,
    ]
);

impl_generic_filter_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    FilterTypedC2, filter_typed, filter_typed_c2, C2, [
    f32 => filter_f32_c2,
    ]
);

impl_generic_filter_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    FilterTypedC3, filter_typed, filter_typed_c3, C3, [
    f32 => filter_f32_c3,
    ]
);

impl_generic_filter_typed!(
    "Convolves the source with the two-dimensional `kernel`.",
    FilterTypedC4, filter_typed, filter_typed_c4, C4, [
    f32 => filter_f32_c4,
    ]
);

impl_generic_filter_typed!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    FilterTypedAc4, filter_typed, filter_typed_ac4, AC4, [
    f32 => filter_f32_ac4,
    ]
);

impl_generic_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    FilterIntegerBorderC1,
    filter_integer_border,
    filter_integer_border_c1,
    C1,
    [
        u8 => filter_border_u8_c1,
        u16 => filter_border_u16_c1,
        i16 => filter_border_i16_c1,
    ]
);

impl_generic_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    FilterIntegerBorderC3,
    filter_integer_border,
    filter_integer_border_c3,
    C3,
    [
        u8 => filter_border_u8_c3,
        u16 => filter_border_u16_c3,
        i16 => filter_border_i16_c3,
    ]
);

impl_generic_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    FilterIntegerBorderC4,
    filter_integer_border,
    filter_integer_border_c4,
    C4,
    [
        u8 => filter_border_u8_c4,
        u16 => filter_border_u16_c4,
        i16 => filter_border_i16_c4,
    ]
);

impl_generic_filter_integer_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    FilterIntegerBorderAc4,
    filter_integer_border,
    filter_integer_border_ac4,
    AC4,
    [
        u8 => filter_border_u8_ac4,
        u16 => filter_border_u16_ac4,
        i16 => filter_border_i16_ac4,
    ]
);

impl_generic_filter_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    FilterTypedBorderC1,
    filter_typed_border,
    filter_typed_border_c1,
    C1,
    [f32 => filter_border_f32_c1]
);

impl_generic_filter_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    FilterTypedBorderC2,
    filter_typed_border,
    filter_typed_border_c2,
    C2,
    [f32 => filter_border_f32_c2]
);

impl_generic_filter_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    FilterTypedBorderC3,
    filter_typed_border,
    filter_typed_border_c3,
    C3,
    [f32 => filter_border_f32_c3]
);

impl_generic_filter_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    FilterTypedBorderC4,
    filter_typed_border,
    filter_typed_border_c4,
    C4,
    [f32 => filter_border_f32_c4]
);

impl_generic_filter_typed_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    FilterTypedBorderAc4,
    filter_typed_border,
    filter_typed_border_ac4,
    AC4,
    [f32 => filter_border_f32_ac4]
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u8_c1,
    u8,
    C1,
    u8,
    C1,
    nppiFilter32f_8u_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u8_c2,
    u8,
    C2,
    u8,
    C2,
    nppiFilter32f_8u_C2R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u8_c3,
    u8,
    C3,
    u8,
    C3,
    nppiFilter32f_8u_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u8_c4,
    u8,
    C4,
    u8,
    C4,
    nppiFilter32f_8u_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter32f_u8_ac4,
    u8,
    AC4,
    u8,
    AC4,
    nppiFilter32f_8u_AC4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i8_c1,
    i8,
    C1,
    i8,
    C1,
    nppiFilter32f_8s_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i8_c2,
    i8,
    C2,
    i8,
    C2,
    nppiFilter32f_8s_C2R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i8_c3,
    i8,
    C3,
    i8,
    C3,
    nppiFilter32f_8s_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i8_c4,
    i8,
    C4,
    i8,
    C4,
    nppiFilter32f_8s_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter32f_i8_ac4,
    i8,
    AC4,
    i8,
    AC4,
    nppiFilter32f_8s_AC4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u16_c1,
    u16,
    C1,
    u16,
    C1,
    nppiFilter32f_16u_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u16_c3,
    u16,
    C3,
    u16,
    C3,
    nppiFilter32f_16u_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u16_c4,
    u16,
    C4,
    u16,
    C4,
    nppiFilter32f_16u_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter32f_u16_ac4,
    u16,
    AC4,
    u16,
    AC4,
    nppiFilter32f_16u_AC4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i16_c1,
    i16,
    C1,
    i16,
    C1,
    nppiFilter32f_16s_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i16_c3,
    i16,
    C3,
    i16,
    C3,
    nppiFilter32f_16s_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i16_c4,
    i16,
    C4,
    i16,
    C4,
    nppiFilter32f_16s_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter32f_i16_ac4,
    i16,
    AC4,
    i16,
    AC4,
    nppiFilter32f_16s_AC4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i32_c1,
    i32,
    C1,
    i32,
    C1,
    nppiFilter32f_32s_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i32_c3,
    i32,
    C3,
    i32,
    C3,
    nppiFilter32f_32s_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i32_c4,
    i32,
    C4,
    i32,
    C4,
    nppiFilter32f_32s_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter32f_i32_ac4,
    i32,
    AC4,
    i32,
    AC4,
    nppiFilter32f_32s_AC4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_f16_c1,
    f16,
    C1,
    f16,
    C1,
    nppiFilter32f_16f_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_f16_c3,
    f16,
    C3,
    f16,
    C3,
    nppiFilter32f_16f_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_f16_c4,
    f16,
    C4,
    f16,
    C4,
    nppiFilter32f_16f_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilter32f_8u16s_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u8_to_i16_c3,
    u8,
    C3,
    i16,
    C3,
    nppiFilter32f_8u16s_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_u8_to_i16_c4,
    u8,
    C4,
    i16,
    C4,
    nppiFilter32f_8u16s_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter32f_u8_to_i16_ac4,
    u8,
    AC4,
    i16,
    AC4,
    nppiFilter32f_8u16s_AC4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilter32f_8s16s_C1R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i8_to_i16_c3,
    i8,
    C3,
    i16,
    C3,
    nppiFilter32f_8s16s_C3R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`.",
    filter32f_i8_to_i16_c4,
    i8,
    C4,
    i16,
    C4,
    nppiFilter32f_8s16s_C4R_Ctx
);

impl_filter32f!(
    "Convolves the source with the two-dimensional `kernel`. The destination alpha channel is not processed.",
    filter32f_i8_to_i16_ac4,
    i8,
    AC4,
    i16,
    AC4,
    nppiFilter32f_8s16s_AC4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u8_c1,
    u8,
    C1,
    u8,
    C1,
    nppiFilterBorder32f_8u_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u8_c2,
    u8,
    C2,
    u8,
    C2,
    nppiFilterBorder32f_8u_C2R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u8_c3,
    u8,
    C3,
    u8,
    C3,
    nppiFilterBorder32f_8u_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u8_c4,
    u8,
    C4,
    u8,
    C4,
    nppiFilterBorder32f_8u_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border32f_u8_ac4,
    u8,
    AC4,
    u8,
    AC4,
    nppiFilterBorder32f_8u_AC4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i8_c1,
    i8,
    C1,
    i8,
    C1,
    nppiFilterBorder32f_8s_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i8_c2,
    i8,
    C2,
    i8,
    C2,
    nppiFilterBorder32f_8s_C2R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i8_c3,
    i8,
    C3,
    i8,
    C3,
    nppiFilterBorder32f_8s_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i8_c4,
    i8,
    C4,
    i8,
    C4,
    nppiFilterBorder32f_8s_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border32f_i8_ac4,
    i8,
    AC4,
    i8,
    AC4,
    nppiFilterBorder32f_8s_AC4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u16_c1,
    u16,
    C1,
    u16,
    C1,
    nppiFilterBorder32f_16u_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u16_c3,
    u16,
    C3,
    u16,
    C3,
    nppiFilterBorder32f_16u_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u16_c4,
    u16,
    C4,
    u16,
    C4,
    nppiFilterBorder32f_16u_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border32f_u16_ac4,
    u16,
    AC4,
    u16,
    AC4,
    nppiFilterBorder32f_16u_AC4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i16_c1,
    i16,
    C1,
    i16,
    C1,
    nppiFilterBorder32f_16s_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i16_c3,
    i16,
    C3,
    i16,
    C3,
    nppiFilterBorder32f_16s_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i16_c4,
    i16,
    C4,
    i16,
    C4,
    nppiFilterBorder32f_16s_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border32f_i16_ac4,
    i16,
    AC4,
    i16,
    AC4,
    nppiFilterBorder32f_16s_AC4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i32_c1,
    i32,
    C1,
    i32,
    C1,
    nppiFilterBorder32f_32s_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i32_c3,
    i32,
    C3,
    i32,
    C3,
    nppiFilterBorder32f_32s_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i32_c4,
    i32,
    C4,
    i32,
    C4,
    nppiFilterBorder32f_32s_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border32f_i32_ac4,
    i32,
    AC4,
    i32,
    AC4,
    nppiFilterBorder32f_32s_AC4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_f16_c1,
    f16,
    C1,
    f16,
    C1,
    nppiFilterBorder32f_16f_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_f16_c3,
    f16,
    C3,
    f16,
    C3,
    nppiFilterBorder32f_16f_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_f16_c4,
    f16,
    C4,
    f16,
    C4,
    nppiFilterBorder32f_16f_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterBorder32f_8u16s_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u8_to_i16_c3,
    u8,
    C3,
    i16,
    C3,
    nppiFilterBorder32f_8u16s_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_u8_to_i16_c4,
    u8,
    C4,
    i16,
    C4,
    nppiFilterBorder32f_8u16s_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border32f_u8_to_i16_ac4,
    u8,
    AC4,
    i16,
    AC4,
    nppiFilterBorder32f_8u16s_AC4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterBorder32f_8s16s_C1R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i8_to_i16_c3,
    i8,
    C3,
    i16,
    C3,
    nppiFilterBorder32f_8s16s_C3R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",
    filter_border32f_i8_to_i16_c4,
    i8,
    C4,
    i16,
    C4,
    nppiFilterBorder32f_8s16s_C4R_Ctx
);

impl_filter32f_border!(
    "Convolves the source with the two-dimensional `kernel`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",
    filter_border32f_i8_to_i16_ac4,
    i8,
    AC4,
    i16,
    AC4,
    nppiFilterBorder32f_8s16s_AC4R_Ctx
);
