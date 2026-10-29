use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_same_size},
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, MaskSize, Point},
};

impl_filter_directional!(
    "Single channel 8-bit unsigned horizontal Sobel filter.",
    filter_sobel_horizontal_u8_c1,
    u8,
    C1,
    nppiFilterSobelHoriz_8u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 8-bit unsigned horizontal Sobel filter.",
    filter_sobel_horizontal_u8_c3,
    u8,
    C3,
    nppiFilterSobelHoriz_8u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned horizontal Sobel filter.",
    filter_sobel_horizontal_u8_c4,
    u8,
    C4,
    nppiFilterSobelHoriz_8u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed horizontal Sobel filter, ignoring alpha channel.",
    filter_sobel_horizontal_u8_ac4,
    u8,
    AC4,
    nppiFilterSobelHoriz_8u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit signed horizontal Sobel filter.",
    filter_sobel_horizontal_i16_c1,
    i16,
    C1,
    nppiFilterSobelHoriz_16s_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit signed horizontal Sobel filter.",
    filter_sobel_horizontal_i16_c3,
    i16,
    C3,
    nppiFilterSobelHoriz_16s_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed horizontal Sobel filter.",
    filter_sobel_horizontal_i16_c4,
    i16,
    C4,
    nppiFilterSobelHoriz_16s_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned horizontal Sobel filter, ignoring alpha channel.",
    filter_sobel_horizontal_i16_ac4,
    i16,
    AC4,
    nppiFilterSobelHoriz_16s_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point horizontal Sobel filter.",
    filter_sobel_horizontal_f32_c1,
    f32,
    C1,
    nppiFilterSobelHoriz_32f_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 32-bit floating-point horizontal Sobel filter.",
    filter_sobel_horizontal_f32_c3,
    f32,
    C3,
    nppiFilterSobelHoriz_32f_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point horizontal Sobel filter.",
    filter_sobel_horizontal_f32_c4,
    f32,
    C4,
    nppiFilterSobelHoriz_32f_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point horizontal Sobel filter, ignoring alpha channel.",
    filter_sobel_horizontal_f32_ac4,
    f32,
    AC4,
    nppiFilterSobelHoriz_32f_AC4R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit unsigned to 16-bit signed horizontal Sobel filter.",
    filter_sobel_horizontal_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelHoriz_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit signed to 16-bit signed horizontal Sobel filter.",
    filter_sobel_horizontal_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelHoriz_8s16s_C1R_Ctx
);

impl_filter_directional!(
    "Single channel 8-bit unsigned vertical Sobel filter.",
    filter_sobel_vertical_u8_c1,
    u8,
    C1,
    nppiFilterSobelVert_8u_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 8-bit unsigned vertical Sobel filter.",
    filter_sobel_vertical_u8_c3,
    u8,
    C3,
    nppiFilterSobelVert_8u_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned vertical Sobel filter.",
    filter_sobel_vertical_u8_c4,
    u8,
    C4,
    nppiFilterSobelVert_8u_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed vertical Sobel filter, ignoring alpha channel.",
    filter_sobel_vertical_u8_ac4,
    u8,
    AC4,
    nppiFilterSobelVert_8u_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 16-bit signed vertical Sobel filter.",
    filter_sobel_vertical_i16_c1,
    i16,
    C1,
    nppiFilterSobelVert_16s_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 16-bit signed vertical Sobel filter.",
    filter_sobel_vertical_i16_c3,
    i16,
    C3,
    nppiFilterSobelVert_16s_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 16-bit signed vertical Sobel filter.",
    filter_sobel_vertical_i16_c4,
    i16,
    C4,
    nppiFilterSobelVert_16s_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 8-bit unsigned vertical Sobel filter, ignoring alpha channel.",
    filter_sobel_vertical_i16_ac4,
    i16,
    AC4,
    nppiFilterSobelVert_16s_AC4R_Ctx
);

impl_filter_directional!(
    "Single channel 32-bit floating-point vertical Sobel filter.",
    filter_sobel_vertical_f32_c1,
    f32,
    C1,
    nppiFilterSobelVert_32f_C1R_Ctx
);

impl_filter_directional!(
    "Three channel 32-bit floating-point vertical Sobel filter.",
    filter_sobel_vertical_f32_c3,
    f32,
    C3,
    nppiFilterSobelVert_32f_C3R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point vertical Sobel filter.",
    filter_sobel_vertical_f32_c4,
    f32,
    C4,
    nppiFilterSobelVert_32f_C4R_Ctx
);

impl_filter_directional!(
    "Four channel 32-bit floating-point vertical Sobel filter, ignoring alpha channel.",
    filter_sobel_vertical_f32_ac4,
    f32,
    AC4,
    nppiFilterSobelVert_32f_AC4R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit unsigned to 16-bit signed vertical Sobel filter.",
    filter_sobel_vertical_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelVert_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed!(
    "Single channel 8-bit signed to 16-bit signed vertical Sobel filter.",
    filter_sobel_vertical_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelVert_8s16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Single channel 8-bit unsigned horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_u8_c1,
    u8,
    C1,
    nppiFilterSobelHorizBorder_8u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 8-bit unsigned horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_u8_c3,
    u8,
    C3,
    nppiFilterSobelHorizBorder_8u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_u8_c4,
    u8,
    C4,
    nppiFilterSobelHorizBorder_8u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed horizontal Sobel filter with border control, ignoring alpha channel.",
    filter_sobel_horizontal_border_u8_ac4,
    u8,
    AC4,
    nppiFilterSobelHorizBorder_8u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit signed horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_i16_c1,
    i16,
    C1,
    nppiFilterSobelHorizBorder_16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit signed horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_i16_c3,
    i16,
    C3,
    nppiFilterSobelHorizBorder_16s_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_i16_c4,
    i16,
    C4,
    nppiFilterSobelHorizBorder_16s_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned horizontal Sobel filter with border control, ignoring alpha channel.",
    filter_sobel_horizontal_border_i16_ac4,
    i16,
    AC4,
    nppiFilterSobelHorizBorder_16s_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizBorder_32f_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 32-bit floating-point horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_f32_c3,
    f32,
    C3,
    nppiFilterSobelHorizBorder_32f_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_f32_c4,
    f32,
    C4,
    nppiFilterSobelHorizBorder_32f_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point horizontal Sobel filter with border control, ignoring alpha channel.",
    filter_sobel_horizontal_border_f32_ac4,
    f32,
    AC4,
    nppiFilterSobelHorizBorder_32f_AC4R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizBorder_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit signed to 16-bit signed horizontal Sobel filter with border control.",
    filter_sobel_horizontal_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizBorder_8s16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Single channel 8-bit unsigned vertical Sobel filter with border control.",
    filter_sobel_vertical_border_u8_c1,
    u8,
    C1,
    nppiFilterSobelVertBorder_8u_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 8-bit unsigned vertical Sobel filter with border control.",
    filter_sobel_vertical_border_u8_c3,
    u8,
    C3,
    nppiFilterSobelVertBorder_8u_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned vertical Sobel filter with border control.",
    filter_sobel_vertical_border_u8_c4,
    u8,
    C4,
    nppiFilterSobelVertBorder_8u_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed vertical Sobel filter with border control, ignoring alpha channel.",
    filter_sobel_vertical_border_u8_ac4,
    u8,
    AC4,
    nppiFilterSobelVertBorder_8u_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 16-bit signed vertical Sobel filter with border control.",
    filter_sobel_vertical_border_i16_c1,
    i16,
    C1,
    nppiFilterSobelVertBorder_16s_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 16-bit signed vertical Sobel filter with border control.",
    filter_sobel_vertical_border_i16_c3,
    i16,
    C3,
    nppiFilterSobelVertBorder_16s_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 16-bit signed vertical Sobel filter with border control.",
    filter_sobel_vertical_border_i16_c4,
    i16,
    C4,
    nppiFilterSobelVertBorder_16s_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 8-bit unsigned vertical Sobel filter with border control, ignoring alpha channel.",
    filter_sobel_vertical_border_i16_ac4,
    i16,
    AC4,
    nppiFilterSobelVertBorder_16s_AC4R_Ctx
);

impl_filter_directional_border!(
    "Single channel 32-bit floating-point vertical Sobel filter with border control.",
    filter_sobel_vertical_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertBorder_32f_C1R_Ctx
);

impl_filter_directional_border!(
    "Three channel 32-bit floating-point vertical Sobel filter with border control.",
    filter_sobel_vertical_border_f32_c3,
    f32,
    C3,
    nppiFilterSobelVertBorder_32f_C3R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point vertical Sobel filter with border control.",
    filter_sobel_vertical_border_f32_c4,
    f32,
    C4,
    nppiFilterSobelVertBorder_32f_C4R_Ctx
);

impl_filter_directional_border!(
    "Four channel 32-bit floating-point vertical Sobel filter with border control, ignoring alpha channel.",
    filter_sobel_vertical_border_f32_ac4,
    f32,
    AC4,
    nppiFilterSobelVertBorder_32f_AC4R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit unsigned to 16-bit signed vertical Sobel filter with border control.",
    filter_sobel_vertical_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertBorder_8u16s_C1R_Ctx
);

impl_filter_masked_kernel_typed_border!(
    "Single channel 8-bit signed to 16-bit signed vertical Sobel filter with border control.",
    filter_sobel_vertical_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertBorder_8s16s_C1R_Ctx
);
