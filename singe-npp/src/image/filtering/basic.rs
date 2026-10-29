use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_border_roi, validate_filter_mask, validate_same_size},
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
    utility::to_usize,
};

impl_filter_box!(
    "Single channel 8-bit unsigned box filter.",
    filter_box_u8_c1,
    u8,
    C1,
    nppiFilterBox_8u_C1R_Ctx
);

impl_filter_box!(
    "Three channel 8-bit unsigned box filter.",
    filter_box_u8_c3,
    u8,
    C3,
    nppiFilterBox_8u_C3R_Ctx
);

impl_filter_box!(
    "Four channel 8-bit unsigned box filter.",
    filter_box_u8_c4,
    u8,
    C4,
    nppiFilterBox_8u_C4R_Ctx
);

impl_filter_box!(
    "Four channel 8-bit unsigned box filter, ignorting alpha channel.",
    filter_box_u8_ac4,
    u8,
    AC4,
    nppiFilterBox_8u_AC4R_Ctx
);

impl_filter_box!(
    "Single channel 16-bit unsigned box filter.",
    filter_box_u16_c1,
    u16,
    C1,
    nppiFilterBox_16u_C1R_Ctx
);

impl_filter_box!(
    "Three channel 16-bit unsigned box filter.",
    filter_box_u16_c3,
    u16,
    C3,
    nppiFilterBox_16u_C3R_Ctx
);

impl_filter_box!(
    "Four channel 16-bit unsigned box filter.",
    filter_box_u16_c4,
    u16,
    C4,
    nppiFilterBox_16u_C4R_Ctx
);

impl_filter_box!(
    "Four channel 16-bit unsigned box filter, ignorting alpha channel.",
    filter_box_u16_ac4,
    u16,
    AC4,
    nppiFilterBox_16u_AC4R_Ctx
);

impl_filter_box!(
    "Single channel 16-bit box filter.",
    filter_box_i16_c1,
    i16,
    C1,
    nppiFilterBox_16s_C1R_Ctx
);

impl_filter_box!(
    "Three channel 16-bit box filter.",
    filter_box_i16_c3,
    i16,
    C3,
    nppiFilterBox_16s_C3R_Ctx
);

impl_filter_box!(
    "Four channel 16-bit box filter.",
    filter_box_i16_c4,
    i16,
    C4,
    nppiFilterBox_16s_C4R_Ctx
);

impl_filter_box!(
    "Four channel 16-bit box filter, ignorting alpha channel.",
    filter_box_i16_ac4,
    i16,
    AC4,
    nppiFilterBox_16s_AC4R_Ctx
);

impl_filter_box!(
    "Single channel 32-bit floating-point box filter.",
    filter_box_f32_c1,
    f32,
    C1,
    nppiFilterBox_32f_C1R_Ctx
);

impl_filter_box!(
    "Three channel 32-bit floating-point box filter.",
    filter_box_f32_c3,
    f32,
    C3,
    nppiFilterBox_32f_C3R_Ctx
);

impl_filter_box!(
    "Four channel 32-bit floating-point box filter.",
    filter_box_f32_c4,
    f32,
    C4,
    nppiFilterBox_32f_C4R_Ctx
);

impl_filter_box!(
    "Four channel 32-bit floating-point box filter, ignorting alpha channel.",
    filter_box_f32_ac4,
    f32,
    AC4,
    nppiFilterBox_32f_AC4R_Ctx
);

impl_filter_box!(
    "Single channel 64-bit floating-point box filter.",
    filter_box_f64_c1,
    f64,
    C1,
    nppiFilterBox_64f_C1R_Ctx
);

impl_filter_box_border!(
    "Single channel 8-bit unsigned box filter with border control.",
    filter_box_border_u8_c1,
    u8,
    C1,
    nppiFilterBoxBorder_8u_C1R_Ctx
);

impl_filter_box_border!(
    "Three channel 8-bit unsigned box filter with border control.",
    filter_box_border_u8_c3,
    u8,
    C3,
    nppiFilterBoxBorder_8u_C3R_Ctx
);

impl_filter_box_border!(
    "Four channel 8-bit unsigned box filter with border control.",
    filter_box_border_u8_c4,
    u8,
    C4,
    nppiFilterBoxBorder_8u_C4R_Ctx
);

impl_filter_box_border!(
    "Four channel 8-bit unsigned box filter with border control, ignoring alpha channel.",
    filter_box_border_u8_ac4,
    u8,
    AC4,
    nppiFilterBoxBorder_8u_AC4R_Ctx
);

impl_filter_box_border!(
    "Single channel 16-bit unsigned box filter with border control.",
    filter_box_border_u16_c1,
    u16,
    C1,
    nppiFilterBoxBorder_16u_C1R_Ctx
);

impl_filter_box_border!(
    "Three channel 16-bit unsigned box filter with border control.",
    filter_box_border_u16_c3,
    u16,
    C3,
    nppiFilterBoxBorder_16u_C3R_Ctx
);

impl_filter_box_border!(
    "Four channel 16-bit unsigned box filter with border control.",
    filter_box_border_u16_c4,
    u16,
    C4,
    nppiFilterBoxBorder_16u_C4R_Ctx
);

impl_filter_box_border!(
    "Four channel 16-bit unsigned box filter with border control, ignoring alpha channel.",
    filter_box_border_u16_ac4,
    u16,
    AC4,
    nppiFilterBoxBorder_16u_AC4R_Ctx
);

impl_filter_box_border!(
    "Single channel 16-bit box filter with border control.",
    filter_box_border_i16_c1,
    i16,
    C1,
    nppiFilterBoxBorder_16s_C1R_Ctx
);

impl_filter_box_border!(
    "Three channel 16-bit box filter with border control.",
    filter_box_border_i16_c3,
    i16,
    C3,
    nppiFilterBoxBorder_16s_C3R_Ctx
);

impl_filter_box_border!(
    "Four channel 16-bit box filter with border control.",
    filter_box_border_i16_c4,
    i16,
    C4,
    nppiFilterBoxBorder_16s_C4R_Ctx
);

impl_filter_box_border!(
    "Four channel 16-bit box filter with border control, ignoring alpha channel.",
    filter_box_border_i16_ac4,
    i16,
    AC4,
    nppiFilterBoxBorder_16s_AC4R_Ctx
);

impl_filter_box_border!(
    "Single channel 32-bit floating-point box filter with border control.",
    filter_box_border_f32_c1,
    f32,
    C1,
    nppiFilterBoxBorder_32f_C1R_Ctx
);

impl_filter_box_border!(
    "Three channel 32-bit floating-point box filter with border control.",
    filter_box_border_f32_c3,
    f32,
    C3,
    nppiFilterBoxBorder_32f_C3R_Ctx
);

impl_filter_box_border!(
    "Four channel 32-bit floating-point box filter with border control.",
    filter_box_border_f32_c4,
    f32,
    C4,
    nppiFilterBoxBorder_32f_C4R_Ctx
);

impl_filter_box_border!(
    "Four channel 32-bit floating-point box filter with border control, ignoring alpha channel.",
    filter_box_border_f32_ac4,
    f32,
    AC4,
    nppiFilterBoxBorder_32f_AC4R_Ctx
);

impl_filter_box_border_advanced!(
    "Single channel 8-bit unsigned box filter with border control.",
    filter_box_border_advanced_u8_c1_buffer_size,
    filter_box_border_advanced_u8_c1,
    u8,
    C1,
    nppiFilterBoxBorderAdvanced_8u_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);

impl_filter_box_border_advanced!(
    "Three channel 8-bit unsigned box filter with border control.",
    filter_box_border_advanced_u8_c3_buffer_size,
    filter_box_border_advanced_u8_c3,
    u8,
    C3,
    nppiFilterBoxBorderAdvanced_8u_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);

impl_filter_box_border_advanced!(
    "Four channel 8-bit unsigned box filter with border control.",
    filter_box_border_advanced_u8_c4_buffer_size,
    filter_box_border_advanced_u8_c4,
    u8,
    C4,
    nppiFilterBoxBorderAdvanced_8u_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);

impl_filter_box_border_advanced!(
    "Single channel 16-bit unsigned box filter with border control.",
    filter_box_border_advanced_u16_c1_buffer_size,
    filter_box_border_advanced_u16_c1,
    u16,
    C1,
    nppiFilterBoxBorderAdvanced_16u_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);

impl_filter_box_border_advanced!(
    "Three channel 16-bit unsigned box filter with border control.",
    filter_box_border_advanced_u16_c3_buffer_size,
    filter_box_border_advanced_u16_c3,
    u16,
    C3,
    nppiFilterBoxBorderAdvanced_16u_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);

impl_filter_box_border_advanced!(
    "Four channel 16-bit unsigned box filter with border control.",
    filter_box_border_advanced_u16_c4_buffer_size,
    filter_box_border_advanced_u16_c4,
    u16,
    C4,
    nppiFilterBoxBorderAdvanced_16u_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);

impl_filter_box_border_advanced!(
    "Single channel 16-bit box filter with border control.",
    filter_box_border_advanced_i16_c1_buffer_size,
    filter_box_border_advanced_i16_c1,
    i16,
    C1,
    nppiFilterBoxBorderAdvanced_16s_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);

impl_filter_box_border_advanced!(
    "Three channel 16-bit box filter with border control.",
    filter_box_border_advanced_i16_c3_buffer_size,
    filter_box_border_advanced_i16_c3,
    i16,
    C3,
    nppiFilterBoxBorderAdvanced_16s_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);

impl_filter_box_border_advanced!(
    "Four channel 16-bit box filter with border control.",
    filter_box_border_advanced_i16_c4_buffer_size,
    filter_box_border_advanced_i16_c4,
    i16,
    C4,
    nppiFilterBoxBorderAdvanced_16s_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);

impl_filter_box_border_advanced!(
    "Single channel 32-bit floating-point box filter with border control.",
    filter_box_border_advanced_f32_c1_buffer_size,
    filter_box_border_advanced_f32_c1,
    f32,
    C1,
    nppiFilterBoxBorderAdvanced_32f_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);

impl_filter_box_border_advanced!(
    "Three channel 32-bit floating-point box filter with border control.",
    filter_box_border_advanced_f32_c3_buffer_size,
    filter_box_border_advanced_f32_c3,
    f32,
    C3,
    nppiFilterBoxBorderAdvanced_32f_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);

impl_filter_box_border_advanced!(
    "Four channel 32-bit floating-point box filter with border control.",
    filter_box_border_advanced_f32_c4_buffer_size,
    filter_box_border_advanced_f32_c4,
    f32,
    C4,
    nppiFilterBoxBorderAdvanced_32f_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);

impl_filter_box_border_advanced!(
    "Single channel 64-bit floating-point box filter with border control.",
    filter_box_border_advanced_f64_c1_buffer_size,
    filter_box_border_advanced_f64_c1,
    f64,
    C1,
    nppiFilterBoxBorderAdvanced_64f_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64,
    1
);

impl_filter_neighborhood!(
    "Single channel 8-bit unsigned maximum filter.",
    filter_max_u8_c1,
    u8,
    C1,
    nppiFilterMax_8u_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 8-bit unsigned maximum filter.",
    filter_max_u8_c3,
    u8,
    C3,
    nppiFilterMax_8u_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 8-bit unsigned maximum filter.",
    filter_max_u8_c4,
    u8,
    C4,
    nppiFilterMax_8u_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 8-bit unsigned maximum filter, ignoring alpha channel.",
    filter_max_u8_ac4,
    u8,
    AC4,
    nppiFilterMax_8u_AC4R_Ctx
);

impl_filter_neighborhood!(
    "Single channel 16-bit unsigned maximum filter.",
    filter_max_u16_c1,
    u16,
    C1,
    nppiFilterMax_16u_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 16-bit unsigned maximum filter.",
    filter_max_u16_c3,
    u16,
    C3,
    nppiFilterMax_16u_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit unsigned maximum filter.",
    filter_max_u16_c4,
    u16,
    C4,
    nppiFilterMax_16u_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit unsigned maximum filter, ignoring alpha channel.",
    filter_max_u16_ac4,
    u16,
    AC4,
    nppiFilterMax_16u_AC4R_Ctx
);

impl_filter_neighborhood!(
    "Single channel 16-bit signed maximum filter.",
    filter_max_i16_c1,
    i16,
    C1,
    nppiFilterMax_16s_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 16-bit signed maximum filter.",
    filter_max_i16_c3,
    i16,
    C3,
    nppiFilterMax_16s_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit signed maximum filter.",
    filter_max_i16_c4,
    i16,
    C4,
    nppiFilterMax_16s_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit signed maximum filter, ignoring alpha channel.",
    filter_max_i16_ac4,
    i16,
    AC4,
    nppiFilterMax_16s_AC4R_Ctx
);

impl_filter_neighborhood!(
    "Single channel 32-bit floating-point maximum filter.",
    filter_max_f32_c1,
    f32,
    C1,
    nppiFilterMax_32f_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 32-bit floating-point maximum filter.",
    filter_max_f32_c3,
    f32,
    C3,
    nppiFilterMax_32f_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 32-bit floating-point maximum filter.",
    filter_max_f32_c4,
    f32,
    C4,
    nppiFilterMax_32f_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 32-bit floating-point maximum filter, ignoring alpha channel.",
    filter_max_f32_ac4,
    f32,
    AC4,
    nppiFilterMax_32f_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 8-bit unsigned maximum filter with border control.",
    filter_max_border_u8_c1,
    u8,
    C1,
    nppiFilterMaxBorder_8u_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 8-bit unsigned maximum filter with border control.",
    filter_max_border_u8_c3,
    u8,
    C3,
    nppiFilterMaxBorder_8u_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 8-bit unsigned maximum filter with border control.",
    filter_max_border_u8_c4,
    u8,
    C4,
    nppiFilterMaxBorder_8u_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 8-bit unsigned maximum filter with border control, ignoring alpha channel.",
    filter_max_border_u8_ac4,
    u8,
    AC4,
    nppiFilterMaxBorder_8u_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 16-bit unsigned maximum filter with border control.",
    filter_max_border_u16_c1,
    u16,
    C1,
    nppiFilterMaxBorder_16u_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 16-bit unsigned maximum filter with border control.",
    filter_max_border_u16_c3,
    u16,
    C3,
    nppiFilterMaxBorder_16u_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit unsigned maximum filter with border control.",
    filter_max_border_u16_c4,
    u16,
    C4,
    nppiFilterMaxBorder_16u_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit unsigned maximum filter with border control, ignoring alpha channel.",
    filter_max_border_u16_ac4,
    u16,
    AC4,
    nppiFilterMaxBorder_16u_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 16-bit signed maximum filter with border control.",
    filter_max_border_i16_c1,
    i16,
    C1,
    nppiFilterMaxBorder_16s_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 16-bit signed maximum filter with border control.",
    filter_max_border_i16_c3,
    i16,
    C3,
    nppiFilterMaxBorder_16s_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit signed maximum filter with border control.",
    filter_max_border_i16_c4,
    i16,
    C4,
    nppiFilterMaxBorder_16s_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit signed maximum filter with border control, ignoring alpha channel.",
    filter_max_border_i16_ac4,
    i16,
    AC4,
    nppiFilterMaxBorder_16s_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 32-bit floating-point maximum filter with border control.",
    filter_max_border_f32_c1,
    f32,
    C1,
    nppiFilterMaxBorder_32f_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 32-bit floating-point maximum filter with border control.",
    filter_max_border_f32_c3,
    f32,
    C3,
    nppiFilterMaxBorder_32f_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 32-bit floating-point maximum filter with border control.",
    filter_max_border_f32_c4,
    f32,
    C4,
    nppiFilterMaxBorder_32f_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 32-bit floating-point maximum filter with border control, ignoring alpha channel.",
    filter_max_border_f32_ac4,
    f32,
    AC4,
    nppiFilterMaxBorder_32f_AC4R_Ctx
);

impl_filter_neighborhood!(
    "Single channel 8-bit unsigned minimum filter.",
    filter_min_u8_c1,
    u8,
    C1,
    nppiFilterMin_8u_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 8-bit unsigned minimum filter.",
    filter_min_u8_c3,
    u8,
    C3,
    nppiFilterMin_8u_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 8-bit unsigned minimum filter.",
    filter_min_u8_c4,
    u8,
    C4,
    nppiFilterMin_8u_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 8-bit unsigned minimum filter, ignoring alpha channel.",
    filter_min_u8_ac4,
    u8,
    AC4,
    nppiFilterMin_8u_AC4R_Ctx
);

impl_filter_neighborhood!(
    "Single channel 16-bit unsigned minimum filter.",
    filter_min_u16_c1,
    u16,
    C1,
    nppiFilterMin_16u_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 16-bit unsigned minimum filter.",
    filter_min_u16_c3,
    u16,
    C3,
    nppiFilterMin_16u_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit unsigned minimum filter.",
    filter_min_u16_c4,
    u16,
    C4,
    nppiFilterMin_16u_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit unsigned minimum filter, ignoring alpha channel.",
    filter_min_u16_ac4,
    u16,
    AC4,
    nppiFilterMin_16u_AC4R_Ctx
);

impl_filter_neighborhood!(
    "Single channel 16-bit signed minimum filter.",
    filter_min_i16_c1,
    i16,
    C1,
    nppiFilterMin_16s_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 16-bit signed minimum filter.",
    filter_min_i16_c3,
    i16,
    C3,
    nppiFilterMin_16s_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit signed minimum filter.",
    filter_min_i16_c4,
    i16,
    C4,
    nppiFilterMin_16s_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 16-bit signed minimum filter, ignoring alpha channel.",
    filter_min_i16_ac4,
    i16,
    AC4,
    nppiFilterMin_16s_AC4R_Ctx
);

impl_filter_neighborhood!(
    "Single channel 32-bit floating-point minimum filter.",
    filter_min_f32_c1,
    f32,
    C1,
    nppiFilterMin_32f_C1R_Ctx
);

impl_filter_neighborhood!(
    "Three channel 32-bit floating-point minimum filter.",
    filter_min_f32_c3,
    f32,
    C3,
    nppiFilterMin_32f_C3R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 32-bit floating-point minimum filter.",
    filter_min_f32_c4,
    f32,
    C4,
    nppiFilterMin_32f_C4R_Ctx
);

impl_filter_neighborhood!(
    "Four channel 32-bit floating-point minimum filter, ignoring alpha channel.",
    filter_min_f32_ac4,
    f32,
    AC4,
    nppiFilterMin_32f_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 8-bit unsigned minimum filter with border control.",
    filter_min_border_u8_c1,
    u8,
    C1,
    nppiFilterMinBorder_8u_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 8-bit unsigned minimum filter with border control.",
    filter_min_border_u8_c3,
    u8,
    C3,
    nppiFilterMinBorder_8u_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 8-bit unsigned minimum filter with border control.",
    filter_min_border_u8_c4,
    u8,
    C4,
    nppiFilterMinBorder_8u_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 8-bit unsigned minimum filter with border control, ignoring alpha channel.",
    filter_min_border_u8_ac4,
    u8,
    AC4,
    nppiFilterMinBorder_8u_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 16-bit unsigned minimum filter with border control.",
    filter_min_border_u16_c1,
    u16,
    C1,
    nppiFilterMinBorder_16u_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 16-bit unsigned minimum filter with border control.",
    filter_min_border_u16_c3,
    u16,
    C3,
    nppiFilterMinBorder_16u_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit unsigned minimum filter with border control.",
    filter_min_border_u16_c4,
    u16,
    C4,
    nppiFilterMinBorder_16u_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit unsigned minimum filter with border control, ignoring alpha channel.",
    filter_min_border_u16_ac4,
    u16,
    AC4,
    nppiFilterMinBorder_16u_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 16-bit signed minimum filter with border control.",
    filter_min_border_i16_c1,
    i16,
    C1,
    nppiFilterMinBorder_16s_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 16-bit signed minimum filter with border control.",
    filter_min_border_i16_c3,
    i16,
    C3,
    nppiFilterMinBorder_16s_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit signed minimum filter with border control.",
    filter_min_border_i16_c4,
    i16,
    C4,
    nppiFilterMinBorder_16s_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 16-bit signed minimum filter with border control, ignoring alpha channel.",
    filter_min_border_i16_ac4,
    i16,
    AC4,
    nppiFilterMinBorder_16s_AC4R_Ctx
);

impl_filter_neighborhood_border!(
    "Single channel 32-bit floating-point minimum filter with border control.",
    filter_min_border_f32_c1,
    f32,
    C1,
    nppiFilterMinBorder_32f_C1R_Ctx
);

impl_filter_neighborhood_border!(
    "Three channel 32-bit floating-point minimum filter with border control.",
    filter_min_border_f32_c3,
    f32,
    C3,
    nppiFilterMinBorder_32f_C3R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 32-bit floating-point minimum filter with border control.",
    filter_min_border_f32_c4,
    f32,
    C4,
    nppiFilterMinBorder_32f_C4R_Ctx
);

impl_filter_neighborhood_border!(
    "Four channel 32-bit floating-point minimum filter with border control, ignoring alpha channel.",
    filter_min_border_f32_ac4,
    f32,
    AC4,
    nppiFilterMinBorder_32f_AC4R_Ctx
);

impl_generic_filter_mask!(
    "Applies a box filter using `mask_size` and `anchor`.",
FilterBoxC1, filter_box, filter_box_c1, C1, [
    u8 => filter_box_u8_c1,
    u16 => filter_box_u16_c1,
    i16 => filter_box_i16_c1,
    f32 => filter_box_f32_c1,
    f64 => filter_box_f64_c1,
]);

impl_generic_filter_mask!(
    "Applies a box filter using `mask_size` and `anchor`.",
FilterBoxC3, filter_box, filter_box_c3, C3, [
    u8 => filter_box_u8_c3,
    u16 => filter_box_u16_c3,
    i16 => filter_box_i16_c3,
    f32 => filter_box_f32_c3,
]);

impl_generic_filter_mask!(
    "Applies a box filter using `mask_size` and `anchor`.",
FilterBoxC4, filter_box, filter_box_c4, C4, [
    u8 => filter_box_u8_c4,
    u16 => filter_box_u16_c4,
    i16 => filter_box_i16_c4,
    f32 => filter_box_f32_c4,
]);

impl_generic_filter_mask!(
    "Applies a box filter using `mask_size` and `anchor`. The destination alpha channel is not processed.",
FilterBoxAc4, filter_box, filter_box_ac4, AC4, [
    u8 => filter_box_u8_ac4,
    u16 => filter_box_u16_ac4,
    i16 => filter_box_i16_ac4,
    f32 => filter_box_f32_ac4,
]);

impl_generic_filter_mask_border!(
    "Applies a box filter using `mask_size` and `anchor`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",

    FilterBoxBorderC1,
    filter_box_border,
    filter_box_border_c1,
    C1,
    [
        u8 => filter_box_border_u8_c1,
        u16 => filter_box_border_u16_c1,
        i16 => filter_box_border_i16_c1,
        f32 => filter_box_border_f32_c1,
    ]
);

impl_generic_filter_mask_border!(
    "Applies a box filter using `mask_size` and `anchor`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",

    FilterBoxBorderC3,
    filter_box_border,
    filter_box_border_c3,
    C3,
    [
        u8 => filter_box_border_u8_c3,
        u16 => filter_box_border_u16_c3,
        i16 => filter_box_border_i16_c3,
        f32 => filter_box_border_f32_c3,
    ]
);

impl_generic_filter_mask_border!(
    "Applies a box filter using `mask_size` and `anchor`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`.",

    FilterBoxBorderC4,
    filter_box_border,
    filter_box_border_c4,
    C4,
    [
        u8 => filter_box_border_u8_c4,
        u16 => filter_box_border_u16_c4,
        i16 => filter_box_border_i16_c4,
        f32 => filter_box_border_f32_c4,
    ]
);

impl_generic_filter_mask_border!(
    "Applies a box filter using `mask_size` and `anchor`. Uses `border_type` at source image boundaries and begins the ROI at `source_offset`. The destination alpha channel is not processed.",

    FilterBoxBorderAc4,
    filter_box_border,
    filter_box_border_ac4,
    AC4,
    [
        u8 => filter_box_border_u8_ac4,
        u16 => filter_box_border_u16_ac4,
        i16 => filter_box_border_i16_ac4,
        f32 => filter_box_border_f32_ac4,
    ]
);

impl_generic_filter_mask!(
    "Applies a maximum filter using `mask_size` and `anchor`.",
FilterMaxC1, filter_max, filter_max_c1, C1, [
    u8 => filter_max_u8_c1,
    u16 => filter_max_u16_c1,
    i16 => filter_max_i16_c1,
    f32 => filter_max_f32_c1,
]);

impl_generic_filter_mask!(
    "Applies a maximum filter using `mask_size` and `anchor`.",
FilterMaxC3, filter_max, filter_max_c3, C3, [
    u8 => filter_max_u8_c3,
    u16 => filter_max_u16_c3,
    i16 => filter_max_i16_c3,
    f32 => filter_max_f32_c3,
]);

impl_generic_filter_mask!(
    "Applies a maximum filter using `mask_size` and `anchor`.",
FilterMaxC4, filter_max, filter_max_c4, C4, [
    u8 => filter_max_u8_c4,
    u16 => filter_max_u16_c4,
    i16 => filter_max_i16_c4,
    f32 => filter_max_f32_c4,
]);

impl_generic_filter_mask!(
    "Applies a maximum filter using `mask_size` and `anchor`. The destination alpha channel is not processed.",
FilterMaxAc4, filter_max, filter_max_ac4, AC4, [
    u8 => filter_max_u8_ac4,
    u16 => filter_max_u16_ac4,
    i16 => filter_max_i16_ac4,
    f32 => filter_max_f32_ac4,
]);

impl_generic_filter_mask!(
    "Applies a minimum filter using `mask_size` and `anchor`.",
FilterMinC1, filter_min, filter_min_c1, C1, [
    u8 => filter_min_u8_c1,
    u16 => filter_min_u16_c1,
    i16 => filter_min_i16_c1,
    f32 => filter_min_f32_c1,
]);

impl_generic_filter_mask!(
    "Applies a minimum filter using `mask_size` and `anchor`.",
FilterMinC3, filter_min, filter_min_c3, C3, [
    u8 => filter_min_u8_c3,
    u16 => filter_min_u16_c3,
    i16 => filter_min_i16_c3,
    f32 => filter_min_f32_c3,
]);

impl_generic_filter_mask!(
    "Applies a minimum filter using `mask_size` and `anchor`.",
FilterMinC4, filter_min, filter_min_c4, C4, [
    u8 => filter_min_u8_c4,
    u16 => filter_min_u16_c4,
    i16 => filter_min_i16_c4,
    f32 => filter_min_f32_c4,
]);

impl_generic_filter_mask!(
    "Applies a minimum filter using `mask_size` and `anchor`. The destination alpha channel is not processed.",
FilterMinAc4, filter_min, filter_min_ac4, AC4, [
    u8 => filter_min_u8_ac4,
    u16 => filter_min_u16_ac4,
    i16 => filter_min_i16_ac4,
    f32 => filter_min_f32_ac4,
]);
