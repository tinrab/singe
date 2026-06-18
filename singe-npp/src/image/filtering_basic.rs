use super::*;

impl_filter_box!(filter_box_u8_c1, u8, C1, nppiFilterBox_8u_C1R_Ctx);
impl_filter_box!(filter_box_u8_c3, u8, C3, nppiFilterBox_8u_C3R_Ctx);
impl_filter_box!(filter_box_u8_c4, u8, C4, nppiFilterBox_8u_C4R_Ctx);
impl_filter_box!(filter_box_u8_ac4, u8, AC4, nppiFilterBox_8u_AC4R_Ctx);
impl_filter_box!(filter_box_u16_c1, u16, C1, nppiFilterBox_16u_C1R_Ctx);
impl_filter_box!(filter_box_u16_c3, u16, C3, nppiFilterBox_16u_C3R_Ctx);
impl_filter_box!(filter_box_u16_c4, u16, C4, nppiFilterBox_16u_C4R_Ctx);
impl_filter_box!(filter_box_u16_ac4, u16, AC4, nppiFilterBox_16u_AC4R_Ctx);
impl_filter_box!(filter_box_i16_c1, i16, C1, nppiFilterBox_16s_C1R_Ctx);
impl_filter_box!(filter_box_i16_c3, i16, C3, nppiFilterBox_16s_C3R_Ctx);
impl_filter_box!(filter_box_i16_c4, i16, C4, nppiFilterBox_16s_C4R_Ctx);
impl_filter_box!(filter_box_i16_ac4, i16, AC4, nppiFilterBox_16s_AC4R_Ctx);
impl_filter_box!(filter_box_f32_c1, f32, C1, nppiFilterBox_32f_C1R_Ctx);
impl_filter_box!(filter_box_f32_c3, f32, C3, nppiFilterBox_32f_C3R_Ctx);
impl_filter_box!(filter_box_f32_c4, f32, C4, nppiFilterBox_32f_C4R_Ctx);
impl_filter_box!(filter_box_f32_ac4, f32, AC4, nppiFilterBox_32f_AC4R_Ctx);
impl_filter_box!(filter_box_f64_c1, f64, C1, nppiFilterBox_64f_C1R_Ctx);

impl_filter_box_border!(
    filter_box_border_u8_c1,
    u8,
    C1,
    nppiFilterBoxBorder_8u_C1R_Ctx
);
impl_filter_box_border!(
    filter_box_border_u8_c3,
    u8,
    C3,
    nppiFilterBoxBorder_8u_C3R_Ctx
);
impl_filter_box_border!(
    filter_box_border_u8_c4,
    u8,
    C4,
    nppiFilterBoxBorder_8u_C4R_Ctx
);
impl_filter_box_border!(
    filter_box_border_u8_ac4,
    u8,
    AC4,
    nppiFilterBoxBorder_8u_AC4R_Ctx
);
impl_filter_box_border!(
    filter_box_border_u16_c1,
    u16,
    C1,
    nppiFilterBoxBorder_16u_C1R_Ctx
);
impl_filter_box_border!(
    filter_box_border_u16_c3,
    u16,
    C3,
    nppiFilterBoxBorder_16u_C3R_Ctx
);
impl_filter_box_border!(
    filter_box_border_u16_c4,
    u16,
    C4,
    nppiFilterBoxBorder_16u_C4R_Ctx
);
impl_filter_box_border!(
    filter_box_border_u16_ac4,
    u16,
    AC4,
    nppiFilterBoxBorder_16u_AC4R_Ctx
);
impl_filter_box_border!(
    filter_box_border_i16_c1,
    i16,
    C1,
    nppiFilterBoxBorder_16s_C1R_Ctx
);
impl_filter_box_border!(
    filter_box_border_i16_c3,
    i16,
    C3,
    nppiFilterBoxBorder_16s_C3R_Ctx
);
impl_filter_box_border!(
    filter_box_border_i16_c4,
    i16,
    C4,
    nppiFilterBoxBorder_16s_C4R_Ctx
);
impl_filter_box_border!(
    filter_box_border_i16_ac4,
    i16,
    AC4,
    nppiFilterBoxBorder_16s_AC4R_Ctx
);
impl_filter_box_border!(
    filter_box_border_f32_c1,
    f32,
    C1,
    nppiFilterBoxBorder_32f_C1R_Ctx
);
impl_filter_box_border!(
    filter_box_border_f32_c3,
    f32,
    C3,
    nppiFilterBoxBorder_32f_C3R_Ctx
);
impl_filter_box_border!(
    filter_box_border_f32_c4,
    f32,
    C4,
    nppiFilterBoxBorder_32f_C4R_Ctx
);
impl_filter_box_border!(
    filter_box_border_f32_ac4,
    f32,
    AC4,
    nppiFilterBoxBorder_32f_AC4R_Ctx
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_u8_c1_buffer_size,
    filter_box_border_advanced_u8_c1,
    u8,
    C1,
    nppiFilterBoxBorderAdvanced_8u_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_u8_c3_buffer_size,
    filter_box_border_advanced_u8_c3,
    u8,
    C3,
    nppiFilterBoxBorderAdvanced_8u_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_u8_c4_buffer_size,
    filter_box_border_advanced_u8_c4,
    u8,
    C4,
    nppiFilterBoxBorderAdvanced_8u_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_u16_c1_buffer_size,
    filter_box_border_advanced_u16_c1,
    u16,
    C1,
    nppiFilterBoxBorderAdvanced_16u_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_u16_c3_buffer_size,
    filter_box_border_advanced_u16_c3,
    u16,
    C3,
    nppiFilterBoxBorderAdvanced_16u_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_u16_c4_buffer_size,
    filter_box_border_advanced_u16_c4,
    u16,
    C4,
    nppiFilterBoxBorderAdvanced_16u_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_i16_c1_buffer_size,
    filter_box_border_advanced_i16_c1,
    i16,
    C1,
    nppiFilterBoxBorderAdvanced_16s_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_i16_c3_buffer_size,
    filter_box_border_advanced_i16_c3,
    i16,
    C3,
    nppiFilterBoxBorderAdvanced_16s_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_i16_c4_buffer_size,
    filter_box_border_advanced_i16_c4,
    i16,
    C4,
    nppiFilterBoxBorderAdvanced_16s_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_f32_c1_buffer_size,
    filter_box_border_advanced_f32_c1,
    f32,
    C1,
    nppiFilterBoxBorderAdvanced_32f_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    1
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_f32_c3_buffer_size,
    filter_box_border_advanced_f32_c3,
    f32,
    C3,
    nppiFilterBoxBorderAdvanced_32f_C3R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    3
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_f32_c4_buffer_size,
    filter_box_border_advanced_f32_c4,
    f32,
    C4,
    nppiFilterBoxBorderAdvanced_32f_C4R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize,
    4
);
impl_filter_box_border_advanced!(
    filter_box_border_advanced_f64_c1_buffer_size,
    filter_box_border_advanced_f64_c1,
    f64,
    C1,
    nppiFilterBoxBorderAdvanced_64f_C1R_Ctx,
    nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64,
    1
);

impl_filter_neighborhood!(filter_max_u8_c1, u8, C1, nppiFilterMax_8u_C1R_Ctx);
impl_filter_neighborhood!(filter_max_u8_c3, u8, C3, nppiFilterMax_8u_C3R_Ctx);
impl_filter_neighborhood!(filter_max_u8_c4, u8, C4, nppiFilterMax_8u_C4R_Ctx);
impl_filter_neighborhood!(filter_max_u8_ac4, u8, AC4, nppiFilterMax_8u_AC4R_Ctx);
impl_filter_neighborhood!(filter_max_u16_c1, u16, C1, nppiFilterMax_16u_C1R_Ctx);
impl_filter_neighborhood!(filter_max_u16_c3, u16, C3, nppiFilterMax_16u_C3R_Ctx);
impl_filter_neighborhood!(filter_max_u16_c4, u16, C4, nppiFilterMax_16u_C4R_Ctx);
impl_filter_neighborhood!(filter_max_u16_ac4, u16, AC4, nppiFilterMax_16u_AC4R_Ctx);
impl_filter_neighborhood!(filter_max_i16_c1, i16, C1, nppiFilterMax_16s_C1R_Ctx);
impl_filter_neighborhood!(filter_max_i16_c3, i16, C3, nppiFilterMax_16s_C3R_Ctx);
impl_filter_neighborhood!(filter_max_i16_c4, i16, C4, nppiFilterMax_16s_C4R_Ctx);
impl_filter_neighborhood!(filter_max_i16_ac4, i16, AC4, nppiFilterMax_16s_AC4R_Ctx);
impl_filter_neighborhood!(filter_max_f32_c1, f32, C1, nppiFilterMax_32f_C1R_Ctx);
impl_filter_neighborhood!(filter_max_f32_c3, f32, C3, nppiFilterMax_32f_C3R_Ctx);
impl_filter_neighborhood!(filter_max_f32_c4, f32, C4, nppiFilterMax_32f_C4R_Ctx);
impl_filter_neighborhood!(filter_max_f32_ac4, f32, AC4, nppiFilterMax_32f_AC4R_Ctx);

impl_filter_neighborhood_border!(
    filter_max_border_u8_c1,
    u8,
    C1,
    nppiFilterMaxBorder_8u_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_u8_c3,
    u8,
    C3,
    nppiFilterMaxBorder_8u_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_u8_c4,
    u8,
    C4,
    nppiFilterMaxBorder_8u_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_u8_ac4,
    u8,
    AC4,
    nppiFilterMaxBorder_8u_AC4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_u16_c1,
    u16,
    C1,
    nppiFilterMaxBorder_16u_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_u16_c3,
    u16,
    C3,
    nppiFilterMaxBorder_16u_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_u16_c4,
    u16,
    C4,
    nppiFilterMaxBorder_16u_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_u16_ac4,
    u16,
    AC4,
    nppiFilterMaxBorder_16u_AC4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_i16_c1,
    i16,
    C1,
    nppiFilterMaxBorder_16s_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_i16_c3,
    i16,
    C3,
    nppiFilterMaxBorder_16s_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_i16_c4,
    i16,
    C4,
    nppiFilterMaxBorder_16s_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_i16_ac4,
    i16,
    AC4,
    nppiFilterMaxBorder_16s_AC4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_f32_c1,
    f32,
    C1,
    nppiFilterMaxBorder_32f_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_f32_c3,
    f32,
    C3,
    nppiFilterMaxBorder_32f_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_f32_c4,
    f32,
    C4,
    nppiFilterMaxBorder_32f_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_max_border_f32_ac4,
    f32,
    AC4,
    nppiFilterMaxBorder_32f_AC4R_Ctx
);

impl_filter_neighborhood!(filter_min_u8_c1, u8, C1, nppiFilterMin_8u_C1R_Ctx);
impl_filter_neighborhood!(filter_min_u8_c3, u8, C3, nppiFilterMin_8u_C3R_Ctx);
impl_filter_neighborhood!(filter_min_u8_c4, u8, C4, nppiFilterMin_8u_C4R_Ctx);
impl_filter_neighborhood!(filter_min_u8_ac4, u8, AC4, nppiFilterMin_8u_AC4R_Ctx);
impl_filter_neighborhood!(filter_min_u16_c1, u16, C1, nppiFilterMin_16u_C1R_Ctx);
impl_filter_neighborhood!(filter_min_u16_c3, u16, C3, nppiFilterMin_16u_C3R_Ctx);
impl_filter_neighborhood!(filter_min_u16_c4, u16, C4, nppiFilterMin_16u_C4R_Ctx);
impl_filter_neighborhood!(filter_min_u16_ac4, u16, AC4, nppiFilterMin_16u_AC4R_Ctx);
impl_filter_neighborhood!(filter_min_i16_c1, i16, C1, nppiFilterMin_16s_C1R_Ctx);
impl_filter_neighborhood!(filter_min_i16_c3, i16, C3, nppiFilterMin_16s_C3R_Ctx);
impl_filter_neighborhood!(filter_min_i16_c4, i16, C4, nppiFilterMin_16s_C4R_Ctx);
impl_filter_neighborhood!(filter_min_i16_ac4, i16, AC4, nppiFilterMin_16s_AC4R_Ctx);
impl_filter_neighborhood!(filter_min_f32_c1, f32, C1, nppiFilterMin_32f_C1R_Ctx);
impl_filter_neighborhood!(filter_min_f32_c3, f32, C3, nppiFilterMin_32f_C3R_Ctx);
impl_filter_neighborhood!(filter_min_f32_c4, f32, C4, nppiFilterMin_32f_C4R_Ctx);
impl_filter_neighborhood!(filter_min_f32_ac4, f32, AC4, nppiFilterMin_32f_AC4R_Ctx);

impl_filter_neighborhood_border!(
    filter_min_border_u8_c1,
    u8,
    C1,
    nppiFilterMinBorder_8u_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_u8_c3,
    u8,
    C3,
    nppiFilterMinBorder_8u_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_u8_c4,
    u8,
    C4,
    nppiFilterMinBorder_8u_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_u8_ac4,
    u8,
    AC4,
    nppiFilterMinBorder_8u_AC4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_u16_c1,
    u16,
    C1,
    nppiFilterMinBorder_16u_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_u16_c3,
    u16,
    C3,
    nppiFilterMinBorder_16u_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_u16_c4,
    u16,
    C4,
    nppiFilterMinBorder_16u_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_u16_ac4,
    u16,
    AC4,
    nppiFilterMinBorder_16u_AC4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_i16_c1,
    i16,
    C1,
    nppiFilterMinBorder_16s_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_i16_c3,
    i16,
    C3,
    nppiFilterMinBorder_16s_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_i16_c4,
    i16,
    C4,
    nppiFilterMinBorder_16s_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_i16_ac4,
    i16,
    AC4,
    nppiFilterMinBorder_16s_AC4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_f32_c1,
    f32,
    C1,
    nppiFilterMinBorder_32f_C1R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_f32_c3,
    f32,
    C3,
    nppiFilterMinBorder_32f_C3R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_f32_c4,
    f32,
    C4,
    nppiFilterMinBorder_32f_C4R_Ctx
);
impl_filter_neighborhood_border!(
    filter_min_border_f32_ac4,
    f32,
    AC4,
    nppiFilterMinBorder_32f_AC4R_Ctx
);

impl_generic_filter_mask!(FilterBoxC1, filter_box, filter_box_c1, C1, [
    u8 => filter_box_u8_c1,
    u16 => filter_box_u16_c1,
    i16 => filter_box_i16_c1,
    f32 => filter_box_f32_c1,
    f64 => filter_box_f64_c1,
]);
impl_generic_filter_mask!(FilterBoxC3, filter_box, filter_box_c3, C3, [
    u8 => filter_box_u8_c3,
    u16 => filter_box_u16_c3,
    i16 => filter_box_i16_c3,
    f32 => filter_box_f32_c3,
]);
impl_generic_filter_mask!(FilterBoxC4, filter_box, filter_box_c4, C4, [
    u8 => filter_box_u8_c4,
    u16 => filter_box_u16_c4,
    i16 => filter_box_i16_c4,
    f32 => filter_box_f32_c4,
]);
impl_generic_filter_mask!(FilterBoxAc4, filter_box, filter_box_ac4, AC4, [
    u8 => filter_box_u8_ac4,
    u16 => filter_box_u16_ac4,
    i16 => filter_box_i16_ac4,
    f32 => filter_box_f32_ac4,
]);
impl_generic_filter_mask_border!(
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
impl_generic_filter_mask!(FilterMaxC1, filter_max, filter_max_c1, C1, [
    u8 => filter_max_u8_c1,
    u16 => filter_max_u16_c1,
    i16 => filter_max_i16_c1,
    f32 => filter_max_f32_c1,
]);
impl_generic_filter_mask!(FilterMaxC3, filter_max, filter_max_c3, C3, [
    u8 => filter_max_u8_c3,
    u16 => filter_max_u16_c3,
    i16 => filter_max_i16_c3,
    f32 => filter_max_f32_c3,
]);
impl_generic_filter_mask!(FilterMaxC4, filter_max, filter_max_c4, C4, [
    u8 => filter_max_u8_c4,
    u16 => filter_max_u16_c4,
    i16 => filter_max_i16_c4,
    f32 => filter_max_f32_c4,
]);
impl_generic_filter_mask!(FilterMaxAc4, filter_max, filter_max_ac4, AC4, [
    u8 => filter_max_u8_ac4,
    u16 => filter_max_u16_ac4,
    i16 => filter_max_i16_ac4,
    f32 => filter_max_f32_ac4,
]);
impl_generic_filter_mask!(FilterMinC1, filter_min, filter_min_c1, C1, [
    u8 => filter_min_u8_c1,
    u16 => filter_min_u16_c1,
    i16 => filter_min_i16_c1,
    f32 => filter_min_f32_c1,
]);
impl_generic_filter_mask!(FilterMinC3, filter_min, filter_min_c3, C3, [
    u8 => filter_min_u8_c3,
    u16 => filter_min_u16_c3,
    i16 => filter_min_i16_c3,
    f32 => filter_min_f32_c3,
]);
impl_generic_filter_mask!(FilterMinC4, filter_min, filter_min_c4, C4, [
    u8 => filter_min_u8_c4,
    u16 => filter_min_u16_c4,
    i16 => filter_min_i16_c4,
    f32 => filter_min_f32_c4,
]);
impl_generic_filter_mask!(FilterMinAc4, filter_min, filter_min_ac4, AC4, [
    u8 => filter_min_u8_ac4,
    u16 => filter_min_u16_ac4,
    i16 => filter_min_i16_ac4,
    f32 => filter_min_f32_ac4,
]);
