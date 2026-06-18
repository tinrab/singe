use super::*;

impl_generic_statistic!(
    MeanC1,
    mean_c1,
    mean_c1_buffer_size,
    C1,
    [
        (u8, mean_u8_c1, mean_u8_c1_buffer_size),
        (u16, mean_u16_c1, mean_u16_c1_buffer_size),
        (i16, mean_i16_c1, mean_i16_c1_buffer_size),
        (f32, mean_f32_c1, mean_f32_c1_buffer_size),
    ]
);
impl_generic_statistic!(
    MeanC3,
    mean_c3,
    mean_c3_buffer_size,
    C3,
    [
        (u8, mean_u8_c3, mean_u8_c3_buffer_size),
        (u16, mean_u16_c3, mean_u16_c3_buffer_size),
        (i16, mean_i16_c3, mean_i16_c3_buffer_size),
        (f32, mean_f32_c3, mean_f32_c3_buffer_size),
    ]
);
impl_generic_statistic!(
    MeanC4,
    mean_c4,
    mean_c4_buffer_size,
    C4,
    [
        (u8, mean_u8_c4, mean_u8_c4_buffer_size),
        (u16, mean_u16_c4, mean_u16_c4_buffer_size),
        (i16, mean_i16_c4, mean_i16_c4_buffer_size),
        (f32, mean_f32_c4, mean_f32_c4_buffer_size),
    ]
);
impl_generic_statistic!(
    MeanAC4,
    mean_ac4,
    mean_ac4_buffer_size,
    AC4,
    [
        (u8, mean_u8_ac4, mean_u8_ac4_buffer_size),
        (u16, mean_u16_ac4, mean_u16_ac4_buffer_size),
        (i16, mean_i16_ac4, mean_i16_ac4_buffer_size),
        (f32, mean_f32_ac4, mean_f32_ac4_buffer_size),
    ]
);

impl_generic_statistic!(
    SumC1,
    sum_c1,
    sum_c1_buffer_size,
    C1,
    [
        (u8, sum_u8_c1, sum_u8_c1_buffer_size),
        (u16, sum_u16_c1, sum_u16_c1_buffer_size),
        (i16, sum_i16_c1, sum_i16_c1_buffer_size),
        (f32, sum_f32_c1, sum_f32_c1_buffer_size),
    ]
);
impl_generic_statistic!(
    SumC3,
    sum_c3,
    sum_c3_buffer_size,
    C3,
    [
        (u8, sum_u8_c3, sum_u8_c3_buffer_size),
        (u16, sum_u16_c3, sum_u16_c3_buffer_size),
        (i16, sum_i16_c3, sum_i16_c3_buffer_size),
        (f32, sum_f32_c3, sum_f32_c3_buffer_size),
    ]
);
impl_generic_statistic!(
    SumC4,
    sum_c4,
    sum_c4_buffer_size,
    C4,
    [
        (u8, sum_u8_c4, sum_u8_c4_buffer_size),
        (u16, sum_u16_c4, sum_u16_c4_buffer_size),
        (i16, sum_i16_c4, sum_i16_c4_buffer_size),
        (f32, sum_f32_c4, sum_f32_c4_buffer_size),
    ]
);
impl_generic_statistic!(
    SumAC4,
    sum_ac4,
    sum_ac4_buffer_size,
    AC4,
    [
        (u8, sum_u8_ac4, sum_u8_ac4_buffer_size),
        (u16, sum_u16_ac4, sum_u16_ac4_buffer_size),
        (i16, sum_i16_ac4, sum_i16_ac4_buffer_size),
        (f32, sum_f32_ac4, sum_f32_ac4_buffer_size),
    ]
);

impl_generic_typed_statistic!(
    MinC1,
    min_c1,
    min_c1_buffer_size,
    C1,
    [
        (u8, u8, min_u8_c1, min_u8_c1_buffer_size),
        (u16, u16, min_u16_c1, min_u16_c1_buffer_size),
        (i16, i16, min_i16_c1, min_i16_c1_buffer_size),
        (f32, f32, min_f32_c1, min_f32_c1_buffer_size),
    ]
);
impl_generic_typed_statistic!(
    MinC3,
    min_c3,
    min_c3_buffer_size,
    C3,
    [
        (u8, u8, min_u8_c3, min_u8_c3_buffer_size),
        (u16, u16, min_u16_c3, min_u16_c3_buffer_size),
        (i16, i16, min_i16_c3, min_i16_c3_buffer_size),
        (f32, f32, min_f32_c3, min_f32_c3_buffer_size),
    ]
);
impl_generic_typed_statistic!(
    MinC4,
    min_c4,
    min_c4_buffer_size,
    C4,
    [
        (u8, u8, min_u8_c4, min_u8_c4_buffer_size),
        (u16, u16, min_u16_c4, min_u16_c4_buffer_size),
        (i16, i16, min_i16_c4, min_i16_c4_buffer_size),
        (f32, f32, min_f32_c4, min_f32_c4_buffer_size),
    ]
);
impl_generic_typed_statistic!(
    MinAC4,
    min_ac4,
    min_ac4_buffer_size,
    AC4,
    [
        (u8, u8, min_u8_ac4, min_u8_ac4_buffer_size),
        (u16, u16, min_u16_ac4, min_u16_ac4_buffer_size),
        (i16, i16, min_i16_ac4, min_i16_ac4_buffer_size),
        (f32, f32, min_f32_ac4, min_f32_ac4_buffer_size),
    ]
);

impl_generic_typed_statistic!(
    MaxC1,
    max_c1,
    max_c1_buffer_size,
    C1,
    [
        (u8, u8, max_u8_c1, max_u8_c1_buffer_size),
        (u16, u16, max_u16_c1, max_u16_c1_buffer_size),
        (i16, i16, max_i16_c1, max_i16_c1_buffer_size),
        (f32, f32, max_f32_c1, max_f32_c1_buffer_size),
    ]
);
impl_generic_typed_statistic!(
    MaxC3,
    max_c3,
    max_c3_buffer_size,
    C3,
    [
        (u8, u8, max_u8_c3, max_u8_c3_buffer_size),
        (u16, u16, max_u16_c3, max_u16_c3_buffer_size),
        (i16, i16, max_i16_c3, max_i16_c3_buffer_size),
        (f32, f32, max_f32_c3, max_f32_c3_buffer_size),
    ]
);
impl_generic_typed_statistic!(
    MaxC4,
    max_c4,
    max_c4_buffer_size,
    C4,
    [
        (u8, u8, max_u8_c4, max_u8_c4_buffer_size),
        (u16, u16, max_u16_c4, max_u16_c4_buffer_size),
        (i16, i16, max_i16_c4, max_i16_c4_buffer_size),
        (f32, f32, max_f32_c4, max_f32_c4_buffer_size),
    ]
);
impl_generic_typed_statistic!(
    MaxAC4,
    max_ac4,
    max_ac4_buffer_size,
    AC4,
    [
        (u8, u8, max_u8_ac4, max_u8_ac4_buffer_size),
        (u16, u16, max_u16_ac4, max_u16_ac4_buffer_size),
        (i16, i16, max_i16_ac4, max_i16_ac4_buffer_size),
        (f32, f32, max_f32_ac4, max_f32_ac4_buffer_size),
    ]
);

impl_generic_typed_pair_statistic!(
    MinMaxC1,
    min_max_c1,
    min_max_c1_buffer_size,
    C1,
    [
        (u8, u8, min_max_u8_c1, min_max_u8_c1_buffer_size),
        (u16, u16, min_max_u16_c1, min_max_u16_c1_buffer_size),
        (i16, i16, min_max_i16_c1, min_max_i16_c1_buffer_size),
        (f32, f32, min_max_f32_c1, min_max_f32_c1_buffer_size),
    ]
);
impl_generic_typed_pair_statistic!(
    MinMaxC3,
    min_max_c3,
    min_max_c3_buffer_size,
    C3,
    [
        (u8, u8, min_max_u8_c3, min_max_u8_c3_buffer_size),
        (u16, u16, min_max_u16_c3, min_max_u16_c3_buffer_size),
        (i16, i16, min_max_i16_c3, min_max_i16_c3_buffer_size),
        (f32, f32, min_max_f32_c3, min_max_f32_c3_buffer_size),
    ]
);
impl_generic_typed_pair_statistic!(
    MinMaxC4,
    min_max_c4,
    min_max_c4_buffer_size,
    C4,
    [
        (u8, u8, min_max_u8_c4, min_max_u8_c4_buffer_size),
        (u16, u16, min_max_u16_c4, min_max_u16_c4_buffer_size),
        (i16, i16, min_max_i16_c4, min_max_i16_c4_buffer_size),
        (f32, f32, min_max_f32_c4, min_max_f32_c4_buffer_size),
    ]
);
impl_generic_typed_pair_statistic!(
    MinMaxAC4,
    min_max_ac4,
    min_max_ac4_buffer_size,
    AC4,
    [
        (u8, u8, min_max_u8_ac4, min_max_u8_ac4_buffer_size),
        (u16, u16, min_max_u16_ac4, min_max_u16_ac4_buffer_size),
        (i16, i16, min_max_i16_ac4, min_max_i16_ac4_buffer_size),
        (f32, f32, min_max_f32_ac4, min_max_f32_ac4_buffer_size),
    ]
);
