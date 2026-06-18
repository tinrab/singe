use super::*;

impl_add_constant_scalar!(and_constant_u8_c1, u8, u8, C1, nppiAndC_8u_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    and_constant_u8_c1_in_place,
    u8,
    u8,
    C1,
    nppiAndC_8u_C1IR_Ctx
);
impl_add_constant_array!(and_constant_u8_c3, u8, u8, C3, 3, nppiAndC_8u_C3R_Ctx);
impl_add_constant_array_in_place!(
    and_constant_u8_c3_in_place,
    u8,
    u8,
    C3,
    3,
    nppiAndC_8u_C3IR_Ctx
);
impl_add_constant_array!(and_constant_u8_ac4, u8, u8, AC4, 3, nppiAndC_8u_AC4R_Ctx);
impl_add_constant_array_in_place!(
    and_constant_u8_ac4_in_place,
    u8,
    u8,
    AC4,
    3,
    nppiAndC_8u_AC4IR_Ctx
);
impl_add_constant_array!(and_constant_u8_c4, u8, u8, C4, 4, nppiAndC_8u_C4R_Ctx);
impl_add_constant_array_in_place!(
    and_constant_u8_c4_in_place,
    u8,
    u8,
    C4,
    4,
    nppiAndC_8u_C4IR_Ctx
);
impl_add_constant_scalar!(and_constant_u16_c1, u16, u16, C1, nppiAndC_16u_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    and_constant_u16_c1_in_place,
    u16,
    u16,
    C1,
    nppiAndC_16u_C1IR_Ctx
);
impl_add_constant_array!(and_constant_u16_c3, u16, u16, C3, 3, nppiAndC_16u_C3R_Ctx);
impl_add_constant_array_in_place!(
    and_constant_u16_c3_in_place,
    u16,
    u16,
    C3,
    3,
    nppiAndC_16u_C3IR_Ctx
);
impl_add_constant_array!(
    and_constant_u16_ac4,
    u16,
    u16,
    AC4,
    3,
    nppiAndC_16u_AC4R_Ctx
);
impl_add_constant_array_in_place!(
    and_constant_u16_ac4_in_place,
    u16,
    u16,
    AC4,
    3,
    nppiAndC_16u_AC4IR_Ctx
);
impl_add_constant_array!(and_constant_u16_c4, u16, u16, C4, 4, nppiAndC_16u_C4R_Ctx);
impl_add_constant_array_in_place!(
    and_constant_u16_c4_in_place,
    u16,
    u16,
    C4,
    4,
    nppiAndC_16u_C4IR_Ctx
);
impl_add_constant_scalar!(and_constant_i32_c1, i32, i32, C1, nppiAndC_32s_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    and_constant_i32_c1_in_place,
    i32,
    i32,
    C1,
    nppiAndC_32s_C1IR_Ctx
);
impl_add_constant_array!(and_constant_i32_c3, i32, i32, C3, 3, nppiAndC_32s_C3R_Ctx);
impl_add_constant_array_in_place!(
    and_constant_i32_c3_in_place,
    i32,
    i32,
    C3,
    3,
    nppiAndC_32s_C3IR_Ctx
);
impl_add_constant_array!(
    and_constant_i32_ac4,
    i32,
    i32,
    AC4,
    3,
    nppiAndC_32s_AC4R_Ctx
);
impl_add_constant_array_in_place!(
    and_constant_i32_ac4_in_place,
    i32,
    i32,
    AC4,
    3,
    nppiAndC_32s_AC4IR_Ctx
);
impl_add_constant_array!(and_constant_i32_c4, i32, i32, C4, 4, nppiAndC_32s_C4R_Ctx);
impl_add_constant_array_in_place!(
    and_constant_i32_c4_in_place,
    i32,
    i32,
    C4,
    4,
    nppiAndC_32s_C4IR_Ctx
);

impl_add_constant_scalar!(or_constant_u8_c1, u8, u8, C1, nppiOrC_8u_C1R_Ctx);
impl_add_constant_scalar_in_place!(or_constant_u8_c1_in_place, u8, u8, C1, nppiOrC_8u_C1IR_Ctx);
impl_add_constant_array!(or_constant_u8_c3, u8, u8, C3, 3, nppiOrC_8u_C3R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_u8_c3_in_place,
    u8,
    u8,
    C3,
    3,
    nppiOrC_8u_C3IR_Ctx
);
impl_add_constant_array!(or_constant_u8_ac4, u8, u8, AC4, 3, nppiOrC_8u_AC4R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_u8_ac4_in_place,
    u8,
    u8,
    AC4,
    3,
    nppiOrC_8u_AC4IR_Ctx
);
impl_add_constant_array!(or_constant_u8_c4, u8, u8, C4, 4, nppiOrC_8u_C4R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_u8_c4_in_place,
    u8,
    u8,
    C4,
    4,
    nppiOrC_8u_C4IR_Ctx
);
impl_add_constant_scalar!(or_constant_u16_c1, u16, u16, C1, nppiOrC_16u_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    or_constant_u16_c1_in_place,
    u16,
    u16,
    C1,
    nppiOrC_16u_C1IR_Ctx
);
impl_add_constant_array!(or_constant_u16_c3, u16, u16, C3, 3, nppiOrC_16u_C3R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_u16_c3_in_place,
    u16,
    u16,
    C3,
    3,
    nppiOrC_16u_C3IR_Ctx
);
impl_add_constant_array!(or_constant_u16_ac4, u16, u16, AC4, 3, nppiOrC_16u_AC4R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_u16_ac4_in_place,
    u16,
    u16,
    AC4,
    3,
    nppiOrC_16u_AC4IR_Ctx
);
impl_add_constant_array!(or_constant_u16_c4, u16, u16, C4, 4, nppiOrC_16u_C4R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_u16_c4_in_place,
    u16,
    u16,
    C4,
    4,
    nppiOrC_16u_C4IR_Ctx
);
impl_add_constant_scalar!(or_constant_i32_c1, i32, i32, C1, nppiOrC_32s_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    or_constant_i32_c1_in_place,
    i32,
    i32,
    C1,
    nppiOrC_32s_C1IR_Ctx
);
impl_add_constant_array!(or_constant_i32_c3, i32, i32, C3, 3, nppiOrC_32s_C3R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_i32_c3_in_place,
    i32,
    i32,
    C3,
    3,
    nppiOrC_32s_C3IR_Ctx
);
impl_add_constant_array!(or_constant_i32_ac4, i32, i32, AC4, 3, nppiOrC_32s_AC4R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_i32_ac4_in_place,
    i32,
    i32,
    AC4,
    3,
    nppiOrC_32s_AC4IR_Ctx
);
impl_add_constant_array!(or_constant_i32_c4, i32, i32, C4, 4, nppiOrC_32s_C4R_Ctx);
impl_add_constant_array_in_place!(
    or_constant_i32_c4_in_place,
    i32,
    i32,
    C4,
    4,
    nppiOrC_32s_C4IR_Ctx
);

impl_add_constant_scalar!(xor_constant_u8_c1, u8, u8, C1, nppiXorC_8u_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    xor_constant_u8_c1_in_place,
    u8,
    u8,
    C1,
    nppiXorC_8u_C1IR_Ctx
);
impl_add_constant_array!(xor_constant_u8_c3, u8, u8, C3, 3, nppiXorC_8u_C3R_Ctx);
impl_add_constant_array_in_place!(
    xor_constant_u8_c3_in_place,
    u8,
    u8,
    C3,
    3,
    nppiXorC_8u_C3IR_Ctx
);
impl_add_constant_array!(xor_constant_u8_ac4, u8, u8, AC4, 3, nppiXorC_8u_AC4R_Ctx);
impl_add_constant_array_in_place!(
    xor_constant_u8_ac4_in_place,
    u8,
    u8,
    AC4,
    3,
    nppiXorC_8u_AC4IR_Ctx
);
impl_add_constant_array!(xor_constant_u8_c4, u8, u8, C4, 4, nppiXorC_8u_C4R_Ctx);
impl_add_constant_array_in_place!(
    xor_constant_u8_c4_in_place,
    u8,
    u8,
    C4,
    4,
    nppiXorC_8u_C4IR_Ctx
);
impl_add_constant_scalar!(xor_constant_u16_c1, u16, u16, C1, nppiXorC_16u_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    xor_constant_u16_c1_in_place,
    u16,
    u16,
    C1,
    nppiXorC_16u_C1IR_Ctx
);
impl_add_constant_array!(xor_constant_u16_c3, u16, u16, C3, 3, nppiXorC_16u_C3R_Ctx);
impl_add_constant_array_in_place!(
    xor_constant_u16_c3_in_place,
    u16,
    u16,
    C3,
    3,
    nppiXorC_16u_C3IR_Ctx
);
impl_add_constant_array!(
    xor_constant_u16_ac4,
    u16,
    u16,
    AC4,
    3,
    nppiXorC_16u_AC4R_Ctx
);
impl_add_constant_array_in_place!(
    xor_constant_u16_ac4_in_place,
    u16,
    u16,
    AC4,
    3,
    nppiXorC_16u_AC4IR_Ctx
);
impl_add_constant_array!(xor_constant_u16_c4, u16, u16, C4, 4, nppiXorC_16u_C4R_Ctx);
impl_add_constant_array_in_place!(
    xor_constant_u16_c4_in_place,
    u16,
    u16,
    C4,
    4,
    nppiXorC_16u_C4IR_Ctx
);
impl_add_constant_scalar!(xor_constant_i32_c1, i32, i32, C1, nppiXorC_32s_C1R_Ctx);
impl_add_constant_scalar_in_place!(
    xor_constant_i32_c1_in_place,
    i32,
    i32,
    C1,
    nppiXorC_32s_C1IR_Ctx
);
impl_add_constant_array!(xor_constant_i32_c3, i32, i32, C3, 3, nppiXorC_32s_C3R_Ctx);
impl_add_constant_array_in_place!(
    xor_constant_i32_c3_in_place,
    i32,
    i32,
    C3,
    3,
    nppiXorC_32s_C3IR_Ctx
);
impl_add_constant_array!(
    xor_constant_i32_ac4,
    i32,
    i32,
    AC4,
    3,
    nppiXorC_32s_AC4R_Ctx
);
impl_add_constant_array_in_place!(
    xor_constant_i32_ac4_in_place,
    i32,
    i32,
    AC4,
    3,
    nppiXorC_32s_AC4IR_Ctx
);
impl_add_constant_array!(xor_constant_i32_c4, i32, i32, C4, 4, nppiXorC_32s_C4R_Ctx);
impl_add_constant_array_in_place!(
    xor_constant_i32_c4_in_place,
    i32,
    i32,
    C4,
    4,
    nppiXorC_32s_C4IR_Ctx
);
