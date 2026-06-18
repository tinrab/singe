use super::*;

impl_unary_operation!(logical_not_u8_c1, u8, C1, nppiNot_8u_C1R_Ctx);
impl_unary_operation_in_place!(logical_not_u8_c1_in_place, u8, C1, nppiNot_8u_C1IR_Ctx);
impl_unary_operation!(logical_not_u8_c3, u8, C3, nppiNot_8u_C3R_Ctx);
impl_unary_operation_in_place!(logical_not_u8_c3_in_place, u8, C3, nppiNot_8u_C3IR_Ctx);
impl_unary_operation!(logical_not_u8_ac4, u8, AC4, nppiNot_8u_AC4R_Ctx);
impl_unary_operation_in_place!(logical_not_u8_ac4_in_place, u8, AC4, nppiNot_8u_AC4IR_Ctx);
impl_unary_operation!(logical_not_u8_c4, u8, C4, nppiNot_8u_C4R_Ctx);
impl_unary_operation_in_place!(logical_not_u8_c4_in_place, u8, C4, nppiNot_8u_C4IR_Ctx);
