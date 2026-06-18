use super::*;

impl_color_twist!(color_twist_f32_c1, f32, C1, nppiColorTwist_32f_C1R_Ctx);
impl_color_twist_in_place!(
    color_twist_f32_c1_in_place,
    f32,
    C1,
    nppiColorTwist_32f_C1IR_Ctx
);
impl_color_twist!(color_twist_f32_c2, f32, C2, nppiColorTwist_32f_C2R_Ctx);
impl_color_twist_in_place!(
    color_twist_f32_c2_in_place,
    f32,
    C2,
    nppiColorTwist_32f_C2IR_Ctx
);
impl_color_twist!(color_twist_f32_c3, f32, C3, nppiColorTwist_32f_C3R_Ctx);
impl_color_twist_in_place!(
    color_twist_f32_c3_in_place,
    f32,
    C3,
    nppiColorTwist_32f_C3IR_Ctx
);
impl_color_twist!(color_twist_f32_c4, f32, C4, nppiColorTwist_32f_C4R_Ctx);
impl_color_twist_in_place!(
    color_twist_f32_c4_in_place,
    f32,
    C4,
    nppiColorTwist_32f_C4IR_Ctx
);
impl_color_twist!(color_twist_f32_ac4, f32, AC4, nppiColorTwist_32f_AC4R_Ctx);
impl_color_twist_in_place!(
    color_twist_f32_ac4_in_place,
    f32,
    AC4,
    nppiColorTwist_32f_AC4IR_Ctx
);
impl_color_twist_with_constants!(
    color_twist_f32_c4_with_constants,
    f32,
    nppiColorTwist_32fC_C4R_Ctx
);
impl_color_twist_with_constants_in_place!(
    color_twist_f32_c4_with_constants_in_place,
    f32,
    nppiColorTwist_32fC_C4IR_Ctx
);
impl_color_twist_planar!(color_twist_f32_p3, f32, 3, nppiColorTwist_32f_P3R_Ctx);
impl_color_twist_planar_in_place!(
    color_twist_f32_p3_in_place,
    f32,
    3,
    nppiColorTwist_32f_IP3R_Ctx
);
impl_color_twist!(color_twist_f16_c1, f16, C1, nppiColorTwist32f_16f_C1R_Ctx);
impl_color_twist_in_place!(
    color_twist_f16_c1_in_place,
    f16,
    C1,
    nppiColorTwist32f_16f_C1IR_Ctx
);
impl_color_twist!(color_twist_f16_c2, f16, C2, nppiColorTwist32f_16f_C2R_Ctx);
impl_color_twist_in_place!(
    color_twist_f16_c2_in_place,
    f16,
    C2,
    nppiColorTwist32f_16f_C2IR_Ctx
);
impl_color_twist!(color_twist_f16_c3, f16, C3, nppiColorTwist32f_16f_C3R_Ctx);
impl_color_twist_in_place!(
    color_twist_f16_c3_in_place,
    f16,
    C3,
    nppiColorTwist32f_16f_C3IR_Ctx
);
impl_color_twist!(color_twist_f16_c4, f16, C4, nppiColorTwist32f_16f_C4R_Ctx);
impl_color_twist_in_place!(
    color_twist_f16_c4_in_place,
    f16,
    C4,
    nppiColorTwist32f_16f_C4IR_Ctx
);
impl_color_twist_with_constants!(
    color_twist_f16_c4_with_constants,
    f16,
    nppiColorTwist32fC_16f_C4R_Ctx
);
impl_color_twist_with_constants_in_place!(
    color_twist_f16_c4_with_constants_in_place,
    f16,
    nppiColorTwist32fC_16f_C4IR_Ctx
);
impl_color_twist!(color_twist_u8_c1, u8, C1, nppiColorTwist32f_8u_C1R_Ctx);
impl_color_twist_in_place!(
    color_twist_u8_c1_in_place,
    u8,
    C1,
    nppiColorTwist32f_8u_C1IR_Ctx
);
impl_color_twist!(color_twist_u8_c2, u8, C2, nppiColorTwist32f_8u_C2R_Ctx);
impl_color_twist_in_place!(
    color_twist_u8_c2_in_place,
    u8,
    C2,
    nppiColorTwist32f_8u_C2IR_Ctx
);
impl_color_twist!(color_twist_u8_c3, u8, C3, nppiColorTwist32f_8u_C3R_Ctx);
impl_color_twist_in_place!(
    color_twist_u8_c3_in_place,
    u8,
    C3,
    nppiColorTwist32f_8u_C3IR_Ctx
);
impl_color_twist!(color_twist_u8_c4, u8, C4, nppiColorTwist32f_8u_C4R_Ctx);
impl_color_twist_in_place!(
    color_twist_u8_c4_in_place,
    u8,
    C4,
    nppiColorTwist32f_8u_C4IR_Ctx
);
impl_color_twist!(color_twist_u8_ac4, u8, AC4, nppiColorTwist32f_8u_AC4R_Ctx);
impl_color_twist_in_place!(
    color_twist_u8_ac4_in_place,
    u8,
    AC4,
    nppiColorTwist32f_8u_AC4IR_Ctx
);
impl_color_twist_with_constants!(
    color_twist_u8_c4_with_constants,
    u8,
    nppiColorTwist32fC_8u_C4R_Ctx
);
impl_color_twist_with_constants_in_place!(
    color_twist_u8_c4_with_constants_in_place,
    u8,
    nppiColorTwist32fC_8u_C4IR_Ctx
);
impl_color_twist_planar!(color_twist_u8_p3, u8, 3, nppiColorTwist32f_8u_P3R_Ctx);
impl_color_twist_planar_in_place!(
    color_twist_u8_p3_in_place,
    u8,
    3,
    nppiColorTwist32f_8u_IP3R_Ctx
);
impl_color_twist!(color_twist_u16_c1, u16, C1, nppiColorTwist32f_16u_C1R_Ctx);
impl_color_twist_in_place!(
    color_twist_u16_c1_in_place,
    u16,
    C1,
    nppiColorTwist32f_16u_C1IR_Ctx
);
impl_color_twist!(color_twist_u16_c2, u16, C2, nppiColorTwist32f_16u_C2R_Ctx);
impl_color_twist_in_place!(
    color_twist_u16_c2_in_place,
    u16,
    C2,
    nppiColorTwist32f_16u_C2IR_Ctx
);
impl_color_twist!(color_twist_u16_c3, u16, C3, nppiColorTwist32f_16u_C3R_Ctx);
impl_color_twist_in_place!(
    color_twist_u16_c3_in_place,
    u16,
    C3,
    nppiColorTwist32f_16u_C3IR_Ctx
);
impl_color_twist!(
    color_twist_u16_ac4,
    u16,
    AC4,
    nppiColorTwist32f_16u_AC4R_Ctx
);
impl_color_twist_in_place!(
    color_twist_u16_ac4_in_place,
    u16,
    AC4,
    nppiColorTwist32f_16u_AC4IR_Ctx
);
impl_color_twist_planar!(color_twist_u16_p3, u16, 3, nppiColorTwist32f_16u_P3R_Ctx);
impl_color_twist_planar_in_place!(
    color_twist_u16_p3_in_place,
    u16,
    3,
    nppiColorTwist32f_16u_IP3R_Ctx
);
impl_color_twist!(color_twist_i8_c1, i8, C1, nppiColorTwist32f_8s_C1R_Ctx);
impl_color_twist_in_place!(
    color_twist_i8_c1_in_place,
    i8,
    C1,
    nppiColorTwist32f_8s_C1IR_Ctx
);
impl_color_twist!(color_twist_i8_c2, i8, C2, nppiColorTwist32f_8s_C2R_Ctx);
impl_color_twist_in_place!(
    color_twist_i8_c2_in_place,
    i8,
    C2,
    nppiColorTwist32f_8s_C2IR_Ctx
);
impl_color_twist!(color_twist_i8_c3, i8, C3, nppiColorTwist32f_8s_C3R_Ctx);
impl_color_twist_in_place!(
    color_twist_i8_c3_in_place,
    i8,
    C3,
    nppiColorTwist32f_8s_C3IR_Ctx
);
impl_color_twist!(color_twist_i8_c4, i8, C4, nppiColorTwist32f_8s_C4R_Ctx);
impl_color_twist_in_place!(
    color_twist_i8_c4_in_place,
    i8,
    C4,
    nppiColorTwist32f_8s_C4IR_Ctx
);
impl_color_twist!(color_twist_i8_ac4, i8, AC4, nppiColorTwist32f_8s_AC4R_Ctx);
impl_color_twist_in_place!(
    color_twist_i8_ac4_in_place,
    i8,
    AC4,
    nppiColorTwist32f_8s_AC4IR_Ctx
);
impl_color_twist_planar!(color_twist_i8_p3, i8, 3, nppiColorTwist32f_8s_P3R_Ctx);
impl_color_twist_planar_in_place!(
    color_twist_i8_p3_in_place,
    i8,
    3,
    nppiColorTwist32f_8s_IP3R_Ctx
);
impl_color_twist!(color_twist_i16_c1, i16, C1, nppiColorTwist32f_16s_C1R_Ctx);
impl_color_twist_in_place!(
    color_twist_i16_c1_in_place,
    i16,
    C1,
    nppiColorTwist32f_16s_C1IR_Ctx
);
impl_color_twist!(color_twist_i16_c2, i16, C2, nppiColorTwist32f_16s_C2R_Ctx);
impl_color_twist_in_place!(
    color_twist_i16_c2_in_place,
    i16,
    C2,
    nppiColorTwist32f_16s_C2IR_Ctx
);
impl_color_twist!(color_twist_i16_c3, i16, C3, nppiColorTwist32f_16s_C3R_Ctx);
impl_color_twist_in_place!(
    color_twist_i16_c3_in_place,
    i16,
    C3,
    nppiColorTwist32f_16s_C3IR_Ctx
);
impl_color_twist!(
    color_twist_i16_ac4,
    i16,
    AC4,
    nppiColorTwist32f_16s_AC4R_Ctx
);
impl_color_twist_in_place!(
    color_twist_i16_ac4_in_place,
    i16,
    AC4,
    nppiColorTwist32f_16s_AC4IR_Ctx
);
impl_color_twist_planar!(color_twist_i16_p3, i16, 3, nppiColorTwist32f_16s_P3R_Ctx);
impl_color_twist_planar_in_place!(
    color_twist_i16_p3_in_place,
    i16,
    3,
    nppiColorTwist32f_16s_IP3R_Ctx
);

impl_generic_color_twist!(ColorTwistC1, color_twist, color_twist_c1, C1, [
    f32 => color_twist_f32_c1,
    f16 => color_twist_f16_c1,
    u8 => color_twist_u8_c1,
    u16 => color_twist_u16_c1,
    i8 => color_twist_i8_c1,
    i16 => color_twist_i16_c1,
]);
impl_generic_color_twist_in_place!(
    ColorTwistC1InPlace,
    color_twist_in_place,
    color_twist_c1_in_place,
    C1,
    [
        f32 => color_twist_f32_c1_in_place,
        f16 => color_twist_f16_c1_in_place,
        u8 => color_twist_u8_c1_in_place,
        u16 => color_twist_u16_c1_in_place,
        i8 => color_twist_i8_c1_in_place,
        i16 => color_twist_i16_c1_in_place,
    ]
);
impl_generic_color_twist!(ColorTwistC2, color_twist, color_twist_c2, C2, [
    f32 => color_twist_f32_c2,
    f16 => color_twist_f16_c2,
    u8 => color_twist_u8_c2,
    u16 => color_twist_u16_c2,
    i8 => color_twist_i8_c2,
    i16 => color_twist_i16_c2,
]);
impl_generic_color_twist_in_place!(
    ColorTwistC2InPlace,
    color_twist_in_place,
    color_twist_c2_in_place,
    C2,
    [
        f32 => color_twist_f32_c2_in_place,
        f16 => color_twist_f16_c2_in_place,
        u8 => color_twist_u8_c2_in_place,
        u16 => color_twist_u16_c2_in_place,
        i8 => color_twist_i8_c2_in_place,
        i16 => color_twist_i16_c2_in_place,
    ]
);
impl_generic_color_twist!(ColorTwistC3, color_twist, color_twist_c3, C3, [
    f32 => color_twist_f32_c3,
    f16 => color_twist_f16_c3,
    u8 => color_twist_u8_c3,
    u16 => color_twist_u16_c3,
    i8 => color_twist_i8_c3,
    i16 => color_twist_i16_c3,
]);
impl_generic_color_twist_in_place!(
    ColorTwistC3InPlace,
    color_twist_in_place,
    color_twist_c3_in_place,
    C3,
    [
        f32 => color_twist_f32_c3_in_place,
        f16 => color_twist_f16_c3_in_place,
        u8 => color_twist_u8_c3_in_place,
        u16 => color_twist_u16_c3_in_place,
        i8 => color_twist_i8_c3_in_place,
        i16 => color_twist_i16_c3_in_place,
    ]
);
impl_generic_color_twist!(ColorTwistC4, color_twist, color_twist_c4, C4, [
    f32 => color_twist_f32_c4,
    f16 => color_twist_f16_c4,
    u8 => color_twist_u8_c4,
    i8 => color_twist_i8_c4,
]);
impl_generic_color_twist_in_place!(
    ColorTwistC4InPlace,
    color_twist_in_place,
    color_twist_c4_in_place,
    C4,
    [
        f32 => color_twist_f32_c4_in_place,
        f16 => color_twist_f16_c4_in_place,
        u8 => color_twist_u8_c4_in_place,
        i8 => color_twist_i8_c4_in_place,
    ]
);
impl_generic_color_twist!(ColorTwistAc4, color_twist, color_twist_ac4, AC4, [
    f32 => color_twist_f32_ac4,
    u8 => color_twist_u8_ac4,
    u16 => color_twist_u16_ac4,
    i8 => color_twist_i8_ac4,
    i16 => color_twist_i16_ac4,
]);
impl_generic_color_twist_in_place!(
    ColorTwistAc4InPlace,
    color_twist_in_place,
    color_twist_ac4_in_place,
    AC4,
    [
        f32 => color_twist_f32_ac4_in_place,
        u8 => color_twist_u8_ac4_in_place,
        u16 => color_twist_u16_ac4_in_place,
        i8 => color_twist_i8_ac4_in_place,
        i16 => color_twist_i16_ac4_in_place,
    ]
);
impl_generic_color_twist_with_constants!(
    ColorTwistWithConstantsC4,
    color_twist_with_constants,
    color_twist_c4_with_constants,
    [
        f32 => color_twist_f32_c4_with_constants,
        f16 => color_twist_f16_c4_with_constants,
        u8 => color_twist_u8_c4_with_constants,
    ]
);
impl_generic_color_twist_with_constants_in_place!(
    ColorTwistWithConstantsC4InPlace,
    color_twist_with_constants_in_place,
    color_twist_c4_with_constants_in_place,
    [
        f32 => color_twist_f32_c4_with_constants_in_place,
        f16 => color_twist_f16_c4_with_constants_in_place,
        u8 => color_twist_u8_c4_with_constants_in_place,
    ]
);

pub trait ColorTwistPlanar<const PLANES: usize>: DataTypeLike {
    fn color_twist_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl ColorTwistPlanar<$planes> for $ty {
            fn color_twist_planar(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
                twist: ColorTwistMatrix,
            ) -> Result<()> {
                $function(stream_context, source, destination, twist)
            }
        }
    };
}

impl_color_twist_planar_dispatch!(f32, 3, color_twist_f32_p3);
impl_color_twist_planar_dispatch!(u8, 3, color_twist_u8_p3);
impl_color_twist_planar_dispatch!(u16, 3, color_twist_u16_p3);
impl_color_twist_planar_dispatch!(i8, 3, color_twist_i8_p3);
impl_color_twist_planar_dispatch!(i16, 3, color_twist_i16_p3);

pub fn color_twist_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
    twist: ColorTwistMatrix,
) -> Result<()>
where
    T: ColorTwistPlanar<PLANES>,
{
    T::color_twist_planar(stream_context, source, destination, twist)
}

pub trait ColorTwistPlanarInPlace<const PLANES: usize>: DataTypeLike {
    fn color_twist_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, PLANES>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_planar_in_place_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl ColorTwistPlanarInPlace<$planes> for $ty {
            fn color_twist_planar_in_place(
                stream_context: &StreamContext,
                image: &mut PlanarImageViewMut<'_, Self, $planes>,
                twist: ColorTwistMatrix,
            ) -> Result<()> {
                $function(stream_context, image, twist)
            }
        }
    };
}

impl_color_twist_planar_in_place_dispatch!(f32, 3, color_twist_f32_p3_in_place);
impl_color_twist_planar_in_place_dispatch!(u8, 3, color_twist_u8_p3_in_place);
impl_color_twist_planar_in_place_dispatch!(u16, 3, color_twist_u16_p3_in_place);
impl_color_twist_planar_in_place_dispatch!(i8, 3, color_twist_i8_p3_in_place);
impl_color_twist_planar_in_place_dispatch!(i16, 3, color_twist_i16_p3_in_place);

pub fn color_twist_planar_in_place<T, const PLANES: usize>(
    stream_context: &StreamContext,
    image: &mut PlanarImageViewMut<'_, T, PLANES>,
    twist: ColorTwistMatrix,
) -> Result<()>
where
    T: ColorTwistPlanarInPlace<PLANES>,
{
    T::color_twist_planar_in_place(stream_context, image, twist)
}
