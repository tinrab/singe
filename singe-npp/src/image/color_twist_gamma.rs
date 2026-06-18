use super::*;

impl_gamma_packed!(gamma_forward_u8_c3, C3, nppiGammaFwd_8u_C3R_Ctx);
impl_gamma_packed_in_place!(gamma_forward_u8_c3_in_place, C3, nppiGammaFwd_8u_C3IR_Ctx);
impl_gamma_packed!(gamma_forward_u8_ac4, AC4, nppiGammaFwd_8u_AC4R_Ctx);
impl_gamma_packed_in_place!(
    gamma_forward_u8_ac4_in_place,
    AC4,
    nppiGammaFwd_8u_AC4IR_Ctx
);
impl_gamma_planar!(gamma_forward_u8_p3, nppiGammaFwd_8u_P3R_Ctx);
impl_gamma_planar_in_place!(gamma_forward_u8_p3_in_place, nppiGammaFwd_8u_IP3R_Ctx);
impl_gamma_packed!(gamma_inverse_u8_c3, C3, nppiGammaInv_8u_C3R_Ctx);
impl_gamma_packed_in_place!(gamma_inverse_u8_c3_in_place, C3, nppiGammaInv_8u_C3IR_Ctx);
impl_gamma_packed!(gamma_inverse_u8_ac4, AC4, nppiGammaInv_8u_AC4R_Ctx);
impl_gamma_packed_in_place!(
    gamma_inverse_u8_ac4_in_place,
    AC4,
    nppiGammaInv_8u_AC4IR_Ctx
);
impl_gamma_planar!(gamma_inverse_u8_p3, nppiGammaInv_8u_P3R_Ctx);
impl_gamma_planar_in_place!(gamma_inverse_u8_p3_in_place, nppiGammaInv_8u_IP3R_Ctx);

impl_generic_gamma!(GammaForwardC3, gamma_forward, gamma_forward_c3, C3, [
    u8 => gamma_forward_u8_c3
]);
impl_generic_gamma_in_place!(
    GammaForwardC3InPlace,
    gamma_forward_in_place,
    gamma_forward_c3_in_place,
    C3,
    [u8 => gamma_forward_u8_c3_in_place]
);
impl_generic_gamma!(GammaForwardAc4, gamma_forward, gamma_forward_ac4, AC4, [
    u8 => gamma_forward_u8_ac4
]);
impl_generic_gamma_in_place!(
    GammaForwardAc4InPlace,
    gamma_forward_in_place,
    gamma_forward_ac4_in_place,
    AC4,
    [u8 => gamma_forward_u8_ac4_in_place]
);
impl_generic_gamma!(GammaInverseC3, gamma_inverse, gamma_inverse_c3, C3, [
    u8 => gamma_inverse_u8_c3
]);
impl_generic_gamma_in_place!(
    GammaInverseC3InPlace,
    gamma_inverse_in_place,
    gamma_inverse_c3_in_place,
    C3,
    [u8 => gamma_inverse_u8_c3_in_place]
);
impl_generic_gamma!(GammaInverseAc4, gamma_inverse, gamma_inverse_ac4, AC4, [
    u8 => gamma_inverse_u8_ac4
]);
impl_generic_gamma_in_place!(
    GammaInverseAc4InPlace,
    gamma_inverse_in_place,
    gamma_inverse_ac4_in_place,
    AC4,
    [u8 => gamma_inverse_u8_ac4_in_place]
);

pub trait GammaForwardPlanar<const PLANES: usize>: DataTypeLike {
    fn gamma_forward_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

impl GammaForwardPlanar<3> for u8 {
    fn gamma_forward_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, 3>,
        destination: &mut PlanarImageViewMut<'_, Self, 3>,
    ) -> Result<()> {
        gamma_forward_u8_p3(stream_context, source, destination)
    }
}

pub fn gamma_forward_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaForwardPlanar<PLANES>,
{
    T::gamma_forward_planar(stream_context, source, destination)
}

pub trait GammaForwardPlanarInPlace<const PLANES: usize>: DataTypeLike {
    fn gamma_forward_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

impl GammaForwardPlanarInPlace<3> for u8 {
    fn gamma_forward_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, 3>,
    ) -> Result<()> {
        gamma_forward_u8_p3_in_place(stream_context, image)
    }
}

pub fn gamma_forward_planar_in_place<T, const PLANES: usize>(
    stream_context: &StreamContext,
    image: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaForwardPlanarInPlace<PLANES>,
{
    T::gamma_forward_planar_in_place(stream_context, image)
}

pub trait GammaInversePlanar<const PLANES: usize>: DataTypeLike {
    fn gamma_inverse_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

impl GammaInversePlanar<3> for u8 {
    fn gamma_inverse_planar(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, Self, 3>,
        destination: &mut PlanarImageViewMut<'_, Self, 3>,
    ) -> Result<()> {
        gamma_inverse_u8_p3(stream_context, source, destination)
    }
}

pub fn gamma_inverse_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaInversePlanar<PLANES>,
{
    T::gamma_inverse_planar(stream_context, source, destination)
}

pub trait GammaInversePlanarInPlace<const PLANES: usize>: DataTypeLike {
    fn gamma_inverse_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

impl GammaInversePlanarInPlace<3> for u8 {
    fn gamma_inverse_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, Self, 3>,
    ) -> Result<()> {
        gamma_inverse_u8_p3_in_place(stream_context, image)
    }
}

pub fn gamma_inverse_planar_in_place<T, const PLANES: usize>(
    stream_context: &StreamContext,
    image: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: GammaInversePlanarInPlace<PLANES>,
{
    T::gamma_inverse_planar_in_place(stream_context, image)
}
