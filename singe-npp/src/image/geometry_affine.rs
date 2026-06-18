use super::*;

impl_warp_affine!(warp_affine_u8_c1, u8, C1, nppiWarpAffine_8u_C1R_Ctx);
impl_warp_affine!(warp_affine_u8_c3, u8, C3, nppiWarpAffine_8u_C3R_Ctx);
impl_warp_affine!(warp_affine_u8_c4, u8, C4, nppiWarpAffine_8u_C4R_Ctx);
impl_warp_affine!(warp_affine_u8_ac4, u8, AC4, nppiWarpAffine_8u_AC4R_Ctx);
impl_warp_affine!(warp_affine_u16_c1, u16, C1, nppiWarpAffine_16u_C1R_Ctx);
impl_warp_affine!(warp_affine_u16_c3, u16, C3, nppiWarpAffine_16u_C3R_Ctx);
impl_warp_affine!(warp_affine_u16_c4, u16, C4, nppiWarpAffine_16u_C4R_Ctx);
impl_warp_affine!(warp_affine_u16_ac4, u16, AC4, nppiWarpAffine_16u_AC4R_Ctx);
impl_warp_affine!(warp_affine_i32_c1, i32, C1, nppiWarpAffine_32s_C1R_Ctx);
impl_warp_affine!(warp_affine_i32_c3, i32, C3, nppiWarpAffine_32s_C3R_Ctx);
impl_warp_affine!(warp_affine_i32_c4, i32, C4, nppiWarpAffine_32s_C4R_Ctx);
impl_warp_affine!(warp_affine_i32_ac4, i32, AC4, nppiWarpAffine_32s_AC4R_Ctx);
impl_warp_affine!(warp_affine_f16_c1, f16, C1, nppiWarpAffine_16f_C1R_Ctx);
impl_warp_affine!(warp_affine_f16_c3, f16, C3, nppiWarpAffine_16f_C3R_Ctx);
impl_warp_affine!(warp_affine_f16_c4, f16, C4, nppiWarpAffine_16f_C4R_Ctx);
impl_warp_affine!(warp_affine_f32_c1, f32, C1, nppiWarpAffine_32f_C1R_Ctx);
impl_warp_affine!(warp_affine_f32_c3, f32, C3, nppiWarpAffine_32f_C3R_Ctx);
impl_warp_affine!(warp_affine_f32_c4, f32, C4, nppiWarpAffine_32f_C4R_Ctx);
impl_warp_affine!(warp_affine_f32_ac4, f32, AC4, nppiWarpAffine_32f_AC4R_Ctx);
impl_warp_affine!(warp_affine_f64_c1, f64, C1, nppiWarpAffine_64f_C1R_Ctx);
impl_warp_affine!(warp_affine_f64_c3, f64, C3, nppiWarpAffine_64f_C3R_Ctx);
impl_warp_affine!(warp_affine_f64_c4, f64, C4, nppiWarpAffine_64f_C4R_Ctx);
impl_warp_affine!(warp_affine_f64_ac4, f64, AC4, nppiWarpAffine_64f_AC4R_Ctx);

pub trait WarpAffineOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_affine(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpAffineOperation<$layout> for $ty {
            fn warp_affine(
                stream_context: &StreamContext,
                warp: &WarpAffine,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()> {
                $function(stream_context, warp, source, destination)
            }
        }
    };
}

impl_warp_affine_operation!(u8, C1, warp_affine_u8_c1);
impl_warp_affine_operation!(u8, C3, warp_affine_u8_c3);
impl_warp_affine_operation!(u8, C4, warp_affine_u8_c4);
impl_warp_affine_operation!(u8, AC4, warp_affine_u8_ac4);
impl_warp_affine_operation!(u16, C1, warp_affine_u16_c1);
impl_warp_affine_operation!(u16, C3, warp_affine_u16_c3);
impl_warp_affine_operation!(u16, C4, warp_affine_u16_c4);
impl_warp_affine_operation!(u16, AC4, warp_affine_u16_ac4);
impl_warp_affine_operation!(i32, C1, warp_affine_i32_c1);
impl_warp_affine_operation!(i32, C3, warp_affine_i32_c3);
impl_warp_affine_operation!(i32, C4, warp_affine_i32_c4);
impl_warp_affine_operation!(i32, AC4, warp_affine_i32_ac4);
impl_warp_affine_operation!(f16, C1, warp_affine_f16_c1);
impl_warp_affine_operation!(f16, C3, warp_affine_f16_c3);
impl_warp_affine_operation!(f16, C4, warp_affine_f16_c4);
impl_warp_affine_operation!(f32, C1, warp_affine_f32_c1);
impl_warp_affine_operation!(f32, C3, warp_affine_f32_c3);
impl_warp_affine_operation!(f32, C4, warp_affine_f32_c4);
impl_warp_affine_operation!(f32, AC4, warp_affine_f32_ac4);
impl_warp_affine_operation!(f64, C1, warp_affine_f64_c1);
impl_warp_affine_operation!(f64, C3, warp_affine_f64_c3);
impl_warp_affine_operation!(f64, C4, warp_affine_f64_c4);
impl_warp_affine_operation!(f64, AC4, warp_affine_f64_ac4);

pub fn warp_affine<T, L>(
    stream_context: &StreamContext,
    warp: &WarpAffine,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: WarpAffineOperation<L>,
    L: ChannelLayout,
{
    T::warp_affine(stream_context, warp, source, destination)
}

impl_warp_affine_batch!(
    warp_affine_batch_u8_c1,
    u8,
    C1,
    nppiWarpAffineBatch_8u_C1R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_u8_c3,
    u8,
    C3,
    nppiWarpAffineBatch_8u_C3R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_u8_c4,
    u8,
    C4,
    nppiWarpAffineBatch_8u_C4R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_u8_ac4,
    u8,
    AC4,
    nppiWarpAffineBatch_8u_AC4R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_f16_c1,
    f16,
    C1,
    nppiWarpAffineBatch_16f_C1R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_f16_c3,
    f16,
    C3,
    nppiWarpAffineBatch_16f_C3R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_f16_c4,
    f16,
    C4,
    nppiWarpAffineBatch_16f_C4R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_f32_c1,
    f32,
    C1,
    nppiWarpAffineBatch_32f_C1R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_f32_c3,
    f32,
    C3,
    nppiWarpAffineBatch_32f_C3R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_f32_c4,
    f32,
    C4,
    nppiWarpAffineBatch_32f_C4R_Ctx
);
impl_warp_affine_batch!(
    warp_affine_batch_f32_ac4,
    f32,
    AC4,
    nppiWarpAffineBatch_32f_AC4R_Ctx
);

pub trait WarpAffineBatchOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_affine_batch(
        stream_context: &StreamContext,
        warp: &WarpAffineBatch,
        coefficients: &[AffineCoefficients],
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageViewMut<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_batch_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpAffineBatchOperation<$layout> for $ty {
            fn warp_affine_batch(
                stream_context: &StreamContext,
                warp: &WarpAffineBatch,
                coefficients: &[AffineCoefficients],
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageViewMut<'_, Self, $layout>],
            ) -> Result<()> {
                $function(stream_context, warp, coefficients, sources, destinations)
            }
        }
    };
}

impl_warp_affine_batch_operation!(u8, C1, warp_affine_batch_u8_c1);
impl_warp_affine_batch_operation!(u8, C3, warp_affine_batch_u8_c3);
impl_warp_affine_batch_operation!(u8, C4, warp_affine_batch_u8_c4);
impl_warp_affine_batch_operation!(u8, AC4, warp_affine_batch_u8_ac4);
impl_warp_affine_batch_operation!(f16, C1, warp_affine_batch_f16_c1);
impl_warp_affine_batch_operation!(f16, C3, warp_affine_batch_f16_c3);
impl_warp_affine_batch_operation!(f16, C4, warp_affine_batch_f16_c4);
impl_warp_affine_batch_operation!(f32, C1, warp_affine_batch_f32_c1);
impl_warp_affine_batch_operation!(f32, C3, warp_affine_batch_f32_c3);
impl_warp_affine_batch_operation!(f32, C4, warp_affine_batch_f32_c4);
impl_warp_affine_batch_operation!(f32, AC4, warp_affine_batch_f32_ac4);

pub fn warp_affine_batch<T, L>(
    stream_context: &StreamContext,
    warp: &WarpAffineBatch,
    coefficients: &[AffineCoefficients],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<()>
where
    T: WarpAffineBatchOperation<L>,
    L: ChannelLayout,
{
    T::warp_affine_batch(stream_context, warp, coefficients, sources, destinations)
}

impl_warp_affine_planar!(warp_affine_u8_p3, u8, 3, nppiWarpAffine_8u_P3R_Ctx);
impl_warp_affine_planar!(warp_affine_u8_p4, u8, 4, nppiWarpAffine_8u_P4R_Ctx);
impl_warp_affine_planar!(warp_affine_u16_p3, u16, 3, nppiWarpAffine_16u_P3R_Ctx);
impl_warp_affine_planar!(warp_affine_u16_p4, u16, 4, nppiWarpAffine_16u_P4R_Ctx);
impl_warp_affine_planar!(warp_affine_i32_p3, i32, 3, nppiWarpAffine_32s_P3R_Ctx);
impl_warp_affine_planar!(warp_affine_i32_p4, i32, 4, nppiWarpAffine_32s_P4R_Ctx);
impl_warp_affine_planar!(warp_affine_f32_p3, f32, 3, nppiWarpAffine_32f_P3R_Ctx);
impl_warp_affine_planar!(warp_affine_f32_p4, f32, 4, nppiWarpAffine_32f_P4R_Ctx);
impl_warp_affine_planar!(warp_affine_f64_p3, f64, 3, nppiWarpAffine_64f_P3R_Ctx);
impl_warp_affine_planar!(warp_affine_f64_p4, f64, 4, nppiWarpAffine_64f_P4R_Ctx);

pub trait WarpAffinePlanar<const PLANES: usize>: DataTypeLike {
    fn warp_affine_planar(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl WarpAffinePlanar<$planes> for $ty {
            fn warp_affine_planar(
                stream_context: &StreamContext,
                warp: &WarpAffine,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()> {
                $function(stream_context, warp, source, destination)
            }
        }
    };
}

impl_warp_affine_planar_dispatch!(u8, 3, warp_affine_u8_p3);
impl_warp_affine_planar_dispatch!(u8, 4, warp_affine_u8_p4);
impl_warp_affine_planar_dispatch!(u16, 3, warp_affine_u16_p3);
impl_warp_affine_planar_dispatch!(u16, 4, warp_affine_u16_p4);
impl_warp_affine_planar_dispatch!(i32, 3, warp_affine_i32_p3);
impl_warp_affine_planar_dispatch!(i32, 4, warp_affine_i32_p4);
impl_warp_affine_planar_dispatch!(f32, 3, warp_affine_f32_p3);
impl_warp_affine_planar_dispatch!(f32, 4, warp_affine_f32_p4);
impl_warp_affine_planar_dispatch!(f64, 3, warp_affine_f64_p3);
impl_warp_affine_planar_dispatch!(f64, 4, warp_affine_f64_p4);

pub fn warp_affine_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpAffine,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpAffinePlanar<PLANES>,
{
    T::warp_affine_planar(stream_context, warp, source, destination)
}

impl_warp_affine_quad!(
    warp_affine_quad_u8_c1,
    u8,
    C1,
    nppiWarpAffineQuad_8u_C1R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_u8_c3,
    u8,
    C3,
    nppiWarpAffineQuad_8u_C3R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_u8_c4,
    u8,
    C4,
    nppiWarpAffineQuad_8u_C4R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_u8_ac4,
    u8,
    AC4,
    nppiWarpAffineQuad_8u_AC4R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_u16_c1,
    u16,
    C1,
    nppiWarpAffineQuad_16u_C1R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_u16_c3,
    u16,
    C3,
    nppiWarpAffineQuad_16u_C3R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_u16_c4,
    u16,
    C4,
    nppiWarpAffineQuad_16u_C4R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_u16_ac4,
    u16,
    AC4,
    nppiWarpAffineQuad_16u_AC4R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_i32_c1,
    i32,
    C1,
    nppiWarpAffineQuad_32s_C1R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_i32_c3,
    i32,
    C3,
    nppiWarpAffineQuad_32s_C3R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_i32_c4,
    i32,
    C4,
    nppiWarpAffineQuad_32s_C4R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_i32_ac4,
    i32,
    AC4,
    nppiWarpAffineQuad_32s_AC4R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_f32_c1,
    f32,
    C1,
    nppiWarpAffineQuad_32f_C1R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_f32_c3,
    f32,
    C3,
    nppiWarpAffineQuad_32f_C3R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_f32_c4,
    f32,
    C4,
    nppiWarpAffineQuad_32f_C4R_Ctx
);
impl_warp_affine_quad!(
    warp_affine_quad_f32_ac4,
    f32,
    AC4,
    nppiWarpAffineQuad_32f_AC4R_Ctx
);

pub trait WarpAffineQuadOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_affine_quad(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_quad_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpAffineQuadOperation<$layout> for $ty {
            fn warp_affine_quad(
                stream_context: &StreamContext,
                warp: &WarpQuad,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()> {
                $function(stream_context, warp, source, destination)
            }
        }
    };
}

impl_warp_affine_quad_operation!(u8, C1, warp_affine_quad_u8_c1);
impl_warp_affine_quad_operation!(u8, C3, warp_affine_quad_u8_c3);
impl_warp_affine_quad_operation!(u8, C4, warp_affine_quad_u8_c4);
impl_warp_affine_quad_operation!(u8, AC4, warp_affine_quad_u8_ac4);
impl_warp_affine_quad_operation!(u16, C1, warp_affine_quad_u16_c1);
impl_warp_affine_quad_operation!(u16, C3, warp_affine_quad_u16_c3);
impl_warp_affine_quad_operation!(u16, C4, warp_affine_quad_u16_c4);
impl_warp_affine_quad_operation!(u16, AC4, warp_affine_quad_u16_ac4);
impl_warp_affine_quad_operation!(i32, C1, warp_affine_quad_i32_c1);
impl_warp_affine_quad_operation!(i32, C3, warp_affine_quad_i32_c3);
impl_warp_affine_quad_operation!(i32, C4, warp_affine_quad_i32_c4);
impl_warp_affine_quad_operation!(i32, AC4, warp_affine_quad_i32_ac4);
impl_warp_affine_quad_operation!(f32, C1, warp_affine_quad_f32_c1);
impl_warp_affine_quad_operation!(f32, C3, warp_affine_quad_f32_c3);
impl_warp_affine_quad_operation!(f32, C4, warp_affine_quad_f32_c4);
impl_warp_affine_quad_operation!(f32, AC4, warp_affine_quad_f32_ac4);

pub fn warp_affine_quad<T, L>(
    stream_context: &StreamContext,
    warp: &WarpQuad,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: WarpAffineQuadOperation<L>,
    L: ChannelLayout,
{
    T::warp_affine_quad(stream_context, warp, source, destination)
}

impl_warp_affine_quad_planar!(warp_affine_quad_u8_p3, u8, 3, nppiWarpAffineQuad_8u_P3R_Ctx);
impl_warp_affine_quad_planar!(warp_affine_quad_u8_p4, u8, 4, nppiWarpAffineQuad_8u_P4R_Ctx);
impl_warp_affine_quad_planar!(
    warp_affine_quad_u16_p3,
    u16,
    3,
    nppiWarpAffineQuad_16u_P3R_Ctx
);
impl_warp_affine_quad_planar!(
    warp_affine_quad_u16_p4,
    u16,
    4,
    nppiWarpAffineQuad_16u_P4R_Ctx
);
impl_warp_affine_quad_planar!(
    warp_affine_quad_i32_p3,
    i32,
    3,
    nppiWarpAffineQuad_32s_P3R_Ctx
);
impl_warp_affine_quad_planar!(
    warp_affine_quad_i32_p4,
    i32,
    4,
    nppiWarpAffineQuad_32s_P4R_Ctx
);
impl_warp_affine_quad_planar!(
    warp_affine_quad_f32_p3,
    f32,
    3,
    nppiWarpAffineQuad_32f_P3R_Ctx
);
impl_warp_affine_quad_planar!(
    warp_affine_quad_f32_p4,
    f32,
    4,
    nppiWarpAffineQuad_32f_P4R_Ctx
);

pub trait WarpAffineQuadPlanar<const PLANES: usize>: DataTypeLike {
    fn warp_affine_quad_planar(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_quad_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl WarpAffineQuadPlanar<$planes> for $ty {
            fn warp_affine_quad_planar(
                stream_context: &StreamContext,
                warp: &WarpQuad,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()> {
                $function(stream_context, warp, source, destination)
            }
        }
    };
}

impl_warp_affine_quad_planar_dispatch!(u8, 3, warp_affine_quad_u8_p3);
impl_warp_affine_quad_planar_dispatch!(u8, 4, warp_affine_quad_u8_p4);
impl_warp_affine_quad_planar_dispatch!(u16, 3, warp_affine_quad_u16_p3);
impl_warp_affine_quad_planar_dispatch!(u16, 4, warp_affine_quad_u16_p4);
impl_warp_affine_quad_planar_dispatch!(i32, 3, warp_affine_quad_i32_p3);
impl_warp_affine_quad_planar_dispatch!(i32, 4, warp_affine_quad_i32_p4);
impl_warp_affine_quad_planar_dispatch!(f32, 3, warp_affine_quad_f32_p3);
impl_warp_affine_quad_planar_dispatch!(f32, 4, warp_affine_quad_f32_p4);

pub fn warp_affine_quad_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpQuad,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpAffineQuadPlanar<PLANES>,
{
    T::warp_affine_quad_planar(stream_context, warp, source, destination)
}
