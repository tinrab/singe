use super::*;

impl_warp_perspective!(
    warp_perspective_u8_c1,
    u8,
    C1,
    nppiWarpPerspective_8u_C1R_Ctx
);
impl_warp_perspective!(
    warp_perspective_u8_c3,
    u8,
    C3,
    nppiWarpPerspective_8u_C3R_Ctx
);
impl_warp_perspective!(
    warp_perspective_u8_c4,
    u8,
    C4,
    nppiWarpPerspective_8u_C4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspective_8u_AC4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_u16_c1,
    u16,
    C1,
    nppiWarpPerspective_16u_C1R_Ctx
);
impl_warp_perspective!(
    warp_perspective_u16_c3,
    u16,
    C3,
    nppiWarpPerspective_16u_C3R_Ctx
);
impl_warp_perspective!(
    warp_perspective_u16_c4,
    u16,
    C4,
    nppiWarpPerspective_16u_C4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_u16_ac4,
    u16,
    AC4,
    nppiWarpPerspective_16u_AC4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_i32_c1,
    i32,
    C1,
    nppiWarpPerspective_32s_C1R_Ctx
);
impl_warp_perspective!(
    warp_perspective_i32_c3,
    i32,
    C3,
    nppiWarpPerspective_32s_C3R_Ctx
);
impl_warp_perspective!(
    warp_perspective_i32_c4,
    i32,
    C4,
    nppiWarpPerspective_32s_C4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_i32_ac4,
    i32,
    AC4,
    nppiWarpPerspective_32s_AC4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_f16_c1,
    f16,
    C1,
    nppiWarpPerspective_16f_C1R_Ctx
);
impl_warp_perspective!(
    warp_perspective_f16_c3,
    f16,
    C3,
    nppiWarpPerspective_16f_C3R_Ctx
);
impl_warp_perspective!(
    warp_perspective_f16_c4,
    f16,
    C4,
    nppiWarpPerspective_16f_C4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_f32_c1,
    f32,
    C1,
    nppiWarpPerspective_32f_C1R_Ctx
);
impl_warp_perspective!(
    warp_perspective_f32_c3,
    f32,
    C3,
    nppiWarpPerspective_32f_C3R_Ctx
);
impl_warp_perspective!(
    warp_perspective_f32_c4,
    f32,
    C4,
    nppiWarpPerspective_32f_C4R_Ctx
);
impl_warp_perspective!(
    warp_perspective_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspective_32f_AC4R_Ctx
);

pub trait WarpPerspectiveOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_perspective(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpPerspectiveOperation<$layout> for $ty {
            fn warp_perspective(
                stream_context: &StreamContext,
                warp: &WarpPerspective,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()> {
                $function(stream_context, warp, source, destination)
            }
        }
    };
}

impl_warp_perspective_operation!(u8, C1, warp_perspective_u8_c1);
impl_warp_perspective_operation!(u8, C3, warp_perspective_u8_c3);
impl_warp_perspective_operation!(u8, C4, warp_perspective_u8_c4);
impl_warp_perspective_operation!(u8, AC4, warp_perspective_u8_ac4);
impl_warp_perspective_operation!(u16, C1, warp_perspective_u16_c1);
impl_warp_perspective_operation!(u16, C3, warp_perspective_u16_c3);
impl_warp_perspective_operation!(u16, C4, warp_perspective_u16_c4);
impl_warp_perspective_operation!(u16, AC4, warp_perspective_u16_ac4);
impl_warp_perspective_operation!(i32, C1, warp_perspective_i32_c1);
impl_warp_perspective_operation!(i32, C3, warp_perspective_i32_c3);
impl_warp_perspective_operation!(i32, C4, warp_perspective_i32_c4);
impl_warp_perspective_operation!(i32, AC4, warp_perspective_i32_ac4);
impl_warp_perspective_operation!(f16, C1, warp_perspective_f16_c1);
impl_warp_perspective_operation!(f16, C3, warp_perspective_f16_c3);
impl_warp_perspective_operation!(f16, C4, warp_perspective_f16_c4);
impl_warp_perspective_operation!(f32, C1, warp_perspective_f32_c1);
impl_warp_perspective_operation!(f32, C3, warp_perspective_f32_c3);
impl_warp_perspective_operation!(f32, C4, warp_perspective_f32_c4);
impl_warp_perspective_operation!(f32, AC4, warp_perspective_f32_ac4);

pub fn warp_perspective<T, L>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: WarpPerspectiveOperation<L>,
    L: ChannelLayout,
{
    T::warp_perspective(stream_context, warp, source, destination)
}

impl_warp_perspective_batch!(
    warp_perspective_batch_u8_c1,
    u8,
    C1,
    nppiWarpPerspectiveBatch_8u_C1R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_u8_c3,
    u8,
    C3,
    nppiWarpPerspectiveBatch_8u_C3R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_u8_c4,
    u8,
    C4,
    nppiWarpPerspectiveBatch_8u_C4R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspectiveBatch_8u_AC4R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_f16_c1,
    f16,
    C1,
    nppiWarpPerspectiveBatch_16f_C1R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_f16_c3,
    f16,
    C3,
    nppiWarpPerspectiveBatch_16f_C3R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_f16_c4,
    f16,
    C4,
    nppiWarpPerspectiveBatch_16f_C4R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_f32_c1,
    f32,
    C1,
    nppiWarpPerspectiveBatch_32f_C1R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_f32_c3,
    f32,
    C3,
    nppiWarpPerspectiveBatch_32f_C3R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_f32_c4,
    f32,
    C4,
    nppiWarpPerspectiveBatch_32f_C4R_Ctx
);
impl_warp_perspective_batch!(
    warp_perspective_batch_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspectiveBatch_32f_AC4R_Ctx
);

pub trait WarpPerspectiveBatchOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_perspective_batch(
        stream_context: &StreamContext,
        warp: &WarpPerspectiveBatch,
        coefficients: &[PerspectiveCoefficients],
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageViewMut<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_batch_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpPerspectiveBatchOperation<$layout> for $ty {
            fn warp_perspective_batch(
                stream_context: &StreamContext,
                warp: &WarpPerspectiveBatch,
                coefficients: &[PerspectiveCoefficients],
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageViewMut<'_, Self, $layout>],
            ) -> Result<()> {
                $function(stream_context, warp, coefficients, sources, destinations)
            }
        }
    };
}

impl_warp_perspective_batch_operation!(u8, C1, warp_perspective_batch_u8_c1);
impl_warp_perspective_batch_operation!(u8, C3, warp_perspective_batch_u8_c3);
impl_warp_perspective_batch_operation!(u8, C4, warp_perspective_batch_u8_c4);
impl_warp_perspective_batch_operation!(u8, AC4, warp_perspective_batch_u8_ac4);
impl_warp_perspective_batch_operation!(f16, C1, warp_perspective_batch_f16_c1);
impl_warp_perspective_batch_operation!(f16, C3, warp_perspective_batch_f16_c3);
impl_warp_perspective_batch_operation!(f16, C4, warp_perspective_batch_f16_c4);
impl_warp_perspective_batch_operation!(f32, C1, warp_perspective_batch_f32_c1);
impl_warp_perspective_batch_operation!(f32, C3, warp_perspective_batch_f32_c3);
impl_warp_perspective_batch_operation!(f32, C4, warp_perspective_batch_f32_c4);
impl_warp_perspective_batch_operation!(f32, AC4, warp_perspective_batch_f32_ac4);

pub fn warp_perspective_batch<T, L>(
    stream_context: &StreamContext,
    warp: &WarpPerspectiveBatch,
    coefficients: &[PerspectiveCoefficients],
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<()>
where
    T: WarpPerspectiveBatchOperation<L>,
    L: ChannelLayout,
{
    T::warp_perspective_batch(stream_context, warp, coefficients, sources, destinations)
}

impl_warp_perspective_planar!(
    warp_perspective_u8_p3,
    u8,
    3,
    nppiWarpPerspective_8u_P3R_Ctx
);
impl_warp_perspective_planar!(
    warp_perspective_u8_p4,
    u8,
    4,
    nppiWarpPerspective_8u_P4R_Ctx
);
impl_warp_perspective_planar!(
    warp_perspective_u16_p3,
    u16,
    3,
    nppiWarpPerspective_16u_P3R_Ctx
);
impl_warp_perspective_planar!(
    warp_perspective_u16_p4,
    u16,
    4,
    nppiWarpPerspective_16u_P4R_Ctx
);
impl_warp_perspective_planar!(
    warp_perspective_i32_p3,
    i32,
    3,
    nppiWarpPerspective_32s_P3R_Ctx
);
impl_warp_perspective_planar!(
    warp_perspective_i32_p4,
    i32,
    4,
    nppiWarpPerspective_32s_P4R_Ctx
);
impl_warp_perspective_planar!(
    warp_perspective_f32_p3,
    f32,
    3,
    nppiWarpPerspective_32f_P3R_Ctx
);
impl_warp_perspective_planar!(
    warp_perspective_f32_p4,
    f32,
    4,
    nppiWarpPerspective_32f_P4R_Ctx
);

pub trait WarpPerspectivePlanar<const PLANES: usize>: DataTypeLike {
    fn warp_perspective_planar(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl WarpPerspectivePlanar<$planes> for $ty {
            fn warp_perspective_planar(
                stream_context: &StreamContext,
                warp: &WarpPerspective,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()> {
                $function(stream_context, warp, source, destination)
            }
        }
    };
}

impl_warp_perspective_planar_dispatch!(u8, 3, warp_perspective_u8_p3);
impl_warp_perspective_planar_dispatch!(u8, 4, warp_perspective_u8_p4);
impl_warp_perspective_planar_dispatch!(u16, 3, warp_perspective_u16_p3);
impl_warp_perspective_planar_dispatch!(u16, 4, warp_perspective_u16_p4);
impl_warp_perspective_planar_dispatch!(i32, 3, warp_perspective_i32_p3);
impl_warp_perspective_planar_dispatch!(i32, 4, warp_perspective_i32_p4);
impl_warp_perspective_planar_dispatch!(f32, 3, warp_perspective_f32_p3);
impl_warp_perspective_planar_dispatch!(f32, 4, warp_perspective_f32_p4);

pub fn warp_perspective_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpPerspectivePlanar<PLANES>,
{
    T::warp_perspective_planar(stream_context, warp, source, destination)
}

impl_warp_perspective_quad!(
    warp_perspective_quad_u8_c1,
    u8,
    C1,
    nppiWarpPerspectiveQuad_8u_C1R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_u8_c3,
    u8,
    C3,
    nppiWarpPerspectiveQuad_8u_C3R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_u8_c4,
    u8,
    C4,
    nppiWarpPerspectiveQuad_8u_C4R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspectiveQuad_8u_AC4R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_u16_c1,
    u16,
    C1,
    nppiWarpPerspectiveQuad_16u_C1R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_u16_c3,
    u16,
    C3,
    nppiWarpPerspectiveQuad_16u_C3R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_u16_c4,
    u16,
    C4,
    nppiWarpPerspectiveQuad_16u_C4R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_u16_ac4,
    u16,
    AC4,
    nppiWarpPerspectiveQuad_16u_AC4R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_i32_c1,
    i32,
    C1,
    nppiWarpPerspectiveQuad_32s_C1R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_i32_c3,
    i32,
    C3,
    nppiWarpPerspectiveQuad_32s_C3R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_i32_c4,
    i32,
    C4,
    nppiWarpPerspectiveQuad_32s_C4R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_i32_ac4,
    i32,
    AC4,
    nppiWarpPerspectiveQuad_32s_AC4R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_f32_c1,
    f32,
    C1,
    nppiWarpPerspectiveQuad_32f_C1R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_f32_c3,
    f32,
    C3,
    nppiWarpPerspectiveQuad_32f_C3R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_f32_c4,
    f32,
    C4,
    nppiWarpPerspectiveQuad_32f_C4R_Ctx
);
impl_warp_perspective_quad!(
    warp_perspective_quad_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspectiveQuad_32f_AC4R_Ctx
);

pub trait WarpPerspectiveQuadOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_perspective_quad(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_quad_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpPerspectiveQuadOperation<$layout> for $ty {
            fn warp_perspective_quad(
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

impl_warp_perspective_quad_operation!(u8, C1, warp_perspective_quad_u8_c1);
impl_warp_perspective_quad_operation!(u8, C3, warp_perspective_quad_u8_c3);
impl_warp_perspective_quad_operation!(u8, C4, warp_perspective_quad_u8_c4);
impl_warp_perspective_quad_operation!(u8, AC4, warp_perspective_quad_u8_ac4);
impl_warp_perspective_quad_operation!(u16, C1, warp_perspective_quad_u16_c1);
impl_warp_perspective_quad_operation!(u16, C3, warp_perspective_quad_u16_c3);
impl_warp_perspective_quad_operation!(u16, C4, warp_perspective_quad_u16_c4);
impl_warp_perspective_quad_operation!(u16, AC4, warp_perspective_quad_u16_ac4);
impl_warp_perspective_quad_operation!(i32, C1, warp_perspective_quad_i32_c1);
impl_warp_perspective_quad_operation!(i32, C3, warp_perspective_quad_i32_c3);
impl_warp_perspective_quad_operation!(i32, C4, warp_perspective_quad_i32_c4);
impl_warp_perspective_quad_operation!(i32, AC4, warp_perspective_quad_i32_ac4);
impl_warp_perspective_quad_operation!(f32, C1, warp_perspective_quad_f32_c1);
impl_warp_perspective_quad_operation!(f32, C3, warp_perspective_quad_f32_c3);
impl_warp_perspective_quad_operation!(f32, C4, warp_perspective_quad_f32_c4);
impl_warp_perspective_quad_operation!(f32, AC4, warp_perspective_quad_f32_ac4);

pub fn warp_perspective_quad<T, L>(
    stream_context: &StreamContext,
    warp: &WarpQuad,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: WarpPerspectiveQuadOperation<L>,
    L: ChannelLayout,
{
    T::warp_perspective_quad(stream_context, warp, source, destination)
}

impl_warp_perspective_quad_planar!(
    warp_perspective_quad_u8_p3,
    u8,
    3,
    nppiWarpPerspectiveQuad_8u_P3R_Ctx
);
impl_warp_perspective_quad_planar!(
    warp_perspective_quad_u8_p4,
    u8,
    4,
    nppiWarpPerspectiveQuad_8u_P4R_Ctx
);
impl_warp_perspective_quad_planar!(
    warp_perspective_quad_u16_p3,
    u16,
    3,
    nppiWarpPerspectiveQuad_16u_P3R_Ctx
);
impl_warp_perspective_quad_planar!(
    warp_perspective_quad_u16_p4,
    u16,
    4,
    nppiWarpPerspectiveQuad_16u_P4R_Ctx
);
impl_warp_perspective_quad_planar!(
    warp_perspective_quad_i32_p3,
    i32,
    3,
    nppiWarpPerspectiveQuad_32s_P3R_Ctx
);
impl_warp_perspective_quad_planar!(
    warp_perspective_quad_i32_p4,
    i32,
    4,
    nppiWarpPerspectiveQuad_32s_P4R_Ctx
);
impl_warp_perspective_quad_planar!(
    warp_perspective_quad_f32_p3,
    f32,
    3,
    nppiWarpPerspectiveQuad_32f_P3R_Ctx
);
impl_warp_perspective_quad_planar!(
    warp_perspective_quad_f32_p4,
    f32,
    4,
    nppiWarpPerspectiveQuad_32f_P4R_Ctx
);

pub trait WarpPerspectiveQuadPlanar<const PLANES: usize>: DataTypeLike {
    fn warp_perspective_quad_planar(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_quad_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl WarpPerspectiveQuadPlanar<$planes> for $ty {
            fn warp_perspective_quad_planar(
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

impl_warp_perspective_quad_planar_dispatch!(u8, 3, warp_perspective_quad_u8_p3);
impl_warp_perspective_quad_planar_dispatch!(u8, 4, warp_perspective_quad_u8_p4);
impl_warp_perspective_quad_planar_dispatch!(u16, 3, warp_perspective_quad_u16_p3);
impl_warp_perspective_quad_planar_dispatch!(u16, 4, warp_perspective_quad_u16_p4);
impl_warp_perspective_quad_planar_dispatch!(i32, 3, warp_perspective_quad_i32_p3);
impl_warp_perspective_quad_planar_dispatch!(i32, 4, warp_perspective_quad_i32_p4);
impl_warp_perspective_quad_planar_dispatch!(f32, 3, warp_perspective_quad_f32_p3);
impl_warp_perspective_quad_planar_dispatch!(f32, 4, warp_perspective_quad_f32_p4);

pub fn warp_perspective_quad_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpQuad,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpPerspectiveQuadPlanar<PLANES>,
{
    T::warp_perspective_quad_planar(stream_context, warp, source, destination)
}
