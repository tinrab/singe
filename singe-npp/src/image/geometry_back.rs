use super::*;

impl_warp_perspective_back!(
    warp_perspective_back_u8_c1,
    u8,
    C1,
    nppiWarpPerspectiveBack_8u_C1R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_u8_c3,
    u8,
    C3,
    nppiWarpPerspectiveBack_8u_C3R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_u8_c4,
    u8,
    C4,
    nppiWarpPerspectiveBack_8u_C4R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_u8_ac4,
    u8,
    AC4,
    nppiWarpPerspectiveBack_8u_AC4R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_u16_c1,
    u16,
    C1,
    nppiWarpPerspectiveBack_16u_C1R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_u16_c3,
    u16,
    C3,
    nppiWarpPerspectiveBack_16u_C3R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_u16_c4,
    u16,
    C4,
    nppiWarpPerspectiveBack_16u_C4R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_u16_ac4,
    u16,
    AC4,
    nppiWarpPerspectiveBack_16u_AC4R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_i32_c1,
    i32,
    C1,
    nppiWarpPerspectiveBack_32s_C1R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_i32_c3,
    i32,
    C3,
    nppiWarpPerspectiveBack_32s_C3R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_i32_c4,
    i32,
    C4,
    nppiWarpPerspectiveBack_32s_C4R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_i32_ac4,
    i32,
    AC4,
    nppiWarpPerspectiveBack_32s_AC4R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_f32_c1,
    f32,
    C1,
    nppiWarpPerspectiveBack_32f_C1R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_f32_c3,
    f32,
    C3,
    nppiWarpPerspectiveBack_32f_C3R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_f32_c4,
    f32,
    C4,
    nppiWarpPerspectiveBack_32f_C4R_Ctx
);
impl_warp_perspective_back!(
    warp_perspective_back_f32_ac4,
    f32,
    AC4,
    nppiWarpPerspectiveBack_32f_AC4R_Ctx
);

pub trait WarpPerspectiveBackOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_perspective_back(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_back_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpPerspectiveBackOperation<$layout> for $ty {
            fn warp_perspective_back(
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

impl_warp_perspective_back_operation!(u8, C1, warp_perspective_back_u8_c1);
impl_warp_perspective_back_operation!(u8, C3, warp_perspective_back_u8_c3);
impl_warp_perspective_back_operation!(u8, C4, warp_perspective_back_u8_c4);
impl_warp_perspective_back_operation!(u8, AC4, warp_perspective_back_u8_ac4);
impl_warp_perspective_back_operation!(u16, C1, warp_perspective_back_u16_c1);
impl_warp_perspective_back_operation!(u16, C3, warp_perspective_back_u16_c3);
impl_warp_perspective_back_operation!(u16, C4, warp_perspective_back_u16_c4);
impl_warp_perspective_back_operation!(u16, AC4, warp_perspective_back_u16_ac4);
impl_warp_perspective_back_operation!(i32, C1, warp_perspective_back_i32_c1);
impl_warp_perspective_back_operation!(i32, C3, warp_perspective_back_i32_c3);
impl_warp_perspective_back_operation!(i32, C4, warp_perspective_back_i32_c4);
impl_warp_perspective_back_operation!(i32, AC4, warp_perspective_back_i32_ac4);
impl_warp_perspective_back_operation!(f32, C1, warp_perspective_back_f32_c1);
impl_warp_perspective_back_operation!(f32, C3, warp_perspective_back_f32_c3);
impl_warp_perspective_back_operation!(f32, C4, warp_perspective_back_f32_c4);
impl_warp_perspective_back_operation!(f32, AC4, warp_perspective_back_f32_ac4);

pub fn warp_perspective_back<T, L>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: WarpPerspectiveBackOperation<L>,
    L: ChannelLayout,
{
    T::warp_perspective_back(stream_context, warp, source, destination)
}

impl_warp_perspective_back_planar!(
    warp_perspective_back_u8_p3,
    u8,
    3,
    nppiWarpPerspectiveBack_8u_P3R_Ctx
);
impl_warp_perspective_back_planar!(
    warp_perspective_back_u8_p4,
    u8,
    4,
    nppiWarpPerspectiveBack_8u_P4R_Ctx
);
impl_warp_perspective_back_planar!(
    warp_perspective_back_u16_p3,
    u16,
    3,
    nppiWarpPerspectiveBack_16u_P3R_Ctx
);
impl_warp_perspective_back_planar!(
    warp_perspective_back_u16_p4,
    u16,
    4,
    nppiWarpPerspectiveBack_16u_P4R_Ctx
);
impl_warp_perspective_back_planar!(
    warp_perspective_back_i32_p3,
    i32,
    3,
    nppiWarpPerspectiveBack_32s_P3R_Ctx
);
impl_warp_perspective_back_planar!(
    warp_perspective_back_i32_p4,
    i32,
    4,
    nppiWarpPerspectiveBack_32s_P4R_Ctx
);
impl_warp_perspective_back_planar!(
    warp_perspective_back_f32_p3,
    f32,
    3,
    nppiWarpPerspectiveBack_32f_P3R_Ctx
);
impl_warp_perspective_back_planar!(
    warp_perspective_back_f32_p4,
    f32,
    4,
    nppiWarpPerspectiveBack_32f_P4R_Ctx
);

pub trait WarpPerspectiveBackPlanar<const PLANES: usize>: DataTypeLike {
    fn warp_perspective_back_planar(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_perspective_back_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl WarpPerspectiveBackPlanar<$planes> for $ty {
            fn warp_perspective_back_planar(
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

impl_warp_perspective_back_planar_dispatch!(u8, 3, warp_perspective_back_u8_p3);
impl_warp_perspective_back_planar_dispatch!(u8, 4, warp_perspective_back_u8_p4);
impl_warp_perspective_back_planar_dispatch!(u16, 3, warp_perspective_back_u16_p3);
impl_warp_perspective_back_planar_dispatch!(u16, 4, warp_perspective_back_u16_p4);
impl_warp_perspective_back_planar_dispatch!(i32, 3, warp_perspective_back_i32_p3);
impl_warp_perspective_back_planar_dispatch!(i32, 4, warp_perspective_back_i32_p4);
impl_warp_perspective_back_planar_dispatch!(f32, 3, warp_perspective_back_f32_p3);
impl_warp_perspective_back_planar_dispatch!(f32, 4, warp_perspective_back_f32_p4);

pub fn warp_perspective_back_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpPerspective,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpPerspectiveBackPlanar<PLANES>,
{
    T::warp_perspective_back_planar(stream_context, warp, source, destination)
}

impl_warp_affine_back!(
    warp_affine_back_u8_c1,
    u8,
    C1,
    nppiWarpAffineBack_8u_C1R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_u8_c3,
    u8,
    C3,
    nppiWarpAffineBack_8u_C3R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_u8_c4,
    u8,
    C4,
    nppiWarpAffineBack_8u_C4R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_u8_ac4,
    u8,
    AC4,
    nppiWarpAffineBack_8u_AC4R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_u16_c1,
    u16,
    C1,
    nppiWarpAffineBack_16u_C1R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_u16_c3,
    u16,
    C3,
    nppiWarpAffineBack_16u_C3R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_u16_c4,
    u16,
    C4,
    nppiWarpAffineBack_16u_C4R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_u16_ac4,
    u16,
    AC4,
    nppiWarpAffineBack_16u_AC4R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_i32_c1,
    i32,
    C1,
    nppiWarpAffineBack_32s_C1R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_i32_c3,
    i32,
    C3,
    nppiWarpAffineBack_32s_C3R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_i32_c4,
    i32,
    C4,
    nppiWarpAffineBack_32s_C4R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_i32_ac4,
    i32,
    AC4,
    nppiWarpAffineBack_32s_AC4R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_f32_c1,
    f32,
    C1,
    nppiWarpAffineBack_32f_C1R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_f32_c3,
    f32,
    C3,
    nppiWarpAffineBack_32f_C3R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_f32_c4,
    f32,
    C4,
    nppiWarpAffineBack_32f_C4R_Ctx
);
impl_warp_affine_back!(
    warp_affine_back_f32_ac4,
    f32,
    AC4,
    nppiWarpAffineBack_32f_AC4R_Ctx
);

pub trait WarpAffineBackOperation<L: ChannelLayout>: DataTypeLike {
    fn warp_affine_back(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_back_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl WarpAffineBackOperation<$layout> for $ty {
            fn warp_affine_back(
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

impl_warp_affine_back_operation!(u8, C1, warp_affine_back_u8_c1);
impl_warp_affine_back_operation!(u8, C3, warp_affine_back_u8_c3);
impl_warp_affine_back_operation!(u8, C4, warp_affine_back_u8_c4);
impl_warp_affine_back_operation!(u8, AC4, warp_affine_back_u8_ac4);
impl_warp_affine_back_operation!(u16, C1, warp_affine_back_u16_c1);
impl_warp_affine_back_operation!(u16, C3, warp_affine_back_u16_c3);
impl_warp_affine_back_operation!(u16, C4, warp_affine_back_u16_c4);
impl_warp_affine_back_operation!(u16, AC4, warp_affine_back_u16_ac4);
impl_warp_affine_back_operation!(i32, C1, warp_affine_back_i32_c1);
impl_warp_affine_back_operation!(i32, C3, warp_affine_back_i32_c3);
impl_warp_affine_back_operation!(i32, C4, warp_affine_back_i32_c4);
impl_warp_affine_back_operation!(i32, AC4, warp_affine_back_i32_ac4);
impl_warp_affine_back_operation!(f32, C1, warp_affine_back_f32_c1);
impl_warp_affine_back_operation!(f32, C3, warp_affine_back_f32_c3);
impl_warp_affine_back_operation!(f32, C4, warp_affine_back_f32_c4);
impl_warp_affine_back_operation!(f32, AC4, warp_affine_back_f32_ac4);

pub fn warp_affine_back<T, L>(
    stream_context: &StreamContext,
    warp: &WarpAffine,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: WarpAffineBackOperation<L>,
    L: ChannelLayout,
{
    T::warp_affine_back(stream_context, warp, source, destination)
}

impl_warp_affine_back_planar!(warp_affine_back_u8_p3, u8, 3, nppiWarpAffineBack_8u_P3R_Ctx);
impl_warp_affine_back_planar!(warp_affine_back_u8_p4, u8, 4, nppiWarpAffineBack_8u_P4R_Ctx);
impl_warp_affine_back_planar!(
    warp_affine_back_u16_p3,
    u16,
    3,
    nppiWarpAffineBack_16u_P3R_Ctx
);
impl_warp_affine_back_planar!(
    warp_affine_back_u16_p4,
    u16,
    4,
    nppiWarpAffineBack_16u_P4R_Ctx
);
impl_warp_affine_back_planar!(
    warp_affine_back_i32_p3,
    i32,
    3,
    nppiWarpAffineBack_32s_P3R_Ctx
);
impl_warp_affine_back_planar!(
    warp_affine_back_i32_p4,
    i32,
    4,
    nppiWarpAffineBack_32s_P4R_Ctx
);
impl_warp_affine_back_planar!(
    warp_affine_back_f32_p3,
    f32,
    3,
    nppiWarpAffineBack_32f_P3R_Ctx
);
impl_warp_affine_back_planar!(
    warp_affine_back_f32_p4,
    f32,
    4,
    nppiWarpAffineBack_32f_P4R_Ctx
);

pub trait WarpAffineBackPlanar<const PLANES: usize>: DataTypeLike {
    fn warp_affine_back_planar(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_warp_affine_back_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl WarpAffineBackPlanar<$planes> for $ty {
            fn warp_affine_back_planar(
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

impl_warp_affine_back_planar_dispatch!(u8, 3, warp_affine_back_u8_p3);
impl_warp_affine_back_planar_dispatch!(u8, 4, warp_affine_back_u8_p4);
impl_warp_affine_back_planar_dispatch!(u16, 3, warp_affine_back_u16_p3);
impl_warp_affine_back_planar_dispatch!(u16, 4, warp_affine_back_u16_p4);
impl_warp_affine_back_planar_dispatch!(i32, 3, warp_affine_back_i32_p3);
impl_warp_affine_back_planar_dispatch!(i32, 4, warp_affine_back_i32_p4);
impl_warp_affine_back_planar_dispatch!(f32, 3, warp_affine_back_f32_p3);
impl_warp_affine_back_planar_dispatch!(f32, 4, warp_affine_back_f32_p4);

pub fn warp_affine_back_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    warp: &WarpAffine,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: WarpAffineBackPlanar<PLANES>,
{
    T::warp_affine_back_planar(stream_context, warp, source, destination)
}
