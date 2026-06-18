use super::*;

impl_rotate!(rotate_u8_c1, u8, C1, nppiRotate_8u_C1R_Ctx);
impl_rotate!(rotate_u8_c3, u8, C3, nppiRotate_8u_C3R_Ctx);
impl_rotate!(rotate_u8_c4, u8, C4, nppiRotate_8u_C4R_Ctx);
impl_rotate!(rotate_u8_ac4, u8, AC4, nppiRotate_8u_AC4R_Ctx);
impl_rotate!(rotate_u16_c1, u16, C1, nppiRotate_16u_C1R_Ctx);
impl_rotate!(rotate_u16_c3, u16, C3, nppiRotate_16u_C3R_Ctx);
impl_rotate!(rotate_u16_c4, u16, C4, nppiRotate_16u_C4R_Ctx);
impl_rotate!(rotate_u16_ac4, u16, AC4, nppiRotate_16u_AC4R_Ctx);
impl_rotate!(rotate_f32_c1, f32, C1, nppiRotate_32f_C1R_Ctx);
impl_rotate!(rotate_f32_c3, f32, C3, nppiRotate_32f_C3R_Ctx);
impl_rotate!(rotate_f32_c4, f32, C4, nppiRotate_32f_C4R_Ctx);
impl_rotate!(rotate_f32_ac4, f32, AC4, nppiRotate_32f_AC4R_Ctx);

pub trait RotateOperation<L: ChannelLayout>: DataTypeLike {
    fn rotate(
        stream_context: &StreamContext,
        rotate: &Rotate,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_rotate_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl RotateOperation<$layout> for $ty {
            fn rotate(
                stream_context: &StreamContext,
                rotate: &Rotate,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()> {
                $function(stream_context, rotate, source, destination)
            }
        }
    };
}

impl_rotate_operation!(u8, C1, rotate_u8_c1);
impl_rotate_operation!(u8, C3, rotate_u8_c3);
impl_rotate_operation!(u8, C4, rotate_u8_c4);
impl_rotate_operation!(u8, AC4, rotate_u8_ac4);
impl_rotate_operation!(u16, C1, rotate_u16_c1);
impl_rotate_operation!(u16, C3, rotate_u16_c3);
impl_rotate_operation!(u16, C4, rotate_u16_c4);
impl_rotate_operation!(u16, AC4, rotate_u16_ac4);
impl_rotate_operation!(f32, C1, rotate_f32_c1);
impl_rotate_operation!(f32, C3, rotate_f32_c3);
impl_rotate_operation!(f32, C4, rotate_f32_c4);
impl_rotate_operation!(f32, AC4, rotate_f32_ac4);

pub fn rotate<T, L>(
    stream_context: &StreamContext,
    rotate: &Rotate,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: RotateOperation<L>,
    L: ChannelLayout,
{
    T::rotate(stream_context, rotate, source, destination)
}

impl_remap!(remap_u8_c1, u8, f32, C1, nppiRemap_8u_C1R_Ctx);
impl_remap!(remap_u8_c3, u8, f32, C3, nppiRemap_8u_C3R_Ctx);
impl_remap!(remap_u8_c4, u8, f32, C4, nppiRemap_8u_C4R_Ctx);
impl_remap!(remap_u8_ac4, u8, f32, AC4, nppiRemap_8u_AC4R_Ctx);
impl_remap!(remap_u16_c1, u16, f32, C1, nppiRemap_16u_C1R_Ctx);
impl_remap!(remap_u16_c3, u16, f32, C3, nppiRemap_16u_C3R_Ctx);
impl_remap!(remap_u16_c4, u16, f32, C4, nppiRemap_16u_C4R_Ctx);
impl_remap!(remap_u16_ac4, u16, f32, AC4, nppiRemap_16u_AC4R_Ctx);
impl_remap!(remap_i16_c1, i16, f32, C1, nppiRemap_16s_C1R_Ctx);
impl_remap!(remap_i16_c3, i16, f32, C3, nppiRemap_16s_C3R_Ctx);
impl_remap!(remap_i16_c4, i16, f32, C4, nppiRemap_16s_C4R_Ctx);
impl_remap!(remap_i16_ac4, i16, f32, AC4, nppiRemap_16s_AC4R_Ctx);
impl_remap!(remap_f32_c1, f32, f32, C1, nppiRemap_32f_C1R_Ctx);
impl_remap!(remap_f32_c3, f32, f32, C3, nppiRemap_32f_C3R_Ctx);
impl_remap!(remap_f32_c4, f32, f32, C4, nppiRemap_32f_C4R_Ctx);
impl_remap!(remap_f32_ac4, f32, f32, AC4, nppiRemap_32f_AC4R_Ctx);
impl_remap!(remap_f64_c1, f64, f64, C1, nppiRemap_64f_C1R_Ctx);
impl_remap!(remap_f64_c3, f64, f64, C3, nppiRemap_64f_C3R_Ctx);
impl_remap!(remap_f64_c4, f64, f64, C4, nppiRemap_64f_C4R_Ctx);
impl_remap!(remap_f64_ac4, f64, f64, AC4, nppiRemap_64f_AC4R_Ctx);

pub trait RemapOperation<L: ChannelLayout>: DataTypeLike {
    type Map: DataTypeLike;

    fn remap(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &ImageView<'_, Self, L>,
        x_map: &ImageView<'_, Self::Map, C1>,
        y_map: &ImageView<'_, Self::Map, C1>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_remap_operation {
    ($ty:ty, $map_ty:ty, $layout:ty, $function:ident) => {
        impl RemapOperation<$layout> for $ty {
            type Map = $map_ty;

            fn remap(
                stream_context: &StreamContext,
                remap: &Remap,
                source: &ImageView<'_, Self, $layout>,
                x_map: &ImageView<'_, Self::Map, C1>,
                y_map: &ImageView<'_, Self::Map, C1>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()> {
                $function(stream_context, remap, source, x_map, y_map, destination)
            }
        }
    };
}

impl_remap_operation!(u8, f32, C1, remap_u8_c1);
impl_remap_operation!(u8, f32, C3, remap_u8_c3);
impl_remap_operation!(u8, f32, C4, remap_u8_c4);
impl_remap_operation!(u8, f32, AC4, remap_u8_ac4);
impl_remap_operation!(u16, f32, C1, remap_u16_c1);
impl_remap_operation!(u16, f32, C3, remap_u16_c3);
impl_remap_operation!(u16, f32, C4, remap_u16_c4);
impl_remap_operation!(u16, f32, AC4, remap_u16_ac4);
impl_remap_operation!(i16, f32, C1, remap_i16_c1);
impl_remap_operation!(i16, f32, C3, remap_i16_c3);
impl_remap_operation!(i16, f32, C4, remap_i16_c4);
impl_remap_operation!(i16, f32, AC4, remap_i16_ac4);
impl_remap_operation!(f32, f32, C1, remap_f32_c1);
impl_remap_operation!(f32, f32, C3, remap_f32_c3);
impl_remap_operation!(f32, f32, C4, remap_f32_c4);
impl_remap_operation!(f32, f32, AC4, remap_f32_ac4);
impl_remap_operation!(f64, f64, C1, remap_f64_c1);
impl_remap_operation!(f64, f64, C3, remap_f64_c3);
impl_remap_operation!(f64, f64, C4, remap_f64_c4);
impl_remap_operation!(f64, f64, AC4, remap_f64_ac4);

pub fn remap<T, L>(
    stream_context: &StreamContext,
    remap: &Remap,
    source: &ImageView<'_, T, L>,
    x_map: &ImageView<'_, T::Map, C1>,
    y_map: &ImageView<'_, T::Map, C1>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: RemapOperation<L>,
    L: ChannelLayout,
{
    T::remap(stream_context, remap, source, x_map, y_map, destination)
}

impl_remap_planar!(remap_u8_p3, u8, f32, 3, nppiRemap_8u_P3R_Ctx);
impl_remap_planar!(remap_u8_p4, u8, f32, 4, nppiRemap_8u_P4R_Ctx);
impl_remap_planar!(remap_u16_p3, u16, f32, 3, nppiRemap_16u_P3R_Ctx);
impl_remap_planar!(remap_u16_p4, u16, f32, 4, nppiRemap_16u_P4R_Ctx);
impl_remap_planar!(remap_i16_p3, i16, f32, 3, nppiRemap_16s_P3R_Ctx);
impl_remap_planar!(remap_i16_p4, i16, f32, 4, nppiRemap_16s_P4R_Ctx);
impl_remap_planar!(remap_f32_p3, f32, f32, 3, nppiRemap_32f_P3R_Ctx);
impl_remap_planar!(remap_f32_p4, f32, f32, 4, nppiRemap_32f_P4R_Ctx);
impl_remap_planar!(remap_f64_p3, f64, f64, 3, nppiRemap_64f_P3R_Ctx);
impl_remap_planar!(remap_f64_p4, f64, f64, 4, nppiRemap_64f_P4R_Ctx);

pub trait RemapPlanar<const PLANES: usize>: DataTypeLike {
    type Map: DataTypeLike;

    fn remap_planar(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &PlanarImageView<'_, Self, PLANES>,
        x_map: &ImageView<'_, Self::Map, C1>,
        y_map: &ImageView<'_, Self::Map, C1>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_remap_planar_dispatch {
    ($ty:ty, $map_ty:ty, $planes:literal, $function:ident) => {
        impl RemapPlanar<$planes> for $ty {
            type Map = $map_ty;

            fn remap_planar(
                stream_context: &StreamContext,
                remap: &Remap,
                source: &PlanarImageView<'_, Self, $planes>,
                x_map: &ImageView<'_, Self::Map, C1>,
                y_map: &ImageView<'_, Self::Map, C1>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()> {
                $function(stream_context, remap, source, x_map, y_map, destination)
            }
        }
    };
}

impl_remap_planar_dispatch!(u8, f32, 3, remap_u8_p3);
impl_remap_planar_dispatch!(u8, f32, 4, remap_u8_p4);
impl_remap_planar_dispatch!(u16, f32, 3, remap_u16_p3);
impl_remap_planar_dispatch!(u16, f32, 4, remap_u16_p4);
impl_remap_planar_dispatch!(i16, f32, 3, remap_i16_p3);
impl_remap_planar_dispatch!(i16, f32, 4, remap_i16_p4);
impl_remap_planar_dispatch!(f32, f32, 3, remap_f32_p3);
impl_remap_planar_dispatch!(f32, f32, 4, remap_f32_p4);
impl_remap_planar_dispatch!(f64, f64, 3, remap_f64_p3);
impl_remap_planar_dispatch!(f64, f64, 4, remap_f64_p4);

pub fn remap_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    remap: &Remap,
    source: &PlanarImageView<'_, T, PLANES>,
    x_map: &ImageView<'_, T::Map, C1>,
    y_map: &ImageView<'_, T::Map, C1>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: RemapPlanar<PLANES>,
{
    T::remap_planar(stream_context, remap, source, x_map, y_map, destination)
}
