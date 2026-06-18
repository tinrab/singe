use super::*;

impl_rgb_to_uyvp_packed!(
    rgb_to_uyvp_10u_u8_c3,
    u8,
    C3,
    nppiRGB_8u_ToUYVP_10u_C3C3R_Ctx
);
impl_rgb_to_uyvp_planar!(
    rgb_to_uyvp_10u_u8_c3_to_p3,
    u8,
    C3,
    nppiRGB_8u_ToUYVP_10u_C3P3R_Ctx
);
impl_rgb_to_uyvp_packed!(
    rgba_to_uyvp_10u_u8_ac4,
    u8,
    AC4,
    nppiRGBA_8u_ToUYVP_10u_AC4C3R_Ctx
);
impl_rgb_planar_to_uyvp_packed!(rgb_to_uyvp_10u_u8_p3, u8, nppiRGB_8u_ToUYVP_10u_P3C3R_Ctx);
impl_rgb_to_uyvp_packed!(
    rgb_to_uyvp_10u_u16_c3,
    u16,
    C3,
    nppiRGB_16u_ToUYVP_10u_C3C3R_Ctx
);
impl_rgb_to_uyvp_packed!(
    rgba_to_uyvp_10u_u16_ac4,
    u16,
    AC4,
    nppiRGBA_16u_ToUYVP_10u_AC4C3R_Ctx
);
impl_rgb_planar_to_uyvp_packed!(
    rgb_to_uyvp_10u_u16_p3,
    u16,
    nppiRGB_16u_ToUYVP_10u_P3C3R_Ctx
);
impl_uyvp_to_rgb_packed!(
    uyvp_10u_to_rgb_u8_c3,
    u8,
    C3,
    nppiUYVP_10u_ToRGB_8u_C3C3R_Ctx
);
impl_uyvp_to_rgb_planar!(uyvp_10u_to_rgb_u8_p3, u8, nppiUYVP_10u_ToRGB_8u_C3P3R_Ctx);
impl_uyvp_to_rgb_packed!(
    uyvp_10u_to_rgb_u16_c3,
    u16,
    C3,
    nppiUYVP_10u_ToRGB_16u_C3C3R_Ctx
);
impl_uyvp_to_rgb_planar!(
    uyvp_10u_to_rgb_u16_p3,
    u16,
    nppiUYVP_10u_ToRGB_16u_C3P3R_Ctx
);
impl_uyvp_to_rgb_packed_constant_alpha!(
    uyvp_10u_to_rgb_u16_ac4,
    u16,
    nppiUYVP_10u_ToRGB_16u_C3AC4R_Ctx
);

pub trait RgbToUyvp10u: DataTypeLike {
    fn rgb_to_uyvp_10u(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C3>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_rgb_to_uyvp_10u_dispatch {
    ($ty:ty, $function:ident) => {
        impl RgbToUyvp10u for $ty {
            fn rgb_to_uyvp_10u(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, u8, C3>,
                color_space: ColorSpace,
            ) -> Result<()> {
                $function(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    color_space,
                )
            }
        }
    };
}

impl_rgb_to_uyvp_10u_dispatch!(u8, rgb_to_uyvp_10u_u8_c3);
impl_rgb_to_uyvp_10u_dispatch!(u16, rgb_to_uyvp_10u_u16_c3);

pub fn rgb_to_uyvp_10u<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C3>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: RgbToUyvp10u,
{
    T::rgb_to_uyvp_10u(
        stream_context,
        source,
        source_offset,
        destination,
        color_space,
    )
}

pub trait RgbaToUyvp10u: DataTypeLike {
    fn rgba_to_uyvp_10u(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, AC4>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C3>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_rgba_to_uyvp_10u_dispatch {
    ($ty:ty, $function:ident) => {
        impl RgbaToUyvp10u for $ty {
            fn rgba_to_uyvp_10u(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, AC4>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, u8, C3>,
                color_space: ColorSpace,
            ) -> Result<()> {
                $function(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    color_space,
                )
            }
        }
    };
}

impl_rgba_to_uyvp_10u_dispatch!(u8, rgba_to_uyvp_10u_u8_ac4);
impl_rgba_to_uyvp_10u_dispatch!(u16, rgba_to_uyvp_10u_u16_ac4);

pub fn rgba_to_uyvp_10u<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, AC4>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C3>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: RgbaToUyvp10u,
{
    T::rgba_to_uyvp_10u(
        stream_context,
        source,
        source_offset,
        destination,
        color_space,
    )
}

pub trait RgbPlanarToUyvp10uP3: DataTypeLike {
    fn rgb_planar_to_uyvp_10u_p3(
        stream_context: &StreamContext,
        source_0: &ImageView<'_, Self, C1>,
        source_1: &ImageView<'_, Self, C1>,
        source_2: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C3>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_rgb_planar_to_uyvp_10u_p3_dispatch {
    ($ty:ty, $function:ident) => {
        impl RgbPlanarToUyvp10uP3 for $ty {
            fn rgb_planar_to_uyvp_10u_p3(
                stream_context: &StreamContext,
                source_0: &ImageView<'_, Self, C1>,
                source_1: &ImageView<'_, Self, C1>,
                source_2: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, u8, C3>,
                color_space: ColorSpace,
            ) -> Result<()> {
                $function(
                    stream_context,
                    source_0,
                    source_1,
                    source_2,
                    source_offset,
                    destination,
                    color_space,
                )
            }
        }
    };
}

impl_rgb_planar_to_uyvp_10u_p3_dispatch!(u8, rgb_to_uyvp_10u_u8_p3);
impl_rgb_planar_to_uyvp_10u_p3_dispatch!(u16, rgb_to_uyvp_10u_u16_p3);

pub fn rgb_planar_to_uyvp_10u_p3<T>(
    stream_context: &StreamContext,
    source_0: &ImageView<'_, T, C1>,
    source_1: &ImageView<'_, T, C1>,
    source_2: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C3>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: RgbPlanarToUyvp10uP3,
{
    T::rgb_planar_to_uyvp_10u_p3(
        stream_context,
        source_0,
        source_1,
        source_2,
        source_offset,
        destination,
        color_space,
    )
}

pub trait Uyvp10uToRgb: DataTypeLike {
    fn uyvp_10u_to_rgb(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, C3>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_uyvp_10u_to_rgb_dispatch {
    ($ty:ty, $function:ident) => {
        impl Uyvp10uToRgb for $ty {
            fn uyvp_10u_to_rgb(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, C3>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, C3>,
                color_space: ColorSpace,
            ) -> Result<()> {
                $function(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    color_space,
                )
            }
        }
    };
}

impl_uyvp_10u_to_rgb_dispatch!(u8, uyvp_10u_to_rgb_u8_c3);
impl_uyvp_10u_to_rgb_dispatch!(u16, uyvp_10u_to_rgb_u16_c3);

pub fn uyvp_10u_to_rgb<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, C3>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: Uyvp10uToRgb,
{
    T::uyvp_10u_to_rgb(
        stream_context,
        source,
        source_offset,
        destination,
        color_space,
    )
}

pub trait Uyvp10uToRgbP3: DataTypeLike {
    fn uyvp_10u_to_rgb_p3(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination_0: &mut ImageViewMut<'_, Self, C1>,
        destination_1: &mut ImageViewMut<'_, Self, C1>,
        destination_2: &mut ImageViewMut<'_, Self, C1>,
        color_space: ColorSpace,
    ) -> Result<()>;
}

macro_rules! impl_uyvp_10u_to_rgb_p3_dispatch {
    ($ty:ty, $function:ident) => {
        impl Uyvp10uToRgbP3 for $ty {
            fn uyvp_10u_to_rgb_p3(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, C3>,
                source_offset: Point,
                destination_0: &mut ImageViewMut<'_, Self, C1>,
                destination_1: &mut ImageViewMut<'_, Self, C1>,
                destination_2: &mut ImageViewMut<'_, Self, C1>,
                color_space: ColorSpace,
            ) -> Result<()> {
                $function(
                    stream_context,
                    source,
                    source_offset,
                    destination_0,
                    destination_1,
                    destination_2,
                    color_space,
                )
            }
        }
    };
}

impl_uyvp_10u_to_rgb_p3_dispatch!(u8, uyvp_10u_to_rgb_u8_p3);
impl_uyvp_10u_to_rgb_p3_dispatch!(u16, uyvp_10u_to_rgb_u16_p3);

pub fn uyvp_10u_to_rgb_p3<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_offset: Point,
    destination_0: &mut ImageViewMut<'_, T, C1>,
    destination_1: &mut ImageViewMut<'_, T, C1>,
    destination_2: &mut ImageViewMut<'_, T, C1>,
    color_space: ColorSpace,
) -> Result<()>
where
    T: Uyvp10uToRgbP3,
{
    T::uyvp_10u_to_rgb_p3(
        stream_context,
        source,
        source_offset,
        destination_0,
        destination_1,
        destination_2,
        color_space,
    )
}

pub trait Uyvp10uToRgba: DataTypeLike {
    fn uyvp_10u_to_rgba(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, AC4>,
        color_space: ColorSpace,
        alpha: Self,
    ) -> Result<()>;
}

impl Uyvp10uToRgba for u16 {
    fn uyvp_10u_to_rgba(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, AC4>,
        color_space: ColorSpace,
        alpha: Self,
    ) -> Result<()> {
        uyvp_10u_to_rgb_u16_ac4(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
            alpha,
        )
    }
}

pub fn uyvp_10u_to_rgba<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, AC4>,
    color_space: ColorSpace,
    alpha: T,
) -> Result<()>
where
    T: Uyvp10uToRgba,
{
    T::uyvp_10u_to_rgba(
        stream_context,
        source,
        source_offset,
        destination,
        color_space,
        alpha,
    )
}
