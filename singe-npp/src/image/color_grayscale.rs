use super::*;

impl_rgb_to_gray!(rgb_to_gray_u8_c3, u8, C3, nppiRGBToGray_8u_C3C1R_Ctx);
impl_rgb_to_gray!(rgb_to_gray_u8_ac4, u8, AC4, nppiRGBToGray_8u_AC4C1R_Ctx);
impl_rgb_to_gray!(rgb_to_gray_u16_c3, u16, C3, nppiRGBToGray_16u_C3C1R_Ctx);
impl_rgb_to_gray!(rgb_to_gray_u16_ac4, u16, AC4, nppiRGBToGray_16u_AC4C1R_Ctx);
impl_rgb_to_gray!(rgb_to_gray_i16_c3, i16, C3, nppiRGBToGray_16s_C3C1R_Ctx);
impl_rgb_to_gray!(rgb_to_gray_i16_ac4, i16, AC4, nppiRGBToGray_16s_AC4C1R_Ctx);
impl_rgb_to_gray!(rgb_to_gray_f32_c3, f32, C3, nppiRGBToGray_32f_C3C1R_Ctx);
impl_rgb_to_gray!(rgb_to_gray_f32_ac4, f32, AC4, nppiRGBToGray_32f_AC4C1R_Ctx);
impl_generic_rgb_to_gray!(RgbToGrayC3, rgb_to_gray, rgb_to_gray_c3, C3, [
    u8 => rgb_to_gray_u8_c3,
    u16 => rgb_to_gray_u16_c3,
    i16 => rgb_to_gray_i16_c3,
    f32 => rgb_to_gray_f32_c3,
]);
impl_generic_rgb_to_gray!(RgbToGrayAC4, rgb_to_gray, rgb_to_gray_ac4, AC4, [
    u8 => rgb_to_gray_u8_ac4,
    u16 => rgb_to_gray_u16_ac4,
    i16 => rgb_to_gray_i16_ac4,
    f32 => rgb_to_gray_f32_ac4,
]);
impl_color_to_gray!(color_to_gray_u8_c3, u8, C3, 3, nppiColorToGray_8u_C3C1R_Ctx);
impl_color_to_gray!(
    color_to_gray_u8_ac4,
    u8,
    AC4,
    3,
    nppiColorToGray_8u_AC4C1R_Ctx
);
impl_color_to_gray!(color_to_gray_u8_c4, u8, C4, 4, nppiColorToGray_8u_C4C1R_Ctx);
impl_color_to_gray!(
    color_to_gray_u16_c3,
    u16,
    C3,
    3,
    nppiColorToGray_16u_C3C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_u16_ac4,
    u16,
    AC4,
    3,
    nppiColorToGray_16u_AC4C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_u16_c4,
    u16,
    C4,
    4,
    nppiColorToGray_16u_C4C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_i16_c3,
    i16,
    C3,
    3,
    nppiColorToGray_16s_C3C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_i16_ac4,
    i16,
    AC4,
    3,
    nppiColorToGray_16s_AC4C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_i16_c4,
    i16,
    C4,
    4,
    nppiColorToGray_16s_C4C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_f32_c3,
    f32,
    C3,
    3,
    nppiColorToGray_32f_C3C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_f32_ac4,
    f32,
    AC4,
    3,
    nppiColorToGray_32f_AC4C1R_Ctx
);
impl_color_to_gray!(
    color_to_gray_f32_c4,
    f32,
    C4,
    4,
    nppiColorToGray_32f_C4C1R_Ctx
);
impl_generic_color_to_gray!(ColorToGrayC3, color_to_gray, color_to_gray_c3, C3, 3, [
    u8 => color_to_gray_u8_c3,
    u16 => color_to_gray_u16_c3,
    i16 => color_to_gray_i16_c3,
    f32 => color_to_gray_f32_c3,
]);
impl_generic_color_to_gray!(
    ColorToGrayAC4,
    color_to_gray,
    color_to_gray_ac4,
    AC4,
    3,
    [
        u8 => color_to_gray_u8_ac4,
        u16 => color_to_gray_u16_ac4,
        i16 => color_to_gray_i16_ac4,
        f32 => color_to_gray_f32_ac4,
    ]
);
impl_generic_color_to_gray!(ColorToGrayC4, color_to_gray, color_to_gray_c4, C4, 4, [
    u8 => color_to_gray_u8_c4,
    u16 => color_to_gray_u16_c4,
    i16 => color_to_gray_i16_c4,
    f32 => color_to_gray_f32_c4,
]);
impl_gradient_color_to_gray!(
    gradient_color_to_gray_u8_c3,
    u8,
    nppiGradientColorToGray_8u_C3C1R_Ctx
);
impl_gradient_color_to_gray!(
    gradient_color_to_gray_u16_c3,
    u16,
    nppiGradientColorToGray_16u_C3C1R_Ctx
);
impl_gradient_color_to_gray!(
    gradient_color_to_gray_i16_c3,
    i16,
    nppiGradientColorToGray_16s_C3C1R_Ctx
);
impl_gradient_color_to_gray!(
    gradient_color_to_gray_f32_c3,
    f32,
    nppiGradientColorToGray_32f_C3C1R_Ctx
);

pub trait GradientColorToGray: DataTypeLike {
    fn gradient_color_to_gray(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        destination: &mut ImageViewMut<'_, Self, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

macro_rules! impl_gradient_color_to_gray_dispatch {
    ($ty:ty, $function:ident) => {
        impl GradientColorToGray for $ty {
            fn gradient_color_to_gray(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut ImageViewMut<'_, Self, C1>,
                norm: ImageNormalization,
            ) -> Result<()> {
                $function(stream_context, source, destination, norm)
            }
        }
    };
}

impl_gradient_color_to_gray_dispatch!(u8, gradient_color_to_gray_u8_c3);
impl_gradient_color_to_gray_dispatch!(u16, gradient_color_to_gray_u16_c3);
impl_gradient_color_to_gray_dispatch!(i16, gradient_color_to_gray_i16_c3);
impl_gradient_color_to_gray_dispatch!(f32, gradient_color_to_gray_f32_c3);

pub fn gradient_color_to_gray<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    destination: &mut ImageViewMut<'_, T, C1>,
    norm: ImageNormalization,
) -> Result<()>
where
    T: GradientColorToGray,
{
    T::gradient_color_to_gray(stream_context, source, destination, norm)
}

impl_cfa_to_rgb!(cfa_to_rgb_u8_c3, u8, C3, nppiCFAToRGB_8u_C1C3R_Ctx);
impl_cfa_to_rgba!(cfa_to_rgba_u8_ac4, u8, nppiCFAToRGBA_8u_C1AC4R_Ctx);
impl_cfa_to_rgb!(cfa_to_rgb_u16_c3, u16, C3, nppiCFAToRGB_16u_C1C3R_Ctx);
impl_cfa_to_rgba!(cfa_to_rgba_u16_ac4, u16, nppiCFAToRGBA_16u_C1AC4R_Ctx);
impl_cfa_to_rgb!(cfa_to_rgb_u32_c3, u32, C3, nppiCFAToRGB_32u_C1C3R_Ctx);
impl_cfa_to_rgba!(cfa_to_rgba_u32_ac4, u32, nppiCFAToRGBA_32u_C1AC4R_Ctx);

pub trait CfaToRgb: DataTypeLike {
    fn cfa_to_rgb(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_roi: Rectangle,
        destination: &mut ImageViewMut<'_, Self, C3>,
        grid: BayerGridPosition,
    ) -> Result<()>;
}

macro_rules! impl_cfa_to_rgb_dispatch {
    ($ty:ty, $function:ident) => {
        impl CfaToRgb for $ty {
            fn cfa_to_rgb(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_roi: Rectangle,
                destination: &mut ImageViewMut<'_, Self, C3>,
                grid: BayerGridPosition,
            ) -> Result<()> {
                $function(stream_context, source, source_roi, destination, grid)
            }
        }
    };
}

impl_cfa_to_rgb_dispatch!(u8, cfa_to_rgb_u8_c3);
impl_cfa_to_rgb_dispatch!(u16, cfa_to_rgb_u16_c3);
impl_cfa_to_rgb_dispatch!(u32, cfa_to_rgb_u32_c3);

pub fn cfa_to_rgb<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_roi: Rectangle,
    destination: &mut ImageViewMut<'_, T, C3>,
    grid: BayerGridPosition,
) -> Result<()>
where
    T: CfaToRgb,
{
    T::cfa_to_rgb(stream_context, source, source_roi, destination, grid)
}

pub trait CfaToRgba: DataTypeLike {
    fn cfa_to_rgba(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_roi: Rectangle,
        destination: &mut ImageViewMut<'_, Self, AC4>,
        grid: BayerGridPosition,
        alpha: Self,
    ) -> Result<()>;
}

macro_rules! impl_cfa_to_rgba_dispatch {
    ($ty:ty, $function:ident) => {
        impl CfaToRgba for $ty {
            fn cfa_to_rgba(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_roi: Rectangle,
                destination: &mut ImageViewMut<'_, Self, AC4>,
                grid: BayerGridPosition,
                alpha: Self,
            ) -> Result<()> {
                $function(stream_context, source, source_roi, destination, grid, alpha)
            }
        }
    };
}

impl_cfa_to_rgba_dispatch!(u8, cfa_to_rgba_u8_ac4);
impl_cfa_to_rgba_dispatch!(u16, cfa_to_rgba_u16_ac4);
impl_cfa_to_rgba_dispatch!(u32, cfa_to_rgba_u32_ac4);

pub fn cfa_to_rgba<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_roi: Rectangle,
    destination: &mut ImageViewMut<'_, T, AC4>,
    grid: BayerGridPosition,
    alpha: T,
) -> Result<()>
where
    T: CfaToRgba,
{
    T::cfa_to_rgba(stream_context, source, source_roi, destination, grid, alpha)
}
