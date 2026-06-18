use super::*;

impl_filter_bilateral_gauss_border!(
    filter_bilateral_gauss_border_u8_c1,
    u8,
    C1,
    nppiFilterBilateralGaussBorder_8u_C1R_Ctx
);
impl_filter_bilateral_gauss_border!(
    filter_bilateral_gauss_border_u8_c3,
    u8,
    C3,
    nppiFilterBilateralGaussBorder_8u_C3R_Ctx
);
impl_filter_bilateral_gauss_border!(
    filter_bilateral_gauss_border_u16_c1,
    u16,
    C1,
    nppiFilterBilateralGaussBorder_16u_C1R_Ctx
);
impl_filter_bilateral_gauss_border!(
    filter_bilateral_gauss_border_u16_c3,
    u16,
    C3,
    nppiFilterBilateralGaussBorder_16u_C3R_Ctx
);
impl_filter_bilateral_gauss_border!(
    filter_bilateral_gauss_border_f32_c1,
    f32,
    C1,
    nppiFilterBilateralGaussBorder_32f_C1R_Ctx
);
impl_filter_bilateral_gauss_border!(
    filter_bilateral_gauss_border_f32_c3,
    f32,
    C3,
    nppiFilterBilateralGaussBorder_32f_C3R_Ctx
);

pub trait FilterBilateralGaussBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_bilateral_gauss_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        radius: i32,
        step_between_source_pixels: i32,
        value_square_sigma: f32,
        position_square_sigma: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_bilateral_gauss_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterBilateralGaussBorder<$layout> for $pixel_ty {
            fn filter_bilateral_gauss_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                radius: i32,
                step_between_source_pixels: i32,
                value_square_sigma: f32,
                position_square_sigma: f32,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    radius,
                    step_between_source_pixels,
                    value_square_sigma,
                    position_square_sigma,
                    border_type,
                )
            }
        }
    };
}

impl_filter_bilateral_gauss_border_dispatch!(u8, C1, filter_bilateral_gauss_border_u8_c1);
impl_filter_bilateral_gauss_border_dispatch!(u8, C3, filter_bilateral_gauss_border_u8_c3);
impl_filter_bilateral_gauss_border_dispatch!(u16, C1, filter_bilateral_gauss_border_u16_c1);
impl_filter_bilateral_gauss_border_dispatch!(u16, C3, filter_bilateral_gauss_border_u16_c3);
impl_filter_bilateral_gauss_border_dispatch!(f32, C1, filter_bilateral_gauss_border_f32_c1);
impl_filter_bilateral_gauss_border_dispatch!(f32, C3, filter_bilateral_gauss_border_f32_c3);

pub fn filter_bilateral_gauss_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    radius: i32,
    step_between_source_pixels: i32,
    value_square_sigma: f32,
    position_square_sigma: f32,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterBilateralGaussBorder<L>,
    L: ChannelLayout,
{
    T::filter_bilateral_gauss_border(
        stream_context,
        source,
        source_offset,
        destination,
        radius,
        step_between_source_pixels,
        value_square_sigma,
        position_square_sigma,
        border_type,
    )
}
