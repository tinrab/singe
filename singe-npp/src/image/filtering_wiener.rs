use super::*;

impl_filter_wiener_border!(
    filter_wiener_border_u8_c1,
    u8,
    C1,
    1,
    nppiFilterWienerBorder_8u_C1R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_u8_c3,
    u8,
    C3,
    3,
    nppiFilterWienerBorder_8u_C3R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_u8_c4,
    u8,
    C4,
    4,
    nppiFilterWienerBorder_8u_C4R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_u8_ac4,
    u8,
    AC4,
    3,
    nppiFilterWienerBorder_8u_AC4R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_i16_c1,
    i16,
    C1,
    1,
    nppiFilterWienerBorder_16s_C1R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_i16_c3,
    i16,
    C3,
    3,
    nppiFilterWienerBorder_16s_C3R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_i16_c4,
    i16,
    C4,
    4,
    nppiFilterWienerBorder_16s_C4R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_i16_ac4,
    i16,
    AC4,
    3,
    nppiFilterWienerBorder_16s_AC4R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_f32_c1,
    f32,
    C1,
    1,
    nppiFilterWienerBorder_32f_C1R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_f32_c3,
    f32,
    C3,
    3,
    nppiFilterWienerBorder_32f_C3R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_f32_c4,
    f32,
    C4,
    4,
    nppiFilterWienerBorder_32f_C4R_Ctx
);
impl_filter_wiener_border!(
    filter_wiener_border_f32_ac4,
    f32,
    AC4,
    3,
    nppiFilterWienerBorder_32f_AC4R_Ctx
);

pub trait FilterWienerBorder<L: ChannelLayout, const N: usize>: DataTypeLike {
    fn filter_wiener_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; N],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_wiener_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $noise_len:expr, $name:ident) => {
        impl FilterWienerBorder<$layout, $noise_len> for $pixel_ty {
            fn filter_wiener_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                mask_size: Size,
                anchor: Point,
                noise: &mut [f32; $noise_len],
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    anchor,
                    noise,
                    border_type,
                )
            }
        }
    };
}

impl_filter_wiener_border_dispatch!(u8, C1, 1, filter_wiener_border_u8_c1);
impl_filter_wiener_border_dispatch!(u8, C3, 3, filter_wiener_border_u8_c3);
impl_filter_wiener_border_dispatch!(u8, C4, 4, filter_wiener_border_u8_c4);
impl_filter_wiener_border_dispatch!(u8, AC4, 3, filter_wiener_border_u8_ac4);
impl_filter_wiener_border_dispatch!(i16, C1, 1, filter_wiener_border_i16_c1);
impl_filter_wiener_border_dispatch!(i16, C3, 3, filter_wiener_border_i16_c3);
impl_filter_wiener_border_dispatch!(i16, C4, 4, filter_wiener_border_i16_c4);
impl_filter_wiener_border_dispatch!(i16, AC4, 3, filter_wiener_border_i16_ac4);
impl_filter_wiener_border_dispatch!(f32, C1, 1, filter_wiener_border_f32_c1);
impl_filter_wiener_border_dispatch!(f32, C3, 3, filter_wiener_border_f32_c3);
impl_filter_wiener_border_dispatch!(f32, C4, 4, filter_wiener_border_f32_c4);
impl_filter_wiener_border_dispatch!(f32, AC4, 3, filter_wiener_border_f32_ac4);

pub fn filter_wiener_border<T, L, const N: usize>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: Size,
    anchor: Point,
    noise: &mut [f32; N],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterWienerBorder<L, N>,
    L: ChannelLayout,
{
    T::filter_wiener_border(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        anchor,
        noise,
        border_type,
    )
}
