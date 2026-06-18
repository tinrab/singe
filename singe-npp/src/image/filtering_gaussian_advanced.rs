use super::*;

impl_filter_advanced!(
    filter_gauss_advanced_u8_c1,
    u8,
    C1,
    nppiFilterGaussAdvanced_8u_C1R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_u8_c3,
    u8,
    C3,
    nppiFilterGaussAdvanced_8u_C3R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_u8_c4,
    u8,
    C4,
    nppiFilterGaussAdvanced_8u_C4R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_u8_ac4,
    u8,
    AC4,
    nppiFilterGaussAdvanced_8u_AC4R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_u16_c1,
    u16,
    C1,
    nppiFilterGaussAdvanced_16u_C1R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_u16_c3,
    u16,
    C3,
    nppiFilterGaussAdvanced_16u_C3R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_u16_c4,
    u16,
    C4,
    nppiFilterGaussAdvanced_16u_C4R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_u16_ac4,
    u16,
    AC4,
    nppiFilterGaussAdvanced_16u_AC4R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_i16_c1,
    i16,
    C1,
    nppiFilterGaussAdvanced_16s_C1R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_i16_c3,
    i16,
    C3,
    nppiFilterGaussAdvanced_16s_C3R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_i16_c4,
    i16,
    C4,
    nppiFilterGaussAdvanced_16s_C4R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_i16_ac4,
    i16,
    AC4,
    nppiFilterGaussAdvanced_16s_AC4R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_f32_c1,
    f32,
    C1,
    nppiFilterGaussAdvanced_32f_C1R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_f32_c3,
    f32,
    C3,
    nppiFilterGaussAdvanced_32f_C3R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_f32_c4,
    f32,
    C4,
    nppiFilterGaussAdvanced_32f_C4R_Ctx
);
impl_filter_advanced!(
    filter_gauss_advanced_f32_ac4,
    f32,
    AC4,
    nppiFilterGaussAdvanced_32f_AC4R_Ctx
);

impl_filter_advanced_border!(
    filter_gauss_advanced_border_u8_c1,
    u8,
    C1,
    nppiFilterGaussAdvancedBorder_8u_C1R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_u8_c3,
    u8,
    C3,
    nppiFilterGaussAdvancedBorder_8u_C3R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_u8_c4,
    u8,
    C4,
    nppiFilterGaussAdvancedBorder_8u_C4R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_u8_ac4,
    u8,
    AC4,
    nppiFilterGaussAdvancedBorder_8u_AC4R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_u16_c1,
    u16,
    C1,
    nppiFilterGaussAdvancedBorder_16u_C1R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_u16_c3,
    u16,
    C3,
    nppiFilterGaussAdvancedBorder_16u_C3R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_u16_c4,
    u16,
    C4,
    nppiFilterGaussAdvancedBorder_16u_C4R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_u16_ac4,
    u16,
    AC4,
    nppiFilterGaussAdvancedBorder_16u_AC4R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_i16_c1,
    i16,
    C1,
    nppiFilterGaussAdvancedBorder_16s_C1R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_i16_c3,
    i16,
    C3,
    nppiFilterGaussAdvancedBorder_16s_C3R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_i16_c4,
    i16,
    C4,
    nppiFilterGaussAdvancedBorder_16s_C4R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_i16_ac4,
    i16,
    AC4,
    nppiFilterGaussAdvancedBorder_16s_AC4R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_f32_c1,
    f32,
    C1,
    nppiFilterGaussAdvancedBorder_32f_C1R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_f32_c3,
    f32,
    C3,
    nppiFilterGaussAdvancedBorder_32f_C3R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_f32_c4,
    f32,
    C4,
    nppiFilterGaussAdvancedBorder_32f_C4R_Ctx
);
impl_filter_advanced_border!(
    filter_gauss_advanced_border_f32_ac4,
    f32,
    AC4,
    nppiFilterGaussAdvancedBorder_32f_AC4R_Ctx
);

pub trait FilterGaussAdvanced<L: ChannelLayout>: DataTypeLike {
    fn filter_gauss_advanced(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
        kernel: &[f32],
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_advanced_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterGaussAdvanced<$layout> for $pixel_ty {
            fn filter_gauss_advanced(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                kernel: &[f32],
            ) -> Result<()> {
                $name(stream_context, source, destination, kernel)
            }
        }
    };
}

impl_filter_gauss_advanced_dispatch!(u8, C1, filter_gauss_advanced_u8_c1);
impl_filter_gauss_advanced_dispatch!(u8, C3, filter_gauss_advanced_u8_c3);
impl_filter_gauss_advanced_dispatch!(u8, C4, filter_gauss_advanced_u8_c4);
impl_filter_gauss_advanced_dispatch!(u8, AC4, filter_gauss_advanced_u8_ac4);
impl_filter_gauss_advanced_dispatch!(u16, C1, filter_gauss_advanced_u16_c1);
impl_filter_gauss_advanced_dispatch!(u16, C3, filter_gauss_advanced_u16_c3);
impl_filter_gauss_advanced_dispatch!(u16, C4, filter_gauss_advanced_u16_c4);
impl_filter_gauss_advanced_dispatch!(u16, AC4, filter_gauss_advanced_u16_ac4);
impl_filter_gauss_advanced_dispatch!(i16, C1, filter_gauss_advanced_i16_c1);
impl_filter_gauss_advanced_dispatch!(i16, C3, filter_gauss_advanced_i16_c3);
impl_filter_gauss_advanced_dispatch!(i16, C4, filter_gauss_advanced_i16_c4);
impl_filter_gauss_advanced_dispatch!(i16, AC4, filter_gauss_advanced_i16_ac4);
impl_filter_gauss_advanced_dispatch!(f32, C1, filter_gauss_advanced_f32_c1);
impl_filter_gauss_advanced_dispatch!(f32, C3, filter_gauss_advanced_f32_c3);
impl_filter_gauss_advanced_dispatch!(f32, C4, filter_gauss_advanced_f32_c4);
impl_filter_gauss_advanced_dispatch!(f32, AC4, filter_gauss_advanced_f32_ac4);

pub fn filter_gauss_advanced<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
    kernel: &[f32],
) -> Result<()>
where
    T: FilterGaussAdvanced<L>,
    L: ChannelLayout,
{
    T::filter_gauss_advanced(stream_context, source, destination, kernel)
}

pub trait FilterGaussAdvancedBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_gauss_advanced_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_advanced_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterGaussAdvancedBorder<$layout> for $pixel_ty {
            fn filter_gauss_advanced_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    kernel,
                    border_type,
                )
            }
        }
    };
}

impl_filter_gauss_advanced_border_dispatch!(u8, C1, filter_gauss_advanced_border_u8_c1);
impl_filter_gauss_advanced_border_dispatch!(u8, C3, filter_gauss_advanced_border_u8_c3);
impl_filter_gauss_advanced_border_dispatch!(u8, C4, filter_gauss_advanced_border_u8_c4);
impl_filter_gauss_advanced_border_dispatch!(u8, AC4, filter_gauss_advanced_border_u8_ac4);
impl_filter_gauss_advanced_border_dispatch!(u16, C1, filter_gauss_advanced_border_u16_c1);
impl_filter_gauss_advanced_border_dispatch!(u16, C3, filter_gauss_advanced_border_u16_c3);
impl_filter_gauss_advanced_border_dispatch!(u16, C4, filter_gauss_advanced_border_u16_c4);
impl_filter_gauss_advanced_border_dispatch!(u16, AC4, filter_gauss_advanced_border_u16_ac4);
impl_filter_gauss_advanced_border_dispatch!(i16, C1, filter_gauss_advanced_border_i16_c1);
impl_filter_gauss_advanced_border_dispatch!(i16, C3, filter_gauss_advanced_border_i16_c3);
impl_filter_gauss_advanced_border_dispatch!(i16, C4, filter_gauss_advanced_border_i16_c4);
impl_filter_gauss_advanced_border_dispatch!(i16, AC4, filter_gauss_advanced_border_i16_ac4);
impl_filter_gauss_advanced_border_dispatch!(f32, C1, filter_gauss_advanced_border_f32_c1);
impl_filter_gauss_advanced_border_dispatch!(f32, C3, filter_gauss_advanced_border_f32_c3);
impl_filter_gauss_advanced_border_dispatch!(f32, C4, filter_gauss_advanced_border_f32_c4);
impl_filter_gauss_advanced_border_dispatch!(f32, AC4, filter_gauss_advanced_border_f32_ac4);

pub fn filter_gauss_advanced_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    kernel: &[f32],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterGaussAdvancedBorder<L>,
    L: ChannelLayout,
{
    T::filter_gauss_advanced_border(
        stream_context,
        source,
        source_offset,
        destination,
        kernel,
        border_type,
    )
}
