use super::*;

impl_filter_masked_kernel!(filter_gauss_u8_c1, u8, C1, nppiFilterGauss_8u_C1R_Ctx);
impl_filter_masked_kernel!(filter_gauss_u8_c3, u8, C3, nppiFilterGauss_8u_C3R_Ctx);
impl_filter_masked_kernel!(filter_gauss_u8_c4, u8, C4, nppiFilterGauss_8u_C4R_Ctx);
impl_filter_masked_kernel!(filter_gauss_u8_ac4, u8, AC4, nppiFilterGauss_8u_AC4R_Ctx);
impl_filter_masked_kernel!(filter_gauss_u16_c1, u16, C1, nppiFilterGauss_16u_C1R_Ctx);
impl_filter_masked_kernel!(filter_gauss_u16_c3, u16, C3, nppiFilterGauss_16u_C3R_Ctx);
impl_filter_masked_kernel!(filter_gauss_u16_c4, u16, C4, nppiFilterGauss_16u_C4R_Ctx);
impl_filter_masked_kernel!(filter_gauss_u16_ac4, u16, AC4, nppiFilterGauss_16u_AC4R_Ctx);
impl_filter_masked_kernel!(filter_gauss_i16_c1, i16, C1, nppiFilterGauss_16s_C1R_Ctx);
impl_filter_masked_kernel!(filter_gauss_i16_c3, i16, C3, nppiFilterGauss_16s_C3R_Ctx);
impl_filter_masked_kernel!(filter_gauss_i16_c4, i16, C4, nppiFilterGauss_16s_C4R_Ctx);
impl_filter_masked_kernel!(filter_gauss_i16_ac4, i16, AC4, nppiFilterGauss_16s_AC4R_Ctx);
impl_filter_masked_kernel!(filter_gauss_f32_c1, f32, C1, nppiFilterGauss_32f_C1R_Ctx);
impl_filter_masked_kernel!(filter_gauss_f32_c3, f32, C3, nppiFilterGauss_32f_C3R_Ctx);
impl_filter_masked_kernel!(filter_gauss_f32_c4, f32, C4, nppiFilterGauss_32f_C4R_Ctx);
impl_filter_masked_kernel!(filter_gauss_f32_ac4, f32, AC4, nppiFilterGauss_32f_AC4R_Ctx);

impl_filter_masked_kernel_border!(
    filter_gauss_border_u8_c1,
    u8,
    C1,
    nppiFilterGaussBorder_8u_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_u8_c3,
    u8,
    C3,
    nppiFilterGaussBorder_8u_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_u8_c4,
    u8,
    C4,
    nppiFilterGaussBorder_8u_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_u8_ac4,
    u8,
    AC4,
    nppiFilterGaussBorder_8u_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_u16_c1,
    u16,
    C1,
    nppiFilterGaussBorder_16u_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_u16_c3,
    u16,
    C3,
    nppiFilterGaussBorder_16u_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_u16_c4,
    u16,
    C4,
    nppiFilterGaussBorder_16u_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_u16_ac4,
    u16,
    AC4,
    nppiFilterGaussBorder_16u_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_i16_c1,
    i16,
    C1,
    nppiFilterGaussBorder_16s_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_i16_c3,
    i16,
    C3,
    nppiFilterGaussBorder_16s_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_i16_c4,
    i16,
    C4,
    nppiFilterGaussBorder_16s_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_i16_ac4,
    i16,
    AC4,
    nppiFilterGaussBorder_16s_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_f32_c1,
    f32,
    C1,
    nppiFilterGaussBorder_32f_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_f32_c3,
    f32,
    C3,
    nppiFilterGaussBorder_32f_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_f32_c4,
    f32,
    C4,
    nppiFilterGaussBorder_32f_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_gauss_border_f32_ac4,
    f32,
    AC4,
    nppiFilterGaussBorder_32f_AC4R_Ctx
);

pub trait FilterGauss<L: ChannelLayout>: DataTypeLike {
    fn filter_gauss(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterGauss<$layout> for $pixel_ty {
            fn filter_gauss(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $name(stream_context, source, destination, mask_size)
            }
        }
    };
}

impl_filter_gauss_dispatch!(u8, C1, filter_gauss_u8_c1);
impl_filter_gauss_dispatch!(u8, C3, filter_gauss_u8_c3);
impl_filter_gauss_dispatch!(u8, C4, filter_gauss_u8_c4);
impl_filter_gauss_dispatch!(u8, AC4, filter_gauss_u8_ac4);
impl_filter_gauss_dispatch!(u16, C1, filter_gauss_u16_c1);
impl_filter_gauss_dispatch!(u16, C3, filter_gauss_u16_c3);
impl_filter_gauss_dispatch!(u16, C4, filter_gauss_u16_c4);
impl_filter_gauss_dispatch!(u16, AC4, filter_gauss_u16_ac4);
impl_filter_gauss_dispatch!(i16, C1, filter_gauss_i16_c1);
impl_filter_gauss_dispatch!(i16, C3, filter_gauss_i16_c3);
impl_filter_gauss_dispatch!(i16, C4, filter_gauss_i16_c4);
impl_filter_gauss_dispatch!(i16, AC4, filter_gauss_i16_ac4);
impl_filter_gauss_dispatch!(f32, C1, filter_gauss_f32_c1);
impl_filter_gauss_dispatch!(f32, C3, filter_gauss_f32_c3);
impl_filter_gauss_dispatch!(f32, C4, filter_gauss_f32_c4);
impl_filter_gauss_dispatch!(f32, AC4, filter_gauss_f32_ac4);

pub fn filter_gauss<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: MaskSize,
) -> Result<()>
where
    T: FilterGauss<L>,
    L: ChannelLayout,
{
    T::filter_gauss(stream_context, source, destination, mask_size)
}

pub trait FilterGaussBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_gauss_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterGaussBorder<$layout> for $pixel_ty {
            fn filter_gauss_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }
        }
    };
}

impl_filter_gauss_border_dispatch!(u8, C1, filter_gauss_border_u8_c1);
impl_filter_gauss_border_dispatch!(u8, C3, filter_gauss_border_u8_c3);
impl_filter_gauss_border_dispatch!(u8, C4, filter_gauss_border_u8_c4);
impl_filter_gauss_border_dispatch!(u8, AC4, filter_gauss_border_u8_ac4);
impl_filter_gauss_border_dispatch!(u16, C1, filter_gauss_border_u16_c1);
impl_filter_gauss_border_dispatch!(u16, C3, filter_gauss_border_u16_c3);
impl_filter_gauss_border_dispatch!(u16, C4, filter_gauss_border_u16_c4);
impl_filter_gauss_border_dispatch!(u16, AC4, filter_gauss_border_u16_ac4);
impl_filter_gauss_border_dispatch!(i16, C1, filter_gauss_border_i16_c1);
impl_filter_gauss_border_dispatch!(i16, C3, filter_gauss_border_i16_c3);
impl_filter_gauss_border_dispatch!(i16, C4, filter_gauss_border_i16_c4);
impl_filter_gauss_border_dispatch!(i16, AC4, filter_gauss_border_i16_ac4);
impl_filter_gauss_border_dispatch!(f32, C1, filter_gauss_border_f32_c1);
impl_filter_gauss_border_dispatch!(f32, C3, filter_gauss_border_f32_c3);
impl_filter_gauss_border_dispatch!(f32, C4, filter_gauss_border_f32_c4);
impl_filter_gauss_border_dispatch!(f32, AC4, filter_gauss_border_f32_ac4);

pub fn filter_gauss_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterGaussBorder<L>,
    L: ChannelLayout,
{
    T::filter_gauss_border(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}
