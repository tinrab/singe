use super::*;

impl_filter_masked_kernel!(filter_low_pass_u8_c1, u8, C1, nppiFilterLowPass_8u_C1R_Ctx);
impl_filter_masked_kernel!(filter_low_pass_u8_c3, u8, C3, nppiFilterLowPass_8u_C3R_Ctx);
impl_filter_masked_kernel!(filter_low_pass_u8_c4, u8, C4, nppiFilterLowPass_8u_C4R_Ctx);
impl_filter_masked_kernel!(
    filter_low_pass_u8_ac4,
    u8,
    AC4,
    nppiFilterLowPass_8u_AC4R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_u16_c1,
    u16,
    C1,
    nppiFilterLowPass_16u_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_u16_c3,
    u16,
    C3,
    nppiFilterLowPass_16u_C3R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_u16_c4,
    u16,
    C4,
    nppiFilterLowPass_16u_C4R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_u16_ac4,
    u16,
    AC4,
    nppiFilterLowPass_16u_AC4R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_i16_c1,
    i16,
    C1,
    nppiFilterLowPass_16s_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_i16_c3,
    i16,
    C3,
    nppiFilterLowPass_16s_C3R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_i16_c4,
    i16,
    C4,
    nppiFilterLowPass_16s_C4R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_i16_ac4,
    i16,
    AC4,
    nppiFilterLowPass_16s_AC4R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_f32_c1,
    f32,
    C1,
    nppiFilterLowPass_32f_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_f32_c3,
    f32,
    C3,
    nppiFilterLowPass_32f_C3R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_f32_c4,
    f32,
    C4,
    nppiFilterLowPass_32f_C4R_Ctx
);
impl_filter_masked_kernel!(
    filter_low_pass_f32_ac4,
    f32,
    AC4,
    nppiFilterLowPass_32f_AC4R_Ctx
);

impl_filter_masked_kernel_border!(
    filter_low_pass_border_u8_c1,
    u8,
    C1,
    nppiFilterLowPassBorder_8u_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_u8_c3,
    u8,
    C3,
    nppiFilterLowPassBorder_8u_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_u8_c4,
    u8,
    C4,
    nppiFilterLowPassBorder_8u_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_u8_ac4,
    u8,
    AC4,
    nppiFilterLowPassBorder_8u_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_u16_c1,
    u16,
    C1,
    nppiFilterLowPassBorder_16u_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_u16_c3,
    u16,
    C3,
    nppiFilterLowPassBorder_16u_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_u16_c4,
    u16,
    C4,
    nppiFilterLowPassBorder_16u_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_u16_ac4,
    u16,
    AC4,
    nppiFilterLowPassBorder_16u_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_i16_c1,
    i16,
    C1,
    nppiFilterLowPassBorder_16s_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_i16_c3,
    i16,
    C3,
    nppiFilterLowPassBorder_16s_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_i16_c4,
    i16,
    C4,
    nppiFilterLowPassBorder_16s_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_i16_ac4,
    i16,
    AC4,
    nppiFilterLowPassBorder_16s_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_f32_c1,
    f32,
    C1,
    nppiFilterLowPassBorder_32f_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_f32_c3,
    f32,
    C3,
    nppiFilterLowPassBorder_32f_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_f32_c4,
    f32,
    C4,
    nppiFilterLowPassBorder_32f_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_low_pass_border_f32_ac4,
    f32,
    AC4,
    nppiFilterLowPassBorder_32f_AC4R_Ctx
);

pub trait FilterLowPass<L: ChannelLayout>: DataTypeLike {
    fn filter_low_pass(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

macro_rules! impl_filter_low_pass_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterLowPass<$layout> for $pixel_ty {
            fn filter_low_pass(
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

impl_filter_low_pass_dispatch!(u8, C1, filter_low_pass_u8_c1);
impl_filter_low_pass_dispatch!(u8, C3, filter_low_pass_u8_c3);
impl_filter_low_pass_dispatch!(u8, C4, filter_low_pass_u8_c4);
impl_filter_low_pass_dispatch!(u8, AC4, filter_low_pass_u8_ac4);
impl_filter_low_pass_dispatch!(u16, C1, filter_low_pass_u16_c1);
impl_filter_low_pass_dispatch!(u16, C3, filter_low_pass_u16_c3);
impl_filter_low_pass_dispatch!(u16, C4, filter_low_pass_u16_c4);
impl_filter_low_pass_dispatch!(u16, AC4, filter_low_pass_u16_ac4);
impl_filter_low_pass_dispatch!(i16, C1, filter_low_pass_i16_c1);
impl_filter_low_pass_dispatch!(i16, C3, filter_low_pass_i16_c3);
impl_filter_low_pass_dispatch!(i16, C4, filter_low_pass_i16_c4);
impl_filter_low_pass_dispatch!(i16, AC4, filter_low_pass_i16_ac4);
impl_filter_low_pass_dispatch!(f32, C1, filter_low_pass_f32_c1);
impl_filter_low_pass_dispatch!(f32, C3, filter_low_pass_f32_c3);
impl_filter_low_pass_dispatch!(f32, C4, filter_low_pass_f32_c4);
impl_filter_low_pass_dispatch!(f32, AC4, filter_low_pass_f32_ac4);

pub fn filter_low_pass<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: MaskSize,
) -> Result<()>
where
    T: FilterLowPass<L>,
    L: ChannelLayout,
{
    T::filter_low_pass(stream_context, source, destination, mask_size)
}

pub trait FilterLowPassBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_low_pass_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_low_pass_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterLowPassBorder<$layout> for $pixel_ty {
            fn filter_low_pass_border(
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

impl_filter_low_pass_border_dispatch!(u8, C1, filter_low_pass_border_u8_c1);
impl_filter_low_pass_border_dispatch!(u8, C3, filter_low_pass_border_u8_c3);
impl_filter_low_pass_border_dispatch!(u8, C4, filter_low_pass_border_u8_c4);
impl_filter_low_pass_border_dispatch!(u8, AC4, filter_low_pass_border_u8_ac4);
impl_filter_low_pass_border_dispatch!(u16, C1, filter_low_pass_border_u16_c1);
impl_filter_low_pass_border_dispatch!(u16, C3, filter_low_pass_border_u16_c3);
impl_filter_low_pass_border_dispatch!(u16, C4, filter_low_pass_border_u16_c4);
impl_filter_low_pass_border_dispatch!(u16, AC4, filter_low_pass_border_u16_ac4);
impl_filter_low_pass_border_dispatch!(i16, C1, filter_low_pass_border_i16_c1);
impl_filter_low_pass_border_dispatch!(i16, C3, filter_low_pass_border_i16_c3);
impl_filter_low_pass_border_dispatch!(i16, C4, filter_low_pass_border_i16_c4);
impl_filter_low_pass_border_dispatch!(i16, AC4, filter_low_pass_border_i16_ac4);
impl_filter_low_pass_border_dispatch!(f32, C1, filter_low_pass_border_f32_c1);
impl_filter_low_pass_border_dispatch!(f32, C3, filter_low_pass_border_f32_c3);
impl_filter_low_pass_border_dispatch!(f32, C4, filter_low_pass_border_f32_c4);
impl_filter_low_pass_border_dispatch!(f32, AC4, filter_low_pass_border_f32_ac4);

pub fn filter_low_pass_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterLowPassBorder<L>,
    L: ChannelLayout,
{
    T::filter_low_pass_border(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}
