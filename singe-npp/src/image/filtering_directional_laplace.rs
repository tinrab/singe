use super::*;

impl_filter_masked_kernel!(filter_laplace_u8_c1, u8, C1, nppiFilterLaplace_8u_C1R_Ctx);
impl_filter_masked_kernel!(filter_laplace_u8_c3, u8, C3, nppiFilterLaplace_8u_C3R_Ctx);
impl_filter_masked_kernel!(filter_laplace_u8_c4, u8, C4, nppiFilterLaplace_8u_C4R_Ctx);
impl_filter_masked_kernel!(
    filter_laplace_u8_ac4,
    u8,
    AC4,
    nppiFilterLaplace_8u_AC4R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_i16_c1,
    i16,
    C1,
    nppiFilterLaplace_16s_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_i16_c3,
    i16,
    C3,
    nppiFilterLaplace_16s_C3R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_i16_c4,
    i16,
    C4,
    nppiFilterLaplace_16s_C4R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_i16_ac4,
    i16,
    AC4,
    nppiFilterLaplace_16s_AC4R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_f32_c1,
    f32,
    C1,
    nppiFilterLaplace_32f_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_f32_c3,
    f32,
    C3,
    nppiFilterLaplace_32f_C3R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_f32_c4,
    f32,
    C4,
    nppiFilterLaplace_32f_C4R_Ctx
);
impl_filter_masked_kernel!(
    filter_laplace_f32_ac4,
    f32,
    AC4,
    nppiFilterLaplace_32f_AC4R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_laplace_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterLaplace_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_laplace_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterLaplace_8s16s_C1R_Ctx
);

impl_filter_masked_kernel_border!(
    filter_laplace_border_u8_c1,
    u8,
    C1,
    nppiFilterLaplaceBorder_8u_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_u8_c3,
    u8,
    C3,
    nppiFilterLaplaceBorder_8u_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_u8_c4,
    u8,
    C4,
    nppiFilterLaplaceBorder_8u_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_u8_ac4,
    u8,
    AC4,
    nppiFilterLaplaceBorder_8u_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_i16_c1,
    i16,
    C1,
    nppiFilterLaplaceBorder_16s_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_i16_c3,
    i16,
    C3,
    nppiFilterLaplaceBorder_16s_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_i16_c4,
    i16,
    C4,
    nppiFilterLaplaceBorder_16s_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_i16_ac4,
    i16,
    AC4,
    nppiFilterLaplaceBorder_16s_AC4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_f32_c1,
    f32,
    C1,
    nppiFilterLaplaceBorder_32f_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_f32_c3,
    f32,
    C3,
    nppiFilterLaplaceBorder_32f_C3R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_f32_c4,
    f32,
    C4,
    nppiFilterLaplaceBorder_32f_C4R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_laplace_border_f32_ac4,
    f32,
    AC4,
    nppiFilterLaplaceBorder_32f_AC4R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_laplace_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterLaplaceBorder_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_laplace_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterLaplaceBorder_8s16s_C1R_Ctx
);

pub trait FilterLaplace<L: ChannelLayout>: DataTypeLike {
    fn filter_laplace(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

macro_rules! impl_filter_laplace_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterLaplace<$layout> for $pixel_ty {
            fn filter_laplace(
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

impl_filter_laplace_dispatch!(u8, C1, filter_laplace_u8_c1);
impl_filter_laplace_dispatch!(u8, C3, filter_laplace_u8_c3);
impl_filter_laplace_dispatch!(u8, C4, filter_laplace_u8_c4);
impl_filter_laplace_dispatch!(u8, AC4, filter_laplace_u8_ac4);
impl_filter_laplace_dispatch!(i16, C1, filter_laplace_i16_c1);
impl_filter_laplace_dispatch!(i16, C3, filter_laplace_i16_c3);
impl_filter_laplace_dispatch!(i16, C4, filter_laplace_i16_c4);
impl_filter_laplace_dispatch!(i16, AC4, filter_laplace_i16_ac4);
impl_filter_laplace_dispatch!(f32, C1, filter_laplace_f32_c1);
impl_filter_laplace_dispatch!(f32, C3, filter_laplace_f32_c3);
impl_filter_laplace_dispatch!(f32, C4, filter_laplace_f32_c4);
impl_filter_laplace_dispatch!(f32, AC4, filter_laplace_f32_ac4);

pub fn filter_laplace<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: MaskSize,
) -> Result<()>
where
    T: FilterLaplace<L>,
    L: ChannelLayout,
{
    T::filter_laplace(stream_context, source, destination, mask_size)
}

pub trait FilterLaplaceC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_laplace_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterLaplaceC1,
    filter_laplace_typed,
    u8,
    i16,
    filter_laplace_u8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterLaplaceC1,
    filter_laplace_typed,
    i8,
    i16,
    filter_laplace_i8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterLaplaceC1,
    filter_laplace_typed,
    f32,
    f32,
    filter_laplace_f32_c1
);

pub fn filter_laplace_typed<T: FilterLaplaceC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    T::filter_laplace_typed(stream_context, source, destination, mask_size)
}

pub trait FilterLaplaceBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_laplace_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_laplace_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterLaplaceBorder<$layout> for $pixel_ty {
            fn filter_laplace_border(
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

impl_filter_laplace_border_dispatch!(u8, C1, filter_laplace_border_u8_c1);
impl_filter_laplace_border_dispatch!(u8, C3, filter_laplace_border_u8_c3);
impl_filter_laplace_border_dispatch!(u8, C4, filter_laplace_border_u8_c4);
impl_filter_laplace_border_dispatch!(u8, AC4, filter_laplace_border_u8_ac4);
impl_filter_laplace_border_dispatch!(i16, C1, filter_laplace_border_i16_c1);
impl_filter_laplace_border_dispatch!(i16, C3, filter_laplace_border_i16_c3);
impl_filter_laplace_border_dispatch!(i16, C4, filter_laplace_border_i16_c4);
impl_filter_laplace_border_dispatch!(i16, AC4, filter_laplace_border_i16_ac4);
impl_filter_laplace_border_dispatch!(f32, C1, filter_laplace_border_f32_c1);
impl_filter_laplace_border_dispatch!(f32, C3, filter_laplace_border_f32_c3);
impl_filter_laplace_border_dispatch!(f32, C4, filter_laplace_border_f32_c4);
impl_filter_laplace_border_dispatch!(f32, AC4, filter_laplace_border_f32_ac4);

pub fn filter_laplace_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterLaplaceBorder<L>,
    L: ChannelLayout,
{
    T::filter_laplace_border(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}

pub trait FilterLaplaceBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_laplace_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterLaplaceBorderC1,
    filter_laplace_border_typed,
    u8,
    i16,
    filter_laplace_border_u8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterLaplaceBorderC1,
    filter_laplace_border_typed,
    i8,
    i16,
    filter_laplace_border_i8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterLaplaceBorderC1,
    filter_laplace_border_typed,
    f32,
    f32,
    filter_laplace_border_f32_c1
);

pub fn filter_laplace_border_typed<T: FilterLaplaceBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    T::filter_laplace_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}
