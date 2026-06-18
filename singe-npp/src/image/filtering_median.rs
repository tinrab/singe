use super::*;

impl_filter_median!(
    filter_median_u8_c1_buffer_size,
    filter_median_u8_c1,
    u8,
    C1,
    nppiFilterMedianGetBufferSize_8u_C1R_Ctx,
    nppiFilterMedian_8u_C1R_Ctx
);
impl_filter_median!(
    filter_median_u8_c3_buffer_size,
    filter_median_u8_c3,
    u8,
    C3,
    nppiFilterMedianGetBufferSize_8u_C3R_Ctx,
    nppiFilterMedian_8u_C3R_Ctx
);
impl_filter_median!(
    filter_median_u8_c4_buffer_size,
    filter_median_u8_c4,
    u8,
    C4,
    nppiFilterMedianGetBufferSize_8u_C4R_Ctx,
    nppiFilterMedian_8u_C4R_Ctx
);
impl_filter_median!(
    filter_median_u8_ac4_buffer_size,
    filter_median_u8_ac4,
    u8,
    AC4,
    nppiFilterMedianGetBufferSize_8u_AC4R_Ctx,
    nppiFilterMedian_8u_AC4R_Ctx
);
impl_filter_median!(
    filter_median_u16_c1_buffer_size,
    filter_median_u16_c1,
    u16,
    C1,
    nppiFilterMedianGetBufferSize_16u_C1R_Ctx,
    nppiFilterMedian_16u_C1R_Ctx
);
impl_filter_median!(
    filter_median_u16_c3_buffer_size,
    filter_median_u16_c3,
    u16,
    C3,
    nppiFilterMedianGetBufferSize_16u_C3R_Ctx,
    nppiFilterMedian_16u_C3R_Ctx
);
impl_filter_median!(
    filter_median_u16_c4_buffer_size,
    filter_median_u16_c4,
    u16,
    C4,
    nppiFilterMedianGetBufferSize_16u_C4R_Ctx,
    nppiFilterMedian_16u_C4R_Ctx
);
impl_filter_median!(
    filter_median_u16_ac4_buffer_size,
    filter_median_u16_ac4,
    u16,
    AC4,
    nppiFilterMedianGetBufferSize_16u_AC4R_Ctx,
    nppiFilterMedian_16u_AC4R_Ctx
);
impl_filter_median!(
    filter_median_i16_c1_buffer_size,
    filter_median_i16_c1,
    i16,
    C1,
    nppiFilterMedianGetBufferSize_16s_C1R_Ctx,
    nppiFilterMedian_16s_C1R_Ctx
);
impl_filter_median!(
    filter_median_i16_c3_buffer_size,
    filter_median_i16_c3,
    i16,
    C3,
    nppiFilterMedianGetBufferSize_16s_C3R_Ctx,
    nppiFilterMedian_16s_C3R_Ctx
);
impl_filter_median!(
    filter_median_i16_c4_buffer_size,
    filter_median_i16_c4,
    i16,
    C4,
    nppiFilterMedianGetBufferSize_16s_C4R_Ctx,
    nppiFilterMedian_16s_C4R_Ctx
);
impl_filter_median!(
    filter_median_i16_ac4_buffer_size,
    filter_median_i16_ac4,
    i16,
    AC4,
    nppiFilterMedianGetBufferSize_16s_AC4R_Ctx,
    nppiFilterMedian_16s_AC4R_Ctx
);
impl_filter_median!(
    filter_median_f32_c1_buffer_size,
    filter_median_f32_c1,
    f32,
    C1,
    nppiFilterMedianGetBufferSize_32f_C1R_Ctx,
    nppiFilterMedian_32f_C1R_Ctx
);
impl_filter_median!(
    filter_median_f32_c3_buffer_size,
    filter_median_f32_c3,
    f32,
    C3,
    nppiFilterMedianGetBufferSize_32f_C3R_Ctx,
    nppiFilterMedian_32f_C3R_Ctx
);
impl_filter_median!(
    filter_median_f32_c4_buffer_size,
    filter_median_f32_c4,
    f32,
    C4,
    nppiFilterMedianGetBufferSize_32f_C4R_Ctx,
    nppiFilterMedian_32f_C4R_Ctx
);
impl_filter_median!(
    filter_median_f32_ac4_buffer_size,
    filter_median_f32_ac4,
    f32,
    AC4,
    nppiFilterMedianGetBufferSize_32f_AC4R_Ctx,
    nppiFilterMedian_32f_AC4R_Ctx
);

impl_filter_median_border!(
    filter_median_border_u8_c1_buffer_size,
    filter_median_border_u8_c1,
    u8,
    C1,
    nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx,
    nppiFilterMedianBorder_8u_C1R_Ctx
);
impl_filter_median_border!(
    filter_median_border_u8_c3_buffer_size,
    filter_median_border_u8_c3,
    u8,
    C3,
    nppiFilterMedianBorderGetBufferSize_8u_C3R_Ctx,
    nppiFilterMedianBorder_8u_C3R_Ctx
);
impl_filter_median_border!(
    filter_median_border_u8_c4_buffer_size,
    filter_median_border_u8_c4,
    u8,
    C4,
    nppiFilterMedianBorderGetBufferSize_8u_C4R_Ctx,
    nppiFilterMedianBorder_8u_C4R_Ctx
);
impl_filter_median_border!(
    filter_median_border_u8_ac4_buffer_size,
    filter_median_border_u8_ac4,
    u8,
    AC4,
    nppiFilterMedianBorderGetBufferSize_8u_AC4R_Ctx,
    nppiFilterMedianBorder_8u_AC4R_Ctx
);
impl_filter_median_border!(
    filter_median_border_u16_c1_buffer_size,
    filter_median_border_u16_c1,
    u16,
    C1,
    nppiFilterMedianBorderGetBufferSize_16u_C1R_Ctx,
    nppiFilterMedianBorder_16u_C1R_Ctx
);
impl_filter_median_border!(
    filter_median_border_u16_c3_buffer_size,
    filter_median_border_u16_c3,
    u16,
    C3,
    nppiFilterMedianBorderGetBufferSize_16u_C3R_Ctx,
    nppiFilterMedianBorder_16u_C3R_Ctx
);
impl_filter_median_border!(
    filter_median_border_u16_c4_buffer_size,
    filter_median_border_u16_c4,
    u16,
    C4,
    nppiFilterMedianBorderGetBufferSize_16u_C4R_Ctx,
    nppiFilterMedianBorder_16u_C4R_Ctx
);
impl_filter_median_border!(
    filter_median_border_u16_ac4_buffer_size,
    filter_median_border_u16_ac4,
    u16,
    AC4,
    nppiFilterMedianBorderGetBufferSize_16u_AC4R_Ctx,
    nppiFilterMedianBorder_16u_AC4R_Ctx
);
impl_filter_median_border!(
    filter_median_border_i16_c1_buffer_size,
    filter_median_border_i16_c1,
    i16,
    C1,
    nppiFilterMedianBorderGetBufferSize_16s_C1R_Ctx,
    nppiFilterMedianBorder_16s_C1R_Ctx
);
impl_filter_median_border!(
    filter_median_border_i16_c3_buffer_size,
    filter_median_border_i16_c3,
    i16,
    C3,
    nppiFilterMedianBorderGetBufferSize_16s_C3R_Ctx,
    nppiFilterMedianBorder_16s_C3R_Ctx
);
impl_filter_median_border!(
    filter_median_border_i16_c4_buffer_size,
    filter_median_border_i16_c4,
    i16,
    C4,
    nppiFilterMedianBorderGetBufferSize_16s_C4R_Ctx,
    nppiFilterMedianBorder_16s_C4R_Ctx
);
impl_filter_median_border!(
    filter_median_border_i16_ac4_buffer_size,
    filter_median_border_i16_ac4,
    i16,
    AC4,
    nppiFilterMedianBorderGetBufferSize_16s_AC4R_Ctx,
    nppiFilterMedianBorder_16s_AC4R_Ctx
);
impl_filter_median_border!(
    filter_median_border_f32_c1_buffer_size,
    filter_median_border_f32_c1,
    f32,
    C1,
    nppiFilterMedianBorderGetBufferSize_32f_C1R_Ctx,
    nppiFilterMedianBorder_32f_C1R_Ctx
);
impl_filter_median_border!(
    filter_median_border_f32_c3_buffer_size,
    filter_median_border_f32_c3,
    f32,
    C3,
    nppiFilterMedianBorderGetBufferSize_32f_C3R_Ctx,
    nppiFilterMedianBorder_32f_C3R_Ctx
);
impl_filter_median_border!(
    filter_median_border_f32_c4_buffer_size,
    filter_median_border_f32_c4,
    f32,
    C4,
    nppiFilterMedianBorderGetBufferSize_32f_C4R_Ctx,
    nppiFilterMedianBorder_32f_C4R_Ctx
);
impl_filter_median_border!(
    filter_median_border_f32_ac4_buffer_size,
    filter_median_border_f32_ac4,
    f32,
    AC4,
    nppiFilterMedianBorderGetBufferSize_32f_AC4R_Ctx,
    nppiFilterMedianBorder_32f_AC4R_Ctx
);

pub trait FilterMedian<L: ChannelLayout>: DataTypeLike {
    fn filter_median_buffer_size(
        stream_context: &StreamContext,
        roi: Size,
        mask_size: Size,
    ) -> Result<usize>;

    fn filter_median(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

macro_rules! impl_filter_median_dispatch {
    ($pixel_ty:ty, $layout:ty, $buffer_size_name:ident, $name:ident) => {
        impl FilterMedian<$layout> for $pixel_ty {
            fn filter_median_buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                mask_size: Size,
            ) -> Result<usize> {
                $buffer_size_name(stream_context, roi, mask_size)
            }

            fn filter_median(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                $name(stream_context, source, destination, mask_size, anchor)
            }
        }
    };
}

impl_filter_median_dispatch!(u8, C1, filter_median_u8_c1_buffer_size, filter_median_u8_c1);
impl_filter_median_dispatch!(u8, C3, filter_median_u8_c3_buffer_size, filter_median_u8_c3);
impl_filter_median_dispatch!(u8, C4, filter_median_u8_c4_buffer_size, filter_median_u8_c4);
impl_filter_median_dispatch!(
    u8,
    AC4,
    filter_median_u8_ac4_buffer_size,
    filter_median_u8_ac4
);
impl_filter_median_dispatch!(
    u16,
    C1,
    filter_median_u16_c1_buffer_size,
    filter_median_u16_c1
);
impl_filter_median_dispatch!(
    u16,
    C3,
    filter_median_u16_c3_buffer_size,
    filter_median_u16_c3
);
impl_filter_median_dispatch!(
    u16,
    C4,
    filter_median_u16_c4_buffer_size,
    filter_median_u16_c4
);
impl_filter_median_dispatch!(
    u16,
    AC4,
    filter_median_u16_ac4_buffer_size,
    filter_median_u16_ac4
);
impl_filter_median_dispatch!(
    i16,
    C1,
    filter_median_i16_c1_buffer_size,
    filter_median_i16_c1
);
impl_filter_median_dispatch!(
    i16,
    C3,
    filter_median_i16_c3_buffer_size,
    filter_median_i16_c3
);
impl_filter_median_dispatch!(
    i16,
    C4,
    filter_median_i16_c4_buffer_size,
    filter_median_i16_c4
);
impl_filter_median_dispatch!(
    i16,
    AC4,
    filter_median_i16_ac4_buffer_size,
    filter_median_i16_ac4
);
impl_filter_median_dispatch!(
    f32,
    C1,
    filter_median_f32_c1_buffer_size,
    filter_median_f32_c1
);
impl_filter_median_dispatch!(
    f32,
    C3,
    filter_median_f32_c3_buffer_size,
    filter_median_f32_c3
);
impl_filter_median_dispatch!(
    f32,
    C4,
    filter_median_f32_c4_buffer_size,
    filter_median_f32_c4
);
impl_filter_median_dispatch!(
    f32,
    AC4,
    filter_median_f32_ac4_buffer_size,
    filter_median_f32_ac4
);

pub fn filter_median_buffer_size<T, L>(
    stream_context: &StreamContext,
    roi: Size,
    mask_size: Size,
) -> Result<usize>
where
    T: FilterMedian<L>,
    L: ChannelLayout,
{
    T::filter_median_buffer_size(stream_context, roi, mask_size)
}

pub fn filter_median<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: Size,
    anchor: Point,
) -> Result<()>
where
    T: FilterMedian<L>,
    L: ChannelLayout,
{
    T::filter_median(stream_context, source, destination, mask_size, anchor)
}

pub trait FilterMedianBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_median_border_buffer_size(
        stream_context: &StreamContext,
        roi: Size,
        mask_size: Size,
        border_type: BorderType,
    ) -> Result<usize>;

    fn filter_median_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_median_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $buffer_size_name:ident, $name:ident) => {
        impl FilterMedianBorder<$layout> for $pixel_ty {
            fn filter_median_border_buffer_size(
                stream_context: &StreamContext,
                roi: Size,
                mask_size: Size,
                border_type: BorderType,
            ) -> Result<usize> {
                $buffer_size_name(stream_context, roi, mask_size, border_type)
            }

            fn filter_median_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    anchor,
                    border_type,
                )
            }
        }
    };
}

impl_filter_median_border_dispatch!(
    u8,
    C1,
    filter_median_border_u8_c1_buffer_size,
    filter_median_border_u8_c1
);
impl_filter_median_border_dispatch!(
    u8,
    C3,
    filter_median_border_u8_c3_buffer_size,
    filter_median_border_u8_c3
);
impl_filter_median_border_dispatch!(
    u8,
    C4,
    filter_median_border_u8_c4_buffer_size,
    filter_median_border_u8_c4
);
impl_filter_median_border_dispatch!(
    u8,
    AC4,
    filter_median_border_u8_ac4_buffer_size,
    filter_median_border_u8_ac4
);
impl_filter_median_border_dispatch!(
    u16,
    C1,
    filter_median_border_u16_c1_buffer_size,
    filter_median_border_u16_c1
);
impl_filter_median_border_dispatch!(
    u16,
    C3,
    filter_median_border_u16_c3_buffer_size,
    filter_median_border_u16_c3
);
impl_filter_median_border_dispatch!(
    u16,
    C4,
    filter_median_border_u16_c4_buffer_size,
    filter_median_border_u16_c4
);
impl_filter_median_border_dispatch!(
    u16,
    AC4,
    filter_median_border_u16_ac4_buffer_size,
    filter_median_border_u16_ac4
);
impl_filter_median_border_dispatch!(
    i16,
    C1,
    filter_median_border_i16_c1_buffer_size,
    filter_median_border_i16_c1
);
impl_filter_median_border_dispatch!(
    i16,
    C3,
    filter_median_border_i16_c3_buffer_size,
    filter_median_border_i16_c3
);
impl_filter_median_border_dispatch!(
    i16,
    C4,
    filter_median_border_i16_c4_buffer_size,
    filter_median_border_i16_c4
);
impl_filter_median_border_dispatch!(
    i16,
    AC4,
    filter_median_border_i16_ac4_buffer_size,
    filter_median_border_i16_ac4
);
impl_filter_median_border_dispatch!(
    f32,
    C1,
    filter_median_border_f32_c1_buffer_size,
    filter_median_border_f32_c1
);
impl_filter_median_border_dispatch!(
    f32,
    C3,
    filter_median_border_f32_c3_buffer_size,
    filter_median_border_f32_c3
);
impl_filter_median_border_dispatch!(
    f32,
    C4,
    filter_median_border_f32_c4_buffer_size,
    filter_median_border_f32_c4
);
impl_filter_median_border_dispatch!(
    f32,
    AC4,
    filter_median_border_f32_ac4_buffer_size,
    filter_median_border_f32_ac4
);

pub fn filter_median_border_buffer_size<T, L>(
    stream_context: &StreamContext,
    roi: Size,
    mask_size: Size,
    border_type: BorderType,
) -> Result<usize>
where
    T: FilterMedianBorder<L>,
    L: ChannelLayout,
{
    T::filter_median_border_buffer_size(stream_context, roi, mask_size, border_type)
}

pub fn filter_median_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    mask_size: Size,
    anchor: Point,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterMedianBorder<L>,
    L: ChannelLayout,
{
    T::filter_median_border(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        anchor,
        border_type,
    )
}
