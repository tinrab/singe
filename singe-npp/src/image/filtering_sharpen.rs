use super::*;

impl_filter_directional!(filter_sharpen_u8_c1, u8, C1, nppiFilterSharpen_8u_C1R_Ctx);
impl_filter_directional!(filter_sharpen_u8_c3, u8, C3, nppiFilterSharpen_8u_C3R_Ctx);
impl_filter_directional!(filter_sharpen_u8_c4, u8, C4, nppiFilterSharpen_8u_C4R_Ctx);
impl_filter_directional!(
    filter_sharpen_u8_ac4,
    u8,
    AC4,
    nppiFilterSharpen_8u_AC4R_Ctx
);
impl_filter_directional!(
    filter_sharpen_u16_c1,
    u16,
    C1,
    nppiFilterSharpen_16u_C1R_Ctx
);
impl_filter_directional!(
    filter_sharpen_u16_c3,
    u16,
    C3,
    nppiFilterSharpen_16u_C3R_Ctx
);
impl_filter_directional!(
    filter_sharpen_u16_c4,
    u16,
    C4,
    nppiFilterSharpen_16u_C4R_Ctx
);
impl_filter_directional!(
    filter_sharpen_u16_ac4,
    u16,
    AC4,
    nppiFilterSharpen_16u_AC4R_Ctx
);
impl_filter_directional!(
    filter_sharpen_i16_c1,
    i16,
    C1,
    nppiFilterSharpen_16s_C1R_Ctx
);
impl_filter_directional!(
    filter_sharpen_i16_c3,
    i16,
    C3,
    nppiFilterSharpen_16s_C3R_Ctx
);
impl_filter_directional!(
    filter_sharpen_i16_c4,
    i16,
    C4,
    nppiFilterSharpen_16s_C4R_Ctx
);
impl_filter_directional!(
    filter_sharpen_i16_ac4,
    i16,
    AC4,
    nppiFilterSharpen_16s_AC4R_Ctx
);
impl_filter_directional!(
    filter_sharpen_f32_c1,
    f32,
    C1,
    nppiFilterSharpen_32f_C1R_Ctx
);
impl_filter_directional!(
    filter_sharpen_f32_c3,
    f32,
    C3,
    nppiFilterSharpen_32f_C3R_Ctx
);
impl_filter_directional!(
    filter_sharpen_f32_c4,
    f32,
    C4,
    nppiFilterSharpen_32f_C4R_Ctx
);
impl_filter_directional!(
    filter_sharpen_f32_ac4,
    f32,
    AC4,
    nppiFilterSharpen_32f_AC4R_Ctx
);

impl_filter_directional_border!(
    filter_sharpen_border_u8_c1,
    u8,
    C1,
    nppiFilterSharpenBorder_8u_C1R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_u8_c3,
    u8,
    C3,
    nppiFilterSharpenBorder_8u_C3R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_u8_c4,
    u8,
    C4,
    nppiFilterSharpenBorder_8u_C4R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_u8_ac4,
    u8,
    AC4,
    nppiFilterSharpenBorder_8u_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_u16_c1,
    u16,
    C1,
    nppiFilterSharpenBorder_16u_C1R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_u16_c3,
    u16,
    C3,
    nppiFilterSharpenBorder_16u_C3R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_u16_c4,
    u16,
    C4,
    nppiFilterSharpenBorder_16u_C4R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_u16_ac4,
    u16,
    AC4,
    nppiFilterSharpenBorder_16u_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_i16_c1,
    i16,
    C1,
    nppiFilterSharpenBorder_16s_C1R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_i16_c3,
    i16,
    C3,
    nppiFilterSharpenBorder_16s_C3R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_i16_c4,
    i16,
    C4,
    nppiFilterSharpenBorder_16s_C4R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_i16_ac4,
    i16,
    AC4,
    nppiFilterSharpenBorder_16s_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_f32_c1,
    f32,
    C1,
    nppiFilterSharpenBorder_32f_C1R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_f32_c3,
    f32,
    C3,
    nppiFilterSharpenBorder_32f_C3R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_f32_c4,
    f32,
    C4,
    nppiFilterSharpenBorder_32f_C4R_Ctx
);
impl_filter_directional_border!(
    filter_sharpen_border_f32_ac4,
    f32,
    AC4,
    nppiFilterSharpenBorder_32f_AC4R_Ctx
);

pub trait FilterSharpen<L: ChannelLayout>: DataTypeLike {
    fn filter_sharpen(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

macro_rules! impl_filter_sharpen_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterSharpen<$layout> for $pixel_ty {
            fn filter_sharpen(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()> {
                $name(stream_context, source, destination)
            }
        }
    };
}

impl_filter_sharpen_dispatch!(u8, C1, filter_sharpen_u8_c1);
impl_filter_sharpen_dispatch!(u8, C3, filter_sharpen_u8_c3);
impl_filter_sharpen_dispatch!(u8, C4, filter_sharpen_u8_c4);
impl_filter_sharpen_dispatch!(u8, AC4, filter_sharpen_u8_ac4);
impl_filter_sharpen_dispatch!(u16, C1, filter_sharpen_u16_c1);
impl_filter_sharpen_dispatch!(u16, C3, filter_sharpen_u16_c3);
impl_filter_sharpen_dispatch!(u16, C4, filter_sharpen_u16_c4);
impl_filter_sharpen_dispatch!(u16, AC4, filter_sharpen_u16_ac4);
impl_filter_sharpen_dispatch!(i16, C1, filter_sharpen_i16_c1);
impl_filter_sharpen_dispatch!(i16, C3, filter_sharpen_i16_c3);
impl_filter_sharpen_dispatch!(i16, C4, filter_sharpen_i16_c4);
impl_filter_sharpen_dispatch!(i16, AC4, filter_sharpen_i16_ac4);
impl_filter_sharpen_dispatch!(f32, C1, filter_sharpen_f32_c1);
impl_filter_sharpen_dispatch!(f32, C3, filter_sharpen_f32_c3);
impl_filter_sharpen_dispatch!(f32, C4, filter_sharpen_f32_c4);
impl_filter_sharpen_dispatch!(f32, AC4, filter_sharpen_f32_ac4);

pub fn filter_sharpen<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: FilterSharpen<L>,
    L: ChannelLayout,
{
    T::filter_sharpen(stream_context, source, destination)
}

pub trait FilterSharpenBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_sharpen_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_sharpen_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterSharpenBorder<$layout> for $pixel_ty {
            fn filter_sharpen_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }
        }
    };
}

impl_filter_sharpen_border_dispatch!(u8, C1, filter_sharpen_border_u8_c1);
impl_filter_sharpen_border_dispatch!(u8, C3, filter_sharpen_border_u8_c3);
impl_filter_sharpen_border_dispatch!(u8, C4, filter_sharpen_border_u8_c4);
impl_filter_sharpen_border_dispatch!(u8, AC4, filter_sharpen_border_u8_ac4);
impl_filter_sharpen_border_dispatch!(u16, C1, filter_sharpen_border_u16_c1);
impl_filter_sharpen_border_dispatch!(u16, C3, filter_sharpen_border_u16_c3);
impl_filter_sharpen_border_dispatch!(u16, C4, filter_sharpen_border_u16_c4);
impl_filter_sharpen_border_dispatch!(u16, AC4, filter_sharpen_border_u16_ac4);
impl_filter_sharpen_border_dispatch!(i16, C1, filter_sharpen_border_i16_c1);
impl_filter_sharpen_border_dispatch!(i16, C3, filter_sharpen_border_i16_c3);
impl_filter_sharpen_border_dispatch!(i16, C4, filter_sharpen_border_i16_c4);
impl_filter_sharpen_border_dispatch!(i16, AC4, filter_sharpen_border_i16_ac4);
impl_filter_sharpen_border_dispatch!(f32, C1, filter_sharpen_border_f32_c1);
impl_filter_sharpen_border_dispatch!(f32, C3, filter_sharpen_border_f32_c3);
impl_filter_sharpen_border_dispatch!(f32, C4, filter_sharpen_border_f32_c4);
impl_filter_sharpen_border_dispatch!(f32, AC4, filter_sharpen_border_f32_ac4);

pub fn filter_sharpen_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterSharpenBorder<L>,
    L: ChannelLayout,
{
    T::filter_sharpen_border(
        stream_context,
        source,
        source_offset,
        destination,
        border_type,
    )
}

impl_filter_unsharp_border!(
    filter_unsharp_border_u8_c1_buffer_size,
    filter_unsharp_border_u8_c1,
    u8,
    C1,
    nppiFilterUnsharpGetBufferSize_8u_C1R,
    nppiFilterUnsharpBorder_8u_C1R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_u8_c3_buffer_size,
    filter_unsharp_border_u8_c3,
    u8,
    C3,
    nppiFilterUnsharpGetBufferSize_8u_C3R,
    nppiFilterUnsharpBorder_8u_C3R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_u8_c4_buffer_size,
    filter_unsharp_border_u8_c4,
    u8,
    C4,
    nppiFilterUnsharpGetBufferSize_8u_C4R,
    nppiFilterUnsharpBorder_8u_C4R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_u8_ac4_buffer_size,
    filter_unsharp_border_u8_ac4,
    u8,
    AC4,
    nppiFilterUnsharpGetBufferSize_8u_AC4R,
    nppiFilterUnsharpBorder_8u_AC4R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_u16_c1_buffer_size,
    filter_unsharp_border_u16_c1,
    u16,
    C1,
    nppiFilterUnsharpGetBufferSize_16u_C1R,
    nppiFilterUnsharpBorder_16u_C1R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_u16_c3_buffer_size,
    filter_unsharp_border_u16_c3,
    u16,
    C3,
    nppiFilterUnsharpGetBufferSize_16u_C3R,
    nppiFilterUnsharpBorder_16u_C3R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_u16_c4_buffer_size,
    filter_unsharp_border_u16_c4,
    u16,
    C4,
    nppiFilterUnsharpGetBufferSize_16u_C4R,
    nppiFilterUnsharpBorder_16u_C4R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_u16_ac4_buffer_size,
    filter_unsharp_border_u16_ac4,
    u16,
    AC4,
    nppiFilterUnsharpGetBufferSize_16u_AC4R,
    nppiFilterUnsharpBorder_16u_AC4R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_i16_c1_buffer_size,
    filter_unsharp_border_i16_c1,
    i16,
    C1,
    nppiFilterUnsharpGetBufferSize_16s_C1R,
    nppiFilterUnsharpBorder_16s_C1R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_i16_c3_buffer_size,
    filter_unsharp_border_i16_c3,
    i16,
    C3,
    nppiFilterUnsharpGetBufferSize_16s_C3R,
    nppiFilterUnsharpBorder_16s_C3R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_i16_c4_buffer_size,
    filter_unsharp_border_i16_c4,
    i16,
    C4,
    nppiFilterUnsharpGetBufferSize_16s_C4R,
    nppiFilterUnsharpBorder_16s_C4R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_i16_ac4_buffer_size,
    filter_unsharp_border_i16_ac4,
    i16,
    AC4,
    nppiFilterUnsharpGetBufferSize_16s_AC4R,
    nppiFilterUnsharpBorder_16s_AC4R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_f32_c1_buffer_size,
    filter_unsharp_border_f32_c1,
    f32,
    C1,
    nppiFilterUnsharpGetBufferSize_32f_C1R,
    nppiFilterUnsharpBorder_32f_C1R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_f32_c3_buffer_size,
    filter_unsharp_border_f32_c3,
    f32,
    C3,
    nppiFilterUnsharpGetBufferSize_32f_C3R,
    nppiFilterUnsharpBorder_32f_C3R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_f32_c4_buffer_size,
    filter_unsharp_border_f32_c4,
    f32,
    C4,
    nppiFilterUnsharpGetBufferSize_32f_C4R,
    nppiFilterUnsharpBorder_32f_C4R_Ctx
);
impl_filter_unsharp_border!(
    filter_unsharp_border_f32_ac4_buffer_size,
    filter_unsharp_border_f32_ac4,
    f32,
    AC4,
    nppiFilterUnsharpGetBufferSize_32f_AC4R,
    nppiFilterUnsharpBorder_32f_AC4R_Ctx
);

pub trait FilterUnsharpBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_unsharp_border_buffer_size(radius: f32, sigma: f32) -> Result<usize>;

    fn filter_unsharp_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_unsharp_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $buffer_size_name:ident, $name:ident) => {
        impl FilterUnsharpBorder<$layout> for $pixel_ty {
            fn filter_unsharp_border_buffer_size(radius: f32, sigma: f32) -> Result<usize> {
                $buffer_size_name(radius, sigma)
            }

            fn filter_unsharp_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                radius: f32,
                sigma: f32,
                weight: f32,
                threshold: f32,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    radius,
                    sigma,
                    weight,
                    threshold,
                    border_type,
                )
            }
        }
    };
}

impl_filter_unsharp_border_dispatch!(
    u8,
    C1,
    filter_unsharp_border_u8_c1_buffer_size,
    filter_unsharp_border_u8_c1
);
impl_filter_unsharp_border_dispatch!(
    u8,
    C3,
    filter_unsharp_border_u8_c3_buffer_size,
    filter_unsharp_border_u8_c3
);
impl_filter_unsharp_border_dispatch!(
    u8,
    C4,
    filter_unsharp_border_u8_c4_buffer_size,
    filter_unsharp_border_u8_c4
);
impl_filter_unsharp_border_dispatch!(
    u8,
    AC4,
    filter_unsharp_border_u8_ac4_buffer_size,
    filter_unsharp_border_u8_ac4
);
impl_filter_unsharp_border_dispatch!(
    u16,
    C1,
    filter_unsharp_border_u16_c1_buffer_size,
    filter_unsharp_border_u16_c1
);
impl_filter_unsharp_border_dispatch!(
    u16,
    C3,
    filter_unsharp_border_u16_c3_buffer_size,
    filter_unsharp_border_u16_c3
);
impl_filter_unsharp_border_dispatch!(
    u16,
    C4,
    filter_unsharp_border_u16_c4_buffer_size,
    filter_unsharp_border_u16_c4
);
impl_filter_unsharp_border_dispatch!(
    u16,
    AC4,
    filter_unsharp_border_u16_ac4_buffer_size,
    filter_unsharp_border_u16_ac4
);
impl_filter_unsharp_border_dispatch!(
    i16,
    C1,
    filter_unsharp_border_i16_c1_buffer_size,
    filter_unsharp_border_i16_c1
);
impl_filter_unsharp_border_dispatch!(
    i16,
    C3,
    filter_unsharp_border_i16_c3_buffer_size,
    filter_unsharp_border_i16_c3
);
impl_filter_unsharp_border_dispatch!(
    i16,
    C4,
    filter_unsharp_border_i16_c4_buffer_size,
    filter_unsharp_border_i16_c4
);
impl_filter_unsharp_border_dispatch!(
    i16,
    AC4,
    filter_unsharp_border_i16_ac4_buffer_size,
    filter_unsharp_border_i16_ac4
);
impl_filter_unsharp_border_dispatch!(
    f32,
    C1,
    filter_unsharp_border_f32_c1_buffer_size,
    filter_unsharp_border_f32_c1
);
impl_filter_unsharp_border_dispatch!(
    f32,
    C3,
    filter_unsharp_border_f32_c3_buffer_size,
    filter_unsharp_border_f32_c3
);
impl_filter_unsharp_border_dispatch!(
    f32,
    C4,
    filter_unsharp_border_f32_c4_buffer_size,
    filter_unsharp_border_f32_c4
);
impl_filter_unsharp_border_dispatch!(
    f32,
    AC4,
    filter_unsharp_border_f32_ac4_buffer_size,
    filter_unsharp_border_f32_ac4
);

pub fn filter_unsharp_border_buffer_size<T, L>(radius: f32, sigma: f32) -> Result<usize>
where
    T: FilterUnsharpBorder<L>,
    L: ChannelLayout,
{
    T::filter_unsharp_border_buffer_size(radius, sigma)
}

pub fn filter_unsharp_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    radius: f32,
    sigma: f32,
    weight: f32,
    threshold: f32,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterUnsharpBorder<L>,
    L: ChannelLayout,
{
    T::filter_unsharp_border(
        stream_context,
        source,
        source_offset,
        destination,
        radius,
        sigma,
        weight,
        threshold,
        border_type,
    )
}
