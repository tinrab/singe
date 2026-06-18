use super::*;

impl_filter_directional!(
    filter_prewitt_horizontal_u8_c1,
    u8,
    C1,
    nppiFilterPrewittHoriz_8u_C1R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_u8_c3,
    u8,
    C3,
    nppiFilterPrewittHoriz_8u_C3R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_u8_c4,
    u8,
    C4,
    nppiFilterPrewittHoriz_8u_C4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittHoriz_8u_AC4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_i16_c1,
    i16,
    C1,
    nppiFilterPrewittHoriz_16s_C1R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_i16_c3,
    i16,
    C3,
    nppiFilterPrewittHoriz_16s_C3R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_i16_c4,
    i16,
    C4,
    nppiFilterPrewittHoriz_16s_C4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittHoriz_16s_AC4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_f32_c1,
    f32,
    C1,
    nppiFilterPrewittHoriz_32f_C1R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_f32_c3,
    f32,
    C3,
    nppiFilterPrewittHoriz_32f_C3R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_f32_c4,
    f32,
    C4,
    nppiFilterPrewittHoriz_32f_C4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_horizontal_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittHoriz_32f_AC4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_u8_c1,
    u8,
    C1,
    nppiFilterPrewittVert_8u_C1R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_u8_c3,
    u8,
    C3,
    nppiFilterPrewittVert_8u_C3R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_u8_c4,
    u8,
    C4,
    nppiFilterPrewittVert_8u_C4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittVert_8u_AC4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_i16_c1,
    i16,
    C1,
    nppiFilterPrewittVert_16s_C1R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_i16_c3,
    i16,
    C3,
    nppiFilterPrewittVert_16s_C3R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_i16_c4,
    i16,
    C4,
    nppiFilterPrewittVert_16s_C4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittVert_16s_AC4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_f32_c1,
    f32,
    C1,
    nppiFilterPrewittVert_32f_C1R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_f32_c3,
    f32,
    C3,
    nppiFilterPrewittVert_32f_C3R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_f32_c4,
    f32,
    C4,
    nppiFilterPrewittVert_32f_C4R_Ctx
);
impl_filter_directional!(
    filter_prewitt_vertical_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittVert_32f_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_u8_c1,
    u8,
    C1,
    nppiFilterPrewittHorizBorder_8u_C1R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_u8_c3,
    u8,
    C3,
    nppiFilterPrewittHorizBorder_8u_C3R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_u8_c4,
    u8,
    C4,
    nppiFilterPrewittHorizBorder_8u_C4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittHorizBorder_8u_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_i16_c1,
    i16,
    C1,
    nppiFilterPrewittHorizBorder_16s_C1R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_i16_c3,
    i16,
    C3,
    nppiFilterPrewittHorizBorder_16s_C3R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_i16_c4,
    i16,
    C4,
    nppiFilterPrewittHorizBorder_16s_C4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittHorizBorder_16s_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_f32_c1,
    f32,
    C1,
    nppiFilterPrewittHorizBorder_32f_C1R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_f32_c3,
    f32,
    C3,
    nppiFilterPrewittHorizBorder_32f_C3R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_f32_c4,
    f32,
    C4,
    nppiFilterPrewittHorizBorder_32f_C4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_horizontal_border_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittHorizBorder_32f_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_u8_c1,
    u8,
    C1,
    nppiFilterPrewittVertBorder_8u_C1R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_u8_c3,
    u8,
    C3,
    nppiFilterPrewittVertBorder_8u_C3R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_u8_c4,
    u8,
    C4,
    nppiFilterPrewittVertBorder_8u_C4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_u8_ac4,
    u8,
    AC4,
    nppiFilterPrewittVertBorder_8u_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_i16_c1,
    i16,
    C1,
    nppiFilterPrewittVertBorder_16s_C1R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_i16_c3,
    i16,
    C3,
    nppiFilterPrewittVertBorder_16s_C3R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_i16_c4,
    i16,
    C4,
    nppiFilterPrewittVertBorder_16s_C4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_i16_ac4,
    i16,
    AC4,
    nppiFilterPrewittVertBorder_16s_AC4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_f32_c1,
    f32,
    C1,
    nppiFilterPrewittVertBorder_32f_C1R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_f32_c3,
    f32,
    C3,
    nppiFilterPrewittVertBorder_32f_C3R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_f32_c4,
    f32,
    C4,
    nppiFilterPrewittVertBorder_32f_C4R_Ctx
);
impl_filter_directional_border!(
    filter_prewitt_vertical_border_f32_ac4,
    f32,
    AC4,
    nppiFilterPrewittVertBorder_32f_AC4R_Ctx
);

pub trait FilterPrewittHorizontal<L: ChannelLayout>: DataTypeLike {
    fn filter_prewitt_horizontal(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    C1,
    filter_prewitt_horizontal_u8_c1
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    C3,
    filter_prewitt_horizontal_u8_c3
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    C4,
    filter_prewitt_horizontal_u8_c4
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    u8,
    AC4,
    filter_prewitt_horizontal_u8_ac4
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    C1,
    filter_prewitt_horizontal_i16_c1
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    C3,
    filter_prewitt_horizontal_i16_c3
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    C4,
    filter_prewitt_horizontal_i16_c4
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    i16,
    AC4,
    filter_prewitt_horizontal_i16_ac4
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    C1,
    filter_prewitt_horizontal_f32_c1
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    C3,
    filter_prewitt_horizontal_f32_c3
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    C4,
    filter_prewitt_horizontal_f32_c4
);
impl_directional_filter_dispatch!(
    FilterPrewittHorizontal,
    filter_prewitt_horizontal,
    f32,
    AC4,
    filter_prewitt_horizontal_f32_ac4
);

pub fn filter_prewitt_horizontal<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: FilterPrewittHorizontal<L>,
    L: ChannelLayout,
{
    T::filter_prewitt_horizontal(stream_context, source, destination)
}

pub trait FilterPrewittVertical<L: ChannelLayout>: DataTypeLike {
    fn filter_prewitt_vertical(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    C1,
    filter_prewitt_vertical_u8_c1
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    C3,
    filter_prewitt_vertical_u8_c3
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    C4,
    filter_prewitt_vertical_u8_c4
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    u8,
    AC4,
    filter_prewitt_vertical_u8_ac4
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    C1,
    filter_prewitt_vertical_i16_c1
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    C3,
    filter_prewitt_vertical_i16_c3
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    C4,
    filter_prewitt_vertical_i16_c4
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    i16,
    AC4,
    filter_prewitt_vertical_i16_ac4
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    C1,
    filter_prewitt_vertical_f32_c1
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    C3,
    filter_prewitt_vertical_f32_c3
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    C4,
    filter_prewitt_vertical_f32_c4
);
impl_directional_filter_dispatch!(
    FilterPrewittVertical,
    filter_prewitt_vertical,
    f32,
    AC4,
    filter_prewitt_vertical_f32_ac4
);

pub fn filter_prewitt_vertical<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: FilterPrewittVertical<L>,
    L: ChannelLayout,
{
    T::filter_prewitt_vertical(stream_context, source, destination)
}

pub trait FilterPrewittHorizontalBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_prewitt_horizontal_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    C1,
    filter_prewitt_horizontal_border_u8_c1
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    C3,
    filter_prewitt_horizontal_border_u8_c3
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    C4,
    filter_prewitt_horizontal_border_u8_c4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    u8,
    AC4,
    filter_prewitt_horizontal_border_u8_ac4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    C1,
    filter_prewitt_horizontal_border_i16_c1
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    C3,
    filter_prewitt_horizontal_border_i16_c3
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    C4,
    filter_prewitt_horizontal_border_i16_c4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    i16,
    AC4,
    filter_prewitt_horizontal_border_i16_ac4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    C1,
    filter_prewitt_horizontal_border_f32_c1
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    C3,
    filter_prewitt_horizontal_border_f32_c3
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    C4,
    filter_prewitt_horizontal_border_f32_c4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittHorizontalBorder,
    filter_prewitt_horizontal_border,
    f32,
    AC4,
    filter_prewitt_horizontal_border_f32_ac4
);

pub fn filter_prewitt_horizontal_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterPrewittHorizontalBorder<L>,
    L: ChannelLayout,
{
    T::filter_prewitt_horizontal_border(
        stream_context,
        source,
        source_offset,
        destination,
        border_type,
    )
}

pub trait FilterPrewittVerticalBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_prewitt_vertical_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    C1,
    filter_prewitt_vertical_border_u8_c1
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    C3,
    filter_prewitt_vertical_border_u8_c3
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    C4,
    filter_prewitt_vertical_border_u8_c4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    u8,
    AC4,
    filter_prewitt_vertical_border_u8_ac4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    C1,
    filter_prewitt_vertical_border_i16_c1
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    C3,
    filter_prewitt_vertical_border_i16_c3
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    C4,
    filter_prewitt_vertical_border_i16_c4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    i16,
    AC4,
    filter_prewitt_vertical_border_i16_ac4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    C1,
    filter_prewitt_vertical_border_f32_c1
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    C3,
    filter_prewitt_vertical_border_f32_c3
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    C4,
    filter_prewitt_vertical_border_f32_c4
);
impl_directional_filter_border_dispatch!(
    FilterPrewittVerticalBorder,
    filter_prewitt_vertical_border,
    f32,
    AC4,
    filter_prewitt_vertical_border_f32_ac4
);

pub fn filter_prewitt_vertical_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterPrewittVerticalBorder<L>,
    L: ChannelLayout,
{
    T::filter_prewitt_vertical_border(
        stream_context,
        source,
        source_offset,
        destination,
        border_type,
    )
}
