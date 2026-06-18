use super::*;

pub trait FilterSobelHorizontalBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_sobel_horizontal_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    C1,
    filter_sobel_horizontal_border_u8_c1
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    C3,
    filter_sobel_horizontal_border_u8_c3
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    C4,
    filter_sobel_horizontal_border_u8_c4
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    u8,
    AC4,
    filter_sobel_horizontal_border_u8_ac4
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    C1,
    filter_sobel_horizontal_border_i16_c1
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    C3,
    filter_sobel_horizontal_border_i16_c3
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    C4,
    filter_sobel_horizontal_border_i16_c4
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    i16,
    AC4,
    filter_sobel_horizontal_border_i16_ac4
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    C1,
    filter_sobel_horizontal_border_f32_c1
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    C3,
    filter_sobel_horizontal_border_f32_c3
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    C4,
    filter_sobel_horizontal_border_f32_c4
);
impl_directional_filter_border_dispatch!(
    FilterSobelHorizontalBorder,
    filter_sobel_horizontal_border,
    f32,
    AC4,
    filter_sobel_horizontal_border_f32_ac4
);

pub fn filter_sobel_horizontal_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterSobelHorizontalBorder<L>,
    L: ChannelLayout,
{
    T::filter_sobel_horizontal_border(
        stream_context,
        source,
        source_offset,
        destination,
        border_type,
    )
}

pub trait FilterSobelVerticalBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_sobel_vertical_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    C1,
    filter_sobel_vertical_border_u8_c1
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    C3,
    filter_sobel_vertical_border_u8_c3
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    C4,
    filter_sobel_vertical_border_u8_c4
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    u8,
    AC4,
    filter_sobel_vertical_border_u8_ac4
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    C1,
    filter_sobel_vertical_border_i16_c1
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    C3,
    filter_sobel_vertical_border_i16_c3
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    C4,
    filter_sobel_vertical_border_i16_c4
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    i16,
    AC4,
    filter_sobel_vertical_border_i16_ac4
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    C1,
    filter_sobel_vertical_border_f32_c1
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    C3,
    filter_sobel_vertical_border_f32_c3
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    C4,
    filter_sobel_vertical_border_f32_c4
);
impl_directional_filter_border_dispatch!(
    FilterSobelVerticalBorder,
    filter_sobel_vertical_border,
    f32,
    AC4,
    filter_sobel_vertical_border_f32_ac4
);

pub fn filter_sobel_vertical_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterSobelVerticalBorder<L>,
    L: ChannelLayout,
{
    T::filter_sobel_vertical_border(
        stream_context,
        source,
        source_offset,
        destination,
        border_type,
    )
}
