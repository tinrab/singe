use super::*;

pub trait FilterSobelHorizontal<L: ChannelLayout>: DataTypeLike {
    fn filter_sobel_horizontal(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    C1,
    filter_sobel_horizontal_u8_c1
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    C3,
    filter_sobel_horizontal_u8_c3
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    C4,
    filter_sobel_horizontal_u8_c4
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    u8,
    AC4,
    filter_sobel_horizontal_u8_ac4
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    C1,
    filter_sobel_horizontal_i16_c1
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    C3,
    filter_sobel_horizontal_i16_c3
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    C4,
    filter_sobel_horizontal_i16_c4
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    i16,
    AC4,
    filter_sobel_horizontal_i16_ac4
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    C1,
    filter_sobel_horizontal_f32_c1
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    C3,
    filter_sobel_horizontal_f32_c3
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    C4,
    filter_sobel_horizontal_f32_c4
);
impl_directional_filter_dispatch!(
    FilterSobelHorizontal,
    filter_sobel_horizontal,
    f32,
    AC4,
    filter_sobel_horizontal_f32_ac4
);

pub fn filter_sobel_horizontal<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: FilterSobelHorizontal<L>,
    L: ChannelLayout,
{
    T::filter_sobel_horizontal(stream_context, source, destination)
}

pub trait FilterSobelVertical<L: ChannelLayout>: DataTypeLike {
    fn filter_sobel_vertical(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
    ) -> Result<()>;
}

impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    C1,
    filter_sobel_vertical_u8_c1
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    C3,
    filter_sobel_vertical_u8_c3
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    C4,
    filter_sobel_vertical_u8_c4
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    u8,
    AC4,
    filter_sobel_vertical_u8_ac4
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    C1,
    filter_sobel_vertical_i16_c1
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    C3,
    filter_sobel_vertical_i16_c3
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    C4,
    filter_sobel_vertical_i16_c4
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    i16,
    AC4,
    filter_sobel_vertical_i16_ac4
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    C1,
    filter_sobel_vertical_f32_c1
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    C3,
    filter_sobel_vertical_f32_c3
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    C4,
    filter_sobel_vertical_f32_c4
);
impl_directional_filter_dispatch!(
    FilterSobelVertical,
    filter_sobel_vertical,
    f32,
    AC4,
    filter_sobel_vertical_f32_ac4
);

pub fn filter_sobel_vertical<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
) -> Result<()>
where
    T: FilterSobelVertical<L>,
    L: ChannelLayout,
{
    T::filter_sobel_vertical(stream_context, source, destination)
}
