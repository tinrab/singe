use super::*;

impl_filter_directional_typed!(
    filter_scharr_horizontal_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrHoriz_8u16s_C1R_Ctx
);
impl_filter_directional_typed!(
    filter_scharr_horizontal_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrHoriz_8s16s_C1R_Ctx
);
impl_filter_directional!(
    filter_scharr_horizontal_f32_c1,
    f32,
    C1,
    nppiFilterScharrHoriz_32f_C1R_Ctx
);
impl_filter_directional_typed!(
    filter_scharr_vertical_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrVert_8u16s_C1R_Ctx
);
impl_filter_directional_typed!(
    filter_scharr_vertical_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrVert_8s16s_C1R_Ctx
);
impl_filter_directional!(
    filter_scharr_vertical_f32_c1,
    f32,
    C1,
    nppiFilterScharrVert_32f_C1R_Ctx
);
impl_filter_directional_typed_border!(
    filter_scharr_horizontal_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrHorizBorder_8u16s_C1R_Ctx
);
impl_filter_directional_typed_border!(
    filter_scharr_horizontal_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrHorizBorder_8s16s_C1R_Ctx
);
impl_filter_directional_border!(
    filter_scharr_horizontal_border_f32_c1,
    f32,
    C1,
    nppiFilterScharrHorizBorder_32f_C1R_Ctx
);
impl_filter_directional_typed_border!(
    filter_scharr_vertical_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterScharrVertBorder_8u16s_C1R_Ctx
);
impl_filter_directional_typed_border!(
    filter_scharr_vertical_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterScharrVertBorder_8s16s_C1R_Ctx
);
impl_filter_directional_border!(
    filter_scharr_vertical_border_f32_c1,
    f32,
    C1,
    nppiFilterScharrVertBorder_32f_C1R_Ctx
);

pub trait FilterScharrHorizontalC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_scharr_horizontal_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
    ) -> Result<()>;
}

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrHorizontalC1,
    filter_scharr_horizontal_typed,
    u8,
    i16,
    filter_scharr_horizontal_u8_to_i16_c1
);
impl_directional_filter_typed_c1_dispatch!(
    FilterScharrHorizontalC1,
    filter_scharr_horizontal_typed,
    i8,
    i16,
    filter_scharr_horizontal_i8_to_i16_c1
);
impl_directional_filter_typed_c1_dispatch!(
    FilterScharrHorizontalC1,
    filter_scharr_horizontal_typed,
    f32,
    f32,
    filter_scharr_horizontal_f32_c1
);

pub fn filter_scharr_horizontal_typed<T: FilterScharrHorizontalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
) -> Result<()> {
    T::filter_scharr_horizontal_typed(stream_context, source, destination)
}

pub trait FilterScharrVerticalC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_scharr_vertical_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
    ) -> Result<()>;
}

impl_directional_filter_typed_c1_dispatch!(
    FilterScharrVerticalC1,
    filter_scharr_vertical_typed,
    u8,
    i16,
    filter_scharr_vertical_u8_to_i16_c1
);
impl_directional_filter_typed_c1_dispatch!(
    FilterScharrVerticalC1,
    filter_scharr_vertical_typed,
    i8,
    i16,
    filter_scharr_vertical_i8_to_i16_c1
);
impl_directional_filter_typed_c1_dispatch!(
    FilterScharrVerticalC1,
    filter_scharr_vertical_typed,
    f32,
    f32,
    filter_scharr_vertical_f32_c1
);

pub fn filter_scharr_vertical_typed<T: FilterScharrVerticalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
) -> Result<()> {
    T::filter_scharr_vertical_typed(stream_context, source, destination)
}

pub trait FilterScharrHorizontalBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_scharr_horizontal_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrHorizontalBorderC1,
    filter_scharr_horizontal_border_typed,
    u8,
    i16,
    filter_scharr_horizontal_border_u8_to_i16_c1
);
impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrHorizontalBorderC1,
    filter_scharr_horizontal_border_typed,
    i8,
    i16,
    filter_scharr_horizontal_border_i8_to_i16_c1
);
impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrHorizontalBorderC1,
    filter_scharr_horizontal_border_typed,
    f32,
    f32,
    filter_scharr_horizontal_border_f32_c1
);

pub fn filter_scharr_horizontal_border_typed<T: FilterScharrHorizontalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    border_type: BorderType,
) -> Result<()> {
    T::filter_scharr_horizontal_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        border_type,
    )
}

pub trait FilterScharrVerticalBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_scharr_vertical_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrVerticalBorderC1,
    filter_scharr_vertical_border_typed,
    u8,
    i16,
    filter_scharr_vertical_border_u8_to_i16_c1
);
impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrVerticalBorderC1,
    filter_scharr_vertical_border_typed,
    i8,
    i16,
    filter_scharr_vertical_border_i8_to_i16_c1
);
impl_directional_filter_typed_border_c1_dispatch!(
    FilterScharrVerticalBorderC1,
    filter_scharr_vertical_border_typed,
    f32,
    f32,
    filter_scharr_vertical_border_f32_c1
);

pub fn filter_scharr_vertical_border_typed<T: FilterScharrVerticalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    border_type: BorderType,
) -> Result<()> {
    T::filter_scharr_vertical_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        border_type,
    )
}
