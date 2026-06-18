use super::*;

pub trait FilterSobelHorizontalC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_horizontal_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalC1,
    filter_sobel_horizontal_typed,
    u8,
    i16,
    filter_sobel_horizontal_u8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalC1,
    filter_sobel_horizontal_typed,
    i8,
    i16,
    filter_sobel_horizontal_i8_to_i16_c1
);

pub fn filter_sobel_horizontal_typed<T: FilterSobelHorizontalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    T::filter_sobel_horizontal_typed(stream_context, source, destination, mask_size)
}

pub trait FilterSobelVerticalC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_vertical_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalC1,
    filter_sobel_vertical_typed,
    u8,
    i16,
    filter_sobel_vertical_u8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalC1,
    filter_sobel_vertical_typed,
    i8,
    i16,
    filter_sobel_vertical_i8_to_i16_c1
);

pub fn filter_sobel_vertical_typed<T: FilterSobelVerticalC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    T::filter_sobel_vertical_typed(stream_context, source, destination, mask_size)
}

pub trait FilterSobelHorizontalBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_horizontal_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalBorderC1,
    filter_sobel_horizontal_border_typed,
    u8,
    i16,
    filter_sobel_horizontal_border_u8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalBorderC1,
    filter_sobel_horizontal_border_typed,
    i8,
    i16,
    filter_sobel_horizontal_border_i8_to_i16_c1
);

pub fn filter_sobel_horizontal_border_typed<T: FilterSobelHorizontalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    T::filter_sobel_horizontal_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}

pub trait FilterSobelVerticalBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_vertical_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalBorderC1,
    filter_sobel_vertical_border_typed,
    u8,
    i16,
    filter_sobel_vertical_border_u8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalBorderC1,
    filter_sobel_vertical_border_typed,
    i8,
    i16,
    filter_sobel_vertical_border_i8_to_i16_c1
);

pub fn filter_sobel_vertical_border_typed<T: FilterSobelVerticalBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    T::filter_sobel_vertical_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}
