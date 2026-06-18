use super::*;

impl_filter_masked_kernel!(
    filter_sobel_horizontal_mask_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizMask_32f_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_sobel_vertical_mask_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertMask_32f_C1R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_sobel_horizontal_second_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecond_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_sobel_horizontal_second_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecond_8s16s_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_sobel_horizontal_second_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizSecond_32f_C1R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_sobel_vertical_second_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecond_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_sobel_vertical_second_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecond_8s16s_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_sobel_vertical_second_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertSecond_32f_C1R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_sobel_cross_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelCross_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed!(
    filter_sobel_cross_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelCross_8s16s_C1R_Ctx
);
impl_filter_masked_kernel!(
    filter_sobel_cross_f32_c1,
    f32,
    C1,
    nppiFilterSobelCross_32f_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_sobel_horizontal_mask_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_sobel_vertical_mask_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertMaskBorder_32f_C1R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_sobel_horizontal_second_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecondBorder_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_sobel_horizontal_second_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelHorizSecondBorder_8s16s_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_sobel_horizontal_second_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelHorizSecondBorder_32f_C1R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_sobel_vertical_second_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecondBorder_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_sobel_vertical_second_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelVertSecondBorder_8s16s_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_sobel_vertical_second_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelVertSecondBorder_32f_C1R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_sobel_cross_border_u8_to_i16_c1,
    u8,
    C1,
    i16,
    C1,
    nppiFilterSobelCrossBorder_8u16s_C1R_Ctx
);
impl_filter_masked_kernel_typed_border!(
    filter_sobel_cross_border_i8_to_i16_c1,
    i8,
    C1,
    i16,
    C1,
    nppiFilterSobelCrossBorder_8s16s_C1R_Ctx
);
impl_filter_masked_kernel_border!(
    filter_sobel_cross_border_f32_c1,
    f32,
    C1,
    nppiFilterSobelCrossBorder_32f_C1R_Ctx
);

pub trait FilterSobelHorizontalSecondC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_horizontal_second_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalSecondC1,
    filter_sobel_horizontal_second_typed,
    u8,
    i16,
    filter_sobel_horizontal_second_u8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalSecondC1,
    filter_sobel_horizontal_second_typed,
    i8,
    i16,
    filter_sobel_horizontal_second_i8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelHorizontalSecondC1,
    filter_sobel_horizontal_second_typed,
    f32,
    f32,
    filter_sobel_horizontal_second_f32_c1
);

pub fn filter_sobel_horizontal_second_typed<T: FilterSobelHorizontalSecondC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    T::filter_sobel_horizontal_second_typed(stream_context, source, destination, mask_size)
}

pub trait FilterSobelVerticalSecondC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_vertical_second_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalSecondC1,
    filter_sobel_vertical_second_typed,
    u8,
    i16,
    filter_sobel_vertical_second_u8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalSecondC1,
    filter_sobel_vertical_second_typed,
    i8,
    i16,
    filter_sobel_vertical_second_i8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelVerticalSecondC1,
    filter_sobel_vertical_second_typed,
    f32,
    f32,
    filter_sobel_vertical_second_f32_c1
);

pub fn filter_sobel_vertical_second_typed<T: FilterSobelVerticalSecondC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    T::filter_sobel_vertical_second_typed(stream_context, source, destination, mask_size)
}

pub trait FilterSobelCrossC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_cross_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelCrossC1,
    filter_sobel_cross_typed,
    u8,
    i16,
    filter_sobel_cross_u8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelCrossC1,
    filter_sobel_cross_typed,
    i8,
    i16,
    filter_sobel_cross_i8_to_i16_c1
);
impl_filter_sobel_typed_c1_dispatch!(
    FilterSobelCrossC1,
    filter_sobel_cross_typed,
    f32,
    f32,
    filter_sobel_cross_f32_c1
);

pub fn filter_sobel_cross_typed<T: FilterSobelCrossC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
) -> Result<()> {
    T::filter_sobel_cross_typed(stream_context, source, destination, mask_size)
}

pub trait FilterSobelHorizontalSecondBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_horizontal_second_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalSecondBorderC1,
    filter_sobel_horizontal_second_border_typed,
    u8,
    i16,
    filter_sobel_horizontal_second_border_u8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalSecondBorderC1,
    filter_sobel_horizontal_second_border_typed,
    i8,
    i16,
    filter_sobel_horizontal_second_border_i8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelHorizontalSecondBorderC1,
    filter_sobel_horizontal_second_border_typed,
    f32,
    f32,
    filter_sobel_horizontal_second_border_f32_c1
);

pub fn filter_sobel_horizontal_second_border_typed<T: FilterSobelHorizontalSecondBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    T::filter_sobel_horizontal_second_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}

pub trait FilterSobelVerticalSecondBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_vertical_second_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalSecondBorderC1,
    filter_sobel_vertical_second_border_typed,
    u8,
    i16,
    filter_sobel_vertical_second_border_u8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalSecondBorderC1,
    filter_sobel_vertical_second_border_typed,
    i8,
    i16,
    filter_sobel_vertical_second_border_i8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelVerticalSecondBorderC1,
    filter_sobel_vertical_second_border_typed,
    f32,
    f32,
    filter_sobel_vertical_second_border_f32_c1
);

pub fn filter_sobel_vertical_second_border_typed<T: FilterSobelVerticalSecondBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    T::filter_sobel_vertical_second_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}

pub trait FilterSobelCrossBorderC1: DataTypeLike {
    type Output: DataTypeLike;

    fn filter_sobel_cross_border_typed(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelCrossBorderC1,
    filter_sobel_cross_border_typed,
    u8,
    i16,
    filter_sobel_cross_border_u8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelCrossBorderC1,
    filter_sobel_cross_border_typed,
    i8,
    i16,
    filter_sobel_cross_border_i8_to_i16_c1
);
impl_filter_sobel_typed_border_c1_dispatch!(
    FilterSobelCrossBorderC1,
    filter_sobel_cross_border_typed,
    f32,
    f32,
    filter_sobel_cross_border_f32_c1
);

pub fn filter_sobel_cross_border_typed<T: FilterSobelCrossBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
    mask_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    T::filter_sobel_cross_border_typed(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        border_type,
    )
}
