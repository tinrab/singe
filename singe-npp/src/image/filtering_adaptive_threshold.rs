use super::*;

pub(crate) fn filter_threshold_adaptive_box_border_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C1>,
    mask_size: Size,
    delta: f32,
    value_greater_than: u8,
    value_less_or_equal: u8,
    border_type: BorderType,
) -> Result<()> {
    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_positive_size(mask_size)?;
    unsafe {
        try_ffi!(sys::nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            mask_size.into(),
            delta,
            value_greater_than,
            value_less_or_equal,
            border_type.into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub trait FilterThresholdAdaptiveBoxBorderC1: DataTypeLike {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, C1>,
        mask_size: Size,
        delta: f32,
        value_greater_than: Self,
        value_less_or_equal: Self,
        border_type: BorderType,
    ) -> Result<()>;
}

impl FilterThresholdAdaptiveBoxBorderC1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, C1>,
        mask_size: Size,
        delta: f32,
        value_greater_than: Self,
        value_less_or_equal: Self,
        border_type: BorderType,
    ) -> Result<()> {
        filter_threshold_adaptive_box_border_u8_c1(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            delta,
            value_greater_than,
            value_less_or_equal,
            border_type,
        )
    }
}

pub fn filter_threshold_adaptive_box_border<T: FilterThresholdAdaptiveBoxBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, C1>,
    mask_size: Size,
    delta: f32,
    value_greater_than: T,
    value_less_or_equal: T,
    border_type: BorderType,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        source_offset,
        destination,
        mask_size,
        delta,
        value_greater_than,
        value_less_or_equal,
        border_type,
    )
}
