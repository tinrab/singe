use super::*;

pub fn filter_canny_border_buffer_size(roi: Size) -> Result<usize> {
    let mut bytes = 0_i32;
    unsafe {
        try_ffi!(sys::nppiFilterCannyBorderGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn filter_canny_border_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_canny_thresholds(low_threshold, high_threshold)?;
    let required_bytes = filter_canny_border_buffer_size(destination.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    unsafe {
        try_ffi!(sys::nppiFilterCannyBorder_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            filter_type.into(),
            mask_size.into(),
            low_threshold,
            high_threshold,
            norm.into(),
            border_type.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn filter_canny_border_u8_c3_to_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_canny_thresholds(low_threshold, high_threshold)?;
    let required_bytes = filter_canny_border_buffer_size(destination.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    unsafe {
        try_ffi!(sys::nppiFilterCannyBorder_8u_C3C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            filter_type.into(),
            mask_size.into(),
            low_threshold,
            high_threshold,
            norm.into(),
            border_type.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn filter_canny_border_c3_to_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C3>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    filter_canny_border_u8_c3_to_c1(
        stream_context,
        source,
        source_offset,
        destination,
        filter_type,
        mask_size,
        low_threshold,
        high_threshold,
        norm,
        border_type,
    )
}

pub fn filter_harris_corners_border_buffer_size(roi: Size) -> Result<usize> {
    let mut bytes = 0_i32;
    unsafe {
        try_ffi!(sys::nppiFilterHarrisCornersBorderGetBufferSize(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn filter_harris_corners_border_u8_to_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, f32, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    average_window_size: MaskSize,
    k: f32,
    scale: f32,
    border_type: BorderType,
) -> Result<()> {
    validate_border_roi(source.size(), source_offset, destination.size())?;
    validate_harris_corners_border_parameters(mask_size, average_window_size, border_type)?;
    let required_bytes = filter_harris_corners_border_buffer_size(destination.size())?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    unsafe {
        try_ffi!(sys::nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            source_offset.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            filter_type.into(),
            mask_size.into(),
            average_window_size.into(),
            k,
            scale,
            border_type.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn filter_hough_line_buffer_size(
    roi: Size,
    delta: PointPolar,
    max_line_count: i32,
) -> Result<usize> {
    validate_positive_size(roi)?;
    validate_hough_line_parameters(delta, max_line_count)?;
    let mut bytes = 0_i32;
    unsafe {
        try_ffi!(sys::nppiFilterHoughLineGetBufferSize(
            roi.into(),
            delta.into(),
            max_line_count,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn filter_hough_line_u8_to_polar_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut DeviceMemory<PointPolar>,
    line_count: &mut DeviceMemory<i32>,
) -> Result<()> {
    validate_hough_line_io(source.size(), delta, threshold, lines, line_count)?;
    let max_line_count = i32::try_from(lines.len()).map_err(|_| Error::OutOfRange {
        name: "max line count".into(),
    })?;
    let required_bytes = filter_hough_line_buffer_size(source.size(), delta, max_line_count)?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    unsafe {
        try_ffi!(sys::nppiFilterHoughLine_8u32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            delta.into(),
            threshold,
            lines.as_mut_ptr().cast(),
            max_line_count,
            line_count.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn filter_hough_line_region_u8_to_polar_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut DeviceMemory<PointPolar>,
    destination_roi: [PointPolar; 2],
    line_count: &mut DeviceMemory<i32>,
) -> Result<()> {
    validate_hough_line_io(source.size(), delta, threshold, lines, line_count)?;
    validate_hough_line_destination_roi(destination_roi)?;
    let max_line_count = i32::try_from(lines.len()).map_err(|_| Error::OutOfRange {
        name: "max line count".into(),
    })?;
    let required_bytes = filter_hough_line_buffer_size(source.size(), delta, max_line_count)?;
    let scratch = DeviceMemory::<u8>::create(required_bytes)?;
    let mut destination_roi = destination_roi;
    unsafe {
        try_ffi!(sys::nppiFilterHoughLineRegion_8u32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            source.size().into(),
            delta.into(),
            threshold,
            lines.as_mut_ptr().cast(),
            destination_roi.as_mut_ptr().cast(),
            max_line_count,
            line_count.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub trait FilterCannyBorderC1: DataTypeLike + Sized {
    fn buffer_size(roi: Size) -> Result<usize>;

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl FilterCannyBorderC1 for u8 {
    fn buffer_size(roi: Size) -> Result<usize> {
        filter_canny_border_buffer_size(roi)
    }

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()> {
        filter_canny_border_u8_c1(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            low_threshold,
            high_threshold,
            norm,
            border_type,
        )
    }
}

pub fn filter_canny_border_typed_buffer_size<T: FilterCannyBorderC1>(roi: Size) -> Result<usize> {
    T::buffer_size(roi)
}

pub fn filter_canny_border<T: FilterCannyBorderC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        source_offset,
        destination,
        filter_type,
        mask_size,
        low_threshold,
        high_threshold,
        norm,
        border_type,
    )
}

pub trait FilterCannyBorderC3ToC1: DataTypeLike + Sized {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()>;
}

impl FilterCannyBorderC3ToC1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()> {
        filter_canny_border_u8_c3_to_c1(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            low_threshold,
            high_threshold,
            norm,
            border_type,
        )
    }
}

pub fn filter_canny_border_to_mask<T: FilterCannyBorderC3ToC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C3>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, u8, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    low_threshold: i16,
    high_threshold: i16,
    norm: ImageNormalization,
    border_type: BorderType,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        source_offset,
        destination,
        filter_type,
        mask_size,
        low_threshold,
        high_threshold,
        norm,
        border_type,
    )
}

pub trait FilterHarrisCornersBorderToF32C1: DataTypeLike + Sized {
    fn buffer_size(roi: Size) -> Result<usize>;

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

impl FilterHarrisCornersBorderToF32C1 for u8 {
    fn buffer_size(roi: Size) -> Result<usize> {
        filter_harris_corners_border_buffer_size(roi)
    }

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()> {
        filter_harris_corners_border_u8_to_f32_c1(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            average_window_size,
            k,
            scale,
            border_type,
        )
    }
}

pub fn filter_harris_corners_border_to_f32_buffer_size<T: FilterHarrisCornersBorderToF32C1>(
    roi: Size,
) -> Result<usize> {
    T::buffer_size(roi)
}

pub fn filter_harris_corners_border_to_f32<T: FilterHarrisCornersBorderToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, f32, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    average_window_size: MaskSize,
    k: f32,
    scale: f32,
    border_type: BorderType,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        source_offset,
        destination,
        filter_type,
        mask_size,
        average_window_size,
        k,
        scale,
        border_type,
    )
}

pub trait FilterHarrisCornersBorderToC1<Destination>: DataTypeLike + Sized
where
    Destination: DataTypeLike,
{
    fn buffer_size(roi: Size) -> Result<usize>;

    fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Destination, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()>;
}

impl<T> FilterHarrisCornersBorderToC1<f32> for T
where
    T: FilterHarrisCornersBorderToF32C1,
{
    fn buffer_size(roi: Size) -> Result<usize> {
        T::buffer_size(roi)
    }

    fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()> {
        T::dispatch(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            average_window_size,
            k,
            scale,
            border_type,
        )
    }
}

pub fn filter_harris_corners_border_typed_buffer_size<T, Destination>(roi: Size) -> Result<usize>
where
    T: FilterHarrisCornersBorderToC1<Destination>,
    Destination: DataTypeLike,
{
    T::buffer_size(roi)
}

pub fn filter_harris_corners_border<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, Destination, C1>,
    filter_type: DifferentialKernel,
    mask_size: MaskSize,
    average_window_size: MaskSize,
    k: f32,
    scale: f32,
    border_type: BorderType,
) -> Result<()>
where
    T: FilterHarrisCornersBorderToC1<Destination>,
    Destination: DataTypeLike,
{
    T::dispatch_to(
        stream_context,
        source,
        source_offset,
        destination,
        filter_type,
        mask_size,
        average_window_size,
        k,
        scale,
        border_type,
    )
}

pub trait FilterHoughLineToPolarC1: DataTypeLike + Sized {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut DeviceMemory<PointPolar>,
        line_count: &mut DeviceMemory<i32>,
    ) -> Result<()>;

    fn region_dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut DeviceMemory<PointPolar>,
        destination_roi: [PointPolar; 2],
        line_count: &mut DeviceMemory<i32>,
    ) -> Result<()>;
}

impl FilterHoughLineToPolarC1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut DeviceMemory<PointPolar>,
        line_count: &mut DeviceMemory<i32>,
    ) -> Result<()> {
        filter_hough_line_u8_to_polar_c1(
            stream_context,
            source,
            delta,
            threshold,
            lines,
            line_count,
        )
    }

    fn region_dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        delta: PointPolar,
        threshold: i32,
        lines: &mut DeviceMemory<PointPolar>,
        destination_roi: [PointPolar; 2],
        line_count: &mut DeviceMemory<i32>,
    ) -> Result<()> {
        filter_hough_line_region_u8_to_polar_c1(
            stream_context,
            source,
            delta,
            threshold,
            lines,
            destination_roi,
            line_count,
        )
    }
}

pub fn filter_hough_line_to_polar<T: FilterHoughLineToPolarC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut DeviceMemory<PointPolar>,
    line_count: &mut DeviceMemory<i32>,
) -> Result<()> {
    T::dispatch(stream_context, source, delta, threshold, lines, line_count)
}

pub fn filter_hough_line_region_to_polar<T: FilterHoughLineToPolarC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    delta: PointPolar,
    threshold: i32,
    lines: &mut DeviceMemory<PointPolar>,
    destination_roi: [PointPolar; 2],
    line_count: &mut DeviceMemory<i32>,
) -> Result<()> {
    T::region_dispatch(
        stream_context,
        source,
        delta,
        threshold,
        lines,
        destination_roi,
        line_count,
    )
}
