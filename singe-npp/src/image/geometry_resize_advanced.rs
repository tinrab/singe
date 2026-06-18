use super::*;

pub(crate) fn resize_sqr_pixel_u8_c1_advanced_buffer_size(
    resize: &ResizeSqrPixelAdvanced,
) -> Result<usize> {
    let mut bytes = 0;
    unsafe {
        try_ffi!(sys::nppiResizeAdvancedGetBufferHostSize_8u_C1R(
            rect_size(resize.source_roi).into(),
            rect_size(resize.destination_roi).into(),
            &raw mut bytes,
            i32::from(resize.interpolation),
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub(crate) fn resize_sqr_pixel_u8_c1_advanced_with_scratch(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixelAdvanced,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, u8, C1>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    scratch.require(resize_sqr_pixel_u8_c1_advanced_buffer_size(resize)?)?;

    unsafe {
        try_ffi!(sys::nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(
            source.as_ptr().cast(),
            source.size().into(),
            source.step(),
            resize.source_roi.into(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            resize.destination_roi.into(),
            resize.x_factor,
            resize.y_factor,
            scratch.as_mut_ptr().cast(),
            i32::from(resize.interpolation),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub trait ResizeSqrPixelAdvancedC1: DataTypeLike {
    fn resize_sqr_pixel_c1_advanced_buffer_size(resize: &ResizeSqrPixelAdvanced) -> Result<usize>;

    fn resize_sqr_pixel_c1_advanced_with_scratch(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self, C1>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

impl ResizeSqrPixelAdvancedC1 for u8 {
    fn resize_sqr_pixel_c1_advanced_buffer_size(resize: &ResizeSqrPixelAdvanced) -> Result<usize> {
        resize_sqr_pixel_u8_c1_advanced_buffer_size(resize)
    }

    fn resize_sqr_pixel_c1_advanced_with_scratch(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self, C1>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        resize_sqr_pixel_u8_c1_advanced_with_scratch(
            stream_context,
            resize,
            source,
            destination,
            scratch,
        )
    }
}

pub fn resize_sqr_pixel_c1_advanced_buffer_size<T: ResizeSqrPixelAdvancedC1>(
    resize: &ResizeSqrPixelAdvanced,
) -> Result<usize> {
    T::resize_sqr_pixel_c1_advanced_buffer_size(resize)
}

pub fn resize_sqr_pixel_c1_advanced_with_scratch<T: ResizeSqrPixelAdvancedC1>(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixelAdvanced,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T, C1>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    T::resize_sqr_pixel_c1_advanced_with_scratch(
        stream_context,
        resize,
        source,
        destination,
        scratch,
    )
}

pub fn resize_sqr_pixel_c1_advanced<T: ResizeSqrPixelAdvancedC1>(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixelAdvanced,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T, C1>,
) -> Result<()> {
    let required_bytes = resize_sqr_pixel_c1_advanced_buffer_size::<T>(resize)?;
    let mut scratch = ScratchBuffer::create(required_bytes)?;
    resize_sqr_pixel_c1_advanced_with_scratch(
        stream_context,
        resize,
        source,
        destination,
        &mut scratch,
    )
}
