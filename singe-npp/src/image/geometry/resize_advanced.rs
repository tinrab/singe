use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry::ResizeSqrPixelAdvanced,
        geometry_helpers::rect_size,
        view::{C1, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
    utility::to_usize,
    workspace::ScratchBuffer,
};

/// Returns the scratch size in bytes for advanced single-channel square-pixel resizing.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI sizes or interpolation, or the returned size cannot be represented as `usize`.
pub fn resize_sqr_pixel_u8_c1_advanced_buffer_size(
    resize: &ResizeSqrPixelAdvanced,
) -> Result<usize> {
    let mut bytes = 0;
    // SAFETY: `nppiResizeAdvancedGetBufferHostSize_8u_C1R` receives valid output locations.
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

/// Resizes a single-channel image with the configured scale factors and interpolation using caller-provided scratch.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if the source ROI has invalid dimensions or overflowing endpoints, or the destination ROI lies outside the destination image.
/// Returns an error if CUDA context binding fails, any image or scratch allocation belongs to another context, or scratch is smaller than the queried requirement.
/// Returns an error if the buffer-size query or native resize fails.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and the destination and scratch exclusively writable and mutually disjoint during native use.
/// Retain image and scratch allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and ROI footprints must satisfy the native routine.
pub unsafe fn resize_sqr_pixel_u8_c1_advanced_with_scratch(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixelAdvanced,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C1>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    resize.source_roi.validate()?;
    resize.destination_roi.validate_within(destination.size())?;

    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(scratch)?;

    scratch.require(resize_sqr_pixel_u8_c1_advanced_buffer_size(resize)?)?;

    // SAFETY: `nppiResizeSqrPixel_8u_C1R_Advanced_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Element dispatch for advanced single-channel square-pixel resizing and its scratch-size query.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI and use a compatible NPP routine.
/// They must uphold the validation and completion contracts of the declared methods and the [raw image execution contract](crate::image::raw).
pub unsafe trait ResizeSqrPixelAdvancedC1: DataTypeLike {
    /// Returns the scratch size in bytes for advanced single-channel square-pixel resizing.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI sizes or interpolation, or the returned size cannot be represented as `usize`.
    fn resize_sqr_pixel_c1_advanced_buffer_size(resize: &ResizeSqrPixelAdvanced) -> Result<usize>;

    /// Resizes a single-channel image with the configured scale factors and interpolation using caller-provided scratch.
    ///
    /// Enqueues work on the selected stream.
    /// Pixels outside the transformed source intersection and padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if the source ROI has invalid dimensions or overflowing endpoints, or the destination ROI lies outside the destination image.
    /// Returns an error if CUDA context binding fails, any image or scratch allocation belongs to another context, or scratch is smaller than the queried requirement.
    /// Returns an error if the buffer-size query or native resize fails.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source pixels must be initialized and ready when the selected stream reads them.
    /// Keep the source read-only and the destination and scratch exclusively writable and mutually disjoint during native use.
    /// Retain image and scratch allocations through queued completion, including submission errors and captured graph replay.
    /// Image layouts, byte strides, alignment, and ROI footprints must satisfy the native routine.
    unsafe fn resize_sqr_pixel_c1_advanced_with_scratch(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self, C1>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

unsafe impl ResizeSqrPixelAdvancedC1 for u8 {
    fn resize_sqr_pixel_c1_advanced_buffer_size(resize: &ResizeSqrPixelAdvanced) -> Result<usize> {
        resize_sqr_pixel_u8_c1_advanced_buffer_size(resize)
    }

    unsafe fn resize_sqr_pixel_c1_advanced_with_scratch(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self, C1>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            resize_sqr_pixel_u8_c1_advanced_with_scratch(
                stream_context,
                resize,
                source,
                destination,
                scratch,
            )
        }
    }
}

/// Returns the scratch size in bytes for advanced single-channel square-pixel resizing.
///
/// # Errors
///
/// Returns an error if NPP rejects the ROI sizes or interpolation, or the returned size cannot be represented as `usize`.
pub fn resize_sqr_pixel_c1_advanced_buffer_size<T: ResizeSqrPixelAdvancedC1>(
    resize: &ResizeSqrPixelAdvanced,
) -> Result<usize> {
    T::resize_sqr_pixel_c1_advanced_buffer_size(resize)
}

/// Resizes a single-channel image with the configured scale factors and interpolation using caller-provided scratch.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if the source ROI has invalid dimensions or overflowing endpoints, or the destination ROI lies outside the destination image.
/// Returns an error if CUDA context binding fails, any image or scratch allocation belongs to another context, or scratch is smaller than the queried requirement.
/// Returns an error if the buffer-size query or native resize fails.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and the destination and scratch exclusively writable and mutually disjoint during native use.
/// Retain image and scratch allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and ROI footprints must satisfy the native routine.
pub unsafe fn resize_sqr_pixel_c1_advanced_with_scratch<T: ResizeSqrPixelAdvancedC1>(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixelAdvanced,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T, C1>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::resize_sqr_pixel_c1_advanced_with_scratch(
            stream_context,
            resize,
            source,
            destination,
            scratch,
        )
    }
}

/// Resizes a single-channel image with the configured scale factors and interpolation using an internally allocated scratch buffer.
///
/// Enqueues work on the selected stream.
/// Pixels outside the transformed source intersection and padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if the source ROI has invalid dimensions or overflowing endpoints, or the destination ROI lies outside the destination image.
/// Returns an error if scratch allocation or CUDA context binding fails, or either image belongs to another context.
/// Returns an error if the buffer-size query or native resize fails.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source pixels must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and the destination exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and ROI footprints must satisfy the native routine.
pub unsafe fn resize_sqr_pixel_c1_advanced<T: ResizeSqrPixelAdvancedC1>(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixelAdvanced,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T, C1>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        let required_bytes = resize_sqr_pixel_c1_advanced_buffer_size::<T>(resize)?;
        let mut scratch = ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
        resize_sqr_pixel_c1_advanced_with_scratch(
            stream_context,
            resize,
            source,
            destination,
            &mut scratch,
        )
    }
}
