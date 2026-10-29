//! Raw complex magnitude operations queued on the selected stream.
//! # Safety
//! Retain allocations through completion or captured replay and establish access ordering.
//! Successful execution writes every logical output pixel; initialized values require completion.

use singe_cuda::types::Complex32;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{C1, ImageOutputView, ImageView},
    try_ffi,
    types::{DataTypeLike, Size},
};

/// Dispatches complex magnitude into a real-valued image.
///
/// # Safety
///
/// Implementations must preserve native complex and real element representations and
/// the declared output type. Uphold the method's geometry, memory-access, and queued
/// execution contracts.
pub unsafe trait MagnitudeC1: DataTypeLike + Sized {
    /// Real element type used for each output magnitude sample.
    type Output: DataTypeLike;

    /// Computes the complex magnitude `sqrt(real * real + imaginary * imaginary)`
    /// for each source pixel, writing the real-valued result to `destination`.
    ///
    /// Queues work on the selected stream without synchronizing. Source and destination
    /// must have equal pixel dimensions. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA
    /// context, image dimensions differ, or NPP rejects the magnitude operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw). Source samples must be
    /// initialized and ready for the selected stream. Keep source storage read-only and
    /// destination storage exclusively writable and disjoint from the source during native use.
    /// Retain both allocations through queued completion, including captured graph replay and
    /// submission errors. Image layouts, byte strides, alignment, and allocation bounds must
    /// satisfy the native operation.
    unsafe fn magnitude(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
    ) -> Result<()>;
}

/// Computes the complex magnitude `sqrt(real * real + imaginary * imaginary)`
/// for each source pixel, writing the real-valued result to `destination`.
///
/// Queues work on the selected stream without synchronizing. Source and destination
/// must have equal pixel dimensions. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA
/// context, image dimensions differ, or NPP rejects the magnitude operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Source samples must be
/// initialized and ready for the selected stream. Keep source storage read-only and
/// destination storage exclusively writable and disjoint from the source during native use.
/// Retain both allocations through queued completion, including captured graph replay and
/// submission errors. Image layouts, byte strides, alignment, and allocation bounds must
/// satisfy the native operation.
pub unsafe fn magnitude<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
) -> Result<()>
where
    T: MagnitudeC1,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::magnitude(stream_context, source, destination) }
}

/// Dispatches squared complex magnitude into a real-valued image.
///
/// # Safety
///
/// Implementations must preserve native complex and real element representations and
/// the declared output type. Uphold the method's geometry, memory-access, and queued
/// execution contracts.
pub unsafe trait MagnitudeSquaredC1: DataTypeLike + Sized {
    /// Real element type used for each output magnitude sample.
    type Output: DataTypeLike;

    /// Computes the squared magnitude `real * real + imaginary * imaginary`
    /// for each source pixel, writing the real-valued result to `destination`.
    ///
    /// Queues work on the selected stream without synchronizing. Source and destination
    /// must have equal pixel dimensions. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA
    /// context, image dimensions differ, or NPP rejects the magnitude operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw). Source samples must be
    /// initialized and ready for the selected stream. Keep source storage read-only and
    /// destination storage exclusively writable and disjoint from the source during native use.
    /// Retain both allocations through queued completion, including captured graph replay and
    /// submission errors. Image layouts, byte strides, alignment, and allocation bounds must
    /// satisfy the native operation.
    unsafe fn magnitude_squared(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
    ) -> Result<()>;
}

/// Computes the squared magnitude `real * real + imaginary * imaginary`
/// for each source pixel, writing the real-valued result to `destination`.
///
/// Queues work on the selected stream without synchronizing. Source and destination
/// must have equal pixel dimensions. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA
/// context, image dimensions differ, or NPP rejects the magnitude operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Source samples must be
/// initialized and ready for the selected stream. Keep source storage read-only and
/// destination storage exclusively writable and disjoint from the source during native use.
/// Retain both allocations through queued completion, including captured graph replay and
/// submission errors. Image layouts, byte strides, alignment, and allocation bounds must
/// satisfy the native operation.
pub unsafe fn magnitude_squared<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T::Output, C1>,
) -> Result<()>
where
    T: MagnitudeSquaredC1,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::magnitude_squared(stream_context, source, destination) }
}

/// Computes the complex magnitude `sqrt(real * real + imaginary * imaginary)`
/// for each source pixel, writing the real-valued result to `destination`.
///
/// Queues work on the selected stream without synchronizing. Source and destination
/// must have equal pixel dimensions. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA
/// context, image dimensions differ, or NPP rejects the magnitude operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Source samples must be
/// initialized and ready for the selected stream. Keep source storage read-only and
/// destination storage exclusively writable and disjoint from the source during native use.
/// Retain both allocations through queued completion, including captured graph replay and
/// submission errors. Image layouts, byte strides, alignment, and allocation bounds must
/// satisfy the native operation.
pub unsafe fn magnitude_f32_complex_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, Complex32, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiMagnitude_32fc32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiMagnitude_32fc32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Computes the squared magnitude `real * real + imaginary * imaginary`
/// for each source pixel, writing the real-valued result to `destination`.
///
/// Queues work on the selected stream without synchronizing. Source and destination
/// must have equal pixel dimensions. Row padding remains unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA
/// context, image dimensions differ, or NPP rejects the magnitude operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Source samples must be
/// initialized and ready for the selected stream. Keep source storage read-only and
/// destination storage exclusively writable and disjoint from the source during native use.
/// Retain both allocations through queued completion, including captured graph replay and
/// submission errors. Image layouts, byte strides, alignment, and allocation bounds must
/// satisfy the native operation.
pub unsafe fn magnitude_squared_f32_complex_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, Complex32, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source.size(), destination.size())?;

    // SAFETY: `nppiMagnitudeSqr_32fc32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiMagnitudeSqr_32fc32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

unsafe impl MagnitudeC1 for Complex32 {
    type Output = f32;

    unsafe fn magnitude(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { magnitude_f32_complex_c1(stream_context, source, destination) }
    }
}

unsafe impl MagnitudeSquaredC1 for Complex32 {
    type Output = f32;

    unsafe fn magnitude_squared(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self::Output, C1>,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { magnitude_squared_f32_complex_c1(stream_context, source, destination) }
    }
}

fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}
