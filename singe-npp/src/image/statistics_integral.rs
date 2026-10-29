use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{
            validate_integral_destination_size, validate_rect_std_dev_sources, validate_same_size,
        },
        view::{C1, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Rectangle},
};

/// Computes an integral image, including its top and left borders.
/// Calls `nppiIntegral_8u32s_C1R_Ctx`.
///
/// Adds `value` to the destination integral-image pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn integral_u8_to_i32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, i32, C1>,
    value: i32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_integral_destination_size(source.size(), destination.size())?;
    // SAFETY: `nppiIntegral_8u32s_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiIntegral_8u32s_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            value,
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Computes an integral image, including its top and left borders.
/// Calls `nppiIntegral_8u32f_C1R_Ctx`.
///
/// Adds `value` to the destination integral-image pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn integral_u8_to_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
    value: f32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_integral_destination_size(source.size(), destination.size())?;
    // SAFETY: `nppiIntegral_8u32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiIntegral_8u32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            value,
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Computes an integral image and an integral of squared samples, including their top and left borders.
/// Calls `nppiSqrIntegral_8u32s_C1R_Ctx`.
///
/// Adds `value` to the destination integral-image pixels.
/// Adds `squared_value` to the `squared` output pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The `squared` output must have the same dimensions as `destination`.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn sqr_integral_u8_to_i32_i32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, i32, C1>,
    squared: &mut ImageOutputView<'_, i32, C1>,
    value: i32,
    squared_value: i32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(squared)?;

    validate_integral_destination_size(source.size(), destination.size())?;
    validate_same_size(destination.size(), squared.size())?;
    // SAFETY: `nppiSqrIntegral_8u32s_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiSqrIntegral_8u32s_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            squared.as_mut_ptr().cast(),
            squared.step(),
            source.size().into(),
            value,
            squared_value,
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Computes an integral image and an integral of squared samples, including their top and left borders.
/// Calls `nppiSqrIntegral_8u32s64f_C1R_Ctx`.
///
/// Adds `value` to the destination integral-image pixels.
/// Adds `squared_value` to the `squared` output pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The `squared` output must have the same dimensions as `destination`.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn sqr_integral_u8_to_i32_f64_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, i32, C1>,
    squared: &mut ImageOutputView<'_, f64, C1>,
    value: i32,
    squared_value: f64,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(squared)?;

    validate_integral_destination_size(source.size(), destination.size())?;
    validate_same_size(destination.size(), squared.size())?;
    // SAFETY: `nppiSqrIntegral_8u32s64f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiSqrIntegral_8u32s64f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            squared.as_mut_ptr().cast(),
            squared.step(),
            source.size().into(),
            value,
            squared_value,
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Computes an integral image and an integral of squared samples, including their top and left borders.
/// Calls `nppiSqrIntegral_8u32f64f_C1R_Ctx`.
///
/// Adds `value` to the destination integral-image pixels.
/// Adds `squared_value` to the `squared` output pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The `squared` output must have the same dimensions as `destination`.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn sqr_integral_u8_to_f32_f64_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
    squared: &mut ImageOutputView<'_, f64, C1>,
    value: f32,
    squared_value: f64,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(squared)?;

    validate_integral_destination_size(source.size(), destination.size())?;
    validate_same_size(destination.size(), squared.size())?;
    // SAFETY: `nppiSqrIntegral_8u32f64f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiSqrIntegral_8u32f64f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            squared.as_mut_ptr().cast(),
            squared.step(),
            source.size().into(),
            value,
            squared_value,
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Computes standard deviations over `rectangle` from integral images.
/// Calls `nppiRectStdDev_32f_C1R_Ctx`.
///
/// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
/// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
/// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// The integral inputs must describe the same source image and meet the native arithmetic requirements.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn rect_std_dev_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, f32, C1>,
    squared: &ImageView<'_, f64, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
    rectangle: Rectangle,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(squared)?;
    stream_context.validate_region(destination)?;

    validate_rect_std_dev_sources(source.size(), squared.size(), destination.size(), rectangle)?;
    // SAFETY: `nppiRectStdDev_32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRectStdDev_32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            squared.as_ptr().cast(),
            squared.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            rectangle.into(),
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Computes standard deviations over `rectangle` from integral images.
/// Calls `nppiRectStdDev_32s_C1RSfs_Ctx`.
///
/// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
/// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
/// Applies native fixed-point scaling with `scale_factor`.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
/// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// The integral inputs must describe the same source image and meet the native arithmetic requirements.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn rect_std_dev_i32_scaled_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, i32, C1>,
    squared: &ImageView<'_, i32, C1>,
    destination: &mut ImageOutputView<'_, i32, C1>,
    rectangle: Rectangle,
    scale_factor: i32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(squared)?;
    stream_context.validate_region(destination)?;

    validate_rect_std_dev_sources(source.size(), squared.size(), destination.size(), rectangle)?;
    // SAFETY: `nppiRectStdDev_32s_C1RSfs_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRectStdDev_32s_C1RSfs_Ctx(
            source.as_ptr().cast(),
            source.step(),
            squared.as_ptr().cast(),
            squared.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            rectangle.into(),
            scale_factor,
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Computes standard deviations over `rectangle` from integral images.
/// Calls `nppiRectStdDev_32s32f_C1R_Ctx`.
///
/// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
/// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
/// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// The integral inputs must describe the same source image and meet the native arithmetic requirements.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn rect_std_dev_i32_to_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, i32, C1>,
    squared: &ImageView<'_, f64, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
    rectangle: Rectangle,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(squared)?;
    stream_context.validate_region(destination)?;

    validate_rect_std_dev_sources(source.size(), squared.size(), destination.size(), rectangle)?;
    // SAFETY: `nppiRectStdDev_32s32f_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiRectStdDev_32s32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            squared.as_ptr().cast(),
            squared.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            rectangle.into(),
            stream_context.as_raw(),
        ))
    }?;
    Ok(())
}

/// Dispatches integral-image computation for a supported element type.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait IntegralToI32C1: DataTypeLike {
    /// Computes an integral image, including its top and left borders.
    ///
    /// Adds `value` to the destination integral-image pixels.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The destination width and height must each equal the source dimension plus one.
    /// Returns an error if either addition overflows or the destination has a different size.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        value: i32,
    ) -> Result<()>
    where
        Self: Sized;
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl IntegralToI32C1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        value: i32,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { integral_u8_to_i32_c1(stream_context, source, destination, value) }
    }
}

/// Computes an integral image, including its top and left borders.
///
/// Adds `value` to the destination integral-image pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn integral_to_i32<T: IntegralToI32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, i32, C1>,
    value: i32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, source, destination, value) }
}

/// Dispatches integral-image computation for a supported element type.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait IntegralToF32C1: DataTypeLike {
    /// Computes an integral image, including its top and left borders.
    ///
    /// Adds `value` to the destination integral-image pixels.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The destination width and height must each equal the source dimension plus one.
    /// Returns an error if either addition overflows or the destination has a different size.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
        value: f32,
    ) -> Result<()>
    where
        Self: Sized;
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl IntegralToF32C1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
        value: f32,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { integral_u8_to_f32_c1(stream_context, source, destination, value) }
    }
}

/// Computes an integral image, including its top and left borders.
///
/// Adds `value` to the destination integral-image pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn integral_to_f32<T: IntegralToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
    value: f32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, source, destination, value) }
}

/// Dispatches integral-image computation for a supported element type.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait IntegralC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Computes an integral image, including its top and left borders.
    ///
    /// Adds `value` to the destination integral-image pixels.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The destination width and height must each equal the source dimension plus one.
    /// Returns an error if either addition overflows or the destination has a different size.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch_integral(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Destination, C1>,
        value: Destination,
    ) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_integral_c1 {
    ($destination:ty, $source_trait:ident) => {
        // SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
        unsafe impl<T> IntegralC1<$destination> for T
        where
            T: $source_trait,
        {
            unsafe fn dispatch_integral(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, $destination, C1>,
                value: $destination,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { T::dispatch(stream_context, source, destination, value) }
            }
        }
    };
}

impl_integral_c1!(i32, IntegralToI32C1);

impl_integral_c1!(f32, IntegralToF32C1);

/// Computes an integral image, including its top and left borders.
///
/// Adds `value` to the destination integral-image pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn integral<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, Destination, C1>,
    value: Destination,
) -> Result<()>
where
    T: IntegralC1<Destination>,
    Destination: DataTypeLike,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch_integral(stream_context, source, destination, value) }
}

/// Dispatches integral images and integrals of squared samples.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait SqrIntegralToI32I32C1: DataTypeLike {
    /// Computes an integral image and an integral of squared samples, including their top and left borders.
    ///
    /// Adds `value` to the destination integral-image pixels.
    /// Adds `squared_value` to the `squared` output pixels.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The destination width and height must each equal the source dimension plus one.
    /// Returns an error if either addition overflows or the destination has a different size.
    /// The `squared` output must have the same dimensions as `destination`.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        squared: &mut ImageOutputView<'_, i32, C1>,
        value: i32,
        squared_value: i32,
    ) -> Result<()>
    where
        Self: Sized;
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl SqrIntegralToI32I32C1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        squared: &mut ImageOutputView<'_, i32, C1>,
        value: i32,
        squared_value: i32,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            sqr_integral_u8_to_i32_i32_c1(
                stream_context,
                source,
                destination,
                squared,
                value,
                squared_value,
            )
        }
    }
}

/// Computes an integral image and an integral of squared samples, including their top and left borders.
///
/// Adds `value` to the destination integral-image pixels.
/// Adds `squared_value` to the `squared` output pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The `squared` output must have the same dimensions as `destination`.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn squared_integral_to_i32_i32<T: SqrIntegralToI32I32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, i32, C1>,
    squared: &mut ImageOutputView<'_, i32, C1>,
    value: i32,
    squared_value: i32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(squared)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            destination,
            squared,
            value,
            squared_value,
        )
    }
}

/// Dispatches integral images and integrals of squared samples.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait SqrIntegralToI32F64C1: DataTypeLike {
    /// Computes an integral image and an integral of squared samples, including their top and left borders.
    ///
    /// Adds `value` to the destination integral-image pixels.
    /// Adds `squared_value` to the `squared` output pixels.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The destination width and height must each equal the source dimension plus one.
    /// Returns an error if either addition overflows or the destination has a different size.
    /// The `squared` output must have the same dimensions as `destination`.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        squared: &mut ImageOutputView<'_, f64, C1>,
        value: i32,
        squared_value: f64,
    ) -> Result<()>
    where
        Self: Sized;
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl SqrIntegralToI32F64C1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        squared: &mut ImageOutputView<'_, f64, C1>,
        value: i32,
        squared_value: f64,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            sqr_integral_u8_to_i32_f64_c1(
                stream_context,
                source,
                destination,
                squared,
                value,
                squared_value,
            )
        }
    }
}

/// Computes an integral image and an integral of squared samples, including their top and left borders.
///
/// Adds `value` to the destination integral-image pixels.
/// Adds `squared_value` to the `squared` output pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The `squared` output must have the same dimensions as `destination`.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn squared_integral_to_i32_f64<T: SqrIntegralToI32F64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, i32, C1>,
    squared: &mut ImageOutputView<'_, f64, C1>,
    value: i32,
    squared_value: f64,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(squared)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            destination,
            squared,
            value,
            squared_value,
        )
    }
}

/// Dispatches integral images and integrals of squared samples.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait SqrIntegralToF32F64C1: DataTypeLike {
    /// Computes an integral image and an integral of squared samples, including their top and left borders.
    ///
    /// Adds `value` to the destination integral-image pixels.
    /// Adds `squared_value` to the `squared` output pixels.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The destination width and height must each equal the source dimension plus one.
    /// Returns an error if either addition overflows or the destination has a different size.
    /// The `squared` output must have the same dimensions as `destination`.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
        squared: &mut ImageOutputView<'_, f64, C1>,
        value: f32,
        squared_value: f64,
    ) -> Result<()>
    where
        Self: Sized;
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl SqrIntegralToF32F64C1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
        squared: &mut ImageOutputView<'_, f64, C1>,
        value: f32,
        squared_value: f64,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            sqr_integral_u8_to_f32_f64_c1(
                stream_context,
                source,
                destination,
                squared,
                value,
                squared_value,
            )
        }
    }
}

/// Computes an integral image and an integral of squared samples, including their top and left borders.
///
/// Adds `value` to the destination integral-image pixels.
/// Adds `squared_value` to the `squared` output pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The `squared` output must have the same dimensions as `destination`.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn squared_integral_to_f32_f64<T: SqrIntegralToF32F64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, f32, C1>,
    squared: &mut ImageOutputView<'_, f64, C1>,
    value: f32,
    squared_value: f64,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(squared)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            destination,
            squared,
            value,
            squared_value,
        )
    }
}

/// Dispatches integral images and integrals of squared samples.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait SqrIntegralC1<Destination, Squared>: DataTypeLike
where
    Destination: DataTypeLike,
    Squared: DataTypeLike,
{
    /// Computes an integral image and an integral of squared samples, including their top and left borders.
    ///
    /// Adds `value` to the destination integral-image pixels.
    /// Adds `squared_value` to the `squared` output pixels.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// The destination width and height must each equal the source dimension plus one.
    /// Returns an error if either addition overflows or the destination has a different size.
    /// The `squared` output must have the same dimensions as `destination`.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch_sqr_integral(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Destination, C1>,
        squared: &mut ImageOutputView<'_, Squared, C1>,
        value: Destination,
        squared_value: Squared,
    ) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_sqr_integral_c1 {
    ($destination:ty, $squared:ty, $source_trait:ident) => {
        // SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
        unsafe impl<T> SqrIntegralC1<$destination, $squared> for T
        where
            T: $source_trait,
        {
            unsafe fn dispatch_sqr_integral(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, $destination, C1>,
                squared: &mut ImageOutputView<'_, $squared, C1>,
                value: $destination,
                squared_value: $squared,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    T::dispatch(
                        stream_context,
                        source,
                        destination,
                        squared,
                        value,
                        squared_value,
                    )
                }
            }
        }
    };
}

impl_sqr_integral_c1!(i32, i32, SqrIntegralToI32I32C1);

impl_sqr_integral_c1!(i32, f64, SqrIntegralToI32F64C1);

impl_sqr_integral_c1!(f32, f64, SqrIntegralToF32F64C1);

/// Computes an integral image and an integral of squared samples, including their top and left borders.
///
/// Adds `value` to the destination integral-image pixels.
/// Adds `squared_value` to the `squared` output pixels.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// The destination width and height must each equal the source dimension plus one.
/// Returns an error if either addition overflows or the destination has a different size.
/// The `squared` output must have the same dimensions as `destination`.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn sqr_integral<T, Destination, Squared>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, Destination, C1>,
    squared: &mut ImageOutputView<'_, Squared, C1>,
    value: Destination,
    squared_value: Squared,
) -> Result<()>
where
    T: SqrIntegralC1<Destination, Squared>,
    Destination: DataTypeLike,
    Squared: DataTypeLike,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.validate_region(squared)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch_sqr_integral(
            stream_context,
            source,
            destination,
            squared,
            value,
            squared_value,
        )
    }
}

/// Dispatches rectangle standard deviations from integral images.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait RectStandardDeviation: DataTypeLike {
    /// Element type of the integral of squared source samples.
    type Squared: DataTypeLike;

    /// Element type of the standard-deviation output.
    type Destination: DataTypeLike;

    /// Computes standard deviations over `rectangle` from integral images.
    ///
    /// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
    /// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
    /// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// The integral inputs must describe the same source image and meet the native arithmetic requirements.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self::Squared, C1>,
        destination: &mut ImageOutputView<'_, Self::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()>
    where
        Self: Sized;
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl RectStandardDeviation for f32 {
    type Squared = f64;

    type Destination = Self;

    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self::Squared, C1>,
        destination: &mut ImageOutputView<'_, Self::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { rect_std_dev_f32_c1(stream_context, source, squared, destination, rectangle) }
    }
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl RectStandardDeviation for i32 {
    type Squared = f64;

    type Destination = f32;

    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self::Squared, C1>,
        destination: &mut ImageOutputView<'_, Self::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            rect_std_dev_i32_to_f32_c1(stream_context, source, squared, destination, rectangle)
        }
    }
}

/// Computes standard deviations over `rectangle` from integral images.
///
/// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
/// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
/// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// The integral inputs must describe the same source image and meet the native arithmetic requirements.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn rect_standard_deviation<T: RectStandardDeviation>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    squared: &ImageView<'_, T::Squared, C1>,
    destination: &mut ImageOutputView<'_, T::Destination, C1>,
    rectangle: Rectangle,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(squared)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, source, squared, destination, rectangle) }
}

/// Computes standard deviations over `rectangle` from integral images.
///
/// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
/// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
/// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// The integral inputs must describe the same source image and meet the native arithmetic requirements.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn rect_standard_deviation_to<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    squared: &ImageView<'_, T::Squared, C1>,
    destination: &mut ImageOutputView<'_, Destination, C1>,
    rectangle: Rectangle,
) -> Result<()>
where
    T: RectStandardDeviation<Destination = Destination>,
    Destination: DataTypeLike,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(squared)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, source, squared, destination, rectangle) }
}

/// Dispatches rectangle standard deviations from integral images.
///
/// # Safety
///
/// Implementations must preserve native element representations, arithmetic, output extents, and top and left border semantics.
/// Uphold the documented queued execution contract and initialize every logical output element when the submitted operation completes successfully.
pub unsafe trait RectStandardDeviationScaled: DataTypeLike {
    /// Computes standard deviations over `rectangle` from integral images.
    ///
    /// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
    /// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
    /// Applies native fixed-point scaling with `scale_factor`.
    /// Enqueues work on the selected stream. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
    /// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
    /// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready for native execution.
    /// The integral inputs must describe the same source image and meet the native arithmetic requirements.
    /// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
    /// Retain all allocations through queued completion, including submission errors and every captured replay.
    /// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
    /// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: Sized;
}

// SAFETY: the concrete NPP entry point matches the declared ABI and result ranges.
unsafe impl RectStandardDeviationScaled for i32 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Self, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            rect_std_dev_i32_scaled_c1(
                stream_context,
                source,
                squared,
                destination,
                rectangle,
                scale_factor,
            )
        }
    }
}

/// Computes standard deviations over `rectangle` from integral images.
///
/// `source` contains the integral image and `squared` contains the integral of squared samples for the same underlying image.
/// `rectangle` specifies the window relative to each destination pixel. Its origin must be nonnegative and its width and height must be positive.
/// Applies native fixed-point scaling with `scale_factor`.
/// Enqueues work on the selected stream. Row padding remains unwritten.
///
/// # Errors
///
/// Both input sizes must equal `(destination.width + rectangle.x + rectangle.width, destination.height + rectangle.y + rectangle.height)`.
/// Returns an error if rectangle coordinates are negative, its dimensions are nonpositive, a size calculation overflows, or either input size differs from that exact size.
/// The provided implementations also return errors if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready for native execution.
/// The integral inputs must describe the same source image and meet the native arithmetic requirements.
/// Keep inputs read-only and every output exclusively writable and disjoint from all inputs and other outputs during native use.
/// Retain all allocations through queued completion, including submission errors and every captured replay.
/// Native pixel layouts, byte strides, alignment, arithmetic ranges, and allocation bounds must be valid.
/// Successful submission covers every logical output element; synchronize before treating newly written outputs as initialized.
pub unsafe fn rect_standard_deviation_scaled<T: RectStandardDeviationScaled>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    squared: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, T, C1>,
    rectangle: Rectangle,
    scale_factor: i32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(squared)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::dispatch(
            stream_context,
            source,
            squared,
            destination,
            rectangle,
            scale_factor,
        )
    }
}
