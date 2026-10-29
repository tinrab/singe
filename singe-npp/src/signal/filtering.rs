use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    signal::view::{SignalOutputView, SignalView},
    try_ffi,
    types::DataTypeLike,
    utility::{to_u64, to_usize},
    workspace::ScratchBuffer,
};

/// Returns the scratch capacity, in bytes, for a cumulative sum with one extra output element.
///
/// # Errors
///
/// Returns an error if adding the extra output element overflows, a length or scratch-size conversion fails, or NPP rejects the size query.
pub fn integral_i32_buffer_size(source: &SignalView<'_, i32>) -> Result<usize> {
    let mut bytes = 0;
    let len = source
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::out_of_range("len"))?;
    let raw_len = to_u64(len, "len")?;
    // SAFETY: `nppsIntegralGetBufferSize_32s` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppsIntegralGetBufferSize_32s(raw_len, &raw mut bytes,))?;
    }
    to_usize(bytes, "scratch bytes")
}

/// Computes the native cumulative sum of the source signal.
///
/// The destination must contain `source.len() + 1` elements.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
/// Returns an error if the destination length is not one greater than the source length, a required size or length conversion fails, or NPP reports a failure.
/// An undersized scratch buffer also returns an error.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
/// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
/// `scratch` must belong to the same CUDA context and remain live and exclusively writable through completion, disjoint from source and result storage.
pub unsafe fn integral_i32_with_scratch(
    stream_context: &StreamContext,
    source: &SignalView<'_, i32>,
    destination: &mut SignalOutputView<'_, i32>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_integral_destination_len(source, destination)?;
    let required_bytes = integral_i32_buffer_size(source)?;
    scratch.require(required_bytes)?;

    // SAFETY: `nppsIntegral_32s_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppsIntegral_32s_Ctx(
            source.as_ptr(),
            destination.as_mut_ptr(),
            to_u64(destination.len(), "destination length")?,
            scratch.as_mut_ptr(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Native dispatch for the corresponding named NPP operation.
///
/// # Safety
/// Implementations must match the native ABI, access roles, written ranges,
/// and status semantics. Successful completed execution must initialize its
/// documented outputs; metadata queries must return the required capacity.
pub unsafe trait Integral: DataTypeLike {
    /// Returns the scratch capacity, in bytes, for a cumulative sum with one extra output element.
    ///
    /// # Errors
    ///
    /// Provided implementations return an error if adding the extra output element overflows, a length or scratch-size conversion fails, or NPP rejects the size query.
    fn integral_buffer_size(source: &SignalView<'_, Self>) -> Result<usize>;

    /// Computes the native cumulative sum of the source signal.
    ///
    /// The destination must contain `source.len() + 1` elements.
    ///
    /// # Errors
    ///
    /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
    /// Returns an error if the destination length is not one greater than the source length, a required size or length conversion fails, or NPP reports a failure.
    /// An undersized scratch buffer also returns an error.
    ///
    /// # Safety
    ///
    /// Follow the [raw signal execution contract](crate::signal::raw).
    /// Source values must be initialized and ready when the selected stream reads them.
    /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
    /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
    /// `scratch` must belong to the same CUDA context and remain live and exclusively writable through completion, disjoint from source and result storage.
    unsafe fn integral_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalOutputView<'_, Self>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

unsafe impl Integral for i32 {
    fn integral_buffer_size(source: &SignalView<'_, Self>) -> Result<usize> {
        integral_i32_buffer_size(source)
    }

    unsafe fn integral_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalOutputView<'_, Self>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { integral_i32_with_scratch(stream_context, source, destination, scratch) }
    }
}

/// Returns the scratch capacity, in bytes, for a cumulative sum with one extra output element.
///
/// # Errors
///
/// Returns an error if adding the extra output element overflows, a length or scratch-size conversion fails, or NPP rejects the size query.
pub fn integral_buffer_size<T: Integral>(source: &SignalView<'_, T>) -> Result<usize> {
    T::integral_buffer_size(source)
}

/// Computes the native cumulative sum of the source signal.
///
/// The destination must contain `source.len() + 1` elements.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
/// Returns an error if the destination length is not one greater than the source length, a required size or length conversion fails, or NPP reports a failure.
/// An undersized scratch buffer also returns an error.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
/// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
/// `scratch` must belong to the same CUDA context and remain live and exclusively writable through completion, disjoint from source and result storage.
pub unsafe fn integral_with_scratch<T: Integral>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
    destination: &mut SignalOutputView<'_, T>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::integral_with_scratch(stream_context, source, destination, scratch) }
}

/// Computes the native cumulative sum of the source signal.
///
/// The destination must contain `source.len() + 1` elements.
/// Allocates temporary scratch storage for this call.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
/// Returns an error if the destination length is not one greater than the source length, a required size or length conversion fails, or NPP reports a failure.
/// Scratch allocation errors are propagated.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
/// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
pub unsafe fn integral<T: Integral>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
    destination: &mut SignalOutputView<'_, T>,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        let required_bytes = integral_buffer_size(source)?;
        let mut scratch = ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
        integral_with_scratch(stream_context, source, destination, &mut scratch)
    }
}

fn validate_integral_destination_len<T>(
    source: &SignalView<'_, T>,
    destination: &SignalOutputView<'_, T>,
) -> Result<()> {
    let expected = source
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::out_of_range("len"))?;
    if destination.len() != expected {
        return Err(Error::LengthMismatch {
            name: "destination".into(),
            expected,
            actual: destination.len(),
        });
    }

    Ok(())
}
