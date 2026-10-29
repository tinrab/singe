use singe_npp_sys as sys;

pub use crate::signal::statistics_dispatch::*;
use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
    try_ffi,
    types::DataTypeLike,
    utility::{to_u64, to_usize, validate_same_len},
    workspace::ScratchBuffer,
};

#[macro_use]
mod statistics_macros;

mod scalar_ops;

pub use scalar_ops::*;

mod metric_ops;

pub use metric_ops::*;

/// Native dispatch for the corresponding named NPP operation.
///
/// # Safety
/// Implementations must match the native ABI, access roles, written ranges,
/// and status semantics. Successful completed execution must initialize its
/// documented outputs; metadata queries must return the required capacity.
pub unsafe trait CountInRange: DataTypeLike {
    /// Returns the scratch capacity, in bytes, for counting values in the source range.
    ///
    /// # Errors
    ///
    /// Provided implementations return an error if CUDA context binding fails, a length or scratch-size conversion fails, or NPP rejects the query.
    fn count_in_range_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
    ) -> Result<usize>;

    /// Counts source values in the range selected by `lower_bound` and `upper_bound`.
    ///
    /// Writes the result into the single element of `count`.
    ///
    /// # Errors
    ///
    /// Provided implementations return an error if CUDA context binding fails or the signals belong to another CUDA context.
    /// The count view must contain one element; other lengths return an error.
    /// Returns an error if a length or scratch-size conversion fails, or NPP reports a failure.
    /// An undersized scratch buffer also returns an error.
    ///
    /// # Safety
    ///
    /// Follow the [raw signal execution contract](crate::signal::raw).
    /// Source values must be initialized and ready when the selected stream reads them.
    /// Keep the count storage exclusively writable and disjoint from the source.
    /// Retain the source, count storage, and stream context through queued completion, including submission errors and captured graph replay.
    /// `scratch` must belong to the same CUDA context and remain live and exclusively writable through completion, disjoint from source and result storage.
    unsafe fn count_in_range_to_device_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        count: &mut SignalOutputView<'_, i32>,
        lower_bound: Self,
        upper_bound: Self,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

unsafe impl CountInRange for i32 {
    fn count_in_range_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
    ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
        count_in_range_i32_buffer_size(stream_context, source)
    }

    unsafe fn count_in_range_to_device_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        count: &mut SignalOutputView<'_, i32>,
        lower_bound: Self,
        upper_bound: Self,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(count)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            count_in_range_i32_to_device_with_scratch(
                stream_context,
                source,
                count,
                lower_bound,
                upper_bound,
                scratch,
            )
        }
    }
}

/// Returns the scratch capacity, in bytes, for counting values in the source range.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a length or scratch-size conversion fails, or NPP rejects the query.
pub fn count_in_range_buffer_size<T: CountInRange>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
) -> Result<usize> {
    stream_context.cuda_context().bind()?;
    T::count_in_range_buffer_size(stream_context, source)
}

/// Counts source values in the range selected by `lower_bound` and `upper_bound`.
///
/// Writes the result into the single element of `count`.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or the signals belong to another CUDA context.
/// The count view must contain one element; other lengths return an error.
/// Returns an error if a length or scratch-size conversion fails, or NPP reports a failure.
/// An undersized scratch buffer also returns an error.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the count storage exclusively writable and disjoint from the source.
/// Retain the source, count storage, and stream context through queued completion, including submission errors and captured graph replay.
/// `scratch` must belong to the same CUDA context and remain live and exclusively writable through completion, disjoint from source and result storage.
pub unsafe fn count_in_range_to_device_with_scratch<T: CountInRange>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
    count: &mut SignalOutputView<'_, i32>,
    lower_bound: T,
    upper_bound: T,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(count)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::count_in_range_to_device_with_scratch(
            stream_context,
            source,
            count,
            lower_bound,
            upper_bound,
            scratch,
        )
    }
}

/// Counts source values in the range selected by `lower_bound` and `upper_bound`.
///
/// Writes the result into the single element of `count`.
/// Allocates temporary scratch storage for this call.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or the signals belong to another CUDA context.
/// The count view must contain one element; other lengths return an error.
/// Returns an error if a length or scratch-size conversion fails, or NPP reports a failure.
/// Scratch allocation errors are propagated.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the count storage exclusively writable and disjoint from the source.
/// Retain the source, count storage, and stream context through queued completion, including submission errors and captured graph replay.
pub unsafe fn count_in_range_to_device<T: CountInRange>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
    count: &mut SignalOutputView<'_, i32>,
    lower_bound: T,
    upper_bound: T,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(count)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        let required_bytes = count_in_range_buffer_size(stream_context, source)?;
        let mut scratch = ScratchBuffer::create(stream_context.cuda_context(), required_bytes)?;
        count_in_range_to_device_with_scratch(
            stream_context,
            source,
            count,
            lower_bound,
            upper_bound,
            &mut scratch,
        )
    }
}

impl_pair_statistic!(
    "Finds the minimum and maximum source values.",
    min_max_u8_buffer_size,
    min_max_u8_to_device_with_scratch,
    min_max_u8_to_device,
    u8,
    nppsMinMaxGetBufferSize_8u_Ctx,
    nppsMinMax_8u_Ctx
);

impl_pair_statistic!(
    "Finds the minimum and maximum source values.",
    min_max_u16_buffer_size,
    min_max_u16_to_device_with_scratch,
    min_max_u16_to_device,
    u16,
    nppsMinMaxGetBufferSize_16u_Ctx,
    nppsMinMax_16u_Ctx
);

impl_pair_statistic!(
    "Finds the minimum and maximum source values.",
    min_max_i16_buffer_size,
    min_max_i16_to_device_with_scratch,
    min_max_i16_to_device,
    i16,
    nppsMinMaxGetBufferSize_16s_Ctx,
    nppsMinMax_16s_Ctx
);

impl_pair_statistic!(
    "Finds the minimum and maximum source values.",
    min_max_u32_buffer_size,
    min_max_u32_to_device_with_scratch,
    min_max_u32_to_device,
    u32,
    nppsMinMaxGetBufferSize_32u_Ctx,
    nppsMinMax_32u_Ctx
);

impl_pair_statistic!(
    "Finds the minimum and maximum source values.",
    min_max_i32_buffer_size,
    min_max_i32_to_device_with_scratch,
    min_max_i32_to_device,
    i32,
    nppsMinMaxGetBufferSize_32s_Ctx,
    nppsMinMax_32s_Ctx
);

impl_pair_statistic!(
    "Finds the minimum and maximum source values.",
    min_max_f32_buffer_size,
    min_max_f32_to_device_with_scratch,
    min_max_f32_to_device,
    f32,
    nppsMinMaxGetBufferSize_32f_Ctx,
    nppsMinMax_32f_Ctx
);

impl_pair_statistic!(
    "Finds the minimum and maximum source values.",
    min_max_f64_buffer_size,
    min_max_f64_to_device_with_scratch,
    min_max_f64_to_device,
    f64,
    nppsMinMaxGetBufferSize_64f_Ctx,
    nppsMinMax_64f_Ctx
);

impl_indexed_pair_statistic!(
    "Finds the minimum and maximum source values and their indices.",
    min_max_index_u8_buffer_size,
    min_max_index_u8_to_device_with_scratch,
    min_max_index_u8_to_device,
    u8,
    nppsMinMaxIndxGetBufferSize_8u_Ctx,
    nppsMinMaxIndx_8u_Ctx
);

impl_indexed_pair_statistic!(
    "Finds the minimum and maximum source values and their indices.",
    min_max_index_u16_buffer_size,
    min_max_index_u16_to_device_with_scratch,
    min_max_index_u16_to_device,
    u16,
    nppsMinMaxIndxGetBufferSize_16u_Ctx,
    nppsMinMaxIndx_16u_Ctx
);

impl_indexed_pair_statistic!(
    "Finds the minimum and maximum source values and their indices.",
    min_max_index_i16_buffer_size,
    min_max_index_i16_to_device_with_scratch,
    min_max_index_i16_to_device,
    i16,
    nppsMinMaxIndxGetBufferSize_16s_Ctx,
    nppsMinMaxIndx_16s_Ctx
);

impl_indexed_pair_statistic!(
    "Finds the minimum and maximum source values and their indices.",
    min_max_index_u32_buffer_size,
    min_max_index_u32_to_device_with_scratch,
    min_max_index_u32_to_device,
    u32,
    nppsMinMaxIndxGetBufferSize_32u_Ctx,
    nppsMinMaxIndx_32u_Ctx
);

impl_indexed_pair_statistic!(
    "Finds the minimum and maximum source values and their indices.",
    min_max_index_i32_buffer_size,
    min_max_index_i32_to_device_with_scratch,
    min_max_index_i32_to_device,
    i32,
    nppsMinMaxIndxGetBufferSize_32s_Ctx,
    nppsMinMaxIndx_32s_Ctx
);

impl_indexed_pair_statistic!(
    "Finds the minimum and maximum source values and their indices.",
    min_max_index_f32_buffer_size,
    min_max_index_f32_to_device_with_scratch,
    min_max_index_f32_to_device,
    f32,
    nppsMinMaxIndxGetBufferSize_32f_Ctx,
    nppsMinMaxIndx_32f_Ctx
);

impl_indexed_pair_statistic!(
    "Finds the minimum and maximum source values and their indices.",
    min_max_index_f64_buffer_size,
    min_max_index_f64_to_device_with_scratch,
    min_max_index_f64_to_device,
    f64,
    nppsMinMaxIndxGetBufferSize_64f_Ctx,
    nppsMinMaxIndx_64f_Ctx
);

impl_mean_stddev!(
    "Computes the mean and standard deviation of the source elements.",
    mean_standard_deviation_f32_buffer_size,
    mean_standard_deviation_f32_to_device_with_scratch,
    mean_standard_deviation_f32_to_device,
    f32,
    nppsMeanStdDevGetBufferSize_32f_Ctx,
    nppsMeanStdDev_32f_Ctx
);

impl_mean_stddev!(
    "Computes the mean and standard deviation of the source elements.",
    mean_standard_deviation_f64_buffer_size,
    mean_standard_deviation_f64_to_device_with_scratch,
    mean_standard_deviation_f64_to_device,
    f64,
    nppsMeanStdDevGetBufferSize_64f_Ctx,
    nppsMeanStdDev_64f_Ctx
);

impl_scaled_mean_stddev!(
    "Computes the mean and standard deviation of the source elements.",
    mean_standard_deviation_i16_scaled_buffer_size,
    mean_standard_deviation_i16_scaled_to_device_with_scratch,
    mean_standard_deviation_i16_scaled_to_device,
    i16,
    i16,
    nppsMeanStdDevGetBufferSize_16s_Sfs_Ctx,
    nppsMeanStdDev_16s_Sfs_Ctx
);

impl_scaled_mean_stddev!(
    "Computes the mean and standard deviation of the source elements.",
    mean_standard_deviation_i16_to_i32_scaled_buffer_size,
    mean_standard_deviation_i16_to_i32_scaled_to_device_with_scratch,
    mean_standard_deviation_i16_to_i32_scaled_to_device,
    i16,
    i32,
    nppsMeanStdDevGetBufferSize_16s32s_Sfs_Ctx,
    nppsMeanStdDev_16s32s_Sfs_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the minimum of its current value and the corresponding source element.",
    min_every_u8_in_place,
    u8,
    nppsMinEvery_8u_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the minimum of its current value and the corresponding source element.",
    min_every_u16_in_place,
    u16,
    nppsMinEvery_16u_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the minimum of its current value and the corresponding source element.",
    min_every_i16_in_place,
    i16,
    nppsMinEvery_16s_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the minimum of its current value and the corresponding source element.",
    min_every_i32_in_place,
    i32,
    nppsMinEvery_32s_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the minimum of its current value and the corresponding source element.",
    min_every_f32_in_place,
    f32,
    nppsMinEvery_32f_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the minimum of its current value and the corresponding source element.",
    min_every_f64_in_place,
    f64,
    nppsMinEvery_64f_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the maximum of its current value and the corresponding source element.",
    max_every_u8_in_place,
    u8,
    nppsMaxEvery_8u_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the maximum of its current value and the corresponding source element.",
    max_every_u16_in_place,
    u16,
    nppsMaxEvery_16u_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the maximum of its current value and the corresponding source element.",
    max_every_i16_in_place,
    i16,
    nppsMaxEvery_16s_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the maximum of its current value and the corresponding source element.",
    max_every_i32_in_place,
    i32,
    nppsMaxEvery_32s_I_Ctx
);

impl_every_statistic_in_place!(
    "Replaces each destination element with the maximum of its current value and the corresponding source element.",
    max_every_f32_in_place,
    f32,
    nppsMaxEvery_32f_I_Ctx
);

fn validate_scalar_output<T>(destination: &SignalOutputView<'_, T>, name: &str) -> Result<()> {
    validate_same_len(1, destination.len(), name)
}
