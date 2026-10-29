use singe_cuda::types::{Complex32, Complex64};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
    try_ffi,
    types::{ComparisonOperation, ComplexI16, DataTypeLike, IntoNpp, RoundMode},
    utility::{to_u64, validate_same_len},
};

#[macro_use]
mod conversion_macros;

impl_convert!(
    "Converts source elements from `i8` to `f32`. Writes the result to `destination`.",
    convert_i8_to_f32,
    i8,
    f32,
    nppsConvert_8s32f_Ctx
);

impl_convert!(
    "Converts source elements from `i8` to `i16`. Writes the result to `destination`.",
    convert_i8_to_i16,
    i8,
    i16,
    nppsConvert_8s16s_Ctx
);

impl_convert!(
    "Converts source elements from `u8` to `f32`. Writes the result to `destination`.",
    convert_u8_to_f32,
    u8,
    f32,
    nppsConvert_8u32f_Ctx
);

impl_convert!(
    "Converts source elements from `i16` to `f32`. Writes the result to `destination`.",
    convert_i16_to_f32,
    i16,
    f32,
    nppsConvert_16s32f_Ctx
);

impl_convert!(
    "Converts source elements from `i16` to `i32`. Writes the result to `destination`.",
    convert_i16_to_i32,
    i16,
    i32,
    nppsConvert_16s32s_Ctx
);

impl_convert!(
    "Converts source elements from `u16` to `f32`. Writes the result to `destination`.",
    convert_u16_to_f32,
    u16,
    f32,
    nppsConvert_16u32f_Ctx
);

impl_convert!(
    "Converts source elements from `i32` to `i16`. Writes the result to `destination`.",
    convert_i32_to_i16,
    i32,
    i16,
    nppsConvert_32s16s_Ctx
);

impl_convert!(
    "Converts source elements from `i32` to `f32`. Writes the result to `destination`.",
    convert_i32_to_f32,
    i32,
    f32,
    nppsConvert_32s32f_Ctx
);

impl_convert!(
    "Converts source elements from `i32` to `f64`. Writes the result to `destination`.",
    convert_i32_to_f64,
    i32,
    f64,
    nppsConvert_32s64f_Ctx
);

impl_convert!(
    "Converts source elements from `f32` to `f64`. Writes the result to `destination`.",
    convert_f32_to_f64,
    f32,
    f64,
    nppsConvert_32f64f_Ctx
);

impl_convert!(
    "Converts source elements from `i64` to `f64`. Writes the result to `destination`.",
    convert_i64_to_f64,
    i64,
    f64,
    nppsConvert_64s64f_Ctx
);

impl_convert!(
    "Converts source elements from `f64` to `f32`. Writes the result to `destination`.",
    convert_f64_to_f32,
    f64,
    f32,
    nppsConvert_64f32f_Ctx
);

impl_convert_scaled!(
    "Converts source elements from `i16` to `f32`. Writes the result to `destination`.",
    convert_i16_to_f32_scaled,
    i16,
    f32,
    nppsConvert_16s32f_Sfs_Ctx
);

impl_convert_scaled!(
    "Converts source elements from `i16` to `f64`. Writes the result to `destination`.",
    convert_i16_to_f64_scaled,
    i16,
    f64,
    nppsConvert_16s64f_Sfs_Ctx
);

impl_convert_scaled!(
    "Converts source elements from `i32` to `i16`. Writes the result to `destination`.",
    convert_i32_to_i16_scaled,
    i32,
    i16,
    nppsConvert_32s16s_Sfs_Ctx
);

impl_convert_scaled!(
    "Converts source elements from `i32` to `f32`. Writes the result to `destination`.",
    convert_i32_to_f32_scaled,
    i32,
    f32,
    nppsConvert_32s32f_Sfs_Ctx
);

impl_convert_scaled!(
    "Converts source elements from `i32` to `f64`. Writes the result to `destination`.",
    convert_i32_to_f64_scaled,
    i32,
    f64,
    nppsConvert_32s64f_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `i16` to `i8`. Writes the result to `destination`.",
    convert_i16_to_i8_scaled,
    i16,
    i8,
    nppsConvert_16s8s_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f32` to `i8`. Writes the result to `destination`.",
    convert_f32_to_i8_scaled,
    f32,
    i8,
    nppsConvert_32f8s_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f32` to `u8`. Writes the result to `destination`.",
    convert_f32_to_u8_scaled,
    f32,
    u8,
    nppsConvert_32f8u_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f32` to `i16`. Writes the result to `destination`.",
    convert_f32_to_i16_scaled,
    f32,
    i16,
    nppsConvert_32f16s_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f32` to `u16`. Writes the result to `destination`.",
    convert_f32_to_u16_scaled,
    f32,
    u16,
    nppsConvert_32f16u_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f32` to `i32`. Writes the result to `destination`.",
    convert_f32_to_i32_scaled,
    f32,
    i32,
    nppsConvert_32f32s_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `i64` to `i32`. Writes the result to `destination`.",
    convert_i64_to_i32_scaled,
    i64,
    i32,
    nppsConvert_64s32s_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f64` to `i16`. Writes the result to `destination`.",
    convert_f64_to_i16_scaled,
    f64,
    i16,
    nppsConvert_64f16s_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f64` to `i32`. Writes the result to `destination`.",
    convert_f64_to_i32_scaled,
    f64,
    i32,
    nppsConvert_64f32s_Sfs_Ctx
);

impl_convert_scaled_round!(
    "Converts source elements from `f64` to `i64`. Writes the result to `destination`.",
    convert_f64_to_i64_scaled,
    f64,
    i64,
    nppsConvert_64f64s_Sfs_Ctx
);

macro_rules! impl_signal_convert_to_dispatch {
    ($source_ty:ty, $destination_ty:ty, $direct:ident) => {
        unsafe impl ConvertTo<$destination_ty> for $source_ty {
            unsafe fn convert_to(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, $destination_ty>,
            ) -> Result<()> {
                stream_context.cuda_context().bind()?;
                stream_context.validate_region(source)?;
                stream_context.validate_region(destination)?;
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $direct(stream_context, source, destination) }
            }
        }
    };
}

macro_rules! impl_signal_convert_scaled_to_dispatch {
    ($source_ty:ty, $destination_ty:ty, $direct:ident) => {
        unsafe impl ConvertScaledTo<$destination_ty> for $source_ty {
            unsafe fn convert_scaled_to(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, $destination_ty>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.cuda_context().bind()?;
                stream_context.validate_region(source)?;
                stream_context.validate_region(destination)?;
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { $direct(stream_context, source, destination, scale_factor) }
            }
        }
    };
}

macro_rules! impl_signal_convert_scaled_round_to_dispatch {
    ($source_ty:ty, $destination_ty:ty, $direct:ident) => {
        unsafe impl ConvertScaledRoundTo<$destination_ty> for $source_ty {
            unsafe fn convert_scaled_round_to(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, $destination_ty>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.cuda_context().bind()?;
                stream_context.validate_region(source)?;
                stream_context.validate_region(destination)?;
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $direct(
                        stream_context,
                        source,
                        destination,
                        round_mode,
                        scale_factor,
                    )
                }
            }
        }
    };
}

/// Native dispatch for the corresponding named NPP operation.
///
/// # Safety
/// Implementations must match the native ABI, access roles, written ranges,
/// and status semantics. Successful completed execution must initialize its
/// documented outputs; metadata queries must return the required capacity.
pub unsafe trait ConvertTo<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Converts each source element to the destination element type.
    ///
    /// # Errors
    ///
    /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
    /// Source and destination lengths must match.
    /// Returns an error if the source length cannot be represented by the native interface, or NPP rejects the conversion parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw signal execution contract](crate::signal::raw).
    /// Source values must be initialized and ready when the selected stream reads them.
    /// Keep the source read-only and the destination exclusively writable and disjoint from the source through completion.
    /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
    unsafe fn convert_to(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalOutputView<'_, Destination>,
    ) -> Result<()>;
}

/// Converts each source element to the destination element type.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
/// Source and destination lengths must match.
/// Returns an error if the source length cannot be represented by the native interface, or NPP rejects the conversion parameters.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and the destination exclusively writable and disjoint from the source through completion.
/// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
pub unsafe fn convert_to<Source, Destination>(
    stream_context: &StreamContext,
    source: &SignalView<'_, Source>,
    destination: &mut SignalOutputView<'_, Destination>,
) -> Result<()>
where
    Source: ConvertTo<Destination>,
    Destination: DataTypeLike,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { Source::convert_to(stream_context, source, destination) }
}

/// Native dispatch for the corresponding named NPP operation.
///
/// # Safety
/// Implementations must match the native ABI, access roles, written ranges,
/// and status semantics. Successful completed execution must initialize its
/// documented outputs; metadata queries must return the required capacity.
pub unsafe trait ConvertScaledTo<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Converts each source element to the destination element type.
    /// Uses native scaling with `scale_factor`.
    ///
    /// # Errors
    ///
    /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
    /// Source and destination lengths must match.
    /// Returns an error if the source length cannot be represented by the native interface, or NPP rejects the conversion parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw signal execution contract](crate::signal::raw).
    /// Source values must be initialized and ready when the selected stream reads them.
    /// Keep the source read-only and the destination exclusively writable and disjoint from the source through completion.
    /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
    unsafe fn convert_scaled_to(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalOutputView<'_, Destination>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Converts each source element to the destination element type.
/// Uses native scaling with `scale_factor`.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
/// Source and destination lengths must match.
/// Returns an error if the source length cannot be represented by the native interface, or NPP rejects the conversion parameters.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and the destination exclusively writable and disjoint from the source through completion.
/// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
pub unsafe fn convert_scaled_to<Source, Destination>(
    stream_context: &StreamContext,
    source: &SignalView<'_, Source>,
    destination: &mut SignalOutputView<'_, Destination>,
    scale_factor: i32,
) -> Result<()>
where
    Source: ConvertScaledTo<Destination>,
    Destination: DataTypeLike,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { Source::convert_scaled_to(stream_context, source, destination, scale_factor) }
}

/// Native dispatch for the corresponding named NPP operation.
///
/// # Safety
/// Implementations must match the native ABI, access roles, written ranges,
/// and status semantics. Successful completed execution must initialize its
/// documented outputs; metadata queries must return the required capacity.
pub unsafe trait ConvertScaledRoundTo<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    /// Converts each source element to the destination element type.
    /// Uses native scaling with `scale_factor`.
    /// Uses `round_mode` for rounding to the destination representation.
    ///
    /// # Errors
    ///
    /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
    /// Source and destination lengths must match.
    /// Returns an error if the source length cannot be represented by the native interface, or NPP rejects the conversion parameters.
    ///
    /// # Safety
    ///
    /// Follow the [raw signal execution contract](crate::signal::raw).
    /// Source values must be initialized and ready when the selected stream reads them.
    /// Keep the source read-only and the destination exclusively writable and disjoint from the source through completion.
    /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
    unsafe fn convert_scaled_round_to(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalOutputView<'_, Destination>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Converts each source element to the destination element type.
/// Uses native scaling with `scale_factor`.
/// Uses `round_mode` for rounding to the destination representation.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
/// Source and destination lengths must match.
/// Returns an error if the source length cannot be represented by the native interface, or NPP rejects the conversion parameters.
///
/// # Safety
///
/// Follow the [raw signal execution contract](crate::signal::raw).
/// Source values must be initialized and ready when the selected stream reads them.
/// Keep the source read-only and the destination exclusively writable and disjoint from the source through completion.
/// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
pub unsafe fn convert_scaled_round_to<Source, Destination>(
    stream_context: &StreamContext,
    source: &SignalView<'_, Source>,
    destination: &mut SignalOutputView<'_, Destination>,
    round_mode: RoundMode,
    scale_factor: i32,
) -> Result<()>
where
    Source: ConvertScaledRoundTo<Destination>,
    Destination: DataTypeLike,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        Source::convert_scaled_round_to(
            stream_context,
            source,
            destination,
            round_mode,
            scale_factor,
        )
    }
}

impl_signal_convert_to_dispatch!(i8, f32, convert_i8_to_f32);

impl_signal_convert_to_dispatch!(i8, i16, convert_i8_to_i16);

impl_signal_convert_to_dispatch!(u8, f32, convert_u8_to_f32);

impl_signal_convert_to_dispatch!(i16, f32, convert_i16_to_f32);

impl_signal_convert_to_dispatch!(i16, i32, convert_i16_to_i32);

impl_signal_convert_to_dispatch!(u16, f32, convert_u16_to_f32);

impl_signal_convert_to_dispatch!(i32, i16, convert_i32_to_i16);

impl_signal_convert_to_dispatch!(i32, f32, convert_i32_to_f32);

impl_signal_convert_to_dispatch!(i32, f64, convert_i32_to_f64);

impl_signal_convert_to_dispatch!(f32, f64, convert_f32_to_f64);

impl_signal_convert_to_dispatch!(i64, f64, convert_i64_to_f64);

impl_signal_convert_to_dispatch!(f64, f32, convert_f64_to_f32);

impl_signal_convert_scaled_to_dispatch!(i16, f32, convert_i16_to_f32_scaled);

impl_signal_convert_scaled_to_dispatch!(i16, f64, convert_i16_to_f64_scaled);

impl_signal_convert_scaled_to_dispatch!(i32, i16, convert_i32_to_i16_scaled);

impl_signal_convert_scaled_to_dispatch!(i32, f32, convert_i32_to_f32_scaled);

impl_signal_convert_scaled_to_dispatch!(i32, f64, convert_i32_to_f64_scaled);

impl_signal_convert_scaled_round_to_dispatch!(i16, i8, convert_i16_to_i8_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f32, i8, convert_f32_to_i8_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f32, u8, convert_f32_to_u8_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f32, i16, convert_f32_to_i16_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f32, u16, convert_f32_to_u16_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f32, i32, convert_f32_to_i32_scaled);

impl_signal_convert_scaled_round_to_dispatch!(i64, i32, convert_i64_to_i32_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f64, i16, convert_f64_to_i16_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f64, i32, convert_f64_to_i32_scaled);

impl_signal_convert_scaled_round_to_dispatch!(f64, i64, convert_f64_to_i64_scaled);

impl_threshold!(
    "Applies a constant-level threshold using the selected comparison `operation`. Writes the result to `destination`.",
    threshold_i16,
    i16,
    nppsThreshold_16s_Ctx
);

impl_threshold_in_place!(
    "Applies a constant-level threshold using the selected comparison `operation`. Updates `signal` in place.",
    threshold_i16_in_place,
    i16,
    nppsThreshold_16s_I_Ctx
);

impl_threshold_complex!(
    "Applies a constant-level threshold using the selected comparison `operation`. Writes the result to `destination`.",
    threshold_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_16sc_Ctx
);

impl_threshold_complex_in_place!(
    "Applies a constant-level threshold using the selected comparison `operation`. Updates `signal` in place.",
    threshold_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_16sc_I_Ctx
);

impl_threshold!(
    "Applies a constant-level threshold using the selected comparison `operation`. Writes the result to `destination`.",
    threshold_f32,
    f32,
    nppsThreshold_32f_Ctx
);

impl_threshold_in_place!(
    "Applies a constant-level threshold using the selected comparison `operation`. Updates `signal` in place.",
    threshold_f32_in_place,
    f32,
    nppsThreshold_32f_I_Ctx
);

impl_threshold_complex!(
    "Applies a constant-level threshold using the selected comparison `operation`. Writes the result to `destination`.",
    threshold_f32_complex,
    Complex32,
    f32,
    nppsThreshold_32fc_Ctx
);

impl_threshold_complex_in_place!(
    "Applies a constant-level threshold using the selected comparison `operation`. Updates `signal` in place.",
    threshold_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_32fc_I_Ctx
);

impl_threshold!(
    "Applies a constant-level threshold using the selected comparison `operation`. Writes the result to `destination`.",
    threshold_f64,
    f64,
    nppsThreshold_64f_Ctx
);

impl_threshold_in_place!(
    "Applies a constant-level threshold using the selected comparison `operation`. Updates `signal` in place.",
    threshold_f64_in_place,
    f64,
    nppsThreshold_64f_I_Ctx
);

impl_threshold_complex!(
    "Applies a constant-level threshold using the selected comparison `operation`. Writes the result to `destination`.",
    threshold_f64_complex,
    Complex64,
    f64,
    nppsThreshold_64fc_Ctx
);

impl_threshold_complex_in_place!(
    "Applies a constant-level threshold using the selected comparison `operation`. Updates `signal` in place.",
    threshold_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_64fc_I_Ctx
);

impl_threshold_fixed!(
    "Applies the native less-than threshold at `level`. Writes the result to `destination`.",
    threshold_less_i16,
    i16,
    nppsThreshold_LT_16s_Ctx
);

impl_threshold_fixed_in_place!(
    "Applies the native less-than threshold at `level`. Updates `signal` in place.",
    threshold_less_i16_in_place,
    i16,
    nppsThreshold_LT_16s_I_Ctx
);

impl_threshold_fixed_complex!(
    "Applies the native less-than threshold at `level`. Writes the result to `destination`.",
    threshold_less_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_LT_16sc_Ctx
);

impl_threshold_fixed_complex_in_place!(
    "Applies the native less-than threshold at `level`. Updates `signal` in place.",
    threshold_less_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_LT_16sc_I_Ctx
);

impl_threshold_fixed!(
    "Applies the native less-than threshold at `level`. Writes the result to `destination`.",
    threshold_less_f32,
    f32,
    nppsThreshold_LT_32f_Ctx
);

impl_threshold_fixed_in_place!(
    "Applies the native less-than threshold at `level`. Updates `signal` in place.",
    threshold_less_f32_in_place,
    f32,
    nppsThreshold_LT_32f_I_Ctx
);

impl_threshold_fixed_complex!(
    "Applies the native less-than threshold at `level`. Writes the result to `destination`.",
    threshold_less_f32_complex,
    Complex32,
    f32,
    nppsThreshold_LT_32fc_Ctx
);

impl_threshold_fixed_complex_in_place!(
    "Applies the native less-than threshold at `level`. Updates `signal` in place.",
    threshold_less_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_LT_32fc_I_Ctx
);

impl_threshold_fixed!(
    "Applies the native less-than threshold at `level`. Writes the result to `destination`.",
    threshold_less_f64,
    f64,
    nppsThreshold_LT_64f_Ctx
);

impl_threshold_fixed_in_place!(
    "Applies the native less-than threshold at `level`. Updates `signal` in place.",
    threshold_less_f64_in_place,
    f64,
    nppsThreshold_LT_64f_I_Ctx
);

impl_threshold_fixed_complex!(
    "Applies the native less-than threshold at `level`. Writes the result to `destination`.",
    threshold_less_f64_complex,
    Complex64,
    f64,
    nppsThreshold_LT_64fc_Ctx
);

impl_threshold_fixed_complex_in_place!(
    "Applies the native less-than threshold at `level`. Updates `signal` in place.",
    threshold_less_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_LT_64fc_I_Ctx
);

impl_threshold_fixed!(
    "Applies the native greater-than threshold at `level`. Writes the result to `destination`.",
    threshold_greater_i16,
    i16,
    nppsThreshold_GT_16s_Ctx
);

impl_threshold_fixed_in_place!(
    "Applies the native greater-than threshold at `level`. Updates `signal` in place.",
    threshold_greater_i16_in_place,
    i16,
    nppsThreshold_GT_16s_I_Ctx
);

impl_threshold_fixed_complex!(
    "Applies the native greater-than threshold at `level`. Writes the result to `destination`.",
    threshold_greater_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_GT_16sc_Ctx
);

impl_threshold_fixed_complex_in_place!(
    "Applies the native greater-than threshold at `level`. Updates `signal` in place.",
    threshold_greater_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_GT_16sc_I_Ctx
);

impl_threshold_fixed!(
    "Applies the native greater-than threshold at `level`. Writes the result to `destination`.",
    threshold_greater_f32,
    f32,
    nppsThreshold_GT_32f_Ctx
);

impl_threshold_fixed_in_place!(
    "Applies the native greater-than threshold at `level`. Updates `signal` in place.",
    threshold_greater_f32_in_place,
    f32,
    nppsThreshold_GT_32f_I_Ctx
);

impl_threshold_fixed_complex!(
    "Applies the native greater-than threshold at `level`. Writes the result to `destination`.",
    threshold_greater_f32_complex,
    Complex32,
    f32,
    nppsThreshold_GT_32fc_Ctx
);

impl_threshold_fixed_complex_in_place!(
    "Applies the native greater-than threshold at `level`. Updates `signal` in place.",
    threshold_greater_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_GT_32fc_I_Ctx
);

impl_threshold_fixed!(
    "Applies the native greater-than threshold at `level`. Writes the result to `destination`.",
    threshold_greater_f64,
    f64,
    nppsThreshold_GT_64f_Ctx
);

impl_threshold_fixed_in_place!(
    "Applies the native greater-than threshold at `level`. Updates `signal` in place.",
    threshold_greater_f64_in_place,
    f64,
    nppsThreshold_GT_64f_I_Ctx
);

impl_threshold_fixed_complex!(
    "Applies the native greater-than threshold at `level`. Writes the result to `destination`.",
    threshold_greater_f64_complex,
    Complex64,
    f64,
    nppsThreshold_GT_64fc_Ctx
);

impl_threshold_fixed_complex_in_place!(
    "Applies the native greater-than threshold at `level`. Updates `signal` in place.",
    threshold_greater_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_GT_64fc_I_Ctx
);

impl_threshold_value!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_less_value_i16,
    i16,
    nppsThreshold_LTVal_16s_Ctx
);

impl_threshold_value_in_place!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_less_value_i16_in_place,
    i16,
    nppsThreshold_LTVal_16s_I_Ctx
);

impl_threshold_value_complex!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_less_value_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_LTVal_16sc_Ctx
);

impl_threshold_value_complex_in_place!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_less_value_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_LTVal_16sc_I_Ctx
);

impl_threshold_value!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_less_value_f32,
    f32,
    nppsThreshold_LTVal_32f_Ctx
);

impl_threshold_value_in_place!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_less_value_f32_in_place,
    f32,
    nppsThreshold_LTVal_32f_I_Ctx
);

impl_threshold_value_complex!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_less_value_f32_complex,
    Complex32,
    f32,
    nppsThreshold_LTVal_32fc_Ctx
);

impl_threshold_value_complex_in_place!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_less_value_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_LTVal_32fc_I_Ctx
);

impl_threshold_value!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_less_value_f64,
    f64,
    nppsThreshold_LTVal_64f_Ctx
);

impl_threshold_value_in_place!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_less_value_f64_in_place,
    f64,
    nppsThreshold_LTVal_64f_I_Ctx
);

impl_threshold_value_complex!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_less_value_f64_complex,
    Complex64,
    f64,
    nppsThreshold_LTVal_64fc_Ctx
);

impl_threshold_value_complex_in_place!(
    "Replaces values selected by the native less-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_less_value_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_LTVal_64fc_I_Ctx
);

impl_threshold_value!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_greater_value_i16,
    i16,
    nppsThreshold_GTVal_16s_Ctx
);

impl_threshold_value_in_place!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_greater_value_i16_in_place,
    i16,
    nppsThreshold_GTVal_16s_I_Ctx
);

impl_threshold_value_complex!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_greater_value_i16_complex,
    ComplexI16,
    i16,
    nppsThreshold_GTVal_16sc_Ctx
);

impl_threshold_value_complex_in_place!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_greater_value_i16_complex_in_place,
    ComplexI16,
    i16,
    nppsThreshold_GTVal_16sc_I_Ctx
);

impl_threshold_value!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_greater_value_f32,
    f32,
    nppsThreshold_GTVal_32f_Ctx
);

impl_threshold_value_in_place!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_greater_value_f32_in_place,
    f32,
    nppsThreshold_GTVal_32f_I_Ctx
);

impl_threshold_value_complex!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_greater_value_f32_complex,
    Complex32,
    f32,
    nppsThreshold_GTVal_32fc_Ctx
);

impl_threshold_value_complex_in_place!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_greater_value_f32_complex_in_place,
    Complex32,
    f32,
    nppsThreshold_GTVal_32fc_I_Ctx
);

impl_threshold_value!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_greater_value_f64,
    f64,
    nppsThreshold_GTVal_64f_Ctx
);

impl_threshold_value_in_place!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_greater_value_f64_in_place,
    f64,
    nppsThreshold_GTVal_64f_I_Ctx
);

impl_threshold_value_complex!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.",
    threshold_greater_value_f64_complex,
    Complex64,
    f64,
    nppsThreshold_GTVal_64fc_Ctx
);

impl_threshold_value_complex_in_place!(
    "Replaces values selected by the native greater-than threshold at `level` with `value`. Updates `signal` in place.",
    threshold_greater_value_f64_complex_in_place,
    Complex64,
    f64,
    nppsThreshold_GTVal_64fc_I_Ctx
);

mod conversion_threshold_dispatch;

pub use conversion_threshold_dispatch::*;
