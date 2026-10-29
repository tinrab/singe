//! Raw image operations queued on the selected NPP stream.
//!
//! # Safety
//! Retain every allocation until completion or the last captured replay and establish access ordering.
//! Input and read-write views require initialized pixels; outputs describe writable storage only.
//! Threshold operations write the complete ROI except ignored AC4 alpha; comparison masks write every pixel.
//! Initialized results require successful completion and initialization of any untouched channels.

use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C2, C3, C4, ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    try_ffi,
    types::{DataType, ImageChannels, Size},
};

/// Declares the native element representation used by fused absolute-difference thresholding.
///
/// # Safety
///
/// `DATA_TYPE` must describe the exact size, alignment, numeric representation, and valid
/// output values of `Self` for the native operation.
pub unsafe trait FusedAbsDiffElement: Copy {
    /// Native element type corresponding to `Self`.
    const DATA_TYPE: DataType;
}

unsafe impl FusedAbsDiffElement for u8 {
    const DATA_TYPE: DataType = DataType::U8;
}

unsafe impl FusedAbsDiffElement for i8 {
    const DATA_TYPE: DataType = DataType::I8;
}

unsafe impl FusedAbsDiffElement for u16 {
    const DATA_TYPE: DataType = DataType::U16;
}

unsafe impl FusedAbsDiffElement for i16 {
    const DATA_TYPE: DataType = DataType::I16;
}

unsafe impl FusedAbsDiffElement for u32 {
    const DATA_TYPE: DataType = DataType::U32;
}

unsafe impl FusedAbsDiffElement for i32 {
    const DATA_TYPE: DataType = DataType::I32;
}

unsafe impl FusedAbsDiffElement for u64 {
    const DATA_TYPE: DataType = DataType::U64;
}

unsafe impl FusedAbsDiffElement for i64 {
    const DATA_TYPE: DataType = DataType::I64;
}

unsafe impl FusedAbsDiffElement for f32 {
    const DATA_TYPE: DataType = DataType::F32;
}

unsafe impl FusedAbsDiffElement for f64 {
    const DATA_TYPE: DataType = DataType::F64;
}

/// Declares the native channel layout used by fused absolute-difference thresholding.
///
/// # Safety
///
/// `CHANNELS` must match the physical component count, order, and ignored-alpha semantics
/// of this channel layout.
pub unsafe trait FusedAbsDiffChannels: ChannelLayout {
    /// Native channel-layout selector corresponding to this layout.
    const CHANNELS: ImageChannels;
}

unsafe impl FusedAbsDiffChannels for C1 {
    const CHANNELS: ImageChannels = ImageChannels::C1;
}

unsafe impl FusedAbsDiffChannels for C2 {
    const CHANNELS: ImageChannels = ImageChannels::C2;
}

unsafe impl FusedAbsDiffChannels for C3 {
    const CHANNELS: ImageChannels = ImageChannels::C3;
}

unsafe impl FusedAbsDiffChannels for C4 {
    const CHANNELS: ImageChannels = ImageChannels::C4;
}

unsafe impl FusedAbsDiffChannels for AC4 {
    const CHANNELS: ImageChannels = ImageChannels::AC4;
}

/// Computes the absolute difference of corresponding source samples and applies a strict threshold.
///
/// If the absolute difference is greater than `threshold`, writes `value`; otherwise writes
/// the absolute difference. Writes the result to `destination`.
/// For AC4 layouts, alpha is not processed. Scalar values and numeric type support are
/// forwarded to NPP without numerical validation. Queues work without synchronizing; row
/// padding remains unwritten.
///
/// # Errors
///
/// Returns an error if image dimensions differ, CUDA context binding fails, an image belongs
/// to another CUDA context, or NPP rejects the operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Every input and read-write
/// pixel must be initialized and ready for the selected stream. Keep both sources read-only and destination storage exclusively writable and disjoint
/// from both sources.
/// Retain every allocation through queued completion, including submission errors and
/// captured replay. Native element representations, component layouts, byte strides,
/// alignment, and allocation bounds must be valid.
pub unsafe fn fused_absolute_difference_threshold_greater_value<T, L>(
    stream_context: &StreamContext,
    source1: &ImageView<'_, T, L>,
    source2: &ImageView<'_, T, L>,
    destination: &mut ImageOutputView<'_, T, L>,
    threshold: T,
    value: T,
) -> Result<()>
where
    T: FusedAbsDiffElement,
    L: FusedAbsDiffChannels,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source1)?;
    stream_context.validate_region(source2)?;
    stream_context.validate_region(destination)?;

    validate_same_size(source1.size(), source2.size())?;
    validate_same_size(source1.size(), destination.size())?;

    // SAFETY: `nppiFusedAbsDiff_Threshold_GTVal_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiFusedAbsDiff_Threshold_GTVal_Ctx(
            T::DATA_TYPE.into(),
            <L as FusedAbsDiffChannels>::CHANNELS.into(),
            source1.as_ptr().cast(),
            source1.step(),
            source2.as_ptr().cast(),
            source2.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source1.size().into(),
            (&raw const threshold).cast(),
            (&raw const value).cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Computes the absolute difference of corresponding source samples and applies a strict threshold.
///
/// If the absolute difference is greater than `threshold`, writes `value`; otherwise writes
/// the absolute difference. Updates `source_destination` in place.
/// For AC4 layouts, alpha is not processed. Scalar values and numeric type support are
/// forwarded to NPP without numerical validation. Queues work without synchronizing; row
/// padding remains unwritten.
///
/// # Errors
///
/// Returns an error if image dimensions differ, CUDA context binding fails, an image belongs
/// to another CUDA context, or NPP rejects the operation parameters.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw). Every input and read-write
/// pixel must be initialized and ready for the selected stream. Keep `source_destination` exclusively writable and disjoint from read-only `source2`.
/// Retain every allocation through queued completion, including submission errors and
/// captured replay. Native element representations, component layouts, byte strides,
/// alignment, and allocation bounds must be valid.
pub unsafe fn fused_absolute_difference_threshold_greater_value_in_place<T, L>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, L>,
    source2: &ImageView<'_, T, L>,
    threshold: T,
    value: T,
) -> Result<()>
where
    T: FusedAbsDiffElement,
    L: FusedAbsDiffChannels,
{
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source_destination)?;
    stream_context.validate_region(source2)?;

    validate_same_size(source_destination.size(), source2.size())?;

    // SAFETY: `nppiFusedAbsDiff_Threshold_GTVal_I_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiFusedAbsDiff_Threshold_GTVal_I_Ctx(
            T::DATA_TYPE.into(),
            <L as FusedAbsDiffChannels>::CHANNELS.into(),
            source_destination.as_mut_ptr().cast(),
            source_destination.step(),
            source2.as_ptr().cast(),
            source2.step(),
            source_destination.size().into(),
            (&raw const threshold).cast(),
            (&raw const value).cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

#[macro_use]
mod direct_macros;

#[macro_use]
mod generic_macros;

mod basic;

pub use basic::*;

mod fixed;

pub use fixed::*;

mod value;

pub use value::*;

mod fixed_value;

pub use fixed_value::*;

mod range_value;

pub use range_value::*;

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
