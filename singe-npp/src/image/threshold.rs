use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C2, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    try_ffi,
    types::{ComparisonOperation, DataType, DataTypeLike, ImageChannels, Size},
};

pub trait FusedAbsDiffElement: Copy {
    const DATA_TYPE: DataType;
}

impl FusedAbsDiffElement for u8 {
    const DATA_TYPE: DataType = DataType::U8;
}

impl FusedAbsDiffElement for i8 {
    const DATA_TYPE: DataType = DataType::I8;
}

impl FusedAbsDiffElement for u16 {
    const DATA_TYPE: DataType = DataType::U16;
}

impl FusedAbsDiffElement for i16 {
    const DATA_TYPE: DataType = DataType::I16;
}

impl FusedAbsDiffElement for u32 {
    const DATA_TYPE: DataType = DataType::U32;
}

impl FusedAbsDiffElement for i32 {
    const DATA_TYPE: DataType = DataType::I32;
}

impl FusedAbsDiffElement for u64 {
    const DATA_TYPE: DataType = DataType::U64;
}

impl FusedAbsDiffElement for i64 {
    const DATA_TYPE: DataType = DataType::I64;
}

impl FusedAbsDiffElement for f32 {
    const DATA_TYPE: DataType = DataType::F32;
}

impl FusedAbsDiffElement for f64 {
    const DATA_TYPE: DataType = DataType::F64;
}

pub trait FusedAbsDiffChannels: ChannelLayout {
    const CHANNELS: ImageChannels;
}

impl FusedAbsDiffChannels for C1 {
    const CHANNELS: ImageChannels = ImageChannels::C1;
}

impl FusedAbsDiffChannels for C2 {
    const CHANNELS: ImageChannels = ImageChannels::C2;
}

impl FusedAbsDiffChannels for C3 {
    const CHANNELS: ImageChannels = ImageChannels::C3;
}

impl FusedAbsDiffChannels for C4 {
    const CHANNELS: ImageChannels = ImageChannels::C4;
}

impl FusedAbsDiffChannels for AC4 {
    const CHANNELS: ImageChannels = ImageChannels::AC4;
}

pub fn fused_absolute_difference_threshold_greater_value<T, L>(
    stream_context: &StreamContext,
    source1: &ImageView<'_, T, L>,
    source2: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
    threshold: T,
    value: T,
) -> Result<()>
where
    T: FusedAbsDiffElement,
    L: FusedAbsDiffChannels,
{
    validate_same_size(source1.size(), source2.size())?;
    validate_same_size(source1.size(), destination.size())?;

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

pub fn fused_absolute_difference_threshold_greater_value_in_place<T, L>(
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
    validate_same_size(source_destination.size(), source2.size())?;

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
#[path = "threshold_direct_macros.rs"]
mod direct_macros;

#[macro_use]
#[path = "threshold_generic_macros.rs"]
mod generic_macros;

#[path = "threshold_basic.rs"]
mod basic;
pub use basic::*;

#[path = "threshold_fixed.rs"]
mod fixed;
pub use fixed::*;

#[path = "threshold_value.rs"]
mod value;
pub use value::*;

#[path = "threshold_fixed_value.rs"]
mod fixed_value;
pub use fixed_value::*;

#[path = "threshold_range_value.rs"]
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
