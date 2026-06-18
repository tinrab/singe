use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    signal::view::{SignalView, SignalViewMut},
    try_ffi,
    types::DataTypeLike,
    utility::{to_u64, to_usize},
    workspace::ScratchBuffer,
};

pub(crate) fn integral_i32_buffer_size(source: &SignalView<'_, i32>) -> Result<usize> {
    let mut bytes = 0;
    let len = source
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::OutOfRange { name: "len".into() })?;
    let raw_len = to_u64(len, "len")?;
    unsafe {
        try_ffi!(sys::nppsIntegralGetBufferSize_32s(raw_len, &raw mut bytes,))?;
    }
    to_usize(bytes, "scratch bytes")
}

pub(crate) fn integral_i32_with_scratch(
    stream_context: &StreamContext,
    source: &SignalView<'_, i32>,
    destination: &mut SignalViewMut<'_, i32>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    validate_integral_destination_len(source, destination)?;
    let required_bytes = integral_i32_buffer_size(source)?;
    scratch.require(required_bytes)?;

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

pub trait Integral: DataTypeLike {
    fn integral_buffer_size(source: &SignalView<'_, Self>) -> Result<usize>;

    fn integral_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalViewMut<'_, Self>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

impl Integral for i32 {
    fn integral_buffer_size(source: &SignalView<'_, Self>) -> Result<usize> {
        integral_i32_buffer_size(source)
    }

    fn integral_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, Self>,
        destination: &mut SignalViewMut<'_, Self>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        integral_i32_with_scratch(stream_context, source, destination, scratch)
    }
}

pub fn integral_buffer_size<T: Integral>(source: &SignalView<'_, T>) -> Result<usize> {
    T::integral_buffer_size(source)
}

pub fn integral_with_scratch<T: Integral>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
    destination: &mut SignalViewMut<'_, T>,
    scratch: &mut ScratchBuffer,
) -> Result<()> {
    T::integral_with_scratch(stream_context, source, destination, scratch)
}

pub fn integral<T: Integral>(
    stream_context: &StreamContext,
    source: &SignalView<'_, T>,
    destination: &mut SignalViewMut<'_, T>,
) -> Result<()> {
    let required_bytes = integral_buffer_size(source)?;
    let mut scratch = ScratchBuffer::create(required_bytes)?;
    integral_with_scratch(stream_context, source, destination, &mut scratch)
}

fn validate_integral_destination_len<T>(
    source: &SignalView<'_, T>,
    destination: &SignalViewMut<'_, T>,
) -> Result<()> {
    let expected = source
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::OutOfRange { name: "len".into() })?;
    if destination.len() != expected {
        return Err(Error::LengthMismatch {
            name: "destination".into(),
            expected,
            actual: destination.len(),
        });
    }

    Ok(())
}
