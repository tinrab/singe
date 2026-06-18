use singe_cuda::types::Complex32;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{C1, ImageView, ImageViewMut},
    try_ffi,
    types::{DataTypeLike, Size},
};

pub trait MagnitudeC1: DataTypeLike + Sized {
    type Output: DataTypeLike;

    fn magnitude(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
    ) -> Result<()>;
}

pub fn magnitude<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
) -> Result<()>
where
    T: MagnitudeC1,
{
    T::magnitude(stream_context, source, destination)
}

pub trait MagnitudeSquaredC1: DataTypeLike + Sized {
    type Output: DataTypeLike;

    fn magnitude_squared(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
    ) -> Result<()>;
}

pub fn magnitude_squared<T>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T::Output, C1>,
) -> Result<()>
where
    T: MagnitudeSquaredC1,
{
    T::magnitude_squared(stream_context, source, destination)
}

pub(crate) fn magnitude_f32_complex_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, Complex32, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

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

pub(crate) fn magnitude_squared_f32_complex_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, Complex32, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
) -> Result<()> {
    validate_same_size(source.size(), destination.size())?;

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

impl MagnitudeC1 for Complex32 {
    type Output = f32;

    fn magnitude(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
    ) -> Result<()> {
        magnitude_f32_complex_c1(stream_context, source, destination)
    }
}

impl MagnitudeSquaredC1 for Complex32 {
    type Output = f32;

    fn magnitude_squared(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self::Output, C1>,
    ) -> Result<()> {
        magnitude_squared_f32_complex_c1(stream_context, source, destination)
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
