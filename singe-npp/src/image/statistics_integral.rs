use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    try_ffi,
    types::{DataTypeLike, Rectangle},
};

use super::statistics_validation::*;

pub(crate) fn integral_u8_to_i32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, i32, C1>,
    value: i32,
) -> Result<()> {
    validate_integral_destination_size(source.size(), destination.size())?;
    unsafe {
        try_ffi!(sys::nppiIntegral_8u32s_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            value,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn integral_u8_to_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
    value: f32,
) -> Result<()> {
    validate_integral_destination_size(source.size(), destination.size())?;
    unsafe {
        try_ffi!(sys::nppiIntegral_8u32f_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            value,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn sqr_integral_u8_to_i32_i32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, i32, C1>,
    squared: &mut ImageViewMut<'_, i32, C1>,
    value: i32,
    squared_value: i32,
) -> Result<()> {
    validate_integral_destination_size(source.size(), destination.size())?;
    validate_same_size(destination.size(), squared.size())?;
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
        ))?;
    }
    Ok(())
}

pub(crate) fn sqr_integral_u8_to_i32_f64_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, i32, C1>,
    squared: &mut ImageViewMut<'_, f64, C1>,
    value: i32,
    squared_value: f64,
) -> Result<()> {
    validate_integral_destination_size(source.size(), destination.size())?;
    validate_same_size(destination.size(), squared.size())?;
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
        ))?;
    }
    Ok(())
}

pub(crate) fn sqr_integral_u8_to_f32_f64_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
    squared: &mut ImageViewMut<'_, f64, C1>,
    value: f32,
    squared_value: f64,
) -> Result<()> {
    validate_integral_destination_size(source.size(), destination.size())?;
    validate_same_size(destination.size(), squared.size())?;
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
        ))?;
    }
    Ok(())
}

pub(crate) fn rect_std_dev_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, f32, C1>,
    squared: &ImageView<'_, f64, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
    rectangle: Rectangle,
) -> Result<()> {
    validate_rect_std_dev_sources(source.size(), squared.size(), destination.size(), rectangle)?;
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
        ))?;
    }
    Ok(())
}

pub(crate) fn rect_std_dev_i32_scaled_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, i32, C1>,
    squared: &ImageView<'_, i32, C1>,
    destination: &mut ImageViewMut<'_, i32, C1>,
    rectangle: Rectangle,
    scale_factor: i32,
) -> Result<()> {
    validate_rect_std_dev_sources(source.size(), squared.size(), destination.size(), rectangle)?;
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
        ))?;
    }
    Ok(())
}

pub(crate) fn rect_std_dev_i32_to_f32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, i32, C1>,
    squared: &ImageView<'_, f64, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
    rectangle: Rectangle,
) -> Result<()> {
    validate_rect_std_dev_sources(source.size(), squared.size(), destination.size(), rectangle)?;
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
        ))?;
    }
    Ok(())
}

pub trait IntegralToI32C1: DataTypeLike {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        value: i32,
    ) -> Result<()>
    where
        Self: Sized;
}

impl IntegralToI32C1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        value: i32,
    ) -> Result<()> {
        integral_u8_to_i32_c1(stream_context, source, destination, value)
    }
}

pub fn integral_to_i32<T: IntegralToI32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, i32, C1>,
    value: i32,
) -> Result<()> {
    T::dispatch(stream_context, source, destination, value)
}

pub trait IntegralToF32C1: DataTypeLike {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        value: f32,
    ) -> Result<()>
    where
        Self: Sized;
}

impl IntegralToF32C1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        value: f32,
    ) -> Result<()> {
        integral_u8_to_f32_c1(stream_context, source, destination, value)
    }
}

pub fn integral_to_f32<T: IntegralToF32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
    value: f32,
) -> Result<()> {
    T::dispatch(stream_context, source, destination, value)
}

pub trait IntegralC1<Destination>: DataTypeLike
where
    Destination: DataTypeLike,
{
    fn dispatch_integral(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Destination, C1>,
        value: Destination,
    ) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_integral_c1 {
    ($destination:ty, $source_trait:ident) => {
        impl<T> IntegralC1<$destination> for T
        where
            T: $source_trait,
        {
            fn dispatch_integral(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, $destination, C1>,
                value: $destination,
            ) -> Result<()> {
                T::dispatch(stream_context, source, destination, value)
            }
        }
    };
}

impl_integral_c1!(i32, IntegralToI32C1);
impl_integral_c1!(f32, IntegralToF32C1);

pub fn integral<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, Destination, C1>,
    value: Destination,
) -> Result<()>
where
    T: IntegralC1<Destination>,
    Destination: DataTypeLike,
{
    T::dispatch_integral(stream_context, source, destination, value)
}

pub trait SqrIntegralToI32I32C1: DataTypeLike {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        squared: &mut ImageViewMut<'_, i32, C1>,
        value: i32,
        squared_value: i32,
    ) -> Result<()>
    where
        Self: Sized;
}

impl SqrIntegralToI32I32C1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        squared: &mut ImageViewMut<'_, i32, C1>,
        value: i32,
        squared_value: i32,
    ) -> Result<()> {
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

pub fn squared_integral_to_i32_i32<T: SqrIntegralToI32I32C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, i32, C1>,
    squared: &mut ImageViewMut<'_, i32, C1>,
    value: i32,
    squared_value: i32,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        destination,
        squared,
        value,
        squared_value,
    )
}

pub trait SqrIntegralToI32F64C1: DataTypeLike {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        squared: &mut ImageViewMut<'_, f64, C1>,
        value: i32,
        squared_value: f64,
    ) -> Result<()>
    where
        Self: Sized;
}

impl SqrIntegralToI32F64C1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        squared: &mut ImageViewMut<'_, f64, C1>,
        value: i32,
        squared_value: f64,
    ) -> Result<()> {
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

pub fn squared_integral_to_i32_f64<T: SqrIntegralToI32F64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, i32, C1>,
    squared: &mut ImageViewMut<'_, f64, C1>,
    value: i32,
    squared_value: f64,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        destination,
        squared,
        value,
        squared_value,
    )
}

pub trait SqrIntegralToF32F64C1: DataTypeLike {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        squared: &mut ImageViewMut<'_, f64, C1>,
        value: f32,
        squared_value: f64,
    ) -> Result<()>
    where
        Self: Sized;
}

impl SqrIntegralToF32F64C1 for u8 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        squared: &mut ImageViewMut<'_, f64, C1>,
        value: f32,
        squared_value: f64,
    ) -> Result<()> {
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

pub fn squared_integral_to_f32_f64<T: SqrIntegralToF32F64C1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, f32, C1>,
    squared: &mut ImageViewMut<'_, f64, C1>,
    value: f32,
    squared_value: f64,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        destination,
        squared,
        value,
        squared_value,
    )
}

pub trait SqrIntegralC1<Destination, Squared>: DataTypeLike
where
    Destination: DataTypeLike,
    Squared: DataTypeLike,
{
    fn dispatch_sqr_integral(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Destination, C1>,
        squared: &mut ImageViewMut<'_, Squared, C1>,
        value: Destination,
        squared_value: Squared,
    ) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_sqr_integral_c1 {
    ($destination:ty, $squared:ty, $source_trait:ident) => {
        impl<T> SqrIntegralC1<$destination, $squared> for T
        where
            T: $source_trait,
        {
            fn dispatch_sqr_integral(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, $destination, C1>,
                squared: &mut ImageViewMut<'_, $squared, C1>,
                value: $destination,
                squared_value: $squared,
            ) -> Result<()> {
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
    };
}

impl_sqr_integral_c1!(i32, i32, SqrIntegralToI32I32C1);
impl_sqr_integral_c1!(i32, f64, SqrIntegralToI32F64C1);
impl_sqr_integral_c1!(f32, f64, SqrIntegralToF32F64C1);

pub fn sqr_integral<T, Destination, Squared>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, Destination, C1>,
    squared: &mut ImageViewMut<'_, Squared, C1>,
    value: Destination,
    squared_value: Squared,
) -> Result<()>
where
    T: SqrIntegralC1<Destination, Squared>,
    Destination: DataTypeLike,
    Squared: DataTypeLike,
{
    T::dispatch_sqr_integral(
        stream_context,
        source,
        destination,
        squared,
        value,
        squared_value,
    )
}

pub trait RectStandardDeviation: DataTypeLike {
    type Squared: DataTypeLike;
    type Destination: DataTypeLike;

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self::Squared, C1>,
        destination: &mut ImageViewMut<'_, Self::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()>
    where
        Self: Sized;
}

impl RectStandardDeviation for f32 {
    type Squared = f64;
    type Destination = f32;

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self::Squared, C1>,
        destination: &mut ImageViewMut<'_, Self::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()> {
        rect_std_dev_f32_c1(stream_context, source, squared, destination, rectangle)
    }
}

impl RectStandardDeviation for i32 {
    type Squared = f64;
    type Destination = f32;

    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self::Squared, C1>,
        destination: &mut ImageViewMut<'_, Self::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()> {
        rect_std_dev_i32_to_f32_c1(stream_context, source, squared, destination, rectangle)
    }
}

pub fn rect_standard_deviation<T: RectStandardDeviation>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    squared: &ImageView<'_, T::Squared, C1>,
    destination: &mut ImageViewMut<'_, T::Destination, C1>,
    rectangle: Rectangle,
) -> Result<()> {
    T::dispatch(stream_context, source, squared, destination, rectangle)
}

pub fn rect_standard_deviation_to<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    squared: &ImageView<'_, T::Squared, C1>,
    destination: &mut ImageViewMut<'_, Destination, C1>,
    rectangle: Rectangle,
) -> Result<()>
where
    T: RectStandardDeviation<Destination = Destination>,
    Destination: DataTypeLike,
{
    T::dispatch(stream_context, source, squared, destination, rectangle)
}

pub trait RectStandardDeviationScaled: DataTypeLike {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: Sized;
}

impl RectStandardDeviationScaled for i32 {
    fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        squared: &ImageView<'_, Self, C1>,
        destination: &mut ImageViewMut<'_, Self, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<()> {
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

pub fn rect_standard_deviation_scaled<T: RectStandardDeviationScaled>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    squared: &ImageView<'_, T, C1>,
    destination: &mut ImageViewMut<'_, T, C1>,
    rectangle: Rectangle,
    scale_factor: i32,
) -> Result<()> {
    T::dispatch(
        stream_context,
        source,
        squared,
        destination,
        rectangle,
        scale_factor,
    )
}
