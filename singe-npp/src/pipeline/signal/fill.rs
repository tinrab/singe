use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace},
    signal::{initialization, view::SignalViewMut},
    types::{ComplexI16, ComplexI32, ComplexI64},
};

use super::{SignalBacking, SignalPipeline};

pub trait SetSignal<T> {
    fn set_signal(
        stream_context: &StreamContext,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait ZeroSignal<T> {
    fn zero_signal(
        stream_context: &StreamContext,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

macro_rules! impl_set_signal {
    ($ty:ty, $set:path) => {
        impl<'a> SetSignal<$ty> for SignalPipeline<'a, $ty> {
            fn set_signal(
                stream_context: &StreamContext,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $set(stream_context, value, destination)
            }
        }
    };
}

macro_rules! impl_zero_signal {
    ($ty:ty, $zero:path) => {
        impl<'a> ZeroSignal<$ty> for SignalPipeline<'a, $ty> {
            fn zero_signal(
                stream_context: &StreamContext,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $zero(stream_context, destination)
            }
        }
    };
}

impl_set_signal!(u8, initialization::set_u8);
impl_set_signal!(i8, initialization::set_i8);
impl_set_signal!(u16, initialization::set_u16);
impl_set_signal!(i16, initialization::set_i16);
impl_set_signal!(u32, initialization::set_u32);
impl_set_signal!(i32, initialization::set_i32);
impl_set_signal!(i64, initialization::set_i64);
impl_set_signal!(ComplexI16, initialization::set_i16_complex);
impl_set_signal!(ComplexI32, initialization::set_i32_complex);
impl_set_signal!(ComplexI64, initialization::set_i64_complex);
impl_set_signal!(f32, initialization::set_f32);
impl_set_signal!(Complex32, initialization::set_f32_complex);
impl_set_signal!(f64, initialization::set_f64);
impl_set_signal!(Complex64, initialization::set_f64_complex);

impl_zero_signal!(u8, initialization::zero_u8);
impl_zero_signal!(i16, initialization::zero_i16);
impl_zero_signal!(i32, initialization::zero_i32);
impl_zero_signal!(i64, initialization::zero_i64);
impl_zero_signal!(ComplexI16, initialization::zero_i16_complex);
impl_zero_signal!(ComplexI32, initialization::zero_i32_complex);
impl_zero_signal!(ComplexI64, initialization::zero_i64_complex);
impl_zero_signal!(f32, initialization::zero_f32);
impl_zero_signal!(Complex32, initialization::zero_f32_complex);
impl_zero_signal!(f64, initialization::zero_f64);
impl_zero_signal!(Complex64, initialization::zero_f64_complex);

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: SetSignal<T>,
{
    pub fn set_into(
        stream_context: &StreamContext,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as SetSignal<T>>::set_signal(stream_context, value, destination)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: ZeroSignal<T>,
{
    pub fn zero_into(
        stream_context: &StreamContext,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as ZeroSignal<T>>::zero_signal(stream_context, destination)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: SetSignal<T>,
{
    pub fn set(mut self, value: T) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as SetSignal<T>>::set_signal(self.stream_context, value, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as SetSignal<T>>::set_signal(
                    self.stream_context,
                    value,
                    &mut destination_view,
                )?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ZeroSignal<T>,
{
    pub fn zero(mut self) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                <Self as ZeroSignal<T>>::zero_signal(self.stream_context, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ZeroSignal<T>>::zero_signal(self.stream_context, &mut destination_view)?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
