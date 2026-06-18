use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace},
    signal::{
        arithmetic,
        view::{SignalView, SignalViewMut},
    },
    types::ComplexI16,
};

use super::{SignalBacking, SignalPipeline};

pub trait NormalizeSignal<T> {
    fn normalize_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        subtrahend: T,
        divisor: T,
    ) -> Result<()>;
}

pub trait ComplexNormalizeSignal<T, D> {
    fn normalize_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        subtrahend: T,
        divisor: D,
    ) -> Result<()>;
}

pub trait ScaledNormalizeSignal<T> {
    fn normalize_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        subtrahend: T,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<()>;
}

macro_rules! impl_normalize_signal {
    ($ty:ty, $normalize:path) => {
        impl<'a> NormalizeSignal<$ty> for SignalPipeline<'a, $ty> {
            fn normalize_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                subtrahend: $ty,
                divisor: $ty,
            ) -> Result<()> {
                $normalize(stream_context, source, destination, subtrahend, divisor)
            }
        }
    };
}

macro_rules! impl_scaled_normalize_signal {
    ($ty:ty, $normalize:path) => {
        impl<'a> ScaledNormalizeSignal<$ty> for SignalPipeline<'a, $ty> {
            fn normalize_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                subtrahend: $ty,
                divisor: i32,
                scale_factor: i32,
            ) -> Result<()> {
                $normalize(
                    stream_context,
                    source,
                    destination,
                    subtrahend,
                    divisor,
                    scale_factor,
                )
            }
        }
    };
}

macro_rules! impl_complex_normalize_signal {
    ($ty:ty, $divisor:ty, $normalize:path) => {
        impl<'a> ComplexNormalizeSignal<$ty, $divisor> for SignalPipeline<'a, $ty> {
            fn normalize_complex_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                subtrahend: $ty,
                divisor: $divisor,
            ) -> Result<()> {
                $normalize(stream_context, source, destination, subtrahend, divisor)
            }
        }
    };
}

impl_normalize_signal!(f32, arithmetic::normalize_f32);
impl_normalize_signal!(f64, arithmetic::normalize_f64);
impl_complex_normalize_signal!(Complex32, f32, arithmetic::normalize_f32_complex);
impl_complex_normalize_signal!(Complex64, f64, arithmetic::normalize_f64_complex);
impl_scaled_normalize_signal!(i16, arithmetic::normalize_i16_scaled);
impl_scaled_normalize_signal!(ComplexI16, arithmetic::normalize_i16_complex_scaled);

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: NormalizeSignal<T>,
{
    pub fn normalize(self, subtrahend: T, divisor: T) -> Result<Self> {
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as NormalizeSignal<T>>::normalize_signal(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
            )
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Self: NormalizeSignal<T>,
{
    pub fn normalize_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        subtrahend: T,
        divisor: T,
    ) -> Result<()> {
        <Self as NormalizeSignal<T>>::normalize_signal(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    pub fn normalize_complex<D>(self, subtrahend: T, divisor: D) -> Result<Self>
    where
        D: Copy,
        Self: ComplexNormalizeSignal<T, D>,
    {
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as ComplexNormalizeSignal<T, D>>::normalize_complex_signal(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
            )
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    pub fn normalize_complex_into<D>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        subtrahend: T,
        divisor: D,
    ) -> Result<()>
    where
        D: Copy,
        Self: ComplexNormalizeSignal<T, D>,
    {
        <Self as ComplexNormalizeSignal<T, D>>::normalize_complex_signal(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
        )
    }
}

impl<'a> SignalPipeline<'a, i16> {
    pub fn normalize_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i16>,
        subtrahend: i16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledNormalizeSignal<i16>>::normalize_signal_scaled(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
            scale_factor,
        )
    }

    pub fn normalize_scaled(
        self,
        subtrahend: i16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<Self> {
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as ScaledNormalizeSignal<i16>>::normalize_signal_scaled(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
                scale_factor,
            )
        })
    }
}

impl<'a> SignalPipeline<'a, ComplexI16> {
    pub fn normalize_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI16>,
        destination: &mut SignalViewMut<'_, ComplexI16>,
        subtrahend: ComplexI16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledNormalizeSignal<ComplexI16>>::normalize_signal_scaled(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
            scale_factor,
        )
    }

    pub fn normalize_scaled(
        self,
        subtrahend: ComplexI16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<Self> {
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as ScaledNormalizeSignal<ComplexI16>>::normalize_signal_scaled(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
                scale_factor,
            )
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    fn normalize_out_of_place(
        mut self,
        normalize: impl FnOnce(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.signal::<T>(self.len())?;
        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            normalize(self.stream_context, &source, &mut destination_view)?;
        }
        self.backing = SignalBacking::Owned(destination);
        Ok(self)
    }
}
