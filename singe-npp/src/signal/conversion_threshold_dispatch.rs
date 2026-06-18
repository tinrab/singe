use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalView, SignalViewMut},
    types::{ComparisonOperation, ComplexI16, DataTypeLike},
};

use super::*;

macro_rules! impl_signal_threshold_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            type Level: DataTypeLike;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                level: Self::Level,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                type Level = $level_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    level: Self::Level,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, level, operation)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            level: T::Level,
            operation: ComparisonOperation,
        ) -> Result<()> {
            T::$method(stream_context, source, destination, level, operation)
        }
    };
}

macro_rules! impl_signal_threshold_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            type Level: DataTypeLike;

            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                level: Self::Level,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                type Level = $level_ty;

                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    level: Self::Level,
                    operation: ComparisonOperation,
                ) -> Result<()> {
                    $direct(stream_context, signal, level, operation)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            level: T::Level,
            operation: ComparisonOperation,
        ) -> Result<()> {
            T::$method(stream_context, signal, level, operation)
        }
    };
}

macro_rules! impl_signal_threshold_fixed_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            type Level: DataTypeLike;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                level: Self::Level,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                type Level = $level_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    level: Self::Level,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, level)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            level: T::Level,
        ) -> Result<()> {
            T::$method(stream_context, source, destination, level)
        }
    };
}

macro_rules! impl_signal_threshold_value_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            type Level: DataTypeLike;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                level: Self::Level,
                value: Self,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                type Level = $level_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    level: Self::Level,
                    value: Self,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, level, value)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            level: T::Level,
            value: T,
        ) -> Result<()> {
            T::$method(stream_context, source, destination, level, value)
        }
    };
}

impl_signal_threshold_dispatch!(Threshold, threshold, threshold, [
    i16 => (i16, threshold_i16),
    f32 => (f32, threshold_f32),
    f64 => (f64, threshold_f64),
    ComplexI16 => (i16, threshold_i16_complex),
    Complex32 => (f32, threshold_f32_complex),
    Complex64 => (f64, threshold_f64_complex)
]);

impl_signal_threshold_in_place_dispatch!(ThresholdInPlace, threshold_in_place, threshold_in_place, [
    i16 => (i16, threshold_i16_in_place),
    f32 => (f32, threshold_f32_in_place),
    f64 => (f64, threshold_f64_in_place),
    ComplexI16 => (i16, threshold_i16_complex_in_place),
    Complex32 => (f32, threshold_f32_complex_in_place),
    Complex64 => (f64, threshold_f64_complex_in_place)
]);

impl_signal_threshold_fixed_dispatch!(ThresholdLess, threshold_less, threshold_less, [
    i16 => (i16, threshold_less_i16),
    f32 => (f32, threshold_less_f32),
    f64 => (f64, threshold_less_f64),
    ComplexI16 => (i16, threshold_less_i16_complex),
    Complex32 => (f32, threshold_less_f32_complex),
    Complex64 => (f64, threshold_less_f64_complex)
]);

impl_signal_threshold_fixed_dispatch!(ThresholdGreater, threshold_greater, threshold_greater, [
    i16 => (i16, threshold_greater_i16),
    f32 => (f32, threshold_greater_f32),
    f64 => (f64, threshold_greater_f64),
    ComplexI16 => (i16, threshold_greater_i16_complex),
    Complex32 => (f32, threshold_greater_f32_complex),
    Complex64 => (f64, threshold_greater_f64_complex)
]);

impl_signal_threshold_value_dispatch!(ThresholdLessValue, threshold_less_value, threshold_less_value, [
    i16 => (i16, threshold_less_value_i16),
    f32 => (f32, threshold_less_value_f32),
    f64 => (f64, threshold_less_value_f64),
    ComplexI16 => (i16, threshold_less_value_i16_complex),
    Complex32 => (f32, threshold_less_value_f32_complex),
    Complex64 => (f64, threshold_less_value_f64_complex)
]);

impl_signal_threshold_value_dispatch!(ThresholdGreaterValue, threshold_greater_value, threshold_greater_value, [
    i16 => (i16, threshold_greater_value_i16),
    f32 => (f32, threshold_greater_value_f32),
    f64 => (f64, threshold_greater_value_f64),
    ComplexI16 => (i16, threshold_greater_value_i16_complex),
    Complex32 => (f32, threshold_greater_value_f32_complex),
    Complex64 => (f64, threshold_greater_value_f64_complex)

]);
