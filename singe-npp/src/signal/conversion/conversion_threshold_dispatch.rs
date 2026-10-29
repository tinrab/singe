use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        conversion::{
            threshold_f32, threshold_f32_complex, threshold_f32_complex_in_place,
            threshold_f32_in_place, threshold_f64, threshold_f64_complex,
            threshold_f64_complex_in_place, threshold_f64_in_place, threshold_greater_f32,
            threshold_greater_f32_complex, threshold_greater_f64, threshold_greater_f64_complex,
            threshold_greater_i16, threshold_greater_i16_complex, threshold_greater_value_f32,
            threshold_greater_value_f32_complex, threshold_greater_value_f64,
            threshold_greater_value_f64_complex, threshold_greater_value_i16,
            threshold_greater_value_i16_complex, threshold_i16, threshold_i16_complex,
            threshold_i16_complex_in_place, threshold_i16_in_place, threshold_less_f32,
            threshold_less_f32_complex, threshold_less_f64, threshold_less_f64_complex,
            threshold_less_i16, threshold_less_i16_complex, threshold_less_value_f32,
            threshold_less_value_f32_complex, threshold_less_value_f64,
            threshold_less_value_f64_complex, threshold_less_value_i16,
            threshold_less_value_i16_complex,
        },
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::{ComparisonOperation, ComplexI16, DataTypeLike},
};

macro_rules! impl_signal_threshold_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Scalar type accepted as the threshold level.
            type Level: DataTypeLike;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
            /// Source and destination lengths must match.
            /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the selected stream reads them.
            /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
            /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Self>,
                level: Self::Level,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                type Level = $level_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
                    level: Self::Level,
                    operation: ComparisonOperation,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, level, operation)

        }
    }
            }
        )*

        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
            level: T::Level,
            operation: ComparisonOperation,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination, level, operation)

        }
    }
    };
}

macro_rules! impl_signal_threshold_in_place_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Scalar type accepted as the threshold level.
            type Level: DataTypeLike;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
            /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
            /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                level: Self::Level,
                operation: ComparisonOperation,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                type Level = $level_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    level: Self::Level,
                    operation: ComparisonOperation,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal, level, operation)

        }
    }
            }
        )*

        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// `signal` must contain initialized values ready for the selected stream and remain exclusively writable through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            level: T::Level,
            operation: ComparisonOperation,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, signal, level, operation)

        }
    }
    };
}

macro_rules! impl_signal_threshold_fixed_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Scalar type accepted as the threshold level.
            type Level: DataTypeLike;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
            /// Source and destination lengths must match.
            /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the selected stream reads them.
            /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
            /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Self>,
                level: Self::Level,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                type Level = $level_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
                    level: Self::Level,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, level)

        }
    }
            }
        )*

        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
            level: T::Level,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination, level)

        }
    }
    };
}

macro_rules! impl_signal_threshold_value_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => ($level_ty:ty, $direct:ident)),* $(,)?]) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Scalar type accepted as the threshold level.
            type Level: DataTypeLike;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails or a signal belongs to another CUDA context.
            /// Source and destination lengths must match.
            /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the selected stream reads them.
            /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
            /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Self>,
                level: Self::Level,
                value: Self,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                type Level = $level_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
                    level: Self::Level,
                    value: Self,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, level, value)

        }
    }
            }
        )*

        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a signal belongs to another CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if the signal length cannot be represented by the native interface, or NPP rejects the threshold parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the selected stream reads them.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source through completion.
        /// Retain signal allocations and the stream context through queued completion, including submission errors and captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
            level: T::Level,
            value: T,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination, level, value)

        }
    }
    };
}

impl_signal_threshold_dispatch!("Applies a constant-level threshold using the selected comparison `operation`. Writes the result to `destination`.",
Threshold, threshold, threshold, [
    i16 => (i16, threshold_i16),
    f32 => (f32, threshold_f32),
    f64 => (f64, threshold_f64),
    ComplexI16 => (i16, threshold_i16_complex),
    Complex32 => (f32, threshold_f32_complex),
    Complex64 => (f64, threshold_f64_complex)
]);

impl_signal_threshold_in_place_dispatch!("Applies a constant-level threshold using the selected comparison `operation`. Updates `signal` in place.",
ThresholdInPlace, threshold_in_place, threshold_in_place, [
    i16 => (i16, threshold_i16_in_place),
    f32 => (f32, threshold_f32_in_place),
    f64 => (f64, threshold_f64_in_place),
    ComplexI16 => (i16, threshold_i16_complex_in_place),
    Complex32 => (f32, threshold_f32_complex_in_place),
    Complex64 => (f64, threshold_f64_complex_in_place)
]);

impl_signal_threshold_fixed_dispatch!("Applies the native less-than threshold at `level`. Writes the result to `destination`.",
ThresholdLess, threshold_less, threshold_less, [
    i16 => (i16, threshold_less_i16),
    f32 => (f32, threshold_less_f32),
    f64 => (f64, threshold_less_f64),
    ComplexI16 => (i16, threshold_less_i16_complex),
    Complex32 => (f32, threshold_less_f32_complex),
    Complex64 => (f64, threshold_less_f64_complex)
]);

impl_signal_threshold_fixed_dispatch!("Applies the native greater-than threshold at `level`. Writes the result to `destination`.",
ThresholdGreater, threshold_greater, threshold_greater, [
    i16 => (i16, threshold_greater_i16),
    f32 => (f32, threshold_greater_f32),
    f64 => (f64, threshold_greater_f64),
    ComplexI16 => (i16, threshold_greater_i16_complex),
    Complex32 => (f32, threshold_greater_f32_complex),
    Complex64 => (f64, threshold_greater_f64_complex)
]);

impl_signal_threshold_value_dispatch!("Replaces values selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.",
ThresholdLessValue, threshold_less_value, threshold_less_value, [
    i16 => (i16, threshold_less_value_i16),
    f32 => (f32, threshold_less_value_f32),
    f64 => (f64, threshold_less_value_f64),
    ComplexI16 => (i16, threshold_less_value_i16_complex),
    Complex32 => (f32, threshold_less_value_f32_complex),
    Complex64 => (f64, threshold_less_value_f64_complex)
]);

impl_signal_threshold_value_dispatch!("Replaces values selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.",
ThresholdGreaterValue, threshold_greater_value, threshold_greater_value, [
    i16 => (i16, threshold_greater_value_i16),
    f32 => (f32, threshold_greater_value_f32),
    f64 => (f64, threshold_greater_value_f64),
    ComplexI16 => (i16, threshold_greater_value_i16_complex),
    Complex32 => (f32, threshold_greater_value_f32_complex),
    Complex64 => (f64, threshold_greater_value_f64_complex)

]);
