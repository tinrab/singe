use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        statistics::{
            average_error_f32_buffer_size, average_error_f32_complex_buffer_size,
            average_error_f32_complex_to_device, average_error_f32_complex_to_device_with_scratch,
            average_error_f32_to_device, average_error_f32_to_device_with_scratch,
            average_error_f64_buffer_size, average_error_f64_complex_buffer_size,
            average_error_f64_complex_to_device, average_error_f64_complex_to_device_with_scratch,
            average_error_f64_to_device, average_error_f64_to_device_with_scratch,
            average_error_i8_buffer_size, average_error_i8_to_device,
            average_error_i8_to_device_with_scratch, average_error_i16_buffer_size,
            average_error_i16_complex_buffer_size, average_error_i16_complex_to_device,
            average_error_i16_complex_to_device_with_scratch, average_error_i16_to_device,
            average_error_i16_to_device_with_scratch, average_error_i32_buffer_size,
            average_error_i32_complex_buffer_size, average_error_i32_complex_to_device,
            average_error_i32_complex_to_device_with_scratch, average_error_i32_to_device,
            average_error_i32_to_device_with_scratch, average_error_i64_buffer_size,
            average_error_i64_complex_buffer_size, average_error_i64_complex_to_device,
            average_error_i64_complex_to_device_with_scratch, average_error_i64_to_device,
            average_error_i64_to_device_with_scratch, average_error_u8_buffer_size,
            average_error_u8_to_device, average_error_u8_to_device_with_scratch,
            average_error_u16_buffer_size, average_error_u16_to_device,
            average_error_u16_to_device_with_scratch, average_error_u32_buffer_size,
            average_error_u32_to_device, average_error_u32_to_device_with_scratch,
            average_relative_error_f32_buffer_size, average_relative_error_f32_complex_buffer_size,
            average_relative_error_f32_complex_to_device,
            average_relative_error_f32_complex_to_device_with_scratch,
            average_relative_error_f32_to_device,
            average_relative_error_f32_to_device_with_scratch,
            average_relative_error_f64_buffer_size, average_relative_error_f64_complex_buffer_size,
            average_relative_error_f64_complex_to_device,
            average_relative_error_f64_complex_to_device_with_scratch,
            average_relative_error_f64_to_device,
            average_relative_error_f64_to_device_with_scratch,
            average_relative_error_i8_buffer_size, average_relative_error_i8_to_device,
            average_relative_error_i8_to_device_with_scratch,
            average_relative_error_i16_buffer_size, average_relative_error_i16_complex_buffer_size,
            average_relative_error_i16_complex_to_device,
            average_relative_error_i16_complex_to_device_with_scratch,
            average_relative_error_i16_to_device,
            average_relative_error_i16_to_device_with_scratch,
            average_relative_error_i32_buffer_size, average_relative_error_i32_complex_buffer_size,
            average_relative_error_i32_complex_to_device,
            average_relative_error_i32_complex_to_device_with_scratch,
            average_relative_error_i32_to_device,
            average_relative_error_i32_to_device_with_scratch,
            average_relative_error_i64_buffer_size, average_relative_error_i64_complex_buffer_size,
            average_relative_error_i64_complex_to_device,
            average_relative_error_i64_complex_to_device_with_scratch,
            average_relative_error_i64_to_device,
            average_relative_error_i64_to_device_with_scratch,
            average_relative_error_u8_buffer_size, average_relative_error_u8_to_device,
            average_relative_error_u8_to_device_with_scratch,
            average_relative_error_u16_buffer_size, average_relative_error_u16_to_device,
            average_relative_error_u16_to_device_with_scratch,
            average_relative_error_u32_buffer_size, average_relative_error_u32_to_device,
            average_relative_error_u32_to_device_with_scratch, max_absolute_index_i16_buffer_size,
            max_absolute_index_i16_to_device, max_absolute_index_i16_to_device_with_scratch,
            max_absolute_index_i32_buffer_size, max_absolute_index_i32_to_device,
            max_absolute_index_i32_to_device_with_scratch, max_every_f32_in_place,
            max_every_i16_in_place, max_every_i32_in_place, max_every_u8_in_place,
            max_every_u16_in_place, max_index_f32_buffer_size, max_index_f32_to_device,
            max_index_f32_to_device_with_scratch, max_index_f64_buffer_size,
            max_index_f64_to_device, max_index_f64_to_device_with_scratch,
            max_index_i16_buffer_size, max_index_i16_to_device,
            max_index_i16_to_device_with_scratch, max_index_i32_buffer_size,
            max_index_i32_to_device, max_index_i32_to_device_with_scratch,
            maximum_error_f32_buffer_size, maximum_error_f32_complex_buffer_size,
            maximum_error_f32_complex_to_device, maximum_error_f32_complex_to_device_with_scratch,
            maximum_error_f32_to_device, maximum_error_f32_to_device_with_scratch,
            maximum_error_f64_buffer_size, maximum_error_f64_complex_buffer_size,
            maximum_error_f64_complex_to_device, maximum_error_f64_complex_to_device_with_scratch,
            maximum_error_f64_to_device, maximum_error_f64_to_device_with_scratch,
            maximum_error_i8_buffer_size, maximum_error_i8_to_device,
            maximum_error_i8_to_device_with_scratch, maximum_error_i16_buffer_size,
            maximum_error_i16_complex_buffer_size, maximum_error_i16_complex_to_device,
            maximum_error_i16_complex_to_device_with_scratch, maximum_error_i16_to_device,
            maximum_error_i16_to_device_with_scratch, maximum_error_i32_buffer_size,
            maximum_error_i32_complex_buffer_size, maximum_error_i32_complex_to_device,
            maximum_error_i32_complex_to_device_with_scratch, maximum_error_i32_to_device,
            maximum_error_i32_to_device_with_scratch, maximum_error_i64_buffer_size,
            maximum_error_i64_complex_buffer_size, maximum_error_i64_complex_to_device,
            maximum_error_i64_complex_to_device_with_scratch, maximum_error_i64_to_device,
            maximum_error_i64_to_device_with_scratch, maximum_error_u8_buffer_size,
            maximum_error_u8_to_device, maximum_error_u8_to_device_with_scratch,
            maximum_error_u16_buffer_size, maximum_error_u16_to_device,
            maximum_error_u16_to_device_with_scratch, maximum_error_u32_buffer_size,
            maximum_error_u32_to_device, maximum_error_u32_to_device_with_scratch,
            maximum_relative_error_f32_buffer_size, maximum_relative_error_f32_complex_buffer_size,
            maximum_relative_error_f32_complex_to_device,
            maximum_relative_error_f32_complex_to_device_with_scratch,
            maximum_relative_error_f32_to_device,
            maximum_relative_error_f32_to_device_with_scratch,
            maximum_relative_error_f64_buffer_size, maximum_relative_error_f64_complex_buffer_size,
            maximum_relative_error_f64_complex_to_device,
            maximum_relative_error_f64_complex_to_device_with_scratch,
            maximum_relative_error_f64_to_device,
            maximum_relative_error_f64_to_device_with_scratch,
            maximum_relative_error_i8_buffer_size, maximum_relative_error_i8_to_device,
            maximum_relative_error_i8_to_device_with_scratch,
            maximum_relative_error_i16_buffer_size, maximum_relative_error_i16_complex_buffer_size,
            maximum_relative_error_i16_complex_to_device,
            maximum_relative_error_i16_complex_to_device_with_scratch,
            maximum_relative_error_i16_to_device,
            maximum_relative_error_i16_to_device_with_scratch,
            maximum_relative_error_i32_buffer_size, maximum_relative_error_i32_complex_buffer_size,
            maximum_relative_error_i32_complex_to_device,
            maximum_relative_error_i32_complex_to_device_with_scratch,
            maximum_relative_error_i32_to_device,
            maximum_relative_error_i32_to_device_with_scratch,
            maximum_relative_error_i64_buffer_size, maximum_relative_error_i64_complex_buffer_size,
            maximum_relative_error_i64_complex_to_device,
            maximum_relative_error_i64_complex_to_device_with_scratch,
            maximum_relative_error_i64_to_device,
            maximum_relative_error_i64_to_device_with_scratch,
            maximum_relative_error_u8_buffer_size, maximum_relative_error_u8_to_device,
            maximum_relative_error_u8_to_device_with_scratch,
            maximum_relative_error_u16_buffer_size, maximum_relative_error_u16_to_device,
            maximum_relative_error_u16_to_device_with_scratch,
            maximum_relative_error_u32_buffer_size, maximum_relative_error_u32_to_device,
            maximum_relative_error_u32_to_device_with_scratch,
            mean_standard_deviation_f32_buffer_size, mean_standard_deviation_f32_to_device,
            mean_standard_deviation_f32_to_device_with_scratch,
            mean_standard_deviation_f64_buffer_size, mean_standard_deviation_f64_to_device,
            mean_standard_deviation_f64_to_device_with_scratch,
            mean_standard_deviation_i16_scaled_buffer_size,
            mean_standard_deviation_i16_scaled_to_device,
            mean_standard_deviation_i16_scaled_to_device_with_scratch,
            mean_standard_deviation_i16_to_i32_scaled_buffer_size,
            mean_standard_deviation_i16_to_i32_scaled_to_device,
            mean_standard_deviation_i16_to_i32_scaled_to_device_with_scratch,
            min_absolute_index_i16_buffer_size, min_absolute_index_i16_to_device,
            min_absolute_index_i16_to_device_with_scratch, min_absolute_index_i32_buffer_size,
            min_absolute_index_i32_to_device, min_absolute_index_i32_to_device_with_scratch,
            min_every_f32_in_place, min_every_f64_in_place, min_every_i16_in_place,
            min_every_i32_in_place, min_every_u8_in_place, min_every_u16_in_place,
            min_index_f32_buffer_size, min_index_f32_to_device,
            min_index_f32_to_device_with_scratch, min_index_f64_buffer_size,
            min_index_f64_to_device, min_index_f64_to_device_with_scratch,
            min_index_i16_buffer_size, min_index_i16_to_device,
            min_index_i16_to_device_with_scratch, min_index_i32_buffer_size,
            min_index_i32_to_device, min_index_i32_to_device_with_scratch, min_max_f32_buffer_size,
            min_max_f32_to_device, min_max_f32_to_device_with_scratch, min_max_f64_buffer_size,
            min_max_f64_to_device, min_max_f64_to_device_with_scratch, min_max_i16_buffer_size,
            min_max_i16_to_device, min_max_i16_to_device_with_scratch, min_max_i32_buffer_size,
            min_max_i32_to_device, min_max_i32_to_device_with_scratch,
            min_max_index_f32_buffer_size, min_max_index_f32_to_device,
            min_max_index_f32_to_device_with_scratch, min_max_index_f64_buffer_size,
            min_max_index_f64_to_device, min_max_index_f64_to_device_with_scratch,
            min_max_index_i16_buffer_size, min_max_index_i16_to_device,
            min_max_index_i16_to_device_with_scratch, min_max_index_i32_buffer_size,
            min_max_index_i32_to_device, min_max_index_i32_to_device_with_scratch,
            min_max_index_u8_buffer_size, min_max_index_u8_to_device,
            min_max_index_u8_to_device_with_scratch, min_max_index_u16_buffer_size,
            min_max_index_u16_to_device, min_max_index_u16_to_device_with_scratch,
            min_max_index_u32_buffer_size, min_max_index_u32_to_device,
            min_max_index_u32_to_device_with_scratch, min_max_u8_buffer_size, min_max_u8_to_device,
            min_max_u8_to_device_with_scratch, min_max_u16_buffer_size, min_max_u16_to_device,
            min_max_u16_to_device_with_scratch, min_max_u32_buffer_size, min_max_u32_to_device,
            min_max_u32_to_device_with_scratch, zero_crossing_f32_buffer_size,
            zero_crossing_f32_to_device, zero_crossing_f32_to_device_with_scratch,
            zero_crossing_i16_to_f32_buffer_size, zero_crossing_i16_to_f32_to_device,
            zero_crossing_i16_to_f32_to_device_with_scratch,
        },
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, ComplexI64, DataTypeLike, ZeroCrossingType},
    workspace::ScratchBuffer,
};

#[macro_use]
mod scalar_dispatch;

pub use scalar_dispatch::*;

mod binary_dispatch;

pub use binary_dispatch::*;

macro_rules! impl_signal_indexed_statistic_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Returns the scratch capacity, in bytes, required for the source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
            /// through completion, disjoint from source and result storage.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: &mut SignalOutputView<'_, Self>,
                index: &mut SignalOutputView<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: &mut SignalOutputView<'_, Self>,
                index: &mut SignalOutputView<'_, i32>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: &mut SignalOutputView<'_, Self>,
                    index: &mut SignalOutputView<'_, i32>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(value)?;
        stream_context.validate_region(index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(stream_context, source, value, index, scratch)

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: &mut SignalOutputView<'_, Self>,
                    index: &mut SignalOutputView<'_, i32>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(value)?;
        stream_context.validate_region(index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, value, index)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, required for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
            T::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from source and result storage.
        pub unsafe fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: &mut SignalOutputView<'_, T>,
            index: &mut SignalOutputView<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(value)?;
        stream_context.validate_region(index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$scratch_method(stream_context, source, value, index, scratch)

        }
    }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: &mut SignalOutputView<'_, T>,
            index: &mut SignalOutputView<'_, i32>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(value)?;
        stream_context.validate_region(index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, value, index)

        }
    }
    };
}

impl_signal_indexed_statistic_dispatch!("Finds the maximum source value and its index.",

    MaxIndex,
    max_index_buffer_size,
    max_index_to_device_with_scratch,
    max_index_to_device,
    max_index_buffer_size,
    max_index_to_device_with_scratch,
    max_index_to_device,
    [
        i16 => (max_index_i16_buffer_size, max_index_i16_to_device_with_scratch, max_index_i16_to_device),
        i32 => (max_index_i32_buffer_size, max_index_i32_to_device_with_scratch, max_index_i32_to_device),
        f32 => (max_index_f32_buffer_size, max_index_f32_to_device_with_scratch, max_index_f32_to_device),
        f64 => (max_index_f64_buffer_size, max_index_f64_to_device_with_scratch, max_index_f64_to_device)
    ]
);

impl_signal_indexed_statistic_dispatch!("Finds the minimum source value and its index.",

    MinIndex,
    min_index_buffer_size,
    min_index_to_device_with_scratch,
    min_index_to_device,
    min_index_buffer_size,
    min_index_to_device_with_scratch,
    min_index_to_device,
    [
        i16 => (min_index_i16_buffer_size, min_index_i16_to_device_with_scratch, min_index_i16_to_device),
        i32 => (min_index_i32_buffer_size, min_index_i32_to_device_with_scratch, min_index_i32_to_device),
        f32 => (min_index_f32_buffer_size, min_index_f32_to_device_with_scratch, min_index_f32_to_device),
        f64 => (min_index_f64_buffer_size, min_index_f64_to_device_with_scratch, min_index_f64_to_device)
    ]
);

impl_signal_indexed_statistic_dispatch!("Finds the maximum absolute source value and its index.",

    MaxAbsoluteIndex,
    max_absolute_index_buffer_size,
    max_absolute_index_to_device_with_scratch,
    max_absolute_index_to_device,
    max_absolute_index_buffer_size,
    max_absolute_index_to_device_with_scratch,
    max_absolute_index_to_device,
    [
        i16 => (max_absolute_index_i16_buffer_size, max_absolute_index_i16_to_device_with_scratch, max_absolute_index_i16_to_device),
        i32 => (max_absolute_index_i32_buffer_size, max_absolute_index_i32_to_device_with_scratch, max_absolute_index_i32_to_device)
    ]
);

impl_signal_indexed_statistic_dispatch!("Finds the minimum absolute source value and its index.",

    MinAbsoluteIndex,
    min_absolute_index_buffer_size,
    min_absolute_index_to_device_with_scratch,
    min_absolute_index_to_device,
    min_absolute_index_buffer_size,
    min_absolute_index_to_device_with_scratch,
    min_absolute_index_to_device,
    [
        i16 => (min_absolute_index_i16_buffer_size, min_absolute_index_i16_to_device_with_scratch, min_absolute_index_i16_to_device),
        i32 => (min_absolute_index_i32_buffer_size, min_absolute_index_i32_to_device_with_scratch, min_absolute_index_i32_to_device)
    ]
);

macro_rules! impl_signal_pair_statistic_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Returns the scratch capacity, in bytes, required for the source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
            /// through completion, disjoint from source and result storage.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalOutputView<'_, Self>,
                max: &mut SignalOutputView<'_, Self>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalOutputView<'_, Self>,
                max: &mut SignalOutputView<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalOutputView<'_, Self>,
                    max: &mut SignalOutputView<'_, Self>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(max)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(stream_context, source, min, max, scratch)

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalOutputView<'_, Self>,
                    max: &mut SignalOutputView<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(max)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, min, max)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, required for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
            T::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from source and result storage.
        pub unsafe fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalOutputView<'_, T>,
            max: &mut SignalOutputView<'_, T>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(max)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$scratch_method(stream_context, source, min, max, scratch)

        }
    }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalOutputView<'_, T>,
            max: &mut SignalOutputView<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(max)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, min, max)

        }
    }
    };
}

macro_rules! impl_signal_indexed_pair_statistic_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Returns the scratch capacity, in bytes, required for the source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
            /// through completion, disjoint from source and result storage.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalOutputView<'_, Self>,
                min_index: &mut SignalOutputView<'_, i32>,
                max: &mut SignalOutputView<'_, Self>,
                max_index: &mut SignalOutputView<'_, i32>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                min: &mut SignalOutputView<'_, Self>,
                min_index: &mut SignalOutputView<'_, i32>,
                max: &mut SignalOutputView<'_, Self>,
                max_index: &mut SignalOutputView<'_, i32>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalOutputView<'_, Self>,
                    min_index: &mut SignalOutputView<'_, i32>,
                    max: &mut SignalOutputView<'_, Self>,
                    max_index: &mut SignalOutputView<'_, i32>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(min_index)?;
        stream_context.validate_region(max)?;
        stream_context.validate_region(max_index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(
                        stream_context,
                        source,
                        min,
                        min_index,
                        max,
                        max_index,
                        scratch,
                    )

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    min: &mut SignalOutputView<'_, Self>,
                    min_index: &mut SignalOutputView<'_, i32>,
                    max: &mut SignalOutputView<'_, Self>,
                    max_index: &mut SignalOutputView<'_, i32>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(min_index)?;
        stream_context.validate_region(max)?;
        stream_context.validate_region(max_index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, min, min_index, max, max_index)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, required for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
            T::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from source and result storage.
        pub unsafe fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalOutputView<'_, T>,
            min_index: &mut SignalOutputView<'_, i32>,
            max: &mut SignalOutputView<'_, T>,
            max_index: &mut SignalOutputView<'_, i32>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(min_index)?;
        stream_context.validate_region(max)?;
        stream_context.validate_region(max_index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$scratch_method(
                stream_context,
                source,
                min,
                min_index,
                max,
                max_index,
                scratch,
            )

        }
    }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            min: &mut SignalOutputView<'_, T>,
            min_index: &mut SignalOutputView<'_, i32>,
            max: &mut SignalOutputView<'_, T>,
            max_index: &mut SignalOutputView<'_, i32>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(min)?;
        stream_context.validate_region(min_index)?;
        stream_context.validate_region(max)?;
        stream_context.validate_region(max_index)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, min, min_index, max, max_index)

        }
    }
    };
}

impl_signal_pair_statistic_dispatch!("Finds the minimum and maximum source values.",

    MinMax,
    min_max_buffer_size,
    min_max_to_device_with_scratch,
    min_max_to_device,
    min_max_buffer_size,
    min_max_to_device_with_scratch,
    min_max_to_device,
    [
        u8 => (min_max_u8_buffer_size, min_max_u8_to_device_with_scratch, min_max_u8_to_device),
        u16 => (min_max_u16_buffer_size, min_max_u16_to_device_with_scratch, min_max_u16_to_device),
        i16 => (min_max_i16_buffer_size, min_max_i16_to_device_with_scratch, min_max_i16_to_device),
        u32 => (min_max_u32_buffer_size, min_max_u32_to_device_with_scratch, min_max_u32_to_device),
        i32 => (min_max_i32_buffer_size, min_max_i32_to_device_with_scratch, min_max_i32_to_device),
        f32 => (min_max_f32_buffer_size, min_max_f32_to_device_with_scratch, min_max_f32_to_device),
        f64 => (min_max_f64_buffer_size, min_max_f64_to_device_with_scratch, min_max_f64_to_device)
    ]
);

impl_signal_indexed_pair_statistic_dispatch!("Finds the minimum and maximum source values and their indices.",

    MinMaxIndex,
    min_max_index_buffer_size,
    min_max_index_to_device_with_scratch,
    min_max_index_to_device,
    min_max_index_buffer_size,
    min_max_index_to_device_with_scratch,
    min_max_index_to_device,
    [
        u8 => (min_max_index_u8_buffer_size, min_max_index_u8_to_device_with_scratch, min_max_index_u8_to_device),
        u16 => (min_max_index_u16_buffer_size, min_max_index_u16_to_device_with_scratch, min_max_index_u16_to_device),
        i16 => (min_max_index_i16_buffer_size, min_max_index_i16_to_device_with_scratch, min_max_index_i16_to_device),
        u32 => (min_max_index_u32_buffer_size, min_max_index_u32_to_device_with_scratch, min_max_index_u32_to_device),
        i32 => (min_max_index_i32_buffer_size, min_max_index_i32_to_device_with_scratch, min_max_index_i32_to_device),
        f32 => (min_max_index_f32_buffer_size, min_max_index_f32_to_device_with_scratch, min_max_index_f32_to_device),
        f64 => (min_max_index_f64_buffer_size, min_max_index_f64_to_device_with_scratch, min_max_index_f64_to_device)
    ]
);

macro_rules! impl_signal_mean_standard_deviation_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($source_ty:ty => ($destination_ty:ty, $direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            /// Returns the scratch capacity, in bytes, required for the source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
            /// through completion, disjoint from source and result storage.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                mean: &mut SignalOutputView<'_, Destination>,
                standard_deviation: &mut SignalOutputView<'_, Destination>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                mean: &mut SignalOutputView<'_, Destination>,
                standard_deviation: &mut SignalOutputView<'_, Destination>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    mean: &mut SignalOutputView<'_, $destination_ty>,
                    standard_deviation: &mut SignalOutputView<'_, $destination_ty>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(stream_context, source, mean, standard_deviation, scratch)

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    mean: &mut SignalOutputView<'_, $destination_ty>,
                    standard_deviation: &mut SignalOutputView<'_, $destination_ty>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, mean, standard_deviation)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, required for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        pub fn $buffer_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
        ) -> Result<usize>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
            Source::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from source and result storage.
        pub unsafe fn $scratch_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            mean: &mut SignalOutputView<'_, Destination>,
            standard_deviation: &mut SignalOutputView<'_, Destination>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$scratch_method(stream_context, source, mean, standard_deviation, scratch)

        }
    }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            mean: &mut SignalOutputView<'_, Destination>,
            standard_deviation: &mut SignalOutputView<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, mean, standard_deviation)

        }
    }
    };
}

macro_rules! impl_signal_scaled_mean_standard_deviation_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($source_ty:ty => ($destination_ty:ty, $direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            /// Returns the scratch capacity, in bytes, required for the source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Applies native fixed-point scaling using `scale_factor`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
            /// through completion, disjoint from source and result storage.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                mean: &mut SignalOutputView<'_, Destination>,
                standard_deviation: &mut SignalOutputView<'_, Destination>,
                scale_factor: i32,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Allocates temporary scratch storage for this call.
            /// Applies native fixed-point scaling using `scale_factor`.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                mean: &mut SignalOutputView<'_, Destination>,
                standard_deviation: &mut SignalOutputView<'_, Destination>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    mean: &mut SignalOutputView<'_, $destination_ty>,
                    standard_deviation: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(
                        stream_context,
                        source,
                        mean,
                        standard_deviation,
                        scale_factor,
                        scratch,
                    )

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    mean: &mut SignalOutputView<'_, $destination_ty>,
                    standard_deviation: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, mean, standard_deviation, scale_factor)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, required for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        pub fn $buffer_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
        ) -> Result<usize>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
            Source::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from source and result storage.
        pub unsafe fn $scratch_function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            mean: &mut SignalOutputView<'_, Destination>,
            standard_deviation: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$scratch_method(
                stream_context,
                source,
                mean,
                standard_deviation,
                scale_factor,
                scratch,
            )

        }
    }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Allocates temporary scratch storage for this call.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            mean: &mut SignalOutputView<'_, Destination>,
            standard_deviation: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(mean)?;
        stream_context.validate_region(standard_deviation)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, mean, standard_deviation, scale_factor)

        }
    }
    };
}

impl_signal_mean_standard_deviation_dispatch!("Computes the mean and standard deviation of the source elements.",

    MeanStandardDeviationTo,
    mean_standard_deviation_buffer_size,
    mean_standard_deviation_to_device_with_scratch,
    mean_standard_deviation_to_device,
    mean_standard_deviation_buffer_size,
    mean_standard_deviation_to_device_with_scratch,
    mean_standard_deviation_to_device,
    [
        f32 => (f32, mean_standard_deviation_f32_buffer_size, mean_standard_deviation_f32_to_device_with_scratch, mean_standard_deviation_f32_to_device),
        f64 => (f64, mean_standard_deviation_f64_buffer_size, mean_standard_deviation_f64_to_device_with_scratch, mean_standard_deviation_f64_to_device)
    ]
);

impl_signal_scaled_mean_standard_deviation_dispatch!("Computes the mean and standard deviation of the source elements.",

    MeanStandardDeviationScaledTo,
    mean_standard_deviation_scaled_buffer_size,
    mean_standard_deviation_scaled_to_device_with_scratch,
    mean_standard_deviation_scaled_to_device,
    mean_standard_deviation_scaled_buffer_size,
    mean_standard_deviation_scaled_to_device_with_scratch,
    mean_standard_deviation_scaled_to_device,
    [
        i16 => (i16, mean_standard_deviation_i16_scaled_buffer_size, mean_standard_deviation_i16_scaled_to_device_with_scratch, mean_standard_deviation_i16_scaled_to_device),
        i16 => (i32, mean_standard_deviation_i16_to_i32_scaled_buffer_size, mean_standard_deviation_i16_to_i32_scaled_to_device_with_scratch, mean_standard_deviation_i16_to_i32_scaled_to_device)
    ]
);

macro_rules! impl_signal_every_statistic_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Source and destination lengths must match.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Both signals must contain initialized values ready for the selected stream.
            /// Keep the source read-only and destination exclusively writable and disjoint from the source.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination)

        }
    }
            }
        )*

        #[doc = $doc]
        ///
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Source and destination lengths must match.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Both signals must contain initialized values ready for the selected stream.
        /// Keep the source read-only and destination exclusively writable and disjoint from the source.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination)

        }
    }
    };
}

impl_signal_every_statistic_dispatch!("Replaces each destination element with the minimum of its current value and the corresponding source element.",
MinEveryInPlace, min_every_in_place, min_every_in_place, [
    u8 => min_every_u8_in_place,
    u16 => min_every_u16_in_place,
    i16 => min_every_i16_in_place,
    i32 => min_every_i32_in_place,
    f32 => min_every_f32_in_place,
    f64 => min_every_f64_in_place
]);

impl_signal_every_statistic_dispatch!("Replaces each destination element with the maximum of its current value and the corresponding source element.",
MaxEveryInPlace, max_every_in_place, max_every_in_place, [
    u8 => max_every_u8_in_place,
    u16 => max_every_u16_in_place,
    i16 => max_every_i16_in_place,
    i32 => max_every_i32_in_place,
    f32 => max_every_f32_in_place
]);

macro_rules! impl_signal_zero_crossing_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Returns the scratch capacity, in bytes, required for the source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
            /// through completion, disjoint from source and result storage.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, f32>,
                zero_crossing_type: ZeroCrossingType,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, f32>,
                zero_crossing_type: ZeroCrossingType,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, f32>,
                    zero_crossing_type: ZeroCrossingType,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(
                        stream_context,
                        source,
                        destination,
                        zero_crossing_type,
                        scratch,
                    )

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, f32>,
                    zero_crossing_type: ZeroCrossingType,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, zero_crossing_type)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, required for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
            T::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from source and result storage.
        pub unsafe fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$scratch_method(
                stream_context,
                source,
                destination,
                zero_crossing_type,
                scratch,
            )

        }
    }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, f32>,
            zero_crossing_type: ZeroCrossingType,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination, zero_crossing_type)

        }
    }
    };
}

impl_signal_zero_crossing_dispatch!("Computes the zero-crossing statistic selected by `zero_crossing_type`.",

    ZeroCrossingToF32,
    zero_crossing_to_f32_buffer_size,
    zero_crossing_to_f32_to_device_with_scratch,
    zero_crossing_to_f32_to_device,
    zero_crossing_to_f32_buffer_size,
    zero_crossing_to_f32_to_device_with_scratch,
    zero_crossing_to_f32_to_device,
    [
        i16 => (zero_crossing_i16_to_f32_buffer_size, zero_crossing_i16_to_f32_to_device_with_scratch, zero_crossing_i16_to_f32_to_device),
        f32 => (zero_crossing_f32_buffer_size, zero_crossing_f32_to_device_with_scratch, zero_crossing_f32_to_device)
    ]
);

macro_rules! impl_signal_error_metric_dispatch {
    (
        $doc:literal,
        $trait:ident,
        $buffer_method:ident,
        $scratch_method:ident,
        $method:ident,
        $buffer_function:ident,
        $scratch_function:ident,
        $function:ident,
        [$($ty:ty => ($direct_buffer:ident, $direct_scratch:ident, $direct:ident)),* $(,)?]
    ) => {
        /// Native dispatch for the corresponding named NPP operation.
        ///
        /// # Safety
        /// Implementations must match the native ABI, access roles, written ranges,
        /// and status semantics. Successful completed execution must initialize its
        /// documented outputs; metadata queries must return the required capacity.
        pub unsafe trait $trait: DataTypeLike {
            /// Returns the scratch capacity, in bytes, required for the source length.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            fn $buffer_method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
            ) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// The two sources must have equal lengths.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// An undersized scratch buffer also returns an error.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
            /// through completion, disjoint from source and result storage.
            unsafe fn $scratch_method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, f64>,
                scratch: &mut ScratchBuffer,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Each scalar result view must contain exactly one element.
            /// Allocates temporary scratch storage for this call.
            ///
            /// # Errors
            ///
            /// Provided implementations return an error if CUDA context binding fails.
            /// Signals must belong to the selected CUDA context.
            /// Each scalar result view must contain one element; other lengths return an error.
            /// The two sources must have equal lengths.
            /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
            /// Scratch allocation errors are propagated.
            ///
            /// # Safety
            ///
            /// Follow the [raw signal execution contract](crate::signal::raw).
            /// Source values must be initialized and ready when the stream reads them.
            /// Result storage must be exclusively writable and disjoint from all source and other result storage.
            /// Retain signal allocations and the stream context through queued completion,
            /// including submission errors and every captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source_1: &SignalView<'_, Self>,
                source_2: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, f64>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                fn $buffer_method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
                    $direct_buffer(stream_context, source)
                }

                unsafe fn $scratch_method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, f64>,
                    scratch: &mut ScratchBuffer,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct_scratch(stream_context, source_1, source_2, destination, scratch)

        }
    }

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_1: &SignalView<'_, Self>,
                    source_2: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, f64>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source_1, source_2, destination)

        }
    }
            }
        )*

        /// Returns the scratch capacity, in bytes, required for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        pub fn $buffer_function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
        ) -> Result<usize> {
        stream_context.cuda_context().bind()?;
            T::$buffer_method(stream_context, source)
        }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// The two sources must have equal lengths.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// An undersized scratch buffer also returns an error.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// `scratch` must belong to the same CUDA context and remain live and exclusively writable
        /// through completion, disjoint from source and result storage.
        pub unsafe fn $scratch_function<T: $trait>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, T>,
            source_2: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, f64>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$scratch_method(stream_context, source_1, source_2, destination, scratch)

        }
    }

        #[doc = $doc]
        ///
        /// Each scalar result view must contain exactly one element.
        /// Allocates temporary scratch storage for this call.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails.
        /// Signals must belong to the selected CUDA context.
        /// Each scalar result view must contain one element; other lengths return an error.
        /// The two sources must have equal lengths.
        /// Returns an error if a required length or queried-size conversion fails, or NPP reports a failure.
        /// Scratch allocation errors are propagated.
        ///
        /// # Safety
        ///
        /// Follow the [raw signal execution contract](crate::signal::raw).
        /// Source values must be initialized and ready when the stream reads them.
        /// Result storage must be exclusively writable and disjoint from all source and other result storage.
        /// Retain signal allocations and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_1: &SignalView<'_, T>,
            source_2: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, f64>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source_1)?;
        stream_context.validate_region(source_2)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source_1, source_2, destination)

        }
    }
    };
}

impl_signal_error_metric_dispatch!("Computes the maximum error between corresponding source elements.",

    MaximumError,
    maximum_error_buffer_size,
    maximum_error_to_device_with_scratch,
    maximum_error_to_device,
    maximum_error_buffer_size,
    maximum_error_to_device_with_scratch,
    maximum_error_to_device,
    [
        u8 => (maximum_error_u8_buffer_size, maximum_error_u8_to_device_with_scratch, maximum_error_u8_to_device),
        i8 => (maximum_error_i8_buffer_size, maximum_error_i8_to_device_with_scratch, maximum_error_i8_to_device),
        u16 => (maximum_error_u16_buffer_size, maximum_error_u16_to_device_with_scratch, maximum_error_u16_to_device),
        i16 => (maximum_error_i16_buffer_size, maximum_error_i16_to_device_with_scratch, maximum_error_i16_to_device),
        ComplexI16 => (maximum_error_i16_complex_buffer_size, maximum_error_i16_complex_to_device_with_scratch, maximum_error_i16_complex_to_device),
        u32 => (maximum_error_u32_buffer_size, maximum_error_u32_to_device_with_scratch, maximum_error_u32_to_device),
        i32 => (maximum_error_i32_buffer_size, maximum_error_i32_to_device_with_scratch, maximum_error_i32_to_device),
        ComplexI32 => (maximum_error_i32_complex_buffer_size, maximum_error_i32_complex_to_device_with_scratch, maximum_error_i32_complex_to_device),
        i64 => (maximum_error_i64_buffer_size, maximum_error_i64_to_device_with_scratch, maximum_error_i64_to_device),
        ComplexI64 => (maximum_error_i64_complex_buffer_size, maximum_error_i64_complex_to_device_with_scratch, maximum_error_i64_complex_to_device),
        f32 => (maximum_error_f32_buffer_size, maximum_error_f32_to_device_with_scratch, maximum_error_f32_to_device),
        Complex32 => (maximum_error_f32_complex_buffer_size, maximum_error_f32_complex_to_device_with_scratch, maximum_error_f32_complex_to_device),
        f64 => (maximum_error_f64_buffer_size, maximum_error_f64_to_device_with_scratch, maximum_error_f64_to_device),
        Complex64 => (maximum_error_f64_complex_buffer_size, maximum_error_f64_complex_to_device_with_scratch, maximum_error_f64_complex_to_device)
    ]
);

impl_signal_error_metric_dispatch!("Computes the average error between corresponding source elements.",

    AverageError,
    average_error_buffer_size,
    average_error_to_device_with_scratch,
    average_error_to_device,
    average_error_buffer_size,
    average_error_to_device_with_scratch,
    average_error_to_device,
    [
        u8 => (average_error_u8_buffer_size, average_error_u8_to_device_with_scratch, average_error_u8_to_device),
        i8 => (average_error_i8_buffer_size, average_error_i8_to_device_with_scratch, average_error_i8_to_device),
        u16 => (average_error_u16_buffer_size, average_error_u16_to_device_with_scratch, average_error_u16_to_device),
        i16 => (average_error_i16_buffer_size, average_error_i16_to_device_with_scratch, average_error_i16_to_device),
        ComplexI16 => (average_error_i16_complex_buffer_size, average_error_i16_complex_to_device_with_scratch, average_error_i16_complex_to_device),
        u32 => (average_error_u32_buffer_size, average_error_u32_to_device_with_scratch, average_error_u32_to_device),
        i32 => (average_error_i32_buffer_size, average_error_i32_to_device_with_scratch, average_error_i32_to_device),
        ComplexI32 => (average_error_i32_complex_buffer_size, average_error_i32_complex_to_device_with_scratch, average_error_i32_complex_to_device),
        i64 => (average_error_i64_buffer_size, average_error_i64_to_device_with_scratch, average_error_i64_to_device),
        ComplexI64 => (average_error_i64_complex_buffer_size, average_error_i64_complex_to_device_with_scratch, average_error_i64_complex_to_device),
        f32 => (average_error_f32_buffer_size, average_error_f32_to_device_with_scratch, average_error_f32_to_device),
        Complex32 => (average_error_f32_complex_buffer_size, average_error_f32_complex_to_device_with_scratch, average_error_f32_complex_to_device),
        f64 => (average_error_f64_buffer_size, average_error_f64_to_device_with_scratch, average_error_f64_to_device),
        Complex64 => (average_error_f64_complex_buffer_size, average_error_f64_complex_to_device_with_scratch, average_error_f64_complex_to_device)
    ]
);

impl_signal_error_metric_dispatch!("Computes the maximum relative error between the two source signals.",

    MaximumRelativeError,
    maximum_relative_error_buffer_size,
    maximum_relative_error_to_device_with_scratch,
    maximum_relative_error_to_device,
    maximum_relative_error_buffer_size,
    maximum_relative_error_to_device_with_scratch,
    maximum_relative_error_to_device,
    [
        u8 => (maximum_relative_error_u8_buffer_size, maximum_relative_error_u8_to_device_with_scratch, maximum_relative_error_u8_to_device),
        i8 => (maximum_relative_error_i8_buffer_size, maximum_relative_error_i8_to_device_with_scratch, maximum_relative_error_i8_to_device),
        u16 => (maximum_relative_error_u16_buffer_size, maximum_relative_error_u16_to_device_with_scratch, maximum_relative_error_u16_to_device),
        i16 => (maximum_relative_error_i16_buffer_size, maximum_relative_error_i16_to_device_with_scratch, maximum_relative_error_i16_to_device),
        ComplexI16 => (maximum_relative_error_i16_complex_buffer_size, maximum_relative_error_i16_complex_to_device_with_scratch, maximum_relative_error_i16_complex_to_device),
        u32 => (maximum_relative_error_u32_buffer_size, maximum_relative_error_u32_to_device_with_scratch, maximum_relative_error_u32_to_device),
        i32 => (maximum_relative_error_i32_buffer_size, maximum_relative_error_i32_to_device_with_scratch, maximum_relative_error_i32_to_device),
        ComplexI32 => (maximum_relative_error_i32_complex_buffer_size, maximum_relative_error_i32_complex_to_device_with_scratch, maximum_relative_error_i32_complex_to_device),
        i64 => (maximum_relative_error_i64_buffer_size, maximum_relative_error_i64_to_device_with_scratch, maximum_relative_error_i64_to_device),
        ComplexI64 => (maximum_relative_error_i64_complex_buffer_size, maximum_relative_error_i64_complex_to_device_with_scratch, maximum_relative_error_i64_complex_to_device),
        f32 => (maximum_relative_error_f32_buffer_size, maximum_relative_error_f32_to_device_with_scratch, maximum_relative_error_f32_to_device),
        Complex32 => (maximum_relative_error_f32_complex_buffer_size, maximum_relative_error_f32_complex_to_device_with_scratch, maximum_relative_error_f32_complex_to_device),
        f64 => (maximum_relative_error_f64_buffer_size, maximum_relative_error_f64_to_device_with_scratch, maximum_relative_error_f64_to_device),
        Complex64 => (maximum_relative_error_f64_complex_buffer_size, maximum_relative_error_f64_complex_to_device_with_scratch, maximum_relative_error_f64_complex_to_device)
    ]
);

impl_signal_error_metric_dispatch!("Computes the average relative error between the two source signals.",

    AverageRelativeError,
    average_relative_error_buffer_size,
    average_relative_error_to_device_with_scratch,
    average_relative_error_to_device,
    average_relative_error_buffer_size,
    average_relative_error_to_device_with_scratch,
    average_relative_error_to_device,
    [
        u8 => (average_relative_error_u8_buffer_size, average_relative_error_u8_to_device_with_scratch, average_relative_error_u8_to_device),
        i8 => (average_relative_error_i8_buffer_size, average_relative_error_i8_to_device_with_scratch, average_relative_error_i8_to_device),
        u16 => (average_relative_error_u16_buffer_size, average_relative_error_u16_to_device_with_scratch, average_relative_error_u16_to_device),
        i16 => (average_relative_error_i16_buffer_size, average_relative_error_i16_to_device_with_scratch, average_relative_error_i16_to_device),
        ComplexI16 => (average_relative_error_i16_complex_buffer_size, average_relative_error_i16_complex_to_device_with_scratch, average_relative_error_i16_complex_to_device),
        u32 => (average_relative_error_u32_buffer_size, average_relative_error_u32_to_device_with_scratch, average_relative_error_u32_to_device),
        i32 => (average_relative_error_i32_buffer_size, average_relative_error_i32_to_device_with_scratch, average_relative_error_i32_to_device),
        ComplexI32 => (average_relative_error_i32_complex_buffer_size, average_relative_error_i32_complex_to_device_with_scratch, average_relative_error_i32_complex_to_device),
        i64 => (average_relative_error_i64_buffer_size, average_relative_error_i64_to_device_with_scratch, average_relative_error_i64_to_device),
        ComplexI64 => (average_relative_error_i64_complex_buffer_size, average_relative_error_i64_complex_to_device_with_scratch, average_relative_error_i64_complex_to_device),
        f32 => (average_relative_error_f32_buffer_size, average_relative_error_f32_to_device_with_scratch, average_relative_error_f32_to_device),
        Complex32 => (average_relative_error_f32_complex_buffer_size, average_relative_error_f32_complex_to_device_with_scratch, average_relative_error_f32_complex_to_device),
        f64 => (average_relative_error_f64_buffer_size, average_relative_error_f64_to_device_with_scratch, average_relative_error_f64_to_device),
        Complex64 => (average_relative_error_f64_complex_buffer_size, average_relative_error_f64_complex_to_device_with_scratch, average_relative_error_f64_complex_to_device)
    ]
);
