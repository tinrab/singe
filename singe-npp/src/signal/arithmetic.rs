use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
    types::{ComplexI16, ComplexI32, DataTypeLike},
};

#[macro_use]
mod arithmetic_macros;

mod constant_ops;

pub use constant_ops::*;

mod binary_ops;

pub use binary_ops::*;

#[macro_use]
mod unary_ops;

pub use unary_ops::*;

macro_rules! impl_signal_constant_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalOutputView<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalOutputView<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, value, destination)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: T,
            destination: &mut SignalOutputView<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, value, destination)

        }
    }
    };
}

macro_rules! impl_signal_constant_in_place_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep the signal or destination exclusively writable through completion. Any separate
            /// input signal must remain read-only and disjoint from the writable destination.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                value: Self,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    value: Self,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal, value)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep the signal or destination exclusively writable through completion. Any separate
        /// input signal must remain read-only and disjoint from the writable destination.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            value: T,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, signal, value)

        }
    }
    };
}

macro_rules! impl_signal_scaled_constant_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalOutputView<'_, Self>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalOutputView<'_, Self>,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, value, destination, scale_factor)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: T,
            destination: &mut SignalOutputView<'_, T>,
            scale_factor: i32,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, value, destination, scale_factor)

        }
    }
    };
}

macro_rules! impl_signal_scaled_constant_in_place_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep the signal or destination exclusively writable through completion. Any separate
            /// input signal must remain read-only and disjoint from the writable destination.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                value: Self,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    value: Self,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal, value, scale_factor)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep the signal or destination exclusively writable through completion. Any separate
        /// input signal must remain read-only and disjoint from the writable destination.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            value: T,
            scale_factor: i32,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, signal, value, scale_factor)

        }
    }
    };
}

impl_signal_constant_dispatch!("Adds `value` to each source element. Writes the result to `destination`.",
AddConstant, add_constant, add_constant, [
    f32 => add_constant_f32,
    f64 => add_constant_f64,
    Complex32 => add_constant_f32_complex,
    Complex64 => add_constant_f64_complex
]);

impl_signal_constant_dispatch!("Subtracts `value` from each source element. Writes the result to `destination`.",
SubtractConstant, subtract_constant, subtract_constant, [
    f32 => subtract_constant_f32,
    f64 => subtract_constant_f64,
    Complex32 => subtract_constant_f32_complex,
    Complex64 => subtract_constant_f64_complex
]);

impl_signal_constant_dispatch!("Subtracts each source element from `value`. Writes the result to `destination`.",
SubtractFromConstant, subtract_from_constant, subtract_from_constant, [
    f32 => subtract_from_constant_f32,
    f64 => subtract_from_constant_f64,
    Complex32 => subtract_from_constant_f32_complex,
    Complex64 => subtract_from_constant_f64_complex
]);

impl_signal_constant_dispatch!("Multiplies each source element by `value`. Writes the result to `destination`.",
MultiplyConstant, multiply_constant, multiply_constant, [
    f32 => multiply_constant_f32,
    f64 => multiply_constant_f64,
    Complex32 => multiply_constant_f32_complex,
    Complex64 => multiply_constant_f64_complex
]);

impl_signal_constant_dispatch!("Divides each source element by `value`. Writes the result to `destination`.",
DivideConstant, divide_constant, divide_constant, [
    f32 => divide_constant_f32,
    f64 => divide_constant_f64,
    Complex32 => divide_constant_f32_complex,
    Complex64 => divide_constant_f64_complex
]);

impl_signal_constant_dispatch!("Divides `value` by each source element. Writes the result to `destination`.",
DivideIntoConstant, divide_into_constant, divide_into_constant, [
    u16 => divide_into_constant_u16,
    f32 => divide_into_constant_f32
]);

impl_signal_constant_dispatch!("Computes the bitwise AND of each source element and `value`. Writes the result to `destination`.",
AndConstant, and_constant, and_constant, [
    u8 => and_constant_u8,
    u16 => and_constant_u16,
    u32 => and_constant_u32
]);

impl_signal_constant_dispatch!("Computes the bitwise OR of each source element and `value`. Writes the result to `destination`.",
OrConstant, or_constant, or_constant, [
    u8 => or_constant_u8,
    u16 => or_constant_u16,
    u32 => or_constant_u32
]);

impl_signal_constant_dispatch!("Computes the bitwise XOR of each source element and `value`. Writes the result to `destination`.",
XorConstant, xor_constant, xor_constant, [
    u8 => xor_constant_u8,
    u16 => xor_constant_u16,
    u32 => xor_constant_u32
]);

impl_signal_constant_in_place_dispatch!("Adds `value` to each element of `signal`. Updates `signal` in place.",
AddConstantInPlace, add_constant_in_place, add_constant_in_place, [
    f32 => add_constant_f32_in_place,
    f64 => add_constant_f64_in_place,
    Complex32 => add_constant_f32_complex_in_place,
    Complex64 => add_constant_f64_complex_in_place
]);

impl_signal_constant_in_place_dispatch!("Subtracts `value` from each element of `signal`. Updates `signal` in place.",
SubtractConstantInPlace, subtract_constant_in_place, subtract_constant_in_place, [
    f32 => subtract_constant_f32_in_place,
    f64 => subtract_constant_f64_in_place,
    Complex32 => subtract_constant_f32_complex_in_place,
    Complex64 => subtract_constant_f64_complex_in_place
]);

impl_signal_constant_in_place_dispatch!("Subtracts each element of `signal` from `value`. Updates `signal` in place.",
SubtractFromConstantInPlace, subtract_from_constant_in_place, subtract_from_constant_in_place, [
    f32 => subtract_from_constant_f32_in_place,
    f64 => subtract_from_constant_f64_in_place,
    Complex32 => subtract_from_constant_f32_complex_in_place,
    Complex64 => subtract_from_constant_f64_complex_in_place
]);

impl_signal_constant_in_place_dispatch!("Multiplies each element of `signal` by `value`. Updates `signal` in place.",
MultiplyConstantInPlace, multiply_constant_in_place, multiply_constant_in_place, [
    f32 => multiply_constant_f32_in_place,
    f64 => multiply_constant_f64_in_place,
    Complex32 => multiply_constant_f32_complex_in_place,
    Complex64 => multiply_constant_f64_complex_in_place
]);

impl_signal_constant_in_place_dispatch!("Divides each element of `signal` by `value`. Updates `signal` in place.",
DivideConstantInPlace, divide_constant_in_place, divide_constant_in_place, [
    f32 => divide_constant_f32_in_place,
    f64 => divide_constant_f64_in_place,
    Complex32 => divide_constant_f32_complex_in_place,
    Complex64 => divide_constant_f64_complex_in_place
]);

impl_signal_constant_in_place_dispatch!("Divides `value` by each element of `signal`. Updates `signal` in place.",
DivideIntoConstantInPlace, divide_into_constant_in_place, divide_into_constant_in_place, [
    u16 => divide_into_constant_u16_in_place,
    f32 => divide_into_constant_f32_in_place
]);

impl_signal_constant_in_place_dispatch!("Computes the bitwise AND of each element of `signal` and `value`. Updates `signal` in place.",
AndConstantInPlace, and_constant_in_place, and_constant_in_place, [
    u8 => and_constant_u8_in_place,
    u16 => and_constant_u16_in_place,
    u32 => and_constant_u32_in_place
]);

impl_signal_constant_in_place_dispatch!("Computes the bitwise OR of each element of `signal` and `value`. Updates `signal` in place.",
OrConstantInPlace, or_constant_in_place, or_constant_in_place, [
    u8 => or_constant_u8_in_place,
    u16 => or_constant_u16_in_place,
    u32 => or_constant_u32_in_place
]);

impl_signal_constant_in_place_dispatch!("Computes the bitwise XOR of each element of `signal` and `value`. Updates `signal` in place.",
XorConstantInPlace, xor_constant_in_place, xor_constant_in_place, [
    u8 => xor_constant_u8_in_place,
    u16 => xor_constant_u16_in_place,
    u32 => xor_constant_u32_in_place
]);

impl_signal_scaled_constant_dispatch!("Adds `value` to each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
AddConstantScaled, add_constant_scaled, add_constant_scaled, [
    u8 => add_constant_u8_scaled,
    u16 => add_constant_u16_scaled,
    i16 => add_constant_i16_scaled,
    i32 => add_constant_i32_scaled,
    ComplexI16 => add_constant_i16_complex_scaled,
    ComplexI32 => add_constant_i32_complex_scaled
]);

impl_signal_scaled_constant_dispatch!("Subtracts `value` from each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SubtractConstantScaled, subtract_constant_scaled, subtract_constant_scaled, [
    u8 => subtract_constant_u8_scaled,
    u16 => subtract_constant_u16_scaled,
    i16 => subtract_constant_i16_scaled,
    i32 => subtract_constant_i32_scaled,
    ComplexI16 => subtract_constant_i16_complex_scaled,
    ComplexI32 => subtract_constant_i32_complex_scaled
]);

impl_signal_scaled_constant_dispatch!("Subtracts each source element from `value`. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SubtractFromConstantScaled, subtract_from_constant_scaled, subtract_from_constant_scaled, [
    u8 => subtract_from_constant_u8_scaled,
    u16 => subtract_from_constant_u16_scaled,
    i16 => subtract_from_constant_i16_scaled,
    i32 => subtract_from_constant_i32_scaled,
    ComplexI16 => subtract_from_constant_i16_complex_scaled,
    ComplexI32 => subtract_from_constant_i32_complex_scaled
]);

impl_signal_scaled_constant_dispatch!("Multiplies each source element by `value`. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
MultiplyConstantScaled, multiply_constant_scaled, multiply_constant_scaled, [
    u8 => multiply_constant_u8_scaled,
    u16 => multiply_constant_u16_scaled,
    i16 => multiply_constant_i16_scaled,
    i32 => multiply_constant_i32_scaled,
    ComplexI16 => multiply_constant_i16_complex_scaled,
    ComplexI32 => multiply_constant_i32_complex_scaled
]);

impl_signal_scaled_constant_dispatch!("Divides each source element by `value`. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
DivideConstantScaled, divide_constant_scaled, divide_constant_scaled, [
    u8 => divide_constant_u8_scaled,
    u16 => divide_constant_u16_scaled,
    i16 => divide_constant_i16_scaled,
    ComplexI16 => divide_constant_i16_complex_scaled
]);

impl_signal_scaled_constant_in_place_dispatch!("Adds `value` to each element of `signal`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
AddConstantScaledInPlace, add_constant_scaled_in_place, add_constant_scaled_in_place, [
    u8 => add_constant_u8_scaled_in_place,
    u16 => add_constant_u16_scaled_in_place,
    i16 => add_constant_i16_scaled_in_place,
    i32 => add_constant_i32_scaled_in_place,
    ComplexI16 => add_constant_i16_complex_scaled_in_place,
    ComplexI32 => add_constant_i32_complex_scaled_in_place
]);

impl_signal_scaled_constant_in_place_dispatch!("Subtracts `value` from each element of `signal`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SubtractConstantScaledInPlace, subtract_constant_scaled_in_place, subtract_constant_scaled_in_place, [
    u8 => subtract_constant_u8_scaled_in_place,
    u16 => subtract_constant_u16_scaled_in_place,
    i16 => subtract_constant_i16_scaled_in_place,
    i32 => subtract_constant_i32_scaled_in_place,
    ComplexI16 => subtract_constant_i16_complex_scaled_in_place,
    ComplexI32 => subtract_constant_i32_complex_scaled_in_place
]);

impl_signal_scaled_constant_in_place_dispatch!("Subtracts each element of `signal` from `value`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SubtractFromConstantScaledInPlace, subtract_from_constant_scaled_in_place, subtract_from_constant_scaled_in_place, [
    u8 => subtract_from_constant_u8_scaled_in_place,
    u16 => subtract_from_constant_u16_scaled_in_place,
    i16 => subtract_from_constant_i16_scaled_in_place,
    i32 => subtract_from_constant_i32_scaled_in_place,
    ComplexI16 => subtract_from_constant_i16_complex_scaled_in_place,
    ComplexI32 => subtract_from_constant_i32_complex_scaled_in_place
]);

impl_signal_scaled_constant_in_place_dispatch!("Multiplies each element of `signal` by `value`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
MultiplyConstantScaledInPlace, multiply_constant_scaled_in_place, multiply_constant_scaled_in_place, [
    u8 => multiply_constant_u8_scaled_in_place,
    u16 => multiply_constant_u16_scaled_in_place,
    i16 => multiply_constant_i16_scaled_in_place,
    i32 => multiply_constant_i32_scaled_in_place,
    ComplexI16 => multiply_constant_i16_complex_scaled_in_place,
    ComplexI32 => multiply_constant_i32_complex_scaled_in_place
]);

impl_signal_scaled_constant_in_place_dispatch!("Divides each element of `signal` by `value`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
DivideConstantScaledInPlace, divide_constant_scaled_in_place, divide_constant_scaled_in_place, [
    u8 => divide_constant_u8_scaled_in_place,
    u16 => divide_constant_u16_scaled_in_place,
    i16 => divide_constant_i16_scaled_in_place,
    ComplexI16 => divide_constant_i16_complex_scaled_in_place
]);

macro_rules! impl_signal_destination_update_constant_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// The accumulating destination must also be initialized and ready before the operation.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, value, destination)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// The accumulating destination must also be initialized and ready before the operation.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: T,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, value, destination)

        }
    }
    };
}

macro_rules! impl_signal_mixed_constant_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalOutputView<'_, Destination>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, value, destination)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            value: Source,
            destination: &mut SignalOutputView<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, value, destination)

        }
    }
    };
}

macro_rules! impl_signal_scaled_mixed_constant_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalOutputView<'_, Destination>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, value, destination, scale_factor)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            value: Source,
            destination: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, value, destination, scale_factor)

        }
    }
    };
}

macro_rules! impl_signal_scaled_mixed_constant_in_place_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($signal_ty:ty => ($value_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait<Value>: DataTypeLike
        where
            Value: DataTypeLike,
        {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep the signal or destination exclusively writable through completion. Any separate
            /// input signal must remain read-only and disjoint from the writable destination.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                value: Value,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$value_ty> for $signal_ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    value: $value_ty,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal, value, scale_factor)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep the signal or destination exclusively writable through completion. Any separate
        /// input signal must remain read-only and disjoint from the writable destination.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<Signal, Value>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, Signal>,
            value: Value,
            scale_factor: i32,
        ) -> Result<()>
        where
            Signal: $trait<Value>,
            Value: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Signal::$method(stream_context, signal, value, scale_factor)

        }
    }
    };
}

impl_signal_destination_update_constant_dispatch!("Adds each source element multiplied by `value` to the corresponding initialized destination element.",
AddProductConstant, add_product_constant, add_product_constant, [
    f32 => add_product_constant_f32
]);

impl_signal_mixed_constant_dispatch!("Multiplies each source element by `value`. Writes the result to `destination`. Uses the source and destination representations selected by this entry point.",
MultiplyConstantLowTo, multiply_constant_low_to, multiply_constant_low_to, [
    f32 => (i16, multiply_constant_f32_to_i16_low)
]);

impl_signal_scaled_mixed_constant_dispatch!("Multiplies each source element by `value`. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type. Uses the source and destination representations selected by this entry point.",
MultiplyConstantScaledTo, multiply_constant_scaled_to, multiply_constant_scaled_to, [
    f32 => (i16, multiply_constant_f32_to_i16_scaled)
]);

impl_signal_scaled_mixed_constant_in_place_dispatch!("Multiplies each element of `signal` by `value`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type. Uses the source and destination representations selected by this entry point.",
MultiplyConstantScaledToInPlace, multiply_constant_scaled_to_in_place, multiply_constant_scaled_to_in_place, [
    i64 => (f64, multiply_constant_f64_to_i64_scaled_in_place)
]);

macro_rules! impl_signal_shift_constant_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                shift: i32,
                destination: &mut SignalOutputView<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    shift: i32,
                    destination: &mut SignalOutputView<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, shift, destination)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            shift: i32,
            destination: &mut SignalOutputView<'_, T>,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, shift, destination)

        }
    }
    };
}

macro_rules! impl_signal_shift_constant_in_place_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep the signal or destination exclusively writable through completion. Any separate
            /// input signal must remain read-only and disjoint from the writable destination.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                shift: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    shift: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal, shift)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep the signal or destination exclusively writable through completion. Any separate
        /// input signal must remain read-only and disjoint from the writable destination.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            shift: i32,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, signal, shift)

        }
    }
    };
}

impl_signal_shift_constant_dispatch!("Shifts each source element left by `shift` bits. Writes the result to `destination`.",
LeftShiftConstant, left_shift_constant, left_shift_constant, [
    u8 => left_shift_constant_u8,
    u16 => left_shift_constant_u16,
    i16 => left_shift_constant_i16,
    u32 => left_shift_constant_u32,
    i32 => left_shift_constant_i32
]);

impl_signal_shift_constant_dispatch!("Shifts each source element right by `shift` bits, using native signed or unsigned shift semantics. Writes the result to `destination`.",
RightShiftConstant, right_shift_constant, right_shift_constant, [
    u8 => right_shift_constant_u8,
    u16 => right_shift_constant_u16,
    i16 => right_shift_constant_i16,
    u32 => right_shift_constant_u32,
    i32 => right_shift_constant_i32
]);

impl_signal_shift_constant_in_place_dispatch!("Shifts each element of `signal` left by `shift` bits. Updates `signal` in place.",
LeftShiftConstantInPlace, left_shift_constant_in_place, left_shift_constant_in_place, [
    u8 => left_shift_constant_u8_in_place,
    u16 => left_shift_constant_u16_in_place,
    i16 => left_shift_constant_i16_in_place,
    u32 => left_shift_constant_u32_in_place,
    i32 => left_shift_constant_i32_in_place
]);

impl_signal_shift_constant_in_place_dispatch!("Shifts each element of `signal` right by `shift` bits, using native signed or unsigned shift semantics. Updates `signal` in place.",
RightShiftConstantInPlace, right_shift_constant_in_place, right_shift_constant_in_place, [
    u8 => right_shift_constant_u8_in_place,
    u16 => right_shift_constant_u16_in_place,
    i16 => right_shift_constant_i16_in_place,
    u32 => right_shift_constant_u32_in_place,
    i32 => right_shift_constant_i32_in_place
]);

macro_rules! impl_signal_to_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Destination>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
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
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, destination)

        }
    }
    };
}

macro_rules! impl_signal_scaled_to_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Destination>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$destination_ty> for $source_ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, scale_factor)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalOutputView<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Source::$method(stream_context, source, destination, scale_factor)

        }
    }
    };
}

impl_signal_to_dispatch!("Computes the exponential of each source element. Writes the result to `destination`.",
ExponentTo, exponent_to, exponent_to, [
    f32 => (f64, exponent_f32_to_f64)
]);

impl_signal_to_dispatch!("Computes the natural logarithm of each source element. Writes the result to `destination`.",
NaturalLogarithmTo, natural_logarithm_to, natural_logarithm_to, [
    f64 => (f32, natural_logarithm_f64_to_f32)
]);

impl_signal_scaled_to_dispatch!("Computes the natural logarithm of each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
NaturalLogarithmScaledTo, natural_logarithm_scaled_to, natural_logarithm_scaled_to, [
    i32 => (i16, natural_logarithm_i32_to_i16_scaled)
]);

impl_signal_scaled_to_dispatch!("Computes the cube root of each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
CubeRootScaledTo, cube_root_scaled_to, cube_root_scaled_to, [
    i32 => (i16, cube_root_i32_to_i16_scaled)
]);

impl_signal_scaled_to_dispatch!("Computes the square root of each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SquareRootScaledTo, square_root_scaled_to, square_root_scaled_to, [
    i32 => (i16, square_root_i32_to_i16_scaled),
    i64 => (i16, square_root_i64_to_i16_scaled)
]);

impl_signal_unary_dispatch!("Computes the cube root of each source element. Writes the result to `destination`.",
CubeRoot, cube_root, cube_root, [
    f32 => cube_root_f32
]);

impl_signal_scaled_unary_dispatch!("Computes ten times the base-ten logarithm of each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
TenTimesLog10Scaled, ten_times_log10_scaled, ten_times_log10_scaled, [
    i32 => ten_times_log10_i32_scaled
]);

impl_signal_scaled_unary_in_place_dispatch!("Computes ten times the base-ten logarithm of each element of `signal`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
TenTimesLog10ScaledInPlace, ten_times_log10_scaled_in_place, ten_times_log10_scaled_in_place, [
    i32 => ten_times_log10_i32_scaled_in_place
]);

macro_rules! impl_signal_normalize_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => ($scale_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            /// Real divisor representation for normalization of this element type.
            type Scale: DataTypeLike;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Self>,
                subtract: Self,
                divide: Self::Scale,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                type Scale = $scale_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
                    subtract: Self,
                    divide: Self::Scale,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, subtract, divide)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
            subtract: T,
            divide: T::Scale,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination, subtract, divide)

        }
    }
    };
}

macro_rules! impl_signal_scaled_normalize_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => ($divisor_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            /// Integer divisor representation for scaled normalization.
            type Divisor: DataTypeLike;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            /// Participating signal lengths must match; differing lengths return an error.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep input signals read-only and destination elements exclusively writable and disjoint
            /// from every input through completion.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalOutputView<'_, Self>,
                subtract: Self,
                divide: Self::Divisor,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                type Divisor = $divisor_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
                    subtract: Self,
                    divide: Self::Divisor,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, source, destination, subtract, divide, scale_factor)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        /// Participating signal lengths must match; differing lengths return an error.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep input signals read-only and destination elements exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
            subtract: T,
            divide: T::Divisor,
            scale_factor: i32,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination, subtract, divide, scale_factor)

        }
    }
    };
}

impl_signal_normalize_dispatch!("Computes `(source - subtract) / divide` for each element. Writes the result to `destination`.",
Normalize, normalize, normalize, [
    f32 => (f32, normalize_f32),
    f64 => (f64, normalize_f64),
    Complex32 => (f32, normalize_f32_complex),
    Complex64 => (f64, normalize_f64_complex)
]);

impl_signal_scaled_normalize_dispatch!("Computes `(source - subtract) / divide` for each element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
NormalizeScaled, normalize_scaled, normalize_scaled, [
    i16 => (i32, normalize_i16_scaled),
    ComplexI16 => (i32, normalize_i16_complex_scaled)
]);

macro_rules! impl_signal_parameterized_unary_in_place_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => ($parameter_ty:ty, $direct:ident)),* $(,)?]) => {
        #[doc = concat!("Element-type dispatch for [`", stringify!($function), "`].")]
        ///
        /// Implementations select a native NPP routine and its scalar or output representation.
        ///
        /// # Safety
        ///
        /// Implementations must match the native element ABI and operand order, and uphold
        /// the method's access, initialization, capacity, and queued-lifetime requirements.
        pub unsafe trait $trait: DataTypeLike {
            /// Scalar parameter representation required by this element type's native operation.
            type Parameter: DataTypeLike;

            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// Provided implementations return errors if CUDA context binding fails, a signal belongs
            /// to another CUDA context, a length cannot be represented by the native interface,
            /// or NPP reports a failure. Success reports native submission.
            ///
            /// # Safety
            ///
            /// Input elements must be initialized and ready when the stream reaches this operation.
            /// Keep the signal or destination exclusively writable through completion. Any separate
            /// input signal must remain read-only and disjoint from the writable destination.
            /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
            /// alignment, representation, and operand requirements. Retain signal allocations and
            /// the stream context through queued completion, including submission errors and every
            /// captured graph replay.
            unsafe fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                parameter: Self::Parameter,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                type Parameter = $parameter_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    parameter: Self::Parameter,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal, parameter)

        }
    }
            }
        )*

        #[doc = $doc]
        #[doc = concat!("Dispatches through [`", stringify!($trait), "`] for the signal element type.")]
        ///
        /// # Errors
        ///
        /// Provided implementations return errors if CUDA context binding fails, a signal belongs
        /// to another CUDA context, a length cannot be represented by the native interface,
        /// or NPP reports a failure. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input elements must be initialized and ready when the stream reaches this operation.
        /// Keep the signal or destination exclusively writable through completion. Any separate
        /// input signal must remain read-only and disjoint from the writable destination.
        /// Addresses, capacities, and scalar operands must satisfy the selected native routine's
        /// alignment, representation, and operand requirements. Retain signal allocations and
        /// the stream context through queued completion, including submission errors and every
        /// captured graph replay.
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            parameter: T::Parameter,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, signal, parameter)

        }
    }
    };
}

impl_signal_parameterized_unary_in_place_dispatch!("Evaluates the native Cauchy robust error function for each element of `signal` using `parameter`. Updates `signal` in place.",
CauchyInPlace, cauchy_in_place, cauchy_in_place, [
    f32 => (f32, cauchy_f32_in_place)
]);

impl_signal_parameterized_unary_in_place_dispatch!("Evaluates the first derivative of the native Cauchy robust error function for each element of `signal` using `parameter`. Updates `signal` in place.",
CauchyDerivativeInPlace, cauchy_derivative_in_place, cauchy_derivative_in_place, [
    f32 => (f32, cauchy_derivative_f32_in_place)
]);
