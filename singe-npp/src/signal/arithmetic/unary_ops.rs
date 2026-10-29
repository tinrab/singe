use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        arithmetic::binary_ops::{
            absolute_f32, absolute_f32_in_place, absolute_f64, absolute_f64_in_place, absolute_i16,
            absolute_i16_in_place, absolute_i32, absolute_i32_in_place, arctangent_f32,
            arctangent_f32_in_place, arctangent_f64, arctangent_f64_in_place, exponent_f32,
            exponent_f32_in_place, exponent_f64, exponent_f64_in_place, exponent_i16_scaled,
            exponent_i16_scaled_in_place, exponent_i32_scaled, exponent_i32_scaled_in_place,
            exponent_i64_scaled, exponent_i64_scaled_in_place, natural_logarithm_f32,
            natural_logarithm_f32_in_place, natural_logarithm_f64, natural_logarithm_f64_in_place,
            natural_logarithm_i16_scaled, natural_logarithm_i16_scaled_in_place,
            natural_logarithm_i32_scaled, natural_logarithm_i32_scaled_in_place, not_u8,
            not_u8_in_place, not_u16, not_u16_in_place, not_u32, not_u32_in_place, square_f32,
            square_f32_complex, square_f32_complex_in_place, square_f32_in_place, square_f64,
            square_f64_complex, square_f64_complex_in_place, square_f64_in_place,
            square_i16_complex_scaled, square_i16_complex_scaled_in_place, square_i16_scaled,
            square_i16_scaled_in_place, square_root_f32, square_root_f32_complex,
            square_root_f32_complex_in_place, square_root_f32_in_place, square_root_f64,
            square_root_f64_complex, square_root_f64_complex_in_place, square_root_f64_in_place,
            square_root_i16_complex_scaled, square_root_i16_complex_scaled_in_place,
            square_root_i16_scaled, square_root_i16_scaled_in_place, square_root_i64_scaled,
            square_root_i64_scaled_in_place, square_root_u8_scaled, square_root_u8_scaled_in_place,
            square_root_u16_scaled, square_root_u16_scaled_in_place, square_u8_scaled,
            square_u8_scaled_in_place, square_u16_scaled, square_u16_scaled_in_place,
        },
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::{ComplexI16, DataTypeLike},
};

macro_rules! impl_signal_unary_dispatch {
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
                destination: &mut SignalOutputView<'_, Self>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
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
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
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

macro_rules! impl_signal_unary_in_place_dispatch {
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
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal)

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
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, signal)

        }
    }
    };
}

macro_rules! impl_signal_scaled_unary_dispatch {
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
                destination: &mut SignalOutputView<'_, Self>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalOutputView<'_, Self>,
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
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalOutputView<'_, T>,
            scale_factor: i32,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(source)?;
        stream_context.validate_region(destination)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, source, destination, scale_factor)

        }
    }
    };
}

macro_rules! impl_signal_scaled_unary_in_place_dispatch {
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
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
                    $direct(stream_context, signal, scale_factor)

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
            scale_factor: i32,
        ) -> Result<()> {
        stream_context.cuda_context().bind()?;
        stream_context.validate_region(signal)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            T::$method(stream_context, signal, scale_factor)

        }
    }
    };
}

impl_signal_unary_dispatch!("Computes the absolute value of each source element. Writes the result to `destination`.",
Absolute, absolute, absolute, [
    i16 => absolute_i16,
    i32 => absolute_i32,
    f32 => absolute_f32,
    f64 => absolute_f64
]);

impl_signal_unary_dispatch!("Computes the exponential of each source element. Writes the result to `destination`.",
Exponent, exponent, exponent, [
    f32 => exponent_f32,
    f64 => exponent_f64
]);

impl_signal_unary_dispatch!("Computes the natural logarithm of each source element. Writes the result to `destination`.",
NaturalLogarithm, natural_logarithm, natural_logarithm, [
    f32 => natural_logarithm_f32,
    f64 => natural_logarithm_f64
]);

impl_signal_unary_dispatch!("Computes the inverse tangent of each source element. Writes the result to `destination`.",
Arctangent, arctangent, arctangent, [
    f32 => arctangent_f32,
    f64 => arctangent_f64
]);

impl_signal_unary_dispatch!("Computes the bitwise complement of each source element. Writes the result to `destination`.",
Not, not, not, [
    u8 => not_u8,
    u16 => not_u16,
    u32 => not_u32
]);

impl_signal_unary_dispatch!("Squares each source element. Writes the result to `destination`.",
Square, square, square, [
    f32 => square_f32,
    f64 => square_f64,
    Complex32 => square_f32_complex,
    Complex64 => square_f64_complex
]);

impl_signal_unary_dispatch!("Computes the square root of each source element. Writes the result to `destination`.",
SquareRoot, square_root, square_root, [
    f32 => square_root_f32,
    f64 => square_root_f64,
    Complex32 => square_root_f32_complex,
    Complex64 => square_root_f64_complex
]);

impl_signal_unary_in_place_dispatch!("Computes the absolute value of each element of `signal`. Updates `signal` in place.",
AbsoluteInPlace, absolute_in_place, absolute_in_place, [
    i16 => absolute_i16_in_place,
    i32 => absolute_i32_in_place,
    f32 => absolute_f32_in_place,
    f64 => absolute_f64_in_place
]);

impl_signal_unary_in_place_dispatch!("Computes the exponential of each element of `signal`. Updates `signal` in place.",
ExponentInPlace, exponent_in_place, exponent_in_place, [
    f32 => exponent_f32_in_place,
    f64 => exponent_f64_in_place
]);

impl_signal_unary_in_place_dispatch!("Computes the natural logarithm of each element of `signal`. Updates `signal` in place.",
NaturalLogarithmInPlace, natural_logarithm_in_place, natural_logarithm_in_place, [
    f32 => natural_logarithm_f32_in_place,
    f64 => natural_logarithm_f64_in_place
]);

impl_signal_unary_in_place_dispatch!("Computes the inverse tangent of each element of `signal`. Updates `signal` in place.",
ArctangentInPlace, arctangent_in_place, arctangent_in_place, [
    f32 => arctangent_f32_in_place,
    f64 => arctangent_f64_in_place
]);

impl_signal_unary_in_place_dispatch!("Computes the bitwise complement of each element of `signal`. Updates `signal` in place.",
NotInPlace, not_in_place, not_in_place, [
    u8 => not_u8_in_place,
    u16 => not_u16_in_place,
    u32 => not_u32_in_place
]);

impl_signal_unary_in_place_dispatch!("Squares each element of `signal`. Updates `signal` in place.",
SquareInPlace, square_in_place, square_in_place, [
    f32 => square_f32_in_place,
    f64 => square_f64_in_place,
    Complex32 => square_f32_complex_in_place,
    Complex64 => square_f64_complex_in_place
]);

impl_signal_unary_in_place_dispatch!("Computes the square root of each element of `signal`. Updates `signal` in place.",
SquareRootInPlace, square_root_in_place, square_root_in_place, [
    f32 => square_root_f32_in_place,
    f64 => square_root_f64_in_place,
    Complex32 => square_root_f32_complex_in_place,
    Complex64 => square_root_f64_complex_in_place
]);

impl_signal_scaled_unary_dispatch!("Computes the exponential of each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
ExponentScaled, exponent_scaled, exponent_scaled, [
    i16 => exponent_i16_scaled,
    i32 => exponent_i32_scaled,
    i64 => exponent_i64_scaled
]);

impl_signal_scaled_unary_dispatch!("Computes the natural logarithm of each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
NaturalLogarithmScaled, natural_logarithm_scaled, natural_logarithm_scaled, [
    i16 => natural_logarithm_i16_scaled,
    i32 => natural_logarithm_i32_scaled
]);

impl_signal_scaled_unary_dispatch!("Squares each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SquareScaled, square_scaled, square_scaled, [
    u8 => square_u8_scaled,
    u16 => square_u16_scaled,
    i16 => square_i16_scaled,
    ComplexI16 => square_i16_complex_scaled
]);

impl_signal_scaled_unary_dispatch!("Computes the square root of each source element. Writes the result to `destination`. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SquareRootScaled, square_root_scaled, square_root_scaled, [
    u8 => square_root_u8_scaled,
    u16 => square_root_u16_scaled,
    i16 => square_root_i16_scaled,
    i64 => square_root_i64_scaled,
    ComplexI16 => square_root_i16_complex_scaled
]);

impl_signal_scaled_unary_in_place_dispatch!("Computes the exponential of each element of `signal`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
ExponentScaledInPlace, exponent_scaled_in_place, exponent_scaled_in_place, [
    i16 => exponent_i16_scaled_in_place,
    i32 => exponent_i32_scaled_in_place,
    i64 => exponent_i64_scaled_in_place
]);

impl_signal_scaled_unary_in_place_dispatch!("Computes the natural logarithm of each element of `signal`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
NaturalLogarithmScaledInPlace, natural_logarithm_scaled_in_place, natural_logarithm_scaled_in_place, [
    i16 => natural_logarithm_i16_scaled_in_place,
    i32 => natural_logarithm_i32_scaled_in_place
]);

impl_signal_scaled_unary_in_place_dispatch!("Squares each element of `signal`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SquareScaledInPlace, square_scaled_in_place, square_scaled_in_place, [
    u8 => square_u8_scaled_in_place,
    u16 => square_u16_scaled_in_place,
    i16 => square_i16_scaled_in_place,
    ComplexI16 => square_i16_complex_scaled_in_place
]);

impl_signal_scaled_unary_in_place_dispatch!("Computes the square root of each element of `signal`. Updates `signal` in place. Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.",
SquareRootScaledInPlace, square_root_scaled_in_place, square_root_scaled_in_place, [
    u8 => square_root_u8_scaled_in_place,
    u16 => square_root_u16_scaled_in_place,
    i16 => square_root_i16_scaled_in_place,
    i64 => square_root_i64_scaled_in_place,
    ComplexI16 => square_root_i16_complex_scaled_in_place
]);
