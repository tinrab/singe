use singe_cuda::types::{Complex32, Complex64};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalView, SignalViewMut},
    try_ffi,
    types::{ComplexI16, ComplexI32, DataTypeLike, IntoNpp, RoundMode},
    utility::{to_u64, validate_same_len},
};

#[macro_use]
#[path = "arithmetic_macros.rs"]
mod arithmetic_macros;

#[path = "arithmetic_constant_ops.rs"]
mod constant_ops;
pub use constant_ops::*;

#[path = "arithmetic_binary_ops.rs"]
mod binary_ops;
pub use binary_ops::*;

#[path = "arithmetic_unary_ops.rs"]
#[macro_use]
mod unary_ops;
pub use unary_ops::*;

macro_rules! impl_signal_constant_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, source, value, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: T,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, source, value, destination)
        }
    };
}

macro_rules! impl_signal_constant_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                value: Self,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    value: Self,
                ) -> Result<()> {
                    $direct(stream_context, signal, value)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            value: T,
        ) -> Result<()> {
            T::$method(stream_context, signal, value)
        }
    };
}

macro_rules! impl_signal_scaled_constant_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalViewMut<'_, Self>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, value, destination, scale_factor)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: T,
            destination: &mut SignalViewMut<'_, T>,
            scale_factor: i32,
        ) -> Result<()> {
            T::$method(stream_context, source, value, destination, scale_factor)
        }
    };
}

macro_rules! impl_signal_scaled_constant_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                value: Self,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    value: Self,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, signal, value, scale_factor)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            value: T,
            scale_factor: i32,
        ) -> Result<()> {
            T::$method(stream_context, signal, value, scale_factor)
        }
    };
}

impl_signal_constant_dispatch!(AddConstant, add_constant, add_constant, [
    f32 => add_constant_f32,
    f64 => add_constant_f64,
    Complex32 => add_constant_f32_complex,
    Complex64 => add_constant_f64_complex
]);
impl_signal_constant_dispatch!(SubtractConstant, subtract_constant, subtract_constant, [
    f32 => subtract_constant_f32,
    f64 => subtract_constant_f64,
    Complex32 => subtract_constant_f32_complex,
    Complex64 => subtract_constant_f64_complex
]);
impl_signal_constant_dispatch!(SubtractFromConstant, subtract_from_constant, subtract_from_constant, [
    f32 => subtract_from_constant_f32,
    f64 => subtract_from_constant_f64,
    Complex32 => subtract_from_constant_f32_complex,
    Complex64 => subtract_from_constant_f64_complex
]);
impl_signal_constant_dispatch!(MultiplyConstant, multiply_constant, multiply_constant, [
    f32 => multiply_constant_f32,
    f64 => multiply_constant_f64,
    Complex32 => multiply_constant_f32_complex,
    Complex64 => multiply_constant_f64_complex
]);
impl_signal_constant_dispatch!(DivideConstant, divide_constant, divide_constant, [
    f32 => divide_constant_f32,
    f64 => divide_constant_f64,
    Complex32 => divide_constant_f32_complex,
    Complex64 => divide_constant_f64_complex
]);
impl_signal_constant_dispatch!(DivideIntoConstant, divide_into_constant, divide_into_constant, [
    u16 => divide_into_constant_u16,
    f32 => divide_into_constant_f32
]);
impl_signal_constant_dispatch!(AndConstant, and_constant, and_constant, [
    u8 => and_constant_u8,
    u16 => and_constant_u16,
    u32 => and_constant_u32
]);
impl_signal_constant_dispatch!(OrConstant, or_constant, or_constant, [
    u8 => or_constant_u8,
    u16 => or_constant_u16,
    u32 => or_constant_u32
]);
impl_signal_constant_dispatch!(XorConstant, xor_constant, xor_constant, [
    u8 => xor_constant_u8,
    u16 => xor_constant_u16,
    u32 => xor_constant_u32
]);

impl_signal_constant_in_place_dispatch!(AddConstantInPlace, add_constant_in_place, add_constant_in_place, [
    f32 => add_constant_f32_in_place,
    f64 => add_constant_f64_in_place,
    Complex32 => add_constant_f32_complex_in_place,
    Complex64 => add_constant_f64_complex_in_place
]);
impl_signal_constant_in_place_dispatch!(SubtractConstantInPlace, subtract_constant_in_place, subtract_constant_in_place, [
    f32 => subtract_constant_f32_in_place,
    f64 => subtract_constant_f64_in_place,
    Complex32 => subtract_constant_f32_complex_in_place,
    Complex64 => subtract_constant_f64_complex_in_place
]);
impl_signal_constant_in_place_dispatch!(SubtractFromConstantInPlace, subtract_from_constant_in_place, subtract_from_constant_in_place, [
    f32 => subtract_from_constant_f32_in_place,
    f64 => subtract_from_constant_f64_in_place,
    Complex32 => subtract_from_constant_f32_complex_in_place,
    Complex64 => subtract_from_constant_f64_complex_in_place
]);
impl_signal_constant_in_place_dispatch!(MultiplyConstantInPlace, multiply_constant_in_place, multiply_constant_in_place, [
    f32 => multiply_constant_f32_in_place,
    f64 => multiply_constant_f64_in_place,
    Complex32 => multiply_constant_f32_complex_in_place,
    Complex64 => multiply_constant_f64_complex_in_place
]);
impl_signal_constant_in_place_dispatch!(DivideConstantInPlace, divide_constant_in_place, divide_constant_in_place, [
    f32 => divide_constant_f32_in_place,
    f64 => divide_constant_f64_in_place,
    Complex32 => divide_constant_f32_complex_in_place,
    Complex64 => divide_constant_f64_complex_in_place
]);
impl_signal_constant_in_place_dispatch!(DivideIntoConstantInPlace, divide_into_constant_in_place, divide_into_constant_in_place, [
    u16 => divide_into_constant_u16_in_place,
    f32 => divide_into_constant_f32_in_place
]);
impl_signal_constant_in_place_dispatch!(AndConstantInPlace, and_constant_in_place, and_constant_in_place, [
    u8 => and_constant_u8_in_place,
    u16 => and_constant_u16_in_place,
    u32 => and_constant_u32_in_place
]);
impl_signal_constant_in_place_dispatch!(OrConstantInPlace, or_constant_in_place, or_constant_in_place, [
    u8 => or_constant_u8_in_place,
    u16 => or_constant_u16_in_place,
    u32 => or_constant_u32_in_place
]);
impl_signal_constant_in_place_dispatch!(XorConstantInPlace, xor_constant_in_place, xor_constant_in_place, [
    u8 => xor_constant_u8_in_place,
    u16 => xor_constant_u16_in_place,
    u32 => xor_constant_u32_in_place
]);

impl_signal_scaled_constant_dispatch!(AddConstantScaled, add_constant_scaled, add_constant_scaled, [
    u8 => add_constant_u8_scaled,
    u16 => add_constant_u16_scaled,
    i16 => add_constant_i16_scaled,
    i32 => add_constant_i32_scaled,
    ComplexI16 => add_constant_i16_complex_scaled,
    ComplexI32 => add_constant_i32_complex_scaled
]);
impl_signal_scaled_constant_dispatch!(SubtractConstantScaled, subtract_constant_scaled, subtract_constant_scaled, [
    u8 => subtract_constant_u8_scaled,
    u16 => subtract_constant_u16_scaled,
    i16 => subtract_constant_i16_scaled,
    i32 => subtract_constant_i32_scaled,
    ComplexI16 => subtract_constant_i16_complex_scaled,
    ComplexI32 => subtract_constant_i32_complex_scaled
]);
impl_signal_scaled_constant_dispatch!(SubtractFromConstantScaled, subtract_from_constant_scaled, subtract_from_constant_scaled, [
    u8 => subtract_from_constant_u8_scaled,
    u16 => subtract_from_constant_u16_scaled,
    i16 => subtract_from_constant_i16_scaled,
    i32 => subtract_from_constant_i32_scaled,
    ComplexI16 => subtract_from_constant_i16_complex_scaled,
    ComplexI32 => subtract_from_constant_i32_complex_scaled
]);
impl_signal_scaled_constant_dispatch!(MultiplyConstantScaled, multiply_constant_scaled, multiply_constant_scaled, [
    u8 => multiply_constant_u8_scaled,
    u16 => multiply_constant_u16_scaled,
    i16 => multiply_constant_i16_scaled,
    i32 => multiply_constant_i32_scaled,
    ComplexI16 => multiply_constant_i16_complex_scaled,
    ComplexI32 => multiply_constant_i32_complex_scaled
]);
impl_signal_scaled_constant_dispatch!(DivideConstantScaled, divide_constant_scaled, divide_constant_scaled, [
    u8 => divide_constant_u8_scaled,
    u16 => divide_constant_u16_scaled,
    i16 => divide_constant_i16_scaled,
    ComplexI16 => divide_constant_i16_complex_scaled
]);

impl_signal_scaled_constant_in_place_dispatch!(AddConstantScaledInPlace, add_constant_scaled_in_place, add_constant_scaled_in_place, [
    u8 => add_constant_u8_scaled_in_place,
    u16 => add_constant_u16_scaled_in_place,
    i16 => add_constant_i16_scaled_in_place,
    i32 => add_constant_i32_scaled_in_place,
    ComplexI16 => add_constant_i16_complex_scaled_in_place,
    ComplexI32 => add_constant_i32_complex_scaled_in_place
]);
impl_signal_scaled_constant_in_place_dispatch!(SubtractConstantScaledInPlace, subtract_constant_scaled_in_place, subtract_constant_scaled_in_place, [
    u8 => subtract_constant_u8_scaled_in_place,
    u16 => subtract_constant_u16_scaled_in_place,
    i16 => subtract_constant_i16_scaled_in_place,
    i32 => subtract_constant_i32_scaled_in_place,
    ComplexI16 => subtract_constant_i16_complex_scaled_in_place,
    ComplexI32 => subtract_constant_i32_complex_scaled_in_place
]);
impl_signal_scaled_constant_in_place_dispatch!(SubtractFromConstantScaledInPlace, subtract_from_constant_scaled_in_place, subtract_from_constant_scaled_in_place, [
    u8 => subtract_from_constant_u8_scaled_in_place,
    u16 => subtract_from_constant_u16_scaled_in_place,
    i16 => subtract_from_constant_i16_scaled_in_place,
    i32 => subtract_from_constant_i32_scaled_in_place,
    ComplexI16 => subtract_from_constant_i16_complex_scaled_in_place,
    ComplexI32 => subtract_from_constant_i32_complex_scaled_in_place
]);
impl_signal_scaled_constant_in_place_dispatch!(MultiplyConstantScaledInPlace, multiply_constant_scaled_in_place, multiply_constant_scaled_in_place, [
    u8 => multiply_constant_u8_scaled_in_place,
    u16 => multiply_constant_u16_scaled_in_place,
    i16 => multiply_constant_i16_scaled_in_place,
    i32 => multiply_constant_i32_scaled_in_place,
    ComplexI16 => multiply_constant_i16_complex_scaled_in_place,
    ComplexI32 => multiply_constant_i32_complex_scaled_in_place
]);
impl_signal_scaled_constant_in_place_dispatch!(DivideConstantScaledInPlace, divide_constant_scaled_in_place, divide_constant_scaled_in_place, [
    u8 => divide_constant_u8_scaled_in_place,
    u16 => divide_constant_u16_scaled_in_place,
    i16 => divide_constant_i16_scaled_in_place,
    ComplexI16 => divide_constant_i16_complex_scaled_in_place
]);

macro_rules! impl_signal_destination_update_constant_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, source, value, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            value: T,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, source, value, destination)
        }
    };
}

macro_rules! impl_signal_mixed_constant_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalViewMut<'_, Destination>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$destination_ty> for $source_ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                ) -> Result<()> {
                    $direct(stream_context, source, value, destination)
                }
            }
        )*

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            value: Source,
            destination: &mut SignalViewMut<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, value, destination)
        }
    };
}

macro_rules! impl_signal_scaled_mixed_constant_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                value: Self,
                destination: &mut SignalViewMut<'_, Destination>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$destination_ty> for $source_ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    value: Self,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, value, destination, scale_factor)
                }
            }
        )*

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            value: Source,
            destination: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, value, destination, scale_factor)
        }
    };
}

macro_rules! impl_signal_scaled_mixed_constant_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($signal_ty:ty => ($value_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait<Value>: DataTypeLike
        where
            Value: DataTypeLike,
        {
            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                value: Value,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$value_ty> for $signal_ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    value: $value_ty,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, signal, value, scale_factor)
                }
            }
        )*

        pub fn $function<Signal, Value>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, Signal>,
            value: Value,
            scale_factor: i32,
        ) -> Result<()>
        where
            Signal: $trait<Value>,
            Value: DataTypeLike,
        {
            Signal::$method(stream_context, signal, value, scale_factor)
        }
    };
}

impl_signal_destination_update_constant_dispatch!(AddProductConstant, add_product_constant, add_product_constant, [
    f32 => add_product_constant_f32
]);
impl_signal_mixed_constant_dispatch!(MultiplyConstantLowTo, multiply_constant_low_to, multiply_constant_low_to, [
    f32 => (i16, multiply_constant_f32_to_i16_low)
]);
impl_signal_scaled_mixed_constant_dispatch!(MultiplyConstantScaledTo, multiply_constant_scaled_to, multiply_constant_scaled_to, [
    f32 => (i16, multiply_constant_f32_to_i16_scaled)
]);
impl_signal_scaled_mixed_constant_in_place_dispatch!(MultiplyConstantScaledToInPlace, multiply_constant_scaled_to_in_place, multiply_constant_scaled_to_in_place, [
    i64 => (f64, multiply_constant_f64_to_i64_scaled_in_place)
]);

macro_rules! impl_signal_shift_constant_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                shift: i32,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    shift: i32,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, source, shift, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            shift: i32,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, source, shift, destination)
        }
    };
}

macro_rules! impl_signal_shift_constant_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                shift: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    shift: i32,
                ) -> Result<()> {
                    $direct(stream_context, signal, shift)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            shift: i32,
        ) -> Result<()> {
            T::$method(stream_context, signal, shift)
        }
    };
}

impl_signal_shift_constant_dispatch!(LeftShiftConstant, left_shift_constant, left_shift_constant, [
    u8 => left_shift_constant_u8,
    u16 => left_shift_constant_u16,
    i16 => left_shift_constant_i16,
    u32 => left_shift_constant_u32,
    i32 => left_shift_constant_i32
]);
impl_signal_shift_constant_dispatch!(RightShiftConstant, right_shift_constant, right_shift_constant, [
    u8 => right_shift_constant_u8,
    u16 => right_shift_constant_u16,
    i16 => right_shift_constant_i16,
    u32 => right_shift_constant_u32,
    i32 => right_shift_constant_i32
]);
impl_signal_shift_constant_in_place_dispatch!(LeftShiftConstantInPlace, left_shift_constant_in_place, left_shift_constant_in_place, [
    u8 => left_shift_constant_u8_in_place,
    u16 => left_shift_constant_u16_in_place,
    i16 => left_shift_constant_i16_in_place,
    u32 => left_shift_constant_u32_in_place,
    i32 => left_shift_constant_i32_in_place
]);
impl_signal_shift_constant_in_place_dispatch!(RightShiftConstantInPlace, right_shift_constant_in_place, right_shift_constant_in_place, [
    u8 => right_shift_constant_u8_in_place,
    u16 => right_shift_constant_u16_in_place,
    i16 => right_shift_constant_i16_in_place,
    u32 => right_shift_constant_u32_in_place,
    i32 => right_shift_constant_i32_in_place
]);

macro_rules! impl_signal_to_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
            ) -> Result<()>;
        }

        $(
            impl $trait<$destination_ty> for $source_ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, destination)
        }
    };
}

macro_rules! impl_signal_scaled_to_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($source_ty:ty => ($destination_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait<Destination>: DataTypeLike
        where
            Destination: DataTypeLike,
        {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Destination>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait<$destination_ty> for $source_ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, $destination_ty>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, scale_factor)
                }
            }
        )*

        pub fn $function<Source, Destination>(
            stream_context: &StreamContext,
            source: &SignalView<'_, Source>,
            destination: &mut SignalViewMut<'_, Destination>,
            scale_factor: i32,
        ) -> Result<()>
        where
            Source: $trait<Destination>,
            Destination: DataTypeLike,
        {
            Source::$method(stream_context, source, destination, scale_factor)
        }
    };
}

impl_signal_to_dispatch!(ExponentTo, exponent_to, exponent_to, [
    f32 => (f64, exponent_f32_to_f64)
]);
impl_signal_to_dispatch!(NaturalLogarithmTo, natural_logarithm_to, natural_logarithm_to, [
    f64 => (f32, natural_logarithm_f64_to_f32)
]);
impl_signal_scaled_to_dispatch!(NaturalLogarithmScaledTo, natural_logarithm_scaled_to, natural_logarithm_scaled_to, [
    i32 => (i16, natural_logarithm_i32_to_i16_scaled)
]);
impl_signal_scaled_to_dispatch!(CubeRootScaledTo, cube_root_scaled_to, cube_root_scaled_to, [
    i32 => (i16, cube_root_i32_to_i16_scaled)
]);
impl_signal_scaled_to_dispatch!(SquareRootScaledTo, square_root_scaled_to, square_root_scaled_to, [
    i32 => (i16, square_root_i32_to_i16_scaled),
    i64 => (i16, square_root_i64_to_i16_scaled)
]);

impl_signal_unary_dispatch!(CubeRoot, cube_root, cube_root, [
    f32 => cube_root_f32
]);
impl_signal_scaled_unary_dispatch!(TenTimesLog10Scaled, ten_times_log10_scaled, ten_times_log10_scaled, [
    i32 => ten_times_log10_i32_scaled
]);
impl_signal_scaled_unary_in_place_dispatch!(TenTimesLog10ScaledInPlace, ten_times_log10_scaled_in_place, ten_times_log10_scaled_in_place, [
    i32 => ten_times_log10_i32_scaled_in_place
]);

macro_rules! impl_signal_normalize_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => ($scale_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            type Scale: DataTypeLike;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                subtract: Self,
                divide: Self::Scale,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                type Scale = $scale_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    subtract: Self,
                    divide: Self::Scale,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, subtract, divide)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            subtract: T,
            divide: T::Scale,
        ) -> Result<()> {
            T::$method(stream_context, source, destination, subtract, divide)
        }
    };
}

macro_rules! impl_signal_scaled_normalize_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => ($divisor_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            type Divisor: DataTypeLike;

            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                subtract: Self,
                divide: Self::Divisor,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                type Divisor = $divisor_ty;

                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    subtract: Self,
                    divide: Self::Divisor,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, subtract, divide, scale_factor)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            subtract: T,
            divide: T::Divisor,
            scale_factor: i32,
        ) -> Result<()> {
            T::$method(stream_context, source, destination, subtract, divide, scale_factor)
        }
    };
}

impl_signal_normalize_dispatch!(Normalize, normalize, normalize, [
    f32 => (f32, normalize_f32),
    f64 => (f64, normalize_f64),
    Complex32 => (f32, normalize_f32_complex),
    Complex64 => (f64, normalize_f64_complex)
]);
impl_signal_scaled_normalize_dispatch!(NormalizeScaled, normalize_scaled, normalize_scaled, [
    i16 => (i32, normalize_i16_scaled),
    ComplexI16 => (i32, normalize_i16_complex_scaled)
]);

macro_rules! impl_signal_parameterized_unary_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => ($parameter_ty:ty, $direct:ident)),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            type Parameter: DataTypeLike;

            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                parameter: Self::Parameter,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                type Parameter = $parameter_ty;

                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    parameter: Self::Parameter,
                ) -> Result<()> {
                    $direct(stream_context, signal, parameter)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            parameter: T::Parameter,
        ) -> Result<()> {
            T::$method(stream_context, signal, parameter)
        }
    };
}

impl_signal_parameterized_unary_in_place_dispatch!(CauchyInPlace, cauchy_in_place, cauchy_in_place, [
    f32 => (f32, cauchy_f32_in_place)
]);
impl_signal_parameterized_unary_in_place_dispatch!(CauchyDerivativeInPlace, cauchy_derivative_in_place, cauchy_derivative_in_place, [
    f32 => (f32, cauchy_derivative_f32_in_place)
]);
