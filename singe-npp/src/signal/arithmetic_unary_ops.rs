use super::*;

macro_rules! impl_signal_unary_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, source, destination)
        }
    };
}

macro_rules! impl_signal_unary_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                ) -> Result<()> {
                    $direct(stream_context, signal)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
        ) -> Result<()> {
            T::$method(stream_context, signal)
        }
    };
}

macro_rules! impl_signal_scaled_unary_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &SignalView<'_, Self>,
                destination: &mut SignalViewMut<'_, Self>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &SignalView<'_, Self>,
                    destination: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, source, destination, scale_factor)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &SignalView<'_, T>,
            destination: &mut SignalViewMut<'_, T>,
            scale_factor: i32,
        ) -> Result<()> {
            T::$method(stream_context, source, destination, scale_factor)
        }
    };
}

macro_rules! impl_signal_scaled_unary_in_place_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, Self>,
                scale_factor: i32,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    signal: &mut SignalViewMut<'_, Self>,
                    scale_factor: i32,
                ) -> Result<()> {
                    $direct(stream_context, signal, scale_factor)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            signal: &mut SignalViewMut<'_, T>,
            scale_factor: i32,
        ) -> Result<()> {
            T::$method(stream_context, signal, scale_factor)
        }
    };
}

impl_signal_unary_dispatch!(Absolute, absolute, absolute, [
    i16 => absolute_i16,
    i32 => absolute_i32,
    f32 => absolute_f32,
    f64 => absolute_f64
]);
impl_signal_unary_dispatch!(Exponent, exponent, exponent, [
    f32 => exponent_f32,
    f64 => exponent_f64
]);
impl_signal_unary_dispatch!(NaturalLogarithm, natural_logarithm, natural_logarithm, [
    f32 => natural_logarithm_f32,
    f64 => natural_logarithm_f64
]);
impl_signal_unary_dispatch!(Arctangent, arctangent, arctangent, [
    f32 => arctangent_f32,
    f64 => arctangent_f64
]);
impl_signal_unary_dispatch!(Not, not, not, [
    u8 => not_u8,
    u16 => not_u16,
    u32 => not_u32
]);
impl_signal_unary_dispatch!(Square, square, square, [
    f32 => square_f32,
    f64 => square_f64,
    Complex32 => square_f32_complex,
    Complex64 => square_f64_complex
]);
impl_signal_unary_dispatch!(SquareRoot, square_root, square_root, [
    f32 => square_root_f32,
    f64 => square_root_f64,
    Complex32 => square_root_f32_complex,
    Complex64 => square_root_f64_complex
]);

impl_signal_unary_in_place_dispatch!(AbsoluteInPlace, absolute_in_place, absolute_in_place, [
    i16 => absolute_i16_in_place,
    i32 => absolute_i32_in_place,
    f32 => absolute_f32_in_place,
    f64 => absolute_f64_in_place
]);
impl_signal_unary_in_place_dispatch!(ExponentInPlace, exponent_in_place, exponent_in_place, [
    f32 => exponent_f32_in_place,
    f64 => exponent_f64_in_place
]);
impl_signal_unary_in_place_dispatch!(NaturalLogarithmInPlace, natural_logarithm_in_place, natural_logarithm_in_place, [
    f32 => natural_logarithm_f32_in_place,
    f64 => natural_logarithm_f64_in_place
]);
impl_signal_unary_in_place_dispatch!(ArctangentInPlace, arctangent_in_place, arctangent_in_place, [
    f32 => arctangent_f32_in_place,
    f64 => arctangent_f64_in_place
]);
impl_signal_unary_in_place_dispatch!(NotInPlace, not_in_place, not_in_place, [
    u8 => not_u8_in_place,
    u16 => not_u16_in_place,
    u32 => not_u32_in_place
]);
impl_signal_unary_in_place_dispatch!(SquareInPlace, square_in_place, square_in_place, [
    f32 => square_f32_in_place,
    f64 => square_f64_in_place,
    Complex32 => square_f32_complex_in_place,
    Complex64 => square_f64_complex_in_place
]);
impl_signal_unary_in_place_dispatch!(SquareRootInPlace, square_root_in_place, square_root_in_place, [
    f32 => square_root_f32_in_place,
    f64 => square_root_f64_in_place,
    Complex32 => square_root_f32_complex_in_place,
    Complex64 => square_root_f64_complex_in_place
]);

impl_signal_scaled_unary_dispatch!(ExponentScaled, exponent_scaled, exponent_scaled, [
    i16 => exponent_i16_scaled,
    i32 => exponent_i32_scaled,
    i64 => exponent_i64_scaled
]);
impl_signal_scaled_unary_dispatch!(NaturalLogarithmScaled, natural_logarithm_scaled, natural_logarithm_scaled, [
    i16 => natural_logarithm_i16_scaled,
    i32 => natural_logarithm_i32_scaled
]);
impl_signal_scaled_unary_dispatch!(SquareScaled, square_scaled, square_scaled, [
    u8 => square_u8_scaled,
    u16 => square_u16_scaled,
    i16 => square_i16_scaled,
    ComplexI16 => square_i16_complex_scaled
]);
impl_signal_scaled_unary_dispatch!(SquareRootScaled, square_root_scaled, square_root_scaled, [
    u8 => square_root_u8_scaled,
    u16 => square_root_u16_scaled,
    i16 => square_root_i16_scaled,
    i64 => square_root_i64_scaled,
    ComplexI16 => square_root_i16_complex_scaled
]);

impl_signal_scaled_unary_in_place_dispatch!(ExponentScaledInPlace, exponent_scaled_in_place, exponent_scaled_in_place, [
    i16 => exponent_i16_scaled_in_place,
    i32 => exponent_i32_scaled_in_place,
    i64 => exponent_i64_scaled_in_place
]);
impl_signal_scaled_unary_in_place_dispatch!(NaturalLogarithmScaledInPlace, natural_logarithm_scaled_in_place, natural_logarithm_scaled_in_place, [
    i16 => natural_logarithm_i16_scaled_in_place,
    i32 => natural_logarithm_i32_scaled_in_place
]);
impl_signal_scaled_unary_in_place_dispatch!(SquareScaledInPlace, square_scaled_in_place, square_scaled_in_place, [
    u8 => square_u8_scaled_in_place,
    u16 => square_u16_scaled_in_place,
    i16 => square_i16_scaled_in_place,
    ComplexI16 => square_i16_complex_scaled_in_place
]);
impl_signal_scaled_unary_in_place_dispatch!(SquareRootScaledInPlace, square_root_scaled_in_place, square_root_scaled_in_place, [
    u8 => square_root_u8_scaled_in_place,
    u16 => square_root_u16_scaled_in_place,
    i16 => square_root_i16_scaled_in_place,
    i64 => square_root_i64_scaled_in_place,
    ComplexI16 => square_root_i16_complex_scaled_in_place
]);
