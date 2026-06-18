use super::*;

macro_rules! impl_scaled_constant_signal {
    (
        $ty:ty,
        $add:path,
        $add_in_place:path,
        $subtract:path,
        $subtract_in_place:path,
        $subtract_from:path,
        $subtract_from_in_place:path,
        $multiply:path,
        $multiply_in_place:path
    ) => {
        impl<'a> ScaledConstantSignal<$ty> for SignalPipeline<'a, $ty> {
            fn add_constant_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $add(stream_context, source, value, destination, scale_factor)
            }

            fn add_constant_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
                scale_factor: i32,
            ) -> Result<()> {
                $add_in_place(stream_context, signal, value, scale_factor)
            }

            fn subtract_constant_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract(stream_context, source, value, destination, scale_factor)
            }

            fn subtract_constant_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_in_place(stream_context, signal, value, scale_factor)
            }

            fn subtract_from_constant_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_from(stream_context, source, value, destination, scale_factor)
            }

            fn subtract_from_constant_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
                scale_factor: i32,
            ) -> Result<()> {
                $subtract_from_in_place(stream_context, signal, value, scale_factor)
            }

            fn multiply_constant_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply(stream_context, source, value, destination, scale_factor)
            }

            fn multiply_constant_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
                scale_factor: i32,
            ) -> Result<()> {
                $multiply_in_place(stream_context, signal, value, scale_factor)
            }
        }
    };
}

macro_rules! impl_divide_scaled_constant_signal {
    ($ty:ty, $divide:path, $divide_in_place:path) => {
        impl<'a> DivideScaledConstantSignal<$ty> for SignalPipeline<'a, $ty> {
            fn divide_constant_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                $divide(stream_context, source, value, destination, scale_factor)
            }

            fn divide_constant_signal_scaled_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
                scale_factor: i32,
            ) -> Result<()> {
                $divide_in_place(stream_context, signal, value, scale_factor)
            }
        }
    };
}

macro_rules! impl_constant_bitwise_signal {
    (
        $ty:ty,
        $and:path,
        $and_in_place:path,
        $or:path,
        $or_in_place:path,
        $xor:path,
        $xor_in_place:path
    ) => {
        impl<'a> ConstantBitwiseSignal<$ty> for SignalPipeline<'a, $ty> {
            fn and_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $and(stream_context, source, value, destination)
            }

            fn and_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $and_in_place(stream_context, signal, value)
            }

            fn or_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $or(stream_context, source, value, destination)
            }

            fn or_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $or_in_place(stream_context, signal, value)
            }

            fn xor_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                value: $ty,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $xor(stream_context, source, value, destination)
            }

            fn xor_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                value: $ty,
            ) -> Result<()> {
                $xor_in_place(stream_context, signal, value)
            }
        }
    };
}

macro_rules! impl_constant_shift_signal {
    (
        $ty:ty,
        $left:path,
        $left_in_place:path,
        $right:path,
        $right_in_place:path
    ) => {
        impl<'a> ConstantShiftSignal<$ty> for SignalPipeline<'a, $ty> {
            fn left_shift_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                shift: i32,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $left(stream_context, source, shift, destination)
            }

            fn left_shift_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                shift: i32,
            ) -> Result<()> {
                $left_in_place(stream_context, signal, shift)
            }

            fn right_shift_constant_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                shift: i32,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $right(stream_context, source, shift, destination)
            }

            fn right_shift_constant_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                shift: i32,
            ) -> Result<()> {
                $right_in_place(stream_context, signal, shift)
            }
        }
    };
}

impl_scaled_constant_signal!(
    u8,
    arithmetic::add_constant_u8_scaled,
    arithmetic::add_constant_u8_scaled_in_place,
    arithmetic::subtract_constant_u8_scaled,
    arithmetic::subtract_constant_u8_scaled_in_place,
    arithmetic::subtract_from_constant_u8_scaled,
    arithmetic::subtract_from_constant_u8_scaled_in_place,
    arithmetic::multiply_constant_u8_scaled,
    arithmetic::multiply_constant_u8_scaled_in_place
);

impl_scaled_constant_signal!(
    u16,
    arithmetic::add_constant_u16_scaled,
    arithmetic::add_constant_u16_scaled_in_place,
    arithmetic::subtract_constant_u16_scaled,
    arithmetic::subtract_constant_u16_scaled_in_place,
    arithmetic::subtract_from_constant_u16_scaled,
    arithmetic::subtract_from_constant_u16_scaled_in_place,
    arithmetic::multiply_constant_u16_scaled,
    arithmetic::multiply_constant_u16_scaled_in_place
);

impl_scaled_constant_signal!(
    i16,
    arithmetic::add_constant_i16_scaled,
    arithmetic::add_constant_i16_scaled_in_place,
    arithmetic::subtract_constant_i16_scaled,
    arithmetic::subtract_constant_i16_scaled_in_place,
    arithmetic::subtract_from_constant_i16_scaled,
    arithmetic::subtract_from_constant_i16_scaled_in_place,
    arithmetic::multiply_constant_i16_scaled,
    arithmetic::multiply_constant_i16_scaled_in_place
);

impl_scaled_constant_signal!(
    i32,
    arithmetic::add_constant_i32_scaled,
    arithmetic::add_constant_i32_scaled_in_place,
    arithmetic::subtract_constant_i32_scaled,
    arithmetic::subtract_constant_i32_scaled_in_place,
    arithmetic::subtract_from_constant_i32_scaled,
    arithmetic::subtract_from_constant_i32_scaled_in_place,
    arithmetic::multiply_constant_i32_scaled,
    arithmetic::multiply_constant_i32_scaled_in_place
);
impl_scaled_constant_signal!(
    ComplexI16,
    arithmetic::add_constant_i16_complex_scaled,
    arithmetic::add_constant_i16_complex_scaled_in_place,
    arithmetic::subtract_constant_i16_complex_scaled,
    arithmetic::subtract_constant_i16_complex_scaled_in_place,
    arithmetic::subtract_from_constant_i16_complex_scaled,
    arithmetic::subtract_from_constant_i16_complex_scaled_in_place,
    arithmetic::multiply_constant_i16_complex_scaled,
    arithmetic::multiply_constant_i16_complex_scaled_in_place
);
impl_scaled_constant_signal!(
    ComplexI32,
    arithmetic::add_constant_i32_complex_scaled,
    arithmetic::add_constant_i32_complex_scaled_in_place,
    arithmetic::subtract_constant_i32_complex_scaled,
    arithmetic::subtract_constant_i32_complex_scaled_in_place,
    arithmetic::subtract_from_constant_i32_complex_scaled,
    arithmetic::subtract_from_constant_i32_complex_scaled_in_place,
    arithmetic::multiply_constant_i32_complex_scaled,
    arithmetic::multiply_constant_i32_complex_scaled_in_place
);

impl_divide_scaled_constant_signal!(
    u8,
    arithmetic::divide_constant_u8_scaled,
    arithmetic::divide_constant_u8_scaled_in_place
);
impl_divide_scaled_constant_signal!(
    u16,
    arithmetic::divide_constant_u16_scaled,
    arithmetic::divide_constant_u16_scaled_in_place
);
impl_divide_scaled_constant_signal!(
    i16,
    arithmetic::divide_constant_i16_scaled,
    arithmetic::divide_constant_i16_scaled_in_place
);
impl_divide_scaled_constant_signal!(
    ComplexI16,
    arithmetic::divide_constant_i16_complex_scaled,
    arithmetic::divide_constant_i16_complex_scaled_in_place
);

impl_constant_bitwise_signal!(
    u8,
    arithmetic::and_constant_u8,
    arithmetic::and_constant_u8_in_place,
    arithmetic::or_constant_u8,
    arithmetic::or_constant_u8_in_place,
    arithmetic::xor_constant_u8,
    arithmetic::xor_constant_u8_in_place
);
impl_constant_bitwise_signal!(
    u16,
    arithmetic::and_constant_u16,
    arithmetic::and_constant_u16_in_place,
    arithmetic::or_constant_u16,
    arithmetic::or_constant_u16_in_place,
    arithmetic::xor_constant_u16,
    arithmetic::xor_constant_u16_in_place
);
impl_constant_bitwise_signal!(
    u32,
    arithmetic::and_constant_u32,
    arithmetic::and_constant_u32_in_place,
    arithmetic::or_constant_u32,
    arithmetic::or_constant_u32_in_place,
    arithmetic::xor_constant_u32,
    arithmetic::xor_constant_u32_in_place
);

impl_constant_shift_signal!(
    u8,
    arithmetic::left_shift_constant_u8,
    arithmetic::left_shift_constant_u8_in_place,
    arithmetic::right_shift_constant_u8,
    arithmetic::right_shift_constant_u8_in_place
);
impl_constant_shift_signal!(
    u16,
    arithmetic::left_shift_constant_u16,
    arithmetic::left_shift_constant_u16_in_place,
    arithmetic::right_shift_constant_u16,
    arithmetic::right_shift_constant_u16_in_place
);
impl_constant_shift_signal!(
    i16,
    arithmetic::left_shift_constant_i16,
    arithmetic::left_shift_constant_i16_in_place,
    arithmetic::right_shift_constant_i16,
    arithmetic::right_shift_constant_i16_in_place
);
impl_constant_shift_signal!(
    u32,
    arithmetic::left_shift_constant_u32,
    arithmetic::left_shift_constant_u32_in_place,
    arithmetic::right_shift_constant_u32,
    arithmetic::right_shift_constant_u32_in_place
);
impl_constant_shift_signal!(
    i32,
    arithmetic::left_shift_constant_i32,
    arithmetic::left_shift_constant_i32_in_place,
    arithmetic::right_shift_constant_i32,
    arithmetic::right_shift_constant_i32_in_place
);
