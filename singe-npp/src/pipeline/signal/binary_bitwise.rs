use super::*;

macro_rules! impl_scaled_round_divide_helpers {
    ($ty:ty, $divide:path, $divide_in_place:path) => {
        impl<'a> SignalPipeline<'a, $ty> {
            pub fn divide_scaled_round_into(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()> {
                $divide(
                    stream_context,
                    left,
                    right,
                    destination,
                    scale_factor,
                    round_mode,
                )
            }

            pub fn divide_scaled_round_in_place(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()> {
                $divide_in_place(
                    stream_context,
                    source,
                    destination,
                    scale_factor,
                    round_mode,
                )
            }
        }
    };
}

impl_scaled_round_divide_helpers!(
    u8,
    arithmetic::divide_u8_scaled_round,
    arithmetic::divide_u8_scaled_round_in_place
);
impl_scaled_round_divide_helpers!(
    u16,
    arithmetic::divide_u16_scaled_round,
    arithmetic::divide_u16_scaled_round_in_place
);
impl_scaled_round_divide_helpers!(
    i16,
    arithmetic::divide_i16_scaled_round,
    arithmetic::divide_i16_scaled_round_in_place
);

pub trait BitwiseSignal<T> {
    fn and_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn and_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn or_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn or_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn xor_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn xor_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

macro_rules! impl_bitwise_signal {
    (
        $ty:ty,
        $and:path,
        $and_in_place:path,
        $or:path,
        $or_in_place:path,
        $xor:path,
        $xor_in_place:path
    ) => {
        impl<'a> BitwiseSignal<$ty> for SignalPipeline<'a, $ty> {
            fn and_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $and(stream_context, left, right, destination)
            }

            fn and_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $and_in_place(stream_context, signal, destination)
            }

            fn or_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $or(stream_context, left, right, destination)
            }

            fn or_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $or_in_place(stream_context, signal, destination)
            }

            fn xor_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $xor(stream_context, left, right, destination)
            }

            fn xor_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                $xor_in_place(stream_context, signal, destination)
            }
        }
    };
}

impl_bitwise_signal!(
    u8,
    arithmetic::and_u8,
    arithmetic::and_u8_in_place,
    arithmetic::or_u8,
    arithmetic::or_u8_in_place,
    arithmetic::xor_u8,
    arithmetic::xor_u8_in_place
);

impl_bitwise_signal!(
    u16,
    arithmetic::and_u16,
    arithmetic::and_u16_in_place,
    arithmetic::or_u16,
    arithmetic::or_u16_in_place,
    arithmetic::xor_u16,
    arithmetic::xor_u16_in_place
);

impl_bitwise_signal!(
    u32,
    arithmetic::and_u32,
    arithmetic::and_u32_in_place,
    arithmetic::or_u32,
    arithmetic::or_u32_in_place,
    arithmetic::xor_u32,
    arithmetic::xor_u32_in_place
);
