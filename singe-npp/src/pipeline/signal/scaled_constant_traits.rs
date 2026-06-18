use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalView, SignalViewMut},
};

pub trait ScaledConstantSignal<T> {
    fn add_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn add_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_from_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_from_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait DivideScaledConstantSignal<T> {
    fn divide_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait ConstantBitwiseSignal<T> {
    fn and_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn and_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    fn or_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn or_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    fn xor_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn xor_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

pub trait ConstantShiftSignal<T> {
    fn left_shift_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn left_shift_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()>;

    fn right_shift_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn right_shift_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()>;
}
