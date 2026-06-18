use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalView, SignalViewMut},
    types::ComparisonOperation,
};

pub trait ConstantSignal<T> {
    fn add_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn add_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    fn subtract_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn subtract_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    fn multiply_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn multiply_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    fn divide_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn divide_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

pub trait SubtractFromConstantSignal<T> {
    fn subtract_from_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn subtract_from_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

pub trait DivideIntoConstantSignal<T> {
    fn divide_into_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn divide_into_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

pub trait UnarySignal<T> {
    fn exponent_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn exponent_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn natural_logarithm_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn natural_logarithm_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn square_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn square_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn square_root_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn square_root_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait ThresholdSignal<T> {
    fn threshold_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()>;

    fn threshold_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

pub trait ComplexThresholdSignal<T, L> {
    fn threshold_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>;

    fn threshold_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

pub trait FixedThresholdSignal<T> {
    fn threshold_less_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()>;

    fn threshold_less_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()>;

    fn threshold_greater_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()>;

    fn threshold_greater_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()>;
}

pub trait ComplexFixedThresholdSignal<T, L> {
    fn threshold_less_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>;

    fn threshold_less_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>;

    fn threshold_greater_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>;

    fn threshold_greater_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>;
}

pub trait ValueThresholdSignal<T> {
    fn threshold_less_value_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;

    fn threshold_less_value_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;

    fn threshold_greater_value_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;

    fn threshold_greater_value_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;
}

pub trait ComplexValueThresholdSignal<T, L> {
    fn threshold_less_value_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;

    fn threshold_less_value_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;

    fn threshold_greater_value_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;

    fn threshold_greater_value_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;
}
