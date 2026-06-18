use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalView, SignalViewMut},
};

pub trait BinarySignal<T> {
    fn add_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn add_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn subtract_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn subtract_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn multiply_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn multiply_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait DivideSignal<T> {
    fn divide_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn divide_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait ScaledBinarySignal<T> {
    fn add_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn add_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait DivideScaledBinarySignal<T> {
    fn divide_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait MixedBinarySignal<T, U> {
    fn add_mixed_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>;

    fn subtract_mixed_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>;

    fn multiply_mixed_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
    ) -> Result<()>;
}

pub trait ScaledMixedBinarySignal<T, U> {
    fn multiply_mixed_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait ScaledHeterogeneousBinarySignal<T, U, V> {
    fn multiply_heterogeneous_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, V>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait MultiplyLowScaledSignal<T> {
    fn multiply_low_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait DivideByScaledSignal<T, U, V> {
    fn divide_by_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, V>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait ScaledHeterogeneousInPlaceSignal<T, U> {
    fn multiply_heterogeneous_signal_scaled_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait AddHeterogeneousInPlaceSignal<T, U> {
    fn add_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait MultiplyHeterogeneousInPlaceSignal<T, U> {
    fn multiply_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait HeterogeneousBinarySignal<T, U, V> {
    fn multiply_heterogeneous_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, V>,
    ) -> Result<()>;
}

pub trait AddProductSignal<T> {
    fn add_product_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait ScaledAddProductSignal<T, U> {
    fn add_product_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}
