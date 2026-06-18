use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        memory::Signal,
        view::{SignalView, SignalViewMut},
    },
    types::ZeroCrossingType,
    workspace::ScratchBuffer,
};

#[derive(Debug)]
pub struct SignalMinMax<T> {
    pub min: Signal<T>,
    pub max: Signal<T>,
}

#[derive(Debug)]
pub struct SignalMeanStandardDeviation<T> {
    pub mean: Signal<T>,
    pub standard_deviation: Signal<T>,
}

#[derive(Debug)]
pub struct SignalIndexedValue<T> {
    pub value: Signal<T>,
    pub index: Signal<i32>,
}

pub trait ScalarStatisticSignal<T> {
    fn sum_buffer_size(stream_context: &StreamContext, source: &SignalView<'_, T>)
    -> Result<usize>;

    fn sum_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn sum_natural_logarithm_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn sum_natural_logarithm_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn mean_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn mean_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn standard_deviation_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn norm_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn norm_inf_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn norm_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn norm_l1_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn norm_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn norm_l2_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait BinaryScalarStatisticSignal<T> {
    fn norm_diff_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn norm_diff_inf_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn norm_diff_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn norm_diff_l1_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn norm_diff_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn norm_diff_l2_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn dot_product_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn dot_product_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait ExtremumStatisticSignal<T> {
    fn max_buffer_size(stream_context: &StreamContext, source: &SignalView<'_, T>)
    -> Result<usize>;

    fn max_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn min_buffer_size(stream_context: &StreamContext, source: &SignalView<'_, T>)
    -> Result<usize>;

    fn min_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait AbsoluteExtremumStatisticSignal<T> {
    fn max_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn max_absolute_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn min_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn min_absolute_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait PairStatisticSignal<T> {
    fn min_max_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn min_max_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        min: &mut SignalViewMut<'_, T>,
        max: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait MeanStandardDeviationStatisticSignal<T> {
    fn mean_standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn mean_standard_deviation_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        mean: &mut SignalViewMut<'_, T>,
        standard_deviation: &mut SignalViewMut<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait IndexedStatisticSignal<T> {
    fn max_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn max_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn min_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn min_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait AbsoluteIndexedStatisticSignal<T> {
    fn max_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn max_absolute_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn min_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn min_absolute_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalViewMut<'_, T>,
        index: &mut SignalViewMut<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait EveryStatisticSignal<T> {
    fn min_every_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    fn max_every_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

pub trait ErrorMetricStatisticSignal<T> {
    fn maximum_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn maximum_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn average_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn average_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn maximum_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn maximum_relative_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    fn average_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn average_relative_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

pub trait ZeroCrossingStatisticSignal<T> {
    fn zero_crossing_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    fn zero_crossing_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, f32>,
        zero_crossing_type: ZeroCrossingType,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}
