use crate::{
    context::StreamContext,
    error::Result,
    signal::{
        memory::Signal,
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::ZeroCrossingType,
    workspace::ScratchBuffer,
};

/// Minimum and maximum results, each stored in an owned one-element device signal.
#[derive(Debug)]
pub struct SignalMinMax<T> {
    /// Minimum source value.
    pub min: Signal<T>,
    /// Maximum source value.
    pub max: Signal<T>,
}

/// Mean and standard-deviation results, each stored in an owned one-element device signal.
#[derive(Debug)]
pub struct SignalMeanStandardDeviation<T> {
    /// Arithmetic mean of the source samples.
    pub mean: Signal<T>,
    /// Standard deviation of the source samples.
    pub standard_deviation: Signal<T>,
}

/// An extremum result and its native source index, each stored in an owned one-element device signal.
#[derive(Debug)]
pub struct SignalIndexedValue<T> {
    /// Extremum value reported by the selected native statistic.
    pub value: Signal<T>,
    /// Native index of the first occurrence of the reported extremum.
    pub index: Signal<i32>,
}

/// Dispatches scalar reductions for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait ScalarStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the sum of the source samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn sum_buffer_size(stream_context: &StreamContext, source: &SignalView<'_, T>)
    -> Result<usize>;

    /// Computes the sum of the source samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn sum_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the sum of the natural logarithms of the source samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn sum_natural_logarithm_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the sum of the natural logarithms of the source samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn sum_natural_logarithm_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the arithmetic mean of the source samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn mean_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the arithmetic mean of the source samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn mean_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the standard deviation of the source samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the standard deviation of the source samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn standard_deviation_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the infinity norm of the source signal.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn norm_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the infinity norm of the source signal.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn norm_inf_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the L1 norm of the source signal.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn norm_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the L1 norm of the source signal.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn norm_l1_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the L2 norm of the source signal.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn norm_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the L2 norm of the source signal.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn norm_l2_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches binary scalar reductions for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait BinaryScalarStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the infinity norm of the difference between the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn norm_diff_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the infinity norm of the difference between the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn norm_diff_inf_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the L1 norm of the difference between the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn norm_diff_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the L1 norm of the difference between the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn norm_diff_l1_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the L2 norm of the difference between the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn norm_diff_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the L2 norm of the difference between the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn norm_diff_l2_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the dot product of the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn dot_product_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the dot product of the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn dot_product_with_scratch(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches minimum and maximum reductions for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait ExtremumStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the maximum source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn max_buffer_size(stream_context: &StreamContext, source: &SignalView<'_, T>)
    -> Result<usize>;

    /// Computes the maximum source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn max_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the minimum source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn min_buffer_size(stream_context: &StreamContext, source: &SignalView<'_, T>)
    -> Result<usize>;

    /// Computes the minimum source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn min_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches absolute minimum and maximum reductions for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait AbsoluteExtremumStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the maximum absolute source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn max_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the maximum absolute source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn max_absolute_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the minimum absolute source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn min_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the minimum absolute source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn min_absolute_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches paired minimum and maximum reduction for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait PairStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the minimum and maximum source values.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn min_max_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the minimum and maximum source values.
    ///
    /// Writes one element to each output view: `min`, `max`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn min_max_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        min: &mut SignalOutputView<'_, T>,
        max: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches paired mean and standard-deviation reduction for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait MeanStandardDeviationStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the arithmetic mean and standard deviation of the source samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn mean_standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the arithmetic mean and standard deviation of the source samples.
    ///
    /// Writes one element to each output view: `mean`, `standard_deviation`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn mean_standard_deviation_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        mean: &mut SignalOutputView<'_, T>,
        standard_deviation: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches minimum and maximum reductions with indices for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait IndexedStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the maximum source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn max_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the maximum source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn max_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the minimum source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn min_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the minimum source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn min_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches absolute minimum and maximum reductions with indices for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait AbsoluteIndexedStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the maximum absolute source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn max_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the maximum absolute source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn max_absolute_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the minimum absolute source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn min_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the minimum absolute source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn min_absolute_index_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches elementwise minimum and maximum updates for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait EveryStatisticSignal<T> {
    /// Replaces each destination sample with the elementwise minimum of the source and destination.
    ///
    /// Preserves the signal length and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The source and destination lengths must match.
    fn min_every_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Replaces each destination sample with the elementwise maximum of the source and destination.
    ///
    /// Preserves the signal length and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The source and destination lengths must match.
    fn max_every_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches signal error metrics with double-precision output for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait ErrorMetricStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the largest absolute difference between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn maximum_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the largest absolute difference between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    /// For complex inputs, NPP uses the complex magnitude of the difference.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn maximum_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the mean absolute difference between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn average_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the mean absolute difference between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    /// For complex inputs, NPP uses the complex magnitude of the difference.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn average_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the maximum relative error between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn maximum_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the maximum relative error between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    /// Each relative difference divides the absolute difference by the larger input magnitude.
    /// Zero-denominator behavior is left to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn maximum_relative_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;

    /// Returns the scratch capacity in bytes required for the average relative error between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn average_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the average relative error between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    /// Each relative difference divides the absolute difference by the larger input magnitude.
    /// Zero-denominator behavior is left to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn average_relative_error_with_scratch(
        stream_context: &StreamContext,
        source_1: &SignalView<'_, T>,
        source_2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}

/// Dispatches zero-crossing statistics with single-precision output for pipeline signals.
///
/// # Safety
///
/// Preserve the native element representation and the operation's input and output layouts.
/// Implementations must uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
/// Successful execution must finish native work before the borrowed signals and scratch can be released.
pub unsafe trait ZeroCrossingStatisticSignal<T> {
    /// Returns the scratch capacity in bytes required for the zero-crossing computation.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    fn zero_crossing_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>;

    /// Computes the zero-crossing statistic selected by `zero_crossing_type`.
    ///
    /// Writes one element to each output view: `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    /// `zero_crossing_type` selects the native sign-change rule.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    fn zero_crossing_with_scratch(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f32>,
        zero_crossing_type: ZeroCrossingType,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>;
}
