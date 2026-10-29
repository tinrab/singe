use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
    types::ComparisonOperation,
};

/// Dispatches completed addition, subtraction, multiplication, and division by a constant.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ConstantSignal<T> {
    /// Adds `value` to each source element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn add_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Adds `value` to each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    /// Subtracts `value` from each source element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn subtract_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Subtracts `value` from each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn subtract_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    /// Multiplies each source element by `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn multiply_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Multiplies each signal element by `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    /// Divides each source element by `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn divide_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Divides each signal element by `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn divide_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

/// Dispatches completed subtraction of signal elements from a constant.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait SubtractFromConstantSignal<T> {
    /// Subtracts each source element from `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn subtract_from_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Subtracts each signal element from `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn subtract_from_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

/// Dispatches completed division of a constant by signal elements.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait DivideIntoConstantSignal<T> {
    /// Divides `value` by each source element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn divide_into_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Divides `value` by each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn divide_into_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

/// Dispatches completed exponential, logarithm, square, and square-root operations.
///
/// # Safety
///
/// Implementations must preserve native element representations and operation semantics.
/// Complete borrowed native work on every return path and initialize every output sample on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait UnarySignal<T> {
    /// Computes the natural exponential of each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn exponent_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the natural exponential of each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    fn exponent_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Computes the natural logarithm of each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn natural_logarithm_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the natural logarithm of each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    fn natural_logarithm_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Squares each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn square_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Squares each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    fn square_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Computes the square root of each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn square_root_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the square root of each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    fn square_root_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches completed constant-level threshold operations.
///
/// # Safety
///
/// Implementations must preserve native element representations and threshold semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ThresholdSignal<T> {
    /// Applies the native selected comparison threshold at `level` to each sample. Writes the result to `destination`.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()>;

    /// Applies the native selected comparison threshold at `level` to each sample. Updates the signal in place.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

/// Dispatches completed complex constant-level threshold operations.
///
/// # Safety
///
/// Implementations must preserve native element representations and threshold semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ComplexThresholdSignal<T, L> {
    /// Applies the native selected comparison threshold at `level` to each complex sample. Writes the result to `destination`.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>;

    /// Applies the native selected comparison threshold at `level` to each complex sample. Updates the signal in place.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

/// Dispatches completed constant-level threshold operations.
///
/// # Safety
///
/// Implementations must preserve native element representations and threshold semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait FixedThresholdSignal<T> {
    /// Applies the native less-than threshold at `level` to each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_less_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
    ) -> Result<()>;

    /// Applies the native less-than threshold at `level` to each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_less_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()>;

    /// Applies the native greater-than threshold at `level` to each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_greater_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
    ) -> Result<()>;

    /// Applies the native greater-than threshold at `level` to each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_greater_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()>;
}

/// Dispatches completed complex constant-level threshold operations.
///
/// # Safety
///
/// Implementations must preserve native element representations and threshold semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ComplexFixedThresholdSignal<T, L> {
    /// Applies the native less-than threshold at `level` to each complex sample. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_less_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
    ) -> Result<()>;

    /// Applies the native less-than threshold at `level` to each complex sample. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_less_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>;

    /// Applies the native greater-than threshold at `level` to each complex sample. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_greater_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
    ) -> Result<()>;

    /// Applies the native greater-than threshold at `level` to each complex sample. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_greater_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>;
}

/// Dispatches completed replacement-value threshold operations.
///
/// # Safety
///
/// Implementations must preserve native element representations and threshold semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ValueThresholdSignal<T> {
    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_less_value_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;

    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_less_value_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_greater_value_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_greater_value_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()>;
}

/// Dispatches completed complex replacement-value threshold operations.
///
/// # Safety
///
/// Implementations must preserve native element representations and threshold semantics.
/// Complete borrowed native work on every return path, including errors, and initialize every logical destination element on success.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ComplexValueThresholdSignal<T, L> {
    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_less_value_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;

    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_less_value_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn threshold_greater_value_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    fn threshold_greater_value_complex_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>;
}
