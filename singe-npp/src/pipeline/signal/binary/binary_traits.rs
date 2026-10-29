use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
};

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait BinarySignal<T> {
    /// Adds `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Adds `signal` to `destination` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise difference `left - right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn subtract_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise difference `destination - signal` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn subtract_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Multiplies `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Multiplies `destination` by `signal` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait DivideSignal<T> {
    /// Computes the element-wise quotient `left / right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn divide_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise quotient `destination / signal` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn divide_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledBinarySignal<T> {
    /// Adds `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Adds `signal` to `destination` element by element, updating `destination` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Computes the element-wise difference `left - right` into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn subtract_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Computes the element-wise difference `destination - signal` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn subtract_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Multiplies `destination` by `signal` element by element, updating `destination` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait DivideScaledBinarySignal<T> {
    /// Computes the element-wise quotient `left / right` into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn divide_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Computes the element-wise quotient `destination / signal` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn divide_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait MixedBinarySignal<T, U> {
    /// Adds `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_mixed_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>;

    /// Computes the element-wise difference `left - right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// The provided `u8` to `u16` mixed subtraction dispatch returns `Error::UnsupportedOperation`.
    fn subtract_mixed_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>;

    /// Multiplies `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_mixed_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledMixedBinarySignal<T, U> {
    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_mixed_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledHeterogeneousBinarySignal<T, U, V> {
    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_heterogeneous_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalOutputView<'_, V>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait MultiplyLowScaledSignal<T> {
    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Uses the native low multiplication routine.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_low_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait DivideByScaledSignal<T, U, V> {
    /// Computes the element-wise quotient `left / right` into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn divide_by_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalOutputView<'_, V>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledHeterogeneousInPlaceSignal<T, U> {
    /// Multiplies `destination` by `source` element by element, updating `destination` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_heterogeneous_signal_scaled_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait AddHeterogeneousInPlaceSignal<T, U> {
    /// Adds `source` to `destination` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait MultiplyHeterogeneousInPlaceSignal<T, U> {
    /// Multiplies `destination` by `source` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, U>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches completed binary signal arithmetic for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait HeterogeneousBinarySignal<T, U, V> {
    /// Multiplies `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn multiply_heterogeneous_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, U>,
        destination: &mut SignalOutputView<'_, V>,
    ) -> Result<()>;
}

/// Dispatches completed product accumulation.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait AddProductSignal<T> {
    /// Adds the element-wise product `left * right` to the initialized `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_product_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches completed product accumulation.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledAddProductSignal<T, U> {
    /// Adds the element-wise product `left * right` to the initialized `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn add_product_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}
