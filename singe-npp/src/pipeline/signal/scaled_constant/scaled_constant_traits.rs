use crate::{
    context::StreamContext,
    error::Result,
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
};

/// Dispatches scaled addition, subtraction, and multiplication by a constant.
///
/// # Safety
/// Implementations must preserve native element representations and the documented arithmetic.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledConstantSignal<T> {
    /// Adds `value` to each source element. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn add_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Adds `value` to each signal element. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn add_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;

    /// Subtracts `value` from each source element. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn subtract_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Subtracts `value` from each signal element. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn subtract_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;

    /// Subtracts each source element from `value`. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn subtract_from_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Subtracts each signal element from `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn subtract_from_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;

    /// Multiplies each source element by `value`. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn multiply_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Multiplies each signal element by `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn multiply_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches scaled division by a constant.
///
/// # Safety
/// Implementations must preserve native element representations and the documented arithmetic.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait DivideScaledConstantSignal<T> {
    /// Divides each source element by `value`. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn divide_constant_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Divides each signal element by `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn divide_constant_signal_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches bitwise operations between a signal and a constant.
///
/// # Safety
/// Implementations must preserve native element representations and the documented arithmetic.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ConstantBitwiseSignal<T> {
    /// Computes the bitwise AND of each source element and `value`. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn and_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the bitwise AND of each signal element and `value`. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn and_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    /// Computes the bitwise OR of each source element and `value`. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn or_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the bitwise OR of each signal element and `value`. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn or_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;

    /// Computes the bitwise XOR of each source element and `value`. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn xor_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the bitwise XOR of each signal element and `value`. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn xor_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()>;
}

/// Dispatches constant left and right shifts of signal elements.
///
/// # Safety
/// Implementations must preserve native element representations and the documented arithmetic.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ConstantShiftSignal<T> {
    /// Shifts each source element left by `shift` bits. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn left_shift_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Shifts each signal element left by `shift` bits. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn left_shift_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()>;

    /// Shifts each source element right by `shift` bits. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn right_shift_constant_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Shifts each signal element right by `shift` bits. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn right_shift_constant_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()>;
}
