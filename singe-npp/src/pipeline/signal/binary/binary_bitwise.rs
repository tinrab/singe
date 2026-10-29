use crate::{
    context::StreamContext,
    error::Result,
    pipeline::signal::SignalPipeline,
    signal::{
        arithmetic,
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::RoundMode,
};

macro_rules! impl_scaled_round_divide_helpers {
    ($ty:ty, $divide:path, $divide_in_place:path) => {
        impl<'a> SignalPipeline<'a, $ty> {
            /// Computes `left / right` element by element into `destination`.
            ///
            /// Uses native scaling with `scale_factor` and explicit `round_mode`.
            /// Prepares participating regions, waiting for producers on other streams, rejects capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Participating signal lengths must match.
            /// Returns an error during capture, if CUDA context binding or region validation fails, waiting for a producer fails, a length exceeds the native range, NPP rejects the parameters, or stream synchronization fails.
            #[doc = concat!("Uses [`", stringify!($divide), "`].")]
            pub fn divide_scaled_round_into(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $divide(
                        stream_context,
                        left,
                        right,
                        destination,
                        scale_factor,
                        round_mode,
                    )
                })
            }

            /// Divides `destination` by `source` element by element in place.
            ///
            /// Uses native scaling with `scale_factor` and explicit `round_mode`.
            /// Prepares participating regions, waiting for producers on other streams, rejects capture, and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Participating signal lengths must match.
            /// Returns an error during capture, if CUDA context binding or region validation fails, waiting for a producer fails, a length exceeds the native range, NPP rejects the parameters, or stream synchronization fails.
            #[doc = concat!("Uses [`", stringify!($divide_in_place), "`].")]
            pub fn divide_scaled_round_in_place(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $divide_in_place(
                        stream_context,
                        source,
                        destination,
                        scale_factor,
                        round_mode,
                    )
                })
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

/// Dispatches completed bitwise operations.
///
/// # Safety
///
/// Implementations must preserve native element representations, operand order, and arithmetic semantics.
/// Complete borrowed native work on every return path, including errors. Initialize every logical output element on success, and preserve initialized accumulator semantics for in-place operations.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait BitwiseSignal<T> {
    /// Computes the element-wise bitwise AND of `left` and `right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn and_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise bitwise AND of `destination` and `signal`, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn and_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise bitwise OR of `left` and `right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn or_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise bitwise OR of `destination` and `signal`, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn or_signal_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise bitwise XOR of `left` and `right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    fn xor_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Computes the element-wise bitwise XOR of `destination` and `signal`, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
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
        unsafe impl<'a> BitwiseSignal<$ty> for SignalPipeline<'a, $ty> {
            fn and_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $and(stream_context, left, right, destination) })
            }

            fn and_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(signal)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $and_in_place(stream_context, signal, destination) })
            }

            fn or_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $or(stream_context, left, right, destination) })
            }

            fn or_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(signal)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $or_in_place(stream_context, signal, destination) })
            }

            fn xor_signal(
                stream_context: &StreamContext,
                left: &SignalView<'_, $ty>,
                right: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $xor(stream_context, left, right, destination) })
            }

            fn xor_signal_in_place(
                stream_context: &StreamContext,
                signal: &SignalView<'_, $ty>,
                destination: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(signal)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $xor_in_place(stream_context, signal, destination) })
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
