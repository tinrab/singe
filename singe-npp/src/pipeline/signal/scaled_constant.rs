use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{
        arithmetic,
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::DataTypeLike,
};

type ScaledConstantSignalOperation<T> =
    fn(&StreamContext, &SignalView<'_, T>, T, &mut SignalOutputView<'_, T>, i32) -> Result<()>;

type ScaledConstantSignalToOperation<T> =
    fn(&StreamContext, &SignalView<'_, T>, T, &mut SignalOutputView<'_, T>, i32) -> Result<()>;

impl SignalPipeline<'_, f32> {
    /// Multiplies each source element by `value`. Writes the result to `destination`.
    /// Converts the floating-point product to `i16` with the selected native routine.
    /// Uses the native low conversion routine.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn multiply_constant_f32_to_i16_low_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        value: f32,
        destination: &mut SignalOutputView<'_, i16>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_constant_f32_to_i16_low(stream_context, source, value, destination)
        })
    }

    /// Multiplies each source element by `value`. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    /// Converts the floating-point product to `i16` with the selected native routine.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn multiply_constant_f32_to_i16_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        value: f32,
        destination: &mut SignalOutputView<'_, i16>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_constant_f32_to_i16_scaled(
                stream_context,
                source,
                value,
                destination,
                scale_factor,
            )
        })
    }
}

impl SignalPipeline<'_, f64> {
    /// Multiplies each signal element by `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    /// Uses a `f64` multiplier for the `i64` signal.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn multiply_constant_f64_to_i64_scaled_in_place(
        stream_context: &StreamContext,
        destination: &mut SignalViewMut<'_, i64>,
        value: f64,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_constant_f64_to_i64_scaled_in_place(
                stream_context,
                destination,
                value,
                scale_factor,
            )
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    /// Multiplies each source element by `value`. Writes the result to `destination`.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    /// Uses the native low conversion routine.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn multiply_constant_low_to_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantLowTo<U>,
    {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_constant_low_to(stream_context, source, value, destination)
        })
    }

    /// Multiplies each source element by `value`. Writes the result to `destination`.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    /// Uses the native low conversion routine.
    /// Equivalent to [`Self::multiply_constant_low_to_into`].
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn multiply_constant_to_low_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantLowTo<U>,
    {
        Self::multiply_constant_low_to_into(stream_context, source, value, destination)
    }

    /// Multiplies each source element by `value`. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn multiply_constant_scaled_to_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledTo<U>,
    {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_constant_scaled_to(
                stream_context,
                source,
                value,
                destination,
                scale_factor,
            )
        })
    }

    /// Multiplies each source element by `value`. Writes the result to `destination`.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    /// Equivalent to [`Self::multiply_constant_scaled_to_into`].
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn multiply_constant_to_scaled_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledTo<U>,
    {
        Self::multiply_constant_scaled_to_into(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    /// Multiplies each signal element by `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn multiply_constant_scaled_to_in_place<Value>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: Value,
        scale_factor: i32,
    ) -> Result<()>
    where
        Value: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledToInPlace<Value>,
    {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(signal)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_constant_scaled_to_in_place(
                stream_context,
                signal,
                value,
                scale_factor,
            )
        })
    }

    /// Multiplies each signal element by `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    /// Equivalent to [`Self::multiply_constant_scaled_to_in_place`].
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn multiply_constant_to_scaled_in_place<Value>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: Value,
        scale_factor: i32,
    ) -> Result<()>
    where
        Value: DataTypeLike,
        T: arithmetic::MultiplyConstantScaledToInPlace<Value>,
    {
        Self::multiply_constant_scaled_to_in_place(stream_context, signal, value, scale_factor)
    }

    /// Multiplies each source element by `value`. Returns an owned pipeline containing the result.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    /// Uses the native low conversion routine.
    ///
    /// Allocates destination storage through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn multiply_constant_low_to<U>(self, value: T) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::MultiplyConstantLowTo<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    arithmetic::multiply_constant_low_to(
                        self.stream_context,
                        &source,
                        value,
                        &mut destination_view,
                    )
                }
            })?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Multiplies each source element by `value`. Returns an owned pipeline containing the result.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    /// Uses the source and destination representations selected by the arithmetic dispatch trait.
    ///
    /// Allocates destination storage through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn multiply_constant_scaled_to<U>(
        self,
        value: T,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::MultiplyConstantScaledTo<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    arithmetic::multiply_constant_scaled_to(
                        self.stream_context,
                        &source,
                        value,
                        &mut destination_view,
                        scale_factor,
                    )
                }
            })?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

mod scaled_constant_traits;

use self::scaled_constant_traits::{
    ConstantBitwiseSignal, ConstantShiftSignal, DivideScaledConstantSignal, ScaledConstantSignal,
};

#[macro_use]
mod scaled_constant_macros;

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledConstantSignal<T>,
{
    /// Adds `value` to each source element. Returns an owned pipeline containing the result.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn add_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled_in_place,
        )
    }

    /// Subtracts `value` from each source element. Returns an owned pipeline containing the result.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn subtract_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled_in_place,
        )
    }

    /// Subtracts each source element from `value`. Returns an owned pipeline containing the result.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn subtract_from_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled_in_place,
        )
    }

    /// Multiplies each source element by `value`. Returns an owned pipeline containing the result.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn multiply_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_constant(
            value,
            scale_factor,
            <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled,
            <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled_in_place,
        )
    }

    fn scaled_constant(
        mut self,
        value: T,
        scale_factor: i32,
        operation: ScaledConstantSignalOperation<T>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T, i32) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, &mut signal_view, value, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: ScaledConstantSignal<T>,
{
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
    pub fn add_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    /// Adds `value` to each signal element. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn add_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::add_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }

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
    pub fn subtract_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    /// Subtracts `value` from each signal element. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn subtract_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }

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
    pub fn subtract_from_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    /// Subtracts each signal element from `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn subtract_from_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::subtract_from_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }

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
    pub fn multiply_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    /// Multiplies each signal element by `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn multiply_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledConstantSignal<T>>::multiply_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ConstantBitwiseSignal<T>,
{
    /// Computes the bitwise AND of each source element and `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn bit_and_constant(self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        self.constant_bitwise(
            value,
            <Self as ConstantBitwiseSignal<T>>::and_constant_signal,
            <Self as ConstantBitwiseSignal<T>>::and_constant_signal_in_place,
        )
    }

    /// Computes the bitwise OR of each source element and `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn bit_or_constant(self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        self.constant_bitwise(
            value,
            <Self as ConstantBitwiseSignal<T>>::or_constant_signal,
            <Self as ConstantBitwiseSignal<T>>::or_constant_signal_in_place,
        )
    }

    /// Computes the bitwise XOR of each source element and `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn bit_xor_constant(self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        self.constant_bitwise(
            value,
            <Self as ConstantBitwiseSignal<T>>::xor_constant_signal,
            <Self as ConstantBitwiseSignal<T>>::xor_constant_signal_in_place,
        )
    }

    fn constant_bitwise(
        mut self,
        value: T,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            T,
            &mut SignalOutputView<'_, T>,
        ) -> Result<()>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, &mut signal_view, value)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(self.stream_context, source, value, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: ConstantBitwiseSignal<T>,
{
    /// Computes the bitwise AND of each source element and `value`. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn bit_and_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::and_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Computes the bitwise AND of each signal element and `value`. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn bit_and_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::and_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    /// Computes the bitwise OR of each source element and `value`. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn bit_or_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::or_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Computes the bitwise OR of each signal element and `value`. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn bit_or_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::or_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    /// Computes the bitwise XOR of each source element and `value`. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn bit_xor_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::xor_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Computes the bitwise XOR of each signal element and `value`. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn bit_xor_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantBitwiseSignal<T>>::xor_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ConstantShiftSignal<T>,
{
    /// Shifts each source element left by `shift` bits. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn left_shift_constant(self, shift: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.constant_shift(
            shift,
            <Self as ConstantShiftSignal<T>>::left_shift_constant_signal,
            <Self as ConstantShiftSignal<T>>::left_shift_constant_signal_in_place,
        )
    }

    /// Shifts each source element right by `shift` bits. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn right_shift_constant(self, shift: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.constant_shift(
            shift,
            <Self as ConstantShiftSignal<T>>::right_shift_constant_signal,
            <Self as ConstantShiftSignal<T>>::right_shift_constant_signal_in_place,
        )
    }

    fn constant_shift(
        mut self,
        shift: i32,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            i32,
            &mut SignalOutputView<'_, T>,
        ) -> Result<()>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, i32) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, &mut signal_view, shift)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(self.stream_context, source, shift, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: ConstantShiftSignal<T>,
{
    /// Shifts each source element left by `shift` bits. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn left_shift_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::left_shift_constant_signal(
            stream_context,
            source,
            shift,
            destination,
        )
    }

    /// Shifts each signal element left by `shift` bits. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn left_shift_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::left_shift_constant_signal_in_place(
            stream_context,
            signal,
            shift,
        )
    }

    /// Shifts each source element right by `shift` bits. Writes the result to `destination`.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn right_shift_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        shift: i32,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::right_shift_constant_signal(
            stream_context,
            source,
            shift,
            destination,
        )
    }

    /// Shifts each signal element right by `shift` bits. Updates the signal in place.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn right_shift_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        shift: i32,
    ) -> Result<()> {
        <Self as ConstantShiftSignal<T>>::right_shift_constant_signal_in_place(
            stream_context,
            signal,
            shift,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: DivideScaledConstantSignal<T>,
{
    /// Divides each source element by `value`. Returns an owned pipeline containing the result.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn divide_constant_scaled(self, value: T, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.divide_scaled_constant(
            value,
            scale_factor,
            <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled,
            <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled_in_place,
        )
    }

    fn divide_scaled_constant(
        mut self,
        value: T,
        scale_factor: i32,
        operation: ScaledConstantSignalToOperation<T>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T, i32) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, &mut signal_view, value, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: DivideScaledConstantSignal<T>,
{
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
    pub fn divide_constant_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled(
            stream_context,
            source,
            value,
            destination,
            scale_factor,
        )
    }

    /// Divides each signal element by `value`. Updates the signal in place.
    /// Applies native fixed-point scaling with `scale_factor` and saturation for the destination type.
    ///
    /// Prepares the participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if the signal length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn divide_constant_scaled_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DivideScaledConstantSignal<T>>::divide_constant_signal_scaled_in_place(
            stream_context,
            signal,
            value,
            scale_factor,
        )
    }
}
