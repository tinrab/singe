use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{
            SignalPipeline,
            statistics::{ErrorMetricStatisticSignal, ZeroCrossingStatisticSignal},
        },
        workspace::with_temporary_scratch,
    },
    signal::{
        memory::Signal,
        view::{SignalOutputView, SignalView},
    },
    types::ZeroCrossingType,
    workspace::ScratchBuffer,
};

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<f64>,
    Self: ErrorMetricStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the largest absolute difference between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn maximum_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_error_buffer_size(stream_context, source)
    }

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
    pub fn maximum_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the largest absolute difference between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    /// For complex inputs, NPP uses the complex magnitude of the difference.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn maximum_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_with_scratch,
        )
    }

    /// Computes the largest absolute difference between corresponding input samples.
    ///
    /// Returns an owned one-element `Signal<f64>`.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    /// For complex inputs, NPP uses the complex magnitude of the difference.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn maximum_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.ensure_eager()?;
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_error_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the mean absolute difference between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn average_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::average_error_buffer_size(stream_context, source)
    }

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
    pub fn average_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::average_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the mean absolute difference between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    /// For complex inputs, NPP uses the complex magnitude of the difference.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn average_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_with_scratch,
        )
    }

    /// Computes the mean absolute difference between corresponding input samples.
    ///
    /// Returns an owned one-element `Signal<f64>`.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    /// For complex inputs, NPP uses the complex magnitude of the difference.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn average_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.ensure_eager()?;
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_error_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the maximum relative error between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn maximum_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_buffer_size(
            stream_context,
            source,
        )
    }

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
    pub fn maximum_relative_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the maximum relative error between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    /// Each relative difference divides the absolute difference by the larger input magnitude.
    /// Zero-denominator behavior is left to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn maximum_relative_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_with_scratch,
        )
    }

    /// Computes the maximum relative error between corresponding input samples.
    ///
    /// Returns an owned one-element `Signal<f64>`.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    /// Each relative difference divides the absolute difference by the larger input magnitude.
    /// Zero-denominator behavior is left to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn maximum_relative_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.ensure_eager()?;
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::maximum_relative_error_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the average relative error between corresponding input samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn average_relative_error_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_buffer_size(
            stream_context,
            source,
        )
    }

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
    pub fn average_relative_error_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the average relative error between corresponding input samples.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    /// Each relative difference divides the absolute difference by the larger input magnitude.
    /// Zero-denominator behavior is left to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn average_relative_error_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
    ) -> Result<()> {
        Self::error_metric_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_with_scratch,
        )
    }

    /// Computes the average relative error between corresponding input samples.
    ///
    /// Returns an owned one-element `Signal<f64>`.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    /// Each relative difference divides the absolute difference by the larger input magnitude.
    /// Zero-denominator behavior is left to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn average_relative_error(self, other: &SignalView<'_, T>) -> Result<Signal<f64>> {
        self.ensure_eager()?;
        self.error_metric_statistic(
            other,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_buffer_size,
            <Self as ErrorMetricStatisticSignal<T>>::average_relative_error_with_scratch,
        )
    }

    fn error_metric_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f64>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: ErrorMetricStatisticIntoCallback<T>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(other)?;
            stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe { statistic(stream_context, source, other, destination, scratch) }
            })
        })
    }

    fn error_metric_statistic(
        self,
        other: &SignalView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: ErrorMetricStatisticIntoCallback<T>,
    ) -> Result<Signal<f64>> {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<f64>(self.stream_context.cuda_context(), 1)?;
        let required_bytes = {
            let source = self.view();
            buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(other)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe {
                    statistic(
                        self.stream_context,
                        &source,
                        other,
                        &mut destination_view,
                        &mut scratch,
                    )
                }
            })
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        // SAFETY: the completed operation above wrote every element.
        Ok(unsafe { destination.assume_init() })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<f32>,
    Self: ZeroCrossingStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the zero-crossing computation.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn zero_crossing_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_buffer_size(stream_context, source)
    }

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
    pub fn zero_crossing_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f32>,
        zero_crossing_type: ZeroCrossingType,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_with_scratch(
            stream_context,
            source,
            destination,
            zero_crossing_type,
            scratch,
        )
    }

    /// Computes the zero-crossing statistic selected by `zero_crossing_type`.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    /// `zero_crossing_type` selects the native sign-change rule.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn zero_crossing_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, f32>,
        zero_crossing_type: ZeroCrossingType,
    ) -> Result<()> {
        let required_bytes = <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_buffer_size(
            stream_context,
            source,
        )?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_with_scratch(
                stream_context,
                source,
                destination,
                zero_crossing_type,
                scratch,
            )
        })
    }

    /// Computes the zero-crossing statistic selected by `zero_crossing_type`.
    ///
    /// Returns an owned one-element `Signal<f32>`.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    /// `zero_crossing_type` selects the native sign-change rule.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn zero_crossing(self, zero_crossing_type: ZeroCrossingType) -> Result<Signal<f32>> {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<f32>(self.stream_context.cuda_context(), 1)?;
        let required_bytes = {
            let source = self.view();
            <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_buffer_size(
                self.stream_context,
                &source,
            )?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut destination_view = destination.output();
            <Self as ZeroCrossingStatisticSignal<T>>::zero_crossing_with_scratch(
                self.stream_context,
                &source,
                &mut destination_view,
                zero_crossing_type,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        // SAFETY: the completed operation above wrote every element.
        Ok(unsafe { destination.assume_init() })
    }
}

type ErrorMetricStatisticIntoCallback<T> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, f64>,
    &mut ScratchBuffer,
) -> Result<()>;
