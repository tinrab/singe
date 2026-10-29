use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{
            SignalBacking, SignalPipeline,
            statistics::{
                AbsoluteExtremumStatisticSignal, AbsoluteIndexedStatisticSignal,
                ExtremumStatisticSignal, IndexedStatisticSignal,
                MeanStandardDeviationStatisticSignal, PairStatisticSignal, SignalIndexedValue,
                SignalMeanStandardDeviation, SignalMinMax,
            },
        },
        workspace::with_temporary_scratch,
    },
    signal::view::{SignalOutputView, SignalView},
    workspace::ScratchBuffer,
};

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ExtremumStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the maximum source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn max_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ExtremumStatisticSignal<T>>::max_buffer_size(stream_context, source)
    }

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
    pub fn max_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ExtremumStatisticSignal<T>>::max_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the maximum source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn max_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ExtremumStatisticSignal<T>>::max_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::max_with_scratch,
        )
    }

    /// Computes the maximum source value.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn max(self) -> Result<Self> {
        self.ensure_eager()?;
        self.extremum_statistic(
            <Self as ExtremumStatisticSignal<T>>::max_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::max_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the minimum source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn min_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ExtremumStatisticSignal<T>>::min_buffer_size(stream_context, source)
    }

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
    pub fn min_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ExtremumStatisticSignal<T>>::min_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the minimum source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn min_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ExtremumStatisticSignal<T>>::min_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::min_with_scratch,
        )
    }

    /// Computes the minimum source value.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn min(self) -> Result<Self> {
        self.ensure_eager()?;
        self.extremum_statistic(
            <Self as ExtremumStatisticSignal<T>>::min_buffer_size,
            <Self as ExtremumStatisticSignal<T>>::min_with_scratch,
        )
    }

    fn extremum_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: unsafe fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination)?;
            stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe { statistic(stream_context, source, destination, scratch) }
            })
        })
    }

    fn extremum_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: unsafe fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
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
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe {
                    statistic(
                        self.stream_context,
                        &source,
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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: AbsoluteExtremumStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the maximum absolute source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn max_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_buffer_size(
            stream_context,
            source,
        )
    }

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
    pub fn max_absolute_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the maximum absolute source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn max_absolute_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::absolute_extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_with_scratch,
        )
    }

    /// Computes the maximum absolute source value.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn max_absolute(self) -> Result<Self> {
        self.ensure_eager()?;
        self.absolute_extremum_statistic(
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::max_absolute_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the minimum absolute source value.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn min_absolute_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_buffer_size(
            stream_context,
            source,
        )
    }

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
    pub fn min_absolute_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the minimum absolute source value.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn min_absolute_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::absolute_extremum_statistic_into(
            stream_context,
            source,
            destination,
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_with_scratch,
        )
    }

    /// Computes the minimum absolute source value.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn min_absolute(self) -> Result<Self> {
        self.ensure_eager()?;
        self.absolute_extremum_statistic(
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_buffer_size,
            <Self as AbsoluteExtremumStatisticSignal<T>>::min_absolute_with_scratch,
        )
    }

    fn absolute_extremum_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: unsafe fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination)?;
            stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe { statistic(stream_context, source, destination, scratch) }
            })
        })
    }

    fn absolute_extremum_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: unsafe fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, T>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
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
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe {
                    statistic(
                        self.stream_context,
                        &source,
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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: PairStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the minimum and maximum source values.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn min_max_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as PairStatisticSignal<T>>::min_max_buffer_size(stream_context, source)
    }

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
    pub fn min_max_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        min: &mut SignalOutputView<'_, T>,
        max: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as PairStatisticSignal<T>>::min_max_with_scratch(
            stream_context,
            source,
            min,
            max,
            scratch,
        )
    }

    /// Computes the minimum and maximum source values.
    ///
    /// Writes one element to each output view: `min`, `max`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn min_max_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        min: &mut SignalOutputView<'_, T>,
        max: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        let required_bytes =
            <Self as PairStatisticSignal<T>>::min_max_buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            <Self as PairStatisticSignal<T>>::min_max_with_scratch(
                stream_context,
                source,
                min,
                max,
                scratch,
            )
        })
    }

    /// Computes the minimum and maximum source values.
    ///
    /// Returns owned one-element `min` and `max` device signals.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn min_max(self) -> Result<SignalMinMax<T>> {
        self.ensure_eager()?;
        let mut min = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
        let mut max = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
        let required_bytes = {
            let source = self.view();
            <Self as PairStatisticSignal<T>>::min_max_buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut min_view = min.output();
            let mut max_view = max.output();
            <Self as PairStatisticSignal<T>>::min_max_with_scratch(
                self.stream_context,
                &source,
                &mut min_view,
                &mut max_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalMinMax {
            // SAFETY: the completed operation above wrote every element.
            min: unsafe { min.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            max: unsafe { max.assume_init() },
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: MeanStandardDeviationStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the arithmetic mean and standard deviation of the source samples.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn mean_standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_buffer_size(
            stream_context,
            source,
        )
    }

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
    pub fn mean_standard_deviation_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        mean: &mut SignalOutputView<'_, T>,
        standard_deviation: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_with_scratch(
            stream_context,
            source,
            mean,
            standard_deviation,
            scratch,
        )
    }

    /// Computes the arithmetic mean and standard deviation of the source samples.
    ///
    /// Writes one element to each output view: `mean`, `standard_deviation`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn mean_standard_deviation_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        mean: &mut SignalOutputView<'_, T>,
        standard_deviation: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        let required_bytes =
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_buffer_size(
                stream_context,
                source,
            )?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_with_scratch(
                stream_context,
                source,
                mean,
                standard_deviation,
                scratch,
            )
        })
    }

    /// Computes the arithmetic mean and standard deviation of the source samples.
    ///
    /// Returns owned one-element `mean` and `standard_deviation` device signals.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn mean_standard_deviation(self) -> Result<SignalMeanStandardDeviation<T>> {
        self.ensure_eager()?;
        let mut mean = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
        let mut standard_deviation = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
        let required_bytes = {
            let source = self.view();
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_buffer_size(
                self.stream_context,
                &source,
            )?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut mean_view = mean.output();
            let mut standard_deviation_view = standard_deviation.output();
            <Self as MeanStandardDeviationStatisticSignal<T>>::mean_standard_deviation_with_scratch(
                self.stream_context,
                &source,
                &mut mean_view,
                &mut standard_deviation_view,
                &mut scratch,
            )
        };

        let recycle_result = self
            .workspace
            .recycle_scratch_after(self.stream_context, scratch);
        operation_result?;
        recycle_result?;

        Ok(SignalMeanStandardDeviation {
            // SAFETY: the completed operation above wrote every element.
            mean: unsafe { mean.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            standard_deviation: unsafe { standard_deviation.assume_init() },
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T> + SignalAllocator<i32>,
    Self: IndexedStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the maximum source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn max_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as IndexedStatisticSignal<T>>::max_index_buffer_size(stream_context, source)
    }

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
    pub fn max_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as IndexedStatisticSignal<T>>::max_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    /// Computes the maximum source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn max_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
    ) -> Result<()> {
        Self::indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as IndexedStatisticSignal<T>>::max_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::max_index_with_scratch,
        )
    }

    /// Computes the maximum source value and the native index of its first occurrence.
    ///
    /// Returns the result in an owned one-element `value` signal and its native source index in an owned one-element `index` signal.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn max_index(self) -> Result<SignalIndexedValue<T>> {
        self.ensure_eager()?;
        self.indexed_statistic(
            <Self as IndexedStatisticSignal<T>>::max_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::max_index_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the minimum source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn min_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as IndexedStatisticSignal<T>>::min_index_buffer_size(stream_context, source)
    }

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
    pub fn min_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as IndexedStatisticSignal<T>>::min_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    /// Computes the minimum source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn min_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
    ) -> Result<()> {
        Self::indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as IndexedStatisticSignal<T>>::min_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::min_index_with_scratch,
        )
    }

    /// Computes the minimum source value and the native index of its first occurrence.
    ///
    /// Returns the result in an owned one-element `value` signal and its native source index in an owned one-element `index` signal.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn min_index(self) -> Result<SignalIndexedValue<T>> {
        self.ensure_eager()?;
        self.indexed_statistic(
            <Self as IndexedStatisticSignal<T>>::min_index_buffer_size,
            <Self as IndexedStatisticSignal<T>>::min_index_with_scratch,
        )
    }

    fn indexed_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: IndexedStatisticIntoCallback<T>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(value)?;
            stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe { statistic(stream_context, source, value, index, scratch) }
            })
        })
    }

    fn indexed_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: IndexedStatisticIntoCallback<T>,
    ) -> Result<SignalIndexedValue<T>> {
        self.ensure_eager()?;
        let mut value = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
        let mut index = self
            .workspace
            .signal_uninit::<i32>(self.stream_context.cuda_context(), 1)?;
        let required_bytes = {
            let source = self.view();
            buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut value_view = value.output();
            let mut index_view = index.output();
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&value_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe {
                    statistic(
                        self.stream_context,
                        &source,
                        &mut value_view,
                        &mut index_view,
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

        Ok(SignalIndexedValue {
            // SAFETY: the completed operation above wrote every element.
            value: unsafe { value.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            index: unsafe { index.assume_init() },
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T> + SignalAllocator<i32>,
    Self: AbsoluteIndexedStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the maximum absolute source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn max_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_buffer_size(
            stream_context,
            source,
        )
    }

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
    pub fn max_absolute_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    /// Computes the maximum absolute source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn max_absolute_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
    ) -> Result<()> {
        Self::absolute_indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_with_scratch,
        )
    }

    /// Computes the maximum absolute source value and the native index of its first occurrence.
    ///
    /// Returns the result in an owned one-element `value` signal and its native source index in an owned one-element `index` signal.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn max_absolute_index(self) -> Result<SignalIndexedValue<T>> {
        self.ensure_eager()?;
        self.absolute_indexed_statistic(
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::max_absolute_index_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the minimum absolute source value and the native index of its first occurrence.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn min_absolute_index_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_buffer_size(
            stream_context,
            source,
        )
    }

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
    pub fn min_absolute_index_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_with_scratch(
            stream_context,
            source,
            value,
            index,
            scratch,
        )
    }

    /// Computes the minimum absolute source value and the native index of its first occurrence.
    ///
    /// Writes one element to each output view: `value`, `index`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn min_absolute_index_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
    ) -> Result<()> {
        Self::absolute_indexed_statistic_into(
            stream_context,
            source,
            value,
            index,
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_with_scratch,
        )
    }

    /// Computes the minimum absolute source value and the native index of its first occurrence.
    ///
    /// Returns the result in an owned one-element `value` signal and its native source index in an owned one-element `index` signal.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn min_absolute_index(self) -> Result<SignalIndexedValue<T>> {
        self.ensure_eager()?;
        self.absolute_indexed_statistic(
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_buffer_size,
            <Self as AbsoluteIndexedStatisticSignal<T>>::min_absolute_index_with_scratch,
        )
    }

    fn absolute_indexed_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: &mut SignalOutputView<'_, T>,
        index: &mut SignalOutputView<'_, i32>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: IndexedStatisticIntoCallback<T>,
    ) -> Result<()> {
        let required_bytes = buffer_size(stream_context, source)?;
        with_temporary_scratch(stream_context, required_bytes, |scratch| {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(value)?;
            stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe { statistic(stream_context, source, value, index, scratch) }
            })
        })
    }

    fn absolute_indexed_statistic(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: IndexedStatisticIntoCallback<T>,
    ) -> Result<SignalIndexedValue<T>> {
        self.ensure_eager()?;
        let mut value = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), 1)?;
        let mut index = self
            .workspace
            .signal_uninit::<i32>(self.stream_context.cuda_context(), 1)?;
        let required_bytes = {
            let source = self.view();
            buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut value_view = value.output();
            let mut index_view = index.output();
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&value_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                unsafe {
                    statistic(
                        self.stream_context,
                        &source,
                        &mut value_view,
                        &mut index_view,
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

        Ok(SignalIndexedValue {
            // SAFETY: the completed operation above wrote every element.
            value: unsafe { value.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            index: unsafe { index.assume_init() },
        })
    }
}

type IndexedStatisticIntoCallback<T> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, T>,
    &mut SignalOutputView<'_, i32>,
    &mut ScratchBuffer,
) -> Result<()>;
