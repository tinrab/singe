use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        signal::{SignalBacking, SignalPipeline, statistics_methods::ScalarStatisticSignal},
        workspace::{SignalAllocator, Workspace, with_temporary_scratch},
    },
    signal::view::{SignalOutputView, SignalView},
    workspace::ScratchBuffer,
};

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScalarStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required to compute the sum of the source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn sum_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::sum_buffer_size(stream_context, source)
    }

    /// Computes the sum of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion before returning.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn sum_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::sum_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the sum of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Allocates temporary scratch and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn sum_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::sum_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_with_scratch,
        )
    }

    /// Computes the sum of the source samples.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn sum(self) -> Result<Self> {
        self.ensure_eager()?;
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::sum_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the sum of the natural logarithms of the source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn sum_natural_logarithm_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_buffer_size(
            stream_context,
            source,
        )
    }

    /// Computes the sum of the natural logarithms of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion before returning.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn sum_natural_logarithm_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the sum of the natural logarithms of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Allocates temporary scratch and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn sum_natural_logarithm_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_with_scratch,
        )
    }

    /// Computes the sum of the natural logarithms of the source samples.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn sum_natural_logarithm(self) -> Result<Self> {
        self.ensure_eager()?;
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_buffer_size,
            <Self as ScalarStatisticSignal<T>>::sum_natural_logarithm_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the arithmetic mean of the source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn mean_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::mean_buffer_size(stream_context, source)
    }

    /// Computes the arithmetic mean of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion before returning.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn mean_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::mean_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the arithmetic mean of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Allocates temporary scratch and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn mean_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::mean_buffer_size,
            <Self as ScalarStatisticSignal<T>>::mean_with_scratch,
        )
    }

    /// Computes the arithmetic mean of the source samples.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn mean(self) -> Result<Self> {
        self.ensure_eager()?;
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::mean_buffer_size,
            <Self as ScalarStatisticSignal<T>>::mean_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the standard deviation of the source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn standard_deviation_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::standard_deviation_buffer_size(stream_context, source)
    }

    /// Computes the standard deviation of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion before returning.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn standard_deviation_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::standard_deviation_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the standard deviation of the source samples.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Allocates temporary scratch and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn standard_deviation_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::standard_deviation_buffer_size,
            <Self as ScalarStatisticSignal<T>>::standard_deviation_with_scratch,
        )
    }

    /// Computes the standard deviation of the source samples.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn standard_deviation(self) -> Result<Self> {
        self.ensure_eager()?;
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::standard_deviation_buffer_size,
            <Self as ScalarStatisticSignal<T>>::standard_deviation_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the infinity norm of the source signal.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn norm_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::norm_inf_buffer_size(stream_context, source)
    }

    /// Computes the infinity norm of the source signal.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion before returning.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn norm_inf_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::norm_inf_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the infinity norm of the source signal.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Allocates temporary scratch and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn norm_inf_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::norm_inf_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_inf_with_scratch,
        )
    }

    /// Computes the infinity norm of the source signal.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn norm_inf(self) -> Result<Self> {
        self.ensure_eager()?;
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::norm_inf_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_inf_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the L1 norm of the source signal.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn norm_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::norm_l1_buffer_size(stream_context, source)
    }

    /// Computes the L1 norm of the source signal.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion before returning.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn norm_l1_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::norm_l1_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the L1 norm of the source signal.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Allocates temporary scratch and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn norm_l1_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::norm_l1_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l1_with_scratch,
        )
    }

    /// Computes the L1 norm of the source signal.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn norm_l1(self) -> Result<Self> {
        self.ensure_eager()?;
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::norm_l1_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l1_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the L2 norm of the source signal.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn norm_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as ScalarStatisticSignal<T>>::norm_l2_buffer_size(stream_context, source)
    }

    /// Computes the L2 norm of the source signal.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Uses caller-provided scratch and waits for successful stream completion before returning.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn norm_l2_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as ScalarStatisticSignal<T>>::norm_l2_with_scratch(
            stream_context,
            source,
            destination,
            scratch,
        )
    }

    /// Computes the L2 norm of the source signal.
    ///
    /// Writes the scalar result into the single element of `destination`.
    /// Allocates temporary scratch and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn norm_l2_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::scalar_statistic_into(
            stream_context,
            source,
            destination,
            <Self as ScalarStatisticSignal<T>>::norm_l2_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l2_with_scratch,
        )
    }

    /// Computes the L2 norm of the source signal.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn norm_l2(self) -> Result<Self> {
        self.ensure_eager()?;
        self.scalar_statistic(
            <Self as ScalarStatisticSignal<T>>::norm_l2_buffer_size,
            <Self as ScalarStatisticSignal<T>>::norm_l2_with_scratch,
        )
    }

    fn scalar_statistic_into(
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

    fn scalar_statistic(
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

macro_rules! impl_statistic_output_pipeline_methods {
    (
        $doc:literal,
        $source_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        /// Returns the required scratch capacity in bytes for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
        #[doc = concat!("Uses [`", stringify!($npp_buffer_size), "`].")]
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($npp_with_scratch), "`].")]
        /// Writes the scalar result into the single element of `destination`.
        /// Uses caller-provided scratch and waits for successful stream completion before returning.
        /// The scratch capacity is checked against the native query.
        /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
        ///
        /// # Errors
        ///
        /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
        /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
        /// Returns an error if `destination` does not contain exactly one element or scratch capacity is insufficient.
        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $npp_with_scratch(stream_context, source, destination, scratch)
            })
        }
    };
}

macro_rules! impl_binary_statistic_output_pipeline_methods {
    (
        $doc:literal,
        $source_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        /// Returns the required scratch capacity in bytes for the source length.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
        #[doc = concat!("Uses [`", stringify!($npp_buffer_size), "`].")]
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($npp_with_scratch), "`].")]
        /// Writes the scalar result into the single element of `destination`.
        /// Uses caller-provided scratch and waits for successful stream completion before returning.
        /// The scratch capacity is checked against the native query.
        /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
        ///
        /// # Errors
        ///
        /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
        /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
        /// Returns an error if `destination` does not contain exactly one element or scratch capacity is insufficient.
        /// The two source lengths must match.
        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source1: &SignalView<'_, $source_ty>,
            source2: &SignalView<'_, $source_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source1)?;
            stream_context.prepare_region(source2)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $npp_with_scratch(stream_context, source1, source2, destination, scratch)
            })
        }
    };
}
