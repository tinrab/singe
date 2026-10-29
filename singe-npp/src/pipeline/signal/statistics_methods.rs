use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{
            SignalBacking, SignalPipeline,
            statistics::{
                BinaryScalarStatisticSignal, EveryStatisticSignal, ScalarStatisticSignal,
            },
        },
        workspace::with_temporary_scratch,
    },
    signal::{
        statistics,
        view::{SignalOutputView, SignalView},
    },
    workspace::ScratchBuffer,
};

#[macro_use]
mod statistics_scalar_methods;

mod statistics_float_complex_methods;

mod statistics_integer_methods;

impl SignalPipeline<'_, i32>
where
    Workspace: SignalAllocator<i32>,
{
    /// Returns the scratch capacity in bytes required for the range count.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn count_in_range_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
    ) -> Result<usize> {
        statistics::count_in_range_buffer_size(stream_context, source)
    }

    /// Computes the number of source samples in the range selected by `lower_bound` and `upper_bound`.
    ///
    /// Writes one element to each output view: `count`.
    /// Allocates temporary scratch and waits for successful stream completion.
    /// Passes both range bounds to NPP without checking their order.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn count_in_range_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        count: &mut SignalOutputView<'_, i32>,
        lower_bound: i32,
        upper_bound: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(count)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::count_in_range_to_device(
                stream_context,
                source,
                count,
                lower_bound,
                upper_bound,
            )
        })
    }

    /// Computes the number of source samples in the range selected by `lower_bound` and `upper_bound`.
    ///
    /// Writes one element to each output view: `count`.
    /// Uses caller-provided scratch and waits for successful stream completion.
    /// The scratch capacity is checked against the native query.
    /// `scratch` must belong to the stream context; this path does not check its CUDA context.
    /// Passes both range bounds to NPP without checking their order.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Every output view must contain exactly one element.
    /// Insufficient scratch capacity returns an error.
    pub fn count_in_range_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        count: &mut SignalOutputView<'_, i32>,
        lower_bound: i32,
        upper_bound: i32,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(count)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::count_in_range_to_device_with_scratch(
                stream_context,
                source,
                count,
                lower_bound,
                upper_bound,
                scratch,
            )
        })
    }

    /// Computes the number of source samples in the range selected by `lower_bound` and `upper_bound`.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    /// Passes both range bounds to NPP without checking their order.
    /// The source must already be ready on the selected stream; this path does not synchronize its producer.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn count_in_range(self, lower_bound: i32, upper_bound: i32) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<i32>(self.stream_context.cuda_context(), 1)?;
        let required_bytes = {
            let source = self.view();
            statistics::count_in_range_buffer_size(self.stream_context, &source)?
        };
        let mut scratch = self
            .workspace
            .scratch(self.stream_context.cuda_context(), required_bytes)?;

        let operation_result = {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    statistics::count_in_range_to_device_with_scratch(
                        self.stream_context,
                        &source,
                        &mut destination_view,
                        lower_bound,
                        upper_bound,
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
    Self: BinaryScalarStatisticSignal<T>,
{
    /// Returns the scratch capacity in bytes required for the infinity norm of the difference between the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn norm_diff_inf_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_buffer_size(stream_context, source)
    }

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
    pub fn norm_diff_inf_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the infinity norm of the difference between the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn norm_diff_inf_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_with_scratch,
        )
    }

    /// Computes the infinity norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_inf(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_inf_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the L1 norm of the difference between the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn norm_diff_l1_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_buffer_size(stream_context, source)
    }

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
    pub fn norm_diff_l1_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the L1 norm of the difference between the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn norm_diff_l1_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_with_scratch,
        )
    }

    /// Computes the L1 norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_l1(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l1_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the L2 norm of the difference between the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn norm_diff_l2_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_buffer_size(stream_context, source)
    }

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
    pub fn norm_diff_l2_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the L2 norm of the difference between the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn norm_diff_l2_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_with_scratch,
        )
    }

    /// Computes the L2 norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_l2(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::norm_diff_l2_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required for the dot product of the input signals.
    ///
    /// The query uses the source length and does not read source samples.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source-length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn dot_product_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize> {
        <Self as BinaryScalarStatisticSignal<T>>::dot_product_buffer_size(stream_context, source)
    }

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
    pub fn dot_product_with_scratch_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        <Self as BinaryScalarStatisticSignal<T>>::dot_product_with_scratch(
            stream_context,
            source,
            other,
            destination,
            scratch,
        )
    }

    /// Computes the dot product of the input signals.
    ///
    /// Writes one element to each output view: `destination`.
    /// Allocates temporary scratch and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Every output view must contain exactly one element.
    /// Scratch allocation can also fail.
    pub fn dot_product_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        Self::binary_scalar_statistic_into(
            stream_context,
            source,
            other,
            destination,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_with_scratch,
        )
    }

    /// Computes the dot product of the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn dot_product(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary_scalar_statistic(
            other,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_buffer_size,
            <Self as BinaryScalarStatisticSignal<T>>::dot_product_with_scratch,
        )
    }

    fn binary_scalar_statistic_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        other: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: BinaryScalarStatisticIntoCallback<T>,
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

    fn binary_scalar_statistic(
        self,
        other: &SignalView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: BinaryScalarStatisticIntoCallback<T>,
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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

type BinaryScalarStatisticIntoCallback<T> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, T>,
    &mut ScratchBuffer,
) -> Result<()>;
