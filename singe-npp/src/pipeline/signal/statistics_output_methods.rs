use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{
        statistics,
        view::{SignalOutputView, SignalView},
    },
    types::DataTypeLike,
    workspace::ScratchBuffer,
};

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
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
    pub fn norm_inf_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormInfTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_inf_buffer_size::<T, U>,
            statistics::norm_inf_to_device_with_scratch::<T, U>,
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
    pub fn norm_l1_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL1To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_l1_buffer_size::<T, U>,
            statistics::norm_l1_to_device_with_scratch::<T, U>,
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
    pub fn norm_l2_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL2To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_l2_buffer_size::<T, U>,
            statistics::norm_l2_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the sum of the source samples.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn sum_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::SumScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_scaled_buffer_size::<T, U>,
            statistics::sum_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the sum of the source samples.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn sum_to_complex_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::SumScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.sum_to_scaled::<U>(scale_factor)
    }

    /// Computes the standard deviation of the source samples.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn standard_deviation_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::StandardDeviationScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::standard_deviation_scaled_buffer_size::<T, U>,
            statistics::standard_deviation_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the infinity norm of the source signal.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn norm_inf_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormInfScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_inf_scaled_buffer_size::<T, U>,
            statistics::norm_inf_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the L1 norm of the source signal.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn norm_l1_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL1ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l1_scaled_buffer_size::<T, U>,
            statistics::norm_l1_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the L2 norm of the source signal.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn norm_l2_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL2ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_scaled_buffer_size::<T, U>,
            statistics::norm_l2_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the squared L2 norm of the source signal.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// Output or scratch allocation can also fail.
    pub fn norm_l2_squared_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormL2SquaredScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_squared_scaled_buffer_size::<T, U>,
            statistics::norm_l2_squared_scaled_to_device_with_scratch::<T, U>,
        )
    }

    pub(crate) fn statistic_output<U>(
        self,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: unsafe fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, U>,
            &mut ScratchBuffer,
        ) -> Result<()>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), 1)?;
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn scaled_statistic_output<U>(
        self,
        scale_factor: i32,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: ScaledStatisticOutputCallback<T, U>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), 1)?;
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
                        scale_factor,
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    /// Computes the infinity norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_inf_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffInfTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_buffer_size::<T, U>,
            statistics::norm_diff_inf_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the L1 norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_l1_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL1To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_buffer_size::<T, U>,
            statistics::norm_diff_l1_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the L2 norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_l2_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL2To<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_buffer_size::<T, U>,
            statistics::norm_diff_l2_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the dot product of the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn dot_product_to<U>(self, other: &SignalView<'_, T>) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_buffer_size::<T, U>,
            statistics::dot_product_to_device_with_scratch::<T, U>,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the dot product of the input signals.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn dot_product_to_buffer_size<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
    ) -> Result<usize>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
    {
        statistics::dot_product_buffer_size::<T, U>(stream_context, source)
    }

    /// Computes the dot product of the input signals.
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
    /// The two source lengths must match.
    /// Returns an error if `destination` does not contain exactly one element.
    /// Returns an error if scratch capacity is insufficient.
    pub fn dot_product_to_with_scratch_into<U>(
        stream_context: &StreamContext,
        source1: &SignalView<'_, T>,
        source2: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
    {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source1)?;
        stream_context.prepare_region(source2)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::dot_product_to_device_with_scratch::<T, U>(
                stream_context,
                source1,
                source2,
                destination,
                scratch,
            )
        })
    }

    /// Computes the dot product of the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn dot_product_to_complex<U>(
        self,
        other: &SignalView<'_, T>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.dot_product_to::<U>(other)
    }

    /// Computes the infinity norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_inf_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffInfScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_inf_scaled_buffer_size::<T, U>,
            statistics::norm_diff_inf_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the L1 norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_l1_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL1ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l1_scaled_buffer_size::<T, U>,
            statistics::norm_diff_l1_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the L2 norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_l2_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL2ScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_scaled_buffer_size::<T, U>,
            statistics::norm_diff_l2_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the squared L2 norm of the difference between the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn norm_diff_l2_squared_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::NormDiffL2SquaredScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_squared_scaled_buffer_size::<T, U>,
            statistics::norm_diff_l2_squared_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the dot product of the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn dot_product_to_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_scaled_buffer_size::<T, U>,
            statistics::dot_product_scaled_to_device_with_scratch::<T, U>,
        )
    }

    /// Computes the dot product of the input signals.
    ///
    /// Returns an owned one-element pipeline using the same stream context and workspace.
    /// Obtains scratch from the workspace and waits for successful stream completion before returning.
    /// Applies native fixed-point scaling using `scale_factor`.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two source lengths must match.
    /// Output or scratch allocation can also fail.
    pub fn dot_product_to_complex_scaled<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        T: statistics::DotProductScaledTo<U>,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        self.dot_product_to_scaled::<U>(other, scale_factor)
    }

    pub(crate) fn binary_statistic_output<U>(
        self,
        other: &SignalView<'_, T>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: BinaryStatisticOutputCallback<T, U>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), 1)?;
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn heterogeneous_binary_statistic_output<U, V>(
        self,
        other: &SignalView<'_, U>,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: HeterogeneousBinaryStatisticOutputCallback<T, U, V>,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<V>(self.stream_context.cuda_context(), 1)?;
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn scaled_binary_statistic_output<U>(
        self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: ScaledBinaryStatisticOutputCallback<T, U>,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), 1)?;
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
                        scale_factor,
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn scaled_heterogeneous_binary_statistic_output<U, V>(
        self,
        other: &SignalView<'_, U>,
        scale_factor: i32,
        buffer_size: fn(&StreamContext, &SignalView<'_, T>) -> Result<usize>,
        statistic: ScaledHeterogeneousBinaryStatisticOutputCallback<T, U, V>,
    ) -> Result<SignalPipeline<'a, V>>
    where
        U: Copy,
        V: Copy,
        Workspace: SignalAllocator<V>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<V>(self.stream_context.cuda_context(), 1)?;
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
                        scale_factor,
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

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

type ScaledStatisticOutputCallback<T, U> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, U>,
    i32,
    &mut ScratchBuffer,
) -> Result<()>;

type BinaryStatisticOutputCallback<T, U> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, U>,
    &mut ScratchBuffer,
) -> Result<()>;

type HeterogeneousBinaryStatisticOutputCallback<T, U, V> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, U>,
    &mut SignalOutputView<'_, V>,
    &mut ScratchBuffer,
) -> Result<()>;

type ScaledBinaryStatisticOutputCallback<T, U> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, U>,
    i32,
    &mut ScratchBuffer,
) -> Result<()>;

type ScaledHeterogeneousBinaryStatisticOutputCallback<T, U, V> = unsafe fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, U>,
    &mut SignalOutputView<'_, V>,
    i32,
    &mut ScratchBuffer,
) -> Result<()>;
