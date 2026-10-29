use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::signal::SignalPipeline,
    signal::{
        statistics,
        view::{SignalOutputView, SignalView},
    },
    workspace::ScratchBuffer,
};

impl<'a> SignalPipeline<'a, Complex32> {
    impl_statistic_output_pipeline_methods!(
        "Computes the sum of the source samples.",
        Complex32,
        Complex32,
        sum_complex_buffer_size,
        sum_complex_with_scratch_into,
        statistics::sum_f32_complex_buffer_size,
        statistics::sum_f32_complex_to_device_with_scratch
    );

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
    pub fn sum_complex(self) -> Result<Self> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::sum_f32_complex_buffer_size,
            statistics::sum_f32_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the arithmetic mean of the source samples.",
        Complex32,
        Complex32,
        mean_complex_buffer_size,
        mean_complex_with_scratch_into,
        statistics::mean_f32_complex_buffer_size,
        statistics::mean_f32_complex_to_device_with_scratch
    );

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
    pub fn mean_complex(self) -> Result<Self> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::mean_f32_complex_buffer_size,
            statistics::mean_f32_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the infinity norm of the source signal.",
        Complex32,
        f32,
        norm_inf_to_f32_buffer_size,
        norm_inf_to_f32_with_scratch_into,
        statistics::norm_inf_f32_complex_to_f32_buffer_size,
        statistics::norm_inf_f32_complex_to_f32_to_device_with_scratch
    );

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
    pub fn norm_inf_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_inf_f32_complex_to_f32_buffer_size,
            statistics::norm_inf_f32_complex_to_f32_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the L1 norm of the source signal.",
        Complex32,
        f64,
        norm_l1_to_f64_buffer_size,
        norm_l1_to_f64_with_scratch_into,
        statistics::norm_l1_f32_complex_to_f64_buffer_size,
        statistics::norm_l1_f32_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_l1_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_l1_f32_complex_to_f64_buffer_size,
            statistics::norm_l1_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the L2 norm of the source signal.",
        Complex32,
        f64,
        norm_l2_to_f64_buffer_size,
        norm_l2_to_f64_with_scratch_into,
        statistics::norm_l2_f32_complex_to_f64_buffer_size,
        statistics::norm_l2_f32_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_l2_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_l2_f32_complex_to_f64_buffer_size,
            statistics::norm_l2_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the infinity norm of the difference between the input signals.",
        Complex32,
        f32,
        norm_diff_inf_to_f32_buffer_size,
        norm_diff_inf_to_f32_with_scratch_into,
        statistics::norm_diff_inf_f32_complex_to_f32_buffer_size,
        statistics::norm_diff_inf_f32_complex_to_f32_to_device_with_scratch
    );

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
    pub fn norm_diff_inf_to_f32(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_f32_complex_to_f32_buffer_size,
            statistics::norm_diff_inf_f32_complex_to_f32_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the L1 norm of the difference between the input signals.",
        Complex32,
        f64,
        norm_diff_l1_to_f64_buffer_size,
        norm_diff_l1_to_f64_with_scratch_into,
        statistics::norm_diff_l1_f32_complex_to_f64_buffer_size,
        statistics::norm_diff_l1_f32_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_diff_l1_to_f64(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_f32_complex_to_f64_buffer_size,
            statistics::norm_diff_l1_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the L2 norm of the difference between the input signals.",
        Complex32,
        f64,
        norm_diff_l2_to_f64_buffer_size,
        norm_diff_l2_to_f64_with_scratch_into,
        statistics::norm_diff_l2_f32_complex_to_f64_buffer_size,
        statistics::norm_diff_l2_f32_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_diff_l2_to_f64(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_f32_complex_to_f64_buffer_size,
            statistics::norm_diff_l2_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        Complex32,
        Complex32,
        dot_product_complex_buffer_size,
        dot_product_complex_with_scratch_into,
        statistics::dot_product_f32_complex_buffer_size,
        statistics::dot_product_f32_complex_to_device_with_scratch
    );

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
    pub fn dot_product_complex(self, other: &SignalView<'_, Complex32>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_f32_complex_buffer_size,
            statistics::dot_product_f32_complex_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        Complex32,
        Complex64,
        dot_product_to_f64_complex_buffer_size,
        dot_product_to_f64_complex_with_scratch_into,
        statistics::dot_product_f32_complex_to_f64_complex_buffer_size,
        statistics::dot_product_f32_complex_to_f64_complex_to_device_with_scratch
    );

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
    pub fn dot_product_to_f64_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_f32_complex_to_f64_complex_buffer_size,
            statistics::dot_product_f32_complex_to_f64_complex_to_device_with_scratch,
        )
    }
}

impl<'a> SignalPipeline<'a, Complex64> {
    impl_statistic_output_pipeline_methods!(
        "Computes the sum of the source samples.",
        Complex64,
        Complex64,
        sum_complex_buffer_size,
        sum_complex_with_scratch_into,
        statistics::sum_f64_complex_buffer_size,
        statistics::sum_f64_complex_to_device_with_scratch
    );

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
    pub fn sum_complex(self) -> Result<Self> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::sum_f64_complex_buffer_size,
            statistics::sum_f64_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the arithmetic mean of the source samples.",
        Complex64,
        Complex64,
        mean_complex_buffer_size,
        mean_complex_with_scratch_into,
        statistics::mean_f64_complex_buffer_size,
        statistics::mean_f64_complex_to_device_with_scratch
    );

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
    pub fn mean_complex(self) -> Result<Self> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::mean_f64_complex_buffer_size,
            statistics::mean_f64_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the infinity norm of the source signal.",
        Complex64,
        f64,
        norm_inf_to_f64_buffer_size,
        norm_inf_to_f64_with_scratch_into,
        statistics::norm_inf_f64_complex_to_f64_buffer_size,
        statistics::norm_inf_f64_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_inf_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_inf_f64_complex_to_f64_buffer_size,
            statistics::norm_inf_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the L1 norm of the source signal.",
        Complex64,
        f64,
        norm_l1_to_f64_buffer_size,
        norm_l1_to_f64_with_scratch_into,
        statistics::norm_l1_f64_complex_to_f64_buffer_size,
        statistics::norm_l1_f64_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_l1_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_l1_f64_complex_to_f64_buffer_size,
            statistics::norm_l1_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        "Computes the L2 norm of the source signal.",
        Complex64,
        f64,
        norm_l2_to_f64_buffer_size,
        norm_l2_to_f64_with_scratch_into,
        statistics::norm_l2_f64_complex_to_f64_buffer_size,
        statistics::norm_l2_f64_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_l2_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.statistic_output(
            statistics::norm_l2_f64_complex_to_f64_buffer_size,
            statistics::norm_l2_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the infinity norm of the difference between the input signals.",
        Complex64,
        f64,
        norm_diff_inf_to_f64_buffer_size,
        norm_diff_inf_to_f64_with_scratch_into,
        statistics::norm_diff_inf_f64_complex_to_f64_buffer_size,
        statistics::norm_diff_inf_f64_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_diff_inf_to_f64(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_f64_complex_to_f64_buffer_size,
            statistics::norm_diff_inf_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the L1 norm of the difference between the input signals.",
        Complex64,
        f64,
        norm_diff_l1_to_f64_buffer_size,
        norm_diff_l1_to_f64_with_scratch_into,
        statistics::norm_diff_l1_f64_complex_to_f64_buffer_size,
        statistics::norm_diff_l1_f64_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_diff_l1_to_f64(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_f64_complex_to_f64_buffer_size,
            statistics::norm_diff_l1_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the L2 norm of the difference between the input signals.",
        Complex64,
        f64,
        norm_diff_l2_to_f64_buffer_size,
        norm_diff_l2_to_f64_with_scratch_into,
        statistics::norm_diff_l2_f64_complex_to_f64_buffer_size,
        statistics::norm_diff_l2_f64_complex_to_f64_to_device_with_scratch
    );

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
    pub fn norm_diff_l2_to_f64(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_f64_complex_to_f64_buffer_size,
            statistics::norm_diff_l2_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        Complex64,
        Complex64,
        dot_product_complex_buffer_size,
        dot_product_complex_with_scratch_into,
        statistics::dot_product_f64_complex_buffer_size,
        statistics::dot_product_f64_complex_to_device_with_scratch
    );

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
    pub fn dot_product_complex(self, other: &SignalView<'_, Complex64>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_f64_complex_buffer_size,
            statistics::dot_product_f64_complex_to_device_with_scratch,
        )
    }
}

impl<'a> SignalPipeline<'a, f32> {
    /// Returns the scratch capacity in bytes required to compute the dot product of the input signals.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn dot_product_to_f64_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::dot_product_f32_to_f64_buffer_size(stream_context, source)
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
    pub fn dot_product_to_f64_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, f32>,
        destination: &mut SignalOutputView<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source1)?;
        stream_context.prepare_region(source2)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::dot_product_f32_to_f64_to_device_with_scratch(
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
    pub fn dot_product_to_f64(
        self,
        other: &SignalView<'_, f32>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_f32_to_f64_buffer_size,
            statistics::dot_product_f32_to_f64_to_device_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the dot product of the input signals.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn dot_product_with_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::dot_product_f32_and_f32_complex_to_f32_complex_buffer_size(
            stream_context,
            source,
        )
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
    pub fn dot_product_with_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, Complex32>,
        destination: &mut SignalOutputView<'_, Complex32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source1)?;
        stream_context.prepare_region(source2)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::dot_product_f32_and_f32_complex_to_f32_complex_to_device_with_scratch(
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
    pub fn dot_product_with_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex32>> {
        self.ensure_eager()?;
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_f32_and_f32_complex_to_f32_complex_buffer_size,
            statistics::dot_product_f32_and_f32_complex_to_f32_complex_to_device_with_scratch,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the dot product of the input signals.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn dot_product_with_complex_to_f64_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::dot_product_f32_and_f32_complex_to_f64_complex_buffer_size(
            stream_context,
            source,
        )
    }

    /// Returns the scratch capacity in bytes required to compute the dot product of the input signals.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn dot_product_with_complex_to_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        Self::dot_product_with_complex_to_f64_complex_buffer_size(stream_context, source)
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
    pub fn dot_product_with_complex_to_f64_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, Complex32>,
        destination: &mut SignalOutputView<'_, Complex64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source1)?;
        stream_context.prepare_region(source2)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::dot_product_f32_and_f32_complex_to_f64_complex_to_device_with_scratch(
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
    pub fn dot_product_with_complex_to_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, Complex32>,
        destination: &mut SignalOutputView<'_, Complex64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        Self::dot_product_with_complex_to_f64_complex_with_scratch_into(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
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
    pub fn dot_product_with_complex_to_f64_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.ensure_eager()?;
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_f32_and_f32_complex_to_f64_complex_buffer_size,
            statistics::dot_product_f32_and_f32_complex_to_f64_complex_to_device_with_scratch,
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
    pub fn dot_product_with_complex_to_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.ensure_eager()?;
        self.dot_product_with_complex_to_f64_complex(other)
    }
}

impl<'a> SignalPipeline<'a, f64> {
    /// Returns the scratch capacity in bytes required to compute the dot product of the input signals.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source length or scratch-size conversion fails, or NPP rejects the workspace query.
    pub fn dot_product_with_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
    ) -> Result<usize> {
        statistics::dot_product_f64_and_f64_complex_to_f64_complex_buffer_size(
            stream_context,
            source,
        )
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
    pub fn dot_product_with_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f64>,
        source2: &SignalView<'_, Complex64>,
        destination: &mut SignalOutputView<'_, Complex64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source1)?;
        stream_context.prepare_region(source2)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            statistics::dot_product_f64_and_f64_complex_to_f64_complex_to_device_with_scratch(
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
    pub fn dot_product_with_complex(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.ensure_eager()?;
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_f64_and_f64_complex_to_f64_complex_buffer_size,
            statistics::dot_product_f64_and_f64_complex_to_f64_complex_to_device_with_scratch,
        )
    }
}
