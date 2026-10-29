use singe_cuda::types::Complex32;

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        signal::{
            SignalBacking, SignalPipeline, copy::CopySignal,
            statistics_methods::EveryStatisticSignal,
        },
        workspace::{SignalAllocator, Workspace},
    },
    signal::{
        statistics,
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32, ComplexI64},
    workspace::ScratchBuffer,
};

macro_rules! impl_scaled_statistic_pipeline_methods {
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
        /// Applies native fixed-point scaling using `scale_factor`.
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
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $npp_with_scratch(stream_context, source, destination, scale_factor, scratch)
            })
        }
    };
}

macro_rules! impl_scaled_binary_statistic_output_pipeline_methods {
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
        /// Applies native fixed-point scaling using `scale_factor`.
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
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source1)?;
            stream_context.prepare_region(source2)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $npp_with_scratch(
                    stream_context,
                    source1,
                    source2,
                    destination,
                    scale_factor,
                    scratch,
                )
            })
        }
    };
}

macro_rules! impl_heterogeneous_binary_statistic_output_pipeline_methods {
    (
        $doc:literal,
        $source1_ty:ty,
        $source2_ty:ty,
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
            source: &SignalView<'_, $source1_ty>,
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
            source1: &SignalView<'_, $source1_ty>,
            source2: &SignalView<'_, $source2_ty>,
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

macro_rules! impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods {
    (
        $doc:literal,
        $source1_ty:ty,
        $source2_ty:ty,
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
            source: &SignalView<'_, $source1_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        #[doc = $doc]
        #[doc = concat!("Uses [`", stringify!($npp_with_scratch), "`].")]
        /// Writes the scalar result into the single element of `destination`.
        /// Uses caller-provided scratch and waits for successful stream completion before returning.
        /// The scratch capacity is checked against the native query.
        /// `scratch` must belong to the stream context; its CUDA context is not checked by this path.
        /// Applies native fixed-point scaling using `scale_factor`.
        ///
        /// # Errors
        ///
        /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
        /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
        /// Returns an error if `destination` does not contain exactly one element or scratch capacity is insufficient.
        /// The two source lengths must match.
        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source1: &SignalView<'_, $source1_ty>,
            source2: &SignalView<'_, $source2_ty>,
            destination: &mut SignalOutputView<'_, $destination_ty>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            stream_context.stream().ensure_not_capturing()?;
            stream_context.prepare_region(source1)?;
            stream_context.prepare_region(source2)?;
            stream_context.prepare_region(destination)?;
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            stream_context.sync_scope(|| unsafe {
                $npp_with_scratch(
                    stream_context,
                    source1,
                    source2,
                    destination,
                    scale_factor,
                    scratch,
                )
            })
        }
    };
}

impl<'a> SignalPipeline<'a, i16> {
    impl_scaled_statistic_pipeline_methods!(
        "Computes the sum of the source samples.",
        i16,
        i16,
        sum_scaled_buffer_size,
        sum_scaled_with_scratch_into,
        statistics::sum_i16_scaled_buffer_size,
        statistics::sum_i16_scaled_to_device_with_scratch
    );

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
    pub fn sum_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_scaled_buffer_size,
            statistics::sum_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the sum of the source samples.",
        i16,
        i32,
        sum_to_i32_scaled_buffer_size,
        sum_to_i32_scaled_with_scratch_into,
        statistics::sum_i16_to_i32_scaled_buffer_size,
        statistics::sum_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn sum_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_to_i32_scaled_buffer_size,
            statistics::sum_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the arithmetic mean of the source samples.",
        i16,
        i16,
        mean_scaled_buffer_size,
        mean_scaled_with_scratch_into,
        statistics::mean_i16_scaled_buffer_size,
        statistics::mean_i16_scaled_to_device_with_scratch
    );

    /// Computes the arithmetic mean of the source samples.
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
    pub fn mean_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::mean_i16_scaled_buffer_size,
            statistics::mean_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the standard deviation of the source samples.",
        i16,
        i16,
        standard_deviation_scaled_buffer_size,
        standard_deviation_scaled_with_scratch_into,
        statistics::standard_deviation_i16_scaled_buffer_size,
        statistics::standard_deviation_i16_scaled_to_device_with_scratch
    );

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
    pub fn standard_deviation_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::standard_deviation_i16_scaled_buffer_size,
            statistics::standard_deviation_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the standard deviation of the source samples.",
        i16,
        i32,
        standard_deviation_to_i32_scaled_buffer_size,
        standard_deviation_to_i32_scaled_with_scratch_into,
        statistics::standard_deviation_i16_to_i32_scaled_buffer_size,
        statistics::standard_deviation_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn standard_deviation_to_i32_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::standard_deviation_i16_to_i32_scaled_buffer_size,
            statistics::standard_deviation_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the infinity norm of the source signal.",
        i16,
        i32,
        norm_inf_to_i32_scaled_buffer_size,
        norm_inf_to_i32_scaled_with_scratch_into,
        statistics::norm_inf_i16_to_i32_scaled_buffer_size,
        statistics::norm_inf_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn norm_inf_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_inf_i16_to_i32_scaled_buffer_size,
            statistics::norm_inf_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the L1 norm of the source signal.",
        i16,
        i32,
        norm_l1_to_i32_scaled_buffer_size,
        norm_l1_to_i32_scaled_with_scratch_into,
        statistics::norm_l1_i16_to_i32_scaled_buffer_size,
        statistics::norm_l1_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn norm_l1_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l1_i16_to_i32_scaled_buffer_size,
            statistics::norm_l1_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the L1 norm of the source signal.",
        i16,
        i64,
        norm_l1_to_i64_scaled_buffer_size,
        norm_l1_to_i64_scaled_with_scratch_into,
        statistics::norm_l1_i16_to_i64_scaled_buffer_size,
        statistics::norm_l1_i16_to_i64_scaled_to_device_with_scratch
    );

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
    pub fn norm_l1_to_i64_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i64>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l1_i16_to_i64_scaled_buffer_size,
            statistics::norm_l1_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the L2 norm of the source signal.",
        i16,
        i32,
        norm_l2_to_i32_scaled_buffer_size,
        norm_l2_to_i32_scaled_with_scratch_into,
        statistics::norm_l2_i16_to_i32_scaled_buffer_size,
        statistics::norm_l2_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn norm_l2_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_i16_to_i32_scaled_buffer_size,
            statistics::norm_l2_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the squared L2 norm of the source signal.",
        i16,
        i64,
        norm_l2_squared_to_i64_scaled_buffer_size,
        norm_l2_squared_to_i64_scaled_with_scratch_into,
        statistics::norm_l2_squared_i16_to_i64_scaled_buffer_size,
        statistics::norm_l2_squared_i16_to_i64_scaled_to_device_with_scratch
    );

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
    pub fn norm_l2_squared_to_i64_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_squared_i16_to_i64_scaled_buffer_size,
            statistics::norm_l2_squared_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

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
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_i16_to_f32_buffer_size,
            statistics::norm_diff_inf_i16_to_f32_to_device_with_scratch,
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
    pub fn norm_diff_l1_to_f32(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_i16_to_f32_buffer_size,
            statistics::norm_diff_l1_i16_to_f32_to_device_with_scratch,
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
    pub fn norm_diff_l2_to_f32(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_i16_to_f32_buffer_size,
            statistics::norm_diff_l2_i16_to_f32_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the infinity norm of the difference between the input signals.",
        i16,
        i32,
        norm_diff_inf_to_i32_scaled_buffer_size,
        norm_diff_inf_to_i32_scaled_with_scratch_into,
        statistics::norm_diff_inf_i16_to_i32_scaled_buffer_size,
        statistics::norm_diff_inf_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn norm_diff_inf_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_inf_i16_to_i32_scaled_buffer_size,
            statistics::norm_diff_inf_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the L1 norm of the difference between the input signals.",
        i16,
        i32,
        norm_diff_l1_to_i32_scaled_buffer_size,
        norm_diff_l1_to_i32_scaled_with_scratch_into,
        statistics::norm_diff_l1_i16_to_i32_scaled_buffer_size,
        statistics::norm_diff_l1_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn norm_diff_l1_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l1_i16_to_i32_scaled_buffer_size,
            statistics::norm_diff_l1_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the L1 norm of the difference between the input signals.",
        i16,
        i64,
        norm_diff_l1_to_i64_scaled_buffer_size,
        norm_diff_l1_to_i64_scaled_with_scratch_into,
        statistics::norm_diff_l1_i16_to_i64_scaled_buffer_size,
        statistics::norm_diff_l1_i16_to_i64_scaled_to_device_with_scratch
    );

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
    pub fn norm_diff_l1_to_i64_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l1_i16_to_i64_scaled_buffer_size,
            statistics::norm_diff_l1_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the L2 norm of the difference between the input signals.",
        i16,
        i32,
        norm_diff_l2_to_i32_scaled_buffer_size,
        norm_diff_l2_to_i32_scaled_with_scratch_into,
        statistics::norm_diff_l2_i16_to_i32_scaled_buffer_size,
        statistics::norm_diff_l2_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn norm_diff_l2_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_i16_to_i32_scaled_buffer_size,
            statistics::norm_diff_l2_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the squared L2 norm of the difference between the input signals.",
        i16,
        i64,
        norm_diff_l2_squared_to_i64_scaled_buffer_size,
        norm_diff_l2_squared_to_i64_scaled_with_scratch_into,
        statistics::norm_diff_l2_squared_i16_to_i64_scaled_buffer_size,
        statistics::norm_diff_l2_squared_i16_to_i64_scaled_to_device_with_scratch
    );

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
    pub fn norm_diff_l2_squared_to_i64_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_squared_i16_to_i64_scaled_buffer_size,
            statistics::norm_diff_l2_squared_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        i64,
        dot_product_to_i64_buffer_size,
        dot_product_to_i64_with_scratch_into,
        statistics::dot_product_i16_to_i64_buffer_size,
        statistics::dot_product_i16_to_i64_to_device_with_scratch
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
    pub fn dot_product_to_i64(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_to_i64_buffer_size,
            statistics::dot_product_i16_to_i64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        f32,
        dot_product_to_f32_buffer_size,
        dot_product_to_f32_with_scratch_into,
        statistics::dot_product_i16_to_f32_buffer_size,
        statistics::dot_product_i16_to_f32_to_device_with_scratch
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
    pub fn dot_product_to_f32(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_to_f32_buffer_size,
            statistics::dot_product_i16_to_f32_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        i16,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i16_scaled_buffer_size,
        statistics::dot_product_i16_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_scaled_buffer_size,
            statistics::dot_product_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        i32,
        dot_product_to_i32_scaled_buffer_size,
        dot_product_to_i32_scaled_with_scratch_into,
        statistics::dot_product_i16_to_i32_scaled_buffer_size,
        statistics::dot_product_i16_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_to_i32_scaled_buffer_size,
            statistics::dot_product_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_heterogeneous_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        ComplexI16,
        ComplexI64,
        dot_product_with_complex_to_i64_complex_buffer_size,
        dot_product_with_complex_to_i64_complex_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_to_i64_complex_buffer_size,
        statistics::dot_product_i16_and_i16_complex_to_i64_complex_to_device_with_scratch
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
    pub fn dot_product_with_complex_to_i64_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, ComplexI64>> {
        self.ensure_eager()?;
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_i16_and_i16_complex_to_i64_complex_buffer_size,
            statistics::dot_product_i16_and_i16_complex_to_i64_complex_to_device_with_scratch,
        )
    }

    impl_heterogeneous_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        ComplexI16,
        Complex32,
        dot_product_with_complex_to_f32_complex_buffer_size,
        dot_product_with_complex_to_f32_complex_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_to_f32_complex_buffer_size,
        statistics::dot_product_i16_and_i16_complex_to_f32_complex_to_device_with_scratch
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
    pub fn dot_product_with_complex_to_f32_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, Complex32>> {
        self.ensure_eager()?;
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_i16_and_i16_complex_to_f32_complex_buffer_size,
            statistics::dot_product_i16_and_i16_complex_to_f32_complex_to_device_with_scratch,
        )
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        ComplexI16,
        ComplexI16,
        dot_product_with_complex_scaled_buffer_size,
        dot_product_with_complex_scaled_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_scaled_buffer_size,
        statistics::dot_product_i16_and_i16_complex_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_with_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI16>> {
        self.ensure_eager()?;
        self.scaled_heterogeneous_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_and_i16_complex_scaled_buffer_size,
            statistics::dot_product_i16_and_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        ComplexI16,
        ComplexI32,
        dot_product_with_complex_to_i32_complex_scaled_buffer_size,
        dot_product_with_complex_to_i32_complex_scaled_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_buffer_size,
        statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_with_complex_to_i32_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.ensure_eager()?;
        self.scaled_heterogeneous_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_buffer_size,
            statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
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
    pub fn dot_product_with_complex_to_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.ensure_eager()?;
        self.dot_product_with_complex_to_i32_complex_scaled(other, scale_factor)
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i16,
        i32,
        i32,
        dot_product_with_i32_to_i32_scaled_buffer_size,
        dot_product_with_i32_to_i32_scaled_with_scratch_into,
        statistics::dot_product_i16_and_i32_to_i32_scaled_buffer_size,
        statistics::dot_product_i16_and_i32_to_i32_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_with_i32_to_i32_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.scaled_heterogeneous_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_and_i32_to_i32_scaled_buffer_size,
            statistics::dot_product_i16_and_i32_to_i32_scaled_to_device_with_scratch,
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
    pub fn dot_product_with_i32_to_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.dot_product_with_i32_to_i32_scaled(other, scale_factor)
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
    pub fn dot_product_with_to_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.dot_product_with_i32_to_i32_scaled(other, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i32> {
    impl_scaled_statistic_pipeline_methods!(
        "Computes the sum of the source samples.",
        i32,
        i32,
        sum_scaled_buffer_size,
        sum_scaled_with_scratch_into,
        statistics::sum_i32_scaled_buffer_size,
        statistics::sum_i32_scaled_to_device_with_scratch
    );

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
    pub fn sum_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i32_scaled_buffer_size,
            statistics::sum_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the arithmetic mean of the source samples.",
        i32,
        i32,
        mean_scaled_buffer_size,
        mean_scaled_with_scratch_into,
        statistics::mean_i32_scaled_buffer_size,
        statistics::mean_i32_scaled_to_device_with_scratch
    );

    /// Computes the arithmetic mean of the source samples.
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
    pub fn mean_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::mean_i32_scaled_buffer_size,
            statistics::mean_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i32,
        i32,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i32_scaled_buffer_size,
        statistics::dot_product_i32_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i32_scaled_buffer_size,
            statistics::dot_product_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        i32,
        ComplexI32,
        ComplexI32,
        dot_product_with_complex_scaled_buffer_size,
        dot_product_with_complex_scaled_with_scratch_into,
        statistics::dot_product_i32_and_i32_complex_scaled_buffer_size,
        statistics::dot_product_i32_and_i32_complex_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_with_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.ensure_eager()?;
        self.scaled_heterogeneous_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i32_and_i32_complex_scaled_buffer_size,
            statistics::dot_product_i32_and_i32_complex_scaled_to_device_with_scratch,
        )
    }
}

impl<'a> SignalPipeline<'a, ComplexI16> {
    impl_scaled_statistic_pipeline_methods!(
        "Computes the sum of the source samples.",
        ComplexI16,
        ComplexI16,
        sum_scaled_buffer_size,
        sum_scaled_with_scratch_into,
        statistics::sum_i16_complex_scaled_buffer_size,
        statistics::sum_i16_complex_scaled_to_device_with_scratch
    );

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
    pub fn sum_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_complex_scaled_buffer_size,
            statistics::sum_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the sum of the source samples.",
        ComplexI16,
        ComplexI32,
        sum_to_i32_complex_scaled_buffer_size,
        sum_to_i32_complex_scaled_with_scratch_into,
        statistics::sum_i16_complex_to_i32_complex_scaled_buffer_size,
        statistics::sum_i16_complex_to_i32_complex_scaled_to_device_with_scratch
    );

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
    pub fn sum_to_i32_complex_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_complex_to_i32_complex_scaled_buffer_size,
            statistics::sum_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        "Computes the arithmetic mean of the source samples.",
        ComplexI16,
        ComplexI16,
        mean_scaled_buffer_size,
        mean_scaled_with_scratch_into,
        statistics::mean_i16_complex_scaled_buffer_size,
        statistics::mean_i16_complex_scaled_to_device_with_scratch
    );

    /// Computes the arithmetic mean of the source samples.
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
    pub fn mean_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_statistic_output(
            scale_factor,
            statistics::mean_i16_complex_scaled_buffer_size,
            statistics::mean_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        ComplexI16,
        ComplexI64,
        dot_product_to_i64_complex_buffer_size,
        dot_product_to_i64_complex_with_scratch_into,
        statistics::dot_product_i16_complex_to_i64_complex_buffer_size,
        statistics::dot_product_i16_complex_to_i64_complex_to_device_with_scratch
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
    pub fn dot_product_to_i64_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, ComplexI64>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_complex_to_i64_complex_buffer_size,
            statistics::dot_product_i16_complex_to_i64_complex_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        ComplexI16,
        Complex32,
        dot_product_to_f32_complex_buffer_size,
        dot_product_to_f32_complex_with_scratch_into,
        statistics::dot_product_i16_complex_to_f32_complex_buffer_size,
        statistics::dot_product_i16_complex_to_f32_complex_to_device_with_scratch
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
    pub fn dot_product_to_f32_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, Complex32>> {
        self.ensure_eager()?;
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_complex_to_f32_complex_buffer_size,
            statistics::dot_product_i16_complex_to_f32_complex_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        ComplexI16,
        ComplexI16,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i16_complex_scaled_buffer_size,
        statistics::dot_product_i16_complex_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_complex_scaled_buffer_size,
            statistics::dot_product_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        ComplexI16,
        ComplexI32,
        dot_product_to_i32_complex_scaled_buffer_size,
        dot_product_to_i32_complex_scaled_with_scratch_into,
        statistics::dot_product_i16_complex_to_i32_complex_scaled_buffer_size,
        statistics::dot_product_i16_complex_to_i32_complex_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_to_i32_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_complex_to_i32_complex_scaled_buffer_size,
            statistics::dot_product_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
        )
    }
}

impl SignalPipeline<'_, ComplexI32> {
    impl_scaled_binary_statistic_output_pipeline_methods!(
        "Computes the dot product of the input signals.",
        ComplexI32,
        ComplexI32,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i32_complex_scaled_buffer_size,
        statistics::dot_product_i32_complex_scaled_to_device_with_scratch
    );

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
    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i32_complex_scaled_buffer_size,
            statistics::dot_product_i32_complex_scaled_to_device_with_scratch,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: CopySignal<T> + EveryStatisticSignal<T>,
{
    /// Replaces each sample with the elementwise minimum of this signal and `other`.
    ///
    /// Reuses owned signal storage; a borrowed signal is copied to an owned allocation first.
    /// Preserves the signal length and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two signal lengths must match.
    /// Copying or allocating owned storage can also fail.
    pub fn min_every(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.every_statistic(other, <Self as EveryStatisticSignal<T>>::min_every_in_place)
    }

    /// Replaces each sample with the elementwise maximum of this signal and `other`.
    ///
    /// Reuses owned signal storage; a borrowed signal is copied to an owned allocation first.
    /// Preserves the signal length and waits for successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active, a participating signal belongs to another CUDA context, or CUDA context binding fails.
    /// Producer-stream synchronization, native length and size conversions, workspace queries, NPP execution, and stream synchronization can also fail.
    /// The two signal lengths must match.
    /// Copying or allocating owned storage can also fail.
    pub fn max_every(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.every_statistic(other, <Self as EveryStatisticSignal<T>>::max_every_in_place)
    }

    fn every_statistic(
        mut self,
        other: &SignalView<'_, T>,
        statistic: unsafe fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                self.stream_context.stream().ensure_not_capturing()?;
                self.stream_context.prepare_region(other)?;
                self.stream_context.prepare_region(&signal_view)?;
                self.stream_context.sync_scope(|| {
                    // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                    unsafe { statistic(self.stream_context, other, &mut signal_view) }
                })?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as CopySignal<T>>::copy(self.stream_context, source, &mut destination_view)?;
                // SAFETY: the completed copy initialized the accumulator.
                let mut destination_view = unsafe { destination_view.assume_init() };
                self.stream_context.stream().ensure_not_capturing()?;
                self.stream_context.prepare_region(other)?;
                self.stream_context.prepare_region(&destination_view)?;
                self.stream_context.sync_scope(|| {
                    // SAFETY: all callback inputs, outputs, and scratch remain live through completion.
                    unsafe { statistic(self.stream_context, other, &mut destination_view) }
                })?;
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
