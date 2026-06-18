use super::*;

impl<'a> SignalPipeline<'a, Complex32> {
    impl_statistic_output_pipeline_methods!(
        Complex32,
        Complex32,
        sum_complex_buffer_size,
        sum_complex_with_scratch_into,
        statistics::sum_f32_complex_buffer_size,
        statistics::sum_f32_complex_to_device_with_scratch
    );

    pub fn sum_complex(self) -> Result<Self> {
        self.statistic_output(
            statistics::sum_f32_complex_buffer_size,
            statistics::sum_f32_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex32,
        Complex32,
        mean_complex_buffer_size,
        mean_complex_with_scratch_into,
        statistics::mean_f32_complex_buffer_size,
        statistics::mean_f32_complex_to_device_with_scratch
    );

    pub fn mean_complex(self) -> Result<Self> {
        self.statistic_output(
            statistics::mean_f32_complex_buffer_size,
            statistics::mean_f32_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex32,
        f32,
        norm_inf_to_f32_buffer_size,
        norm_inf_to_f32_with_scratch_into,
        statistics::norm_inf_f32_complex_to_f32_buffer_size,
        statistics::norm_inf_f32_complex_to_f32_to_device_with_scratch
    );

    pub fn norm_inf_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.statistic_output(
            statistics::norm_inf_f32_complex_to_f32_buffer_size,
            statistics::norm_inf_f32_complex_to_f32_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex32,
        f64,
        norm_l1_to_f64_buffer_size,
        norm_l1_to_f64_with_scratch_into,
        statistics::norm_l1_f32_complex_to_f64_buffer_size,
        statistics::norm_l1_f32_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_l1_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.statistic_output(
            statistics::norm_l1_f32_complex_to_f64_buffer_size,
            statistics::norm_l1_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex32,
        f64,
        norm_l2_to_f64_buffer_size,
        norm_l2_to_f64_with_scratch_into,
        statistics::norm_l2_f32_complex_to_f64_buffer_size,
        statistics::norm_l2_f32_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_l2_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.statistic_output(
            statistics::norm_l2_f32_complex_to_f64_buffer_size,
            statistics::norm_l2_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex32,
        f32,
        norm_diff_inf_to_f32_buffer_size,
        norm_diff_inf_to_f32_with_scratch_into,
        statistics::norm_diff_inf_f32_complex_to_f32_buffer_size,
        statistics::norm_diff_inf_f32_complex_to_f32_to_device_with_scratch
    );

    pub fn norm_diff_inf_to_f32(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_f32_complex_to_f32_buffer_size,
            statistics::norm_diff_inf_f32_complex_to_f32_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex32,
        f64,
        norm_diff_l1_to_f64_buffer_size,
        norm_diff_l1_to_f64_with_scratch_into,
        statistics::norm_diff_l1_f32_complex_to_f64_buffer_size,
        statistics::norm_diff_l1_f32_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_diff_l1_to_f64(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_f32_complex_to_f64_buffer_size,
            statistics::norm_diff_l1_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex32,
        f64,
        norm_diff_l2_to_f64_buffer_size,
        norm_diff_l2_to_f64_with_scratch_into,
        statistics::norm_diff_l2_f32_complex_to_f64_buffer_size,
        statistics::norm_diff_l2_f32_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_diff_l2_to_f64(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_f32_complex_to_f64_buffer_size,
            statistics::norm_diff_l2_f32_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex32,
        Complex32,
        dot_product_complex_buffer_size,
        dot_product_complex_with_scratch_into,
        statistics::dot_product_f32_complex_buffer_size,
        statistics::dot_product_f32_complex_to_device_with_scratch
    );

    pub fn dot_product_complex(self, other: &SignalView<'_, Complex32>) -> Result<Self> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_f32_complex_buffer_size,
            statistics::dot_product_f32_complex_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex32,
        Complex64,
        dot_product_to_f64_complex_buffer_size,
        dot_product_to_f64_complex_with_scratch_into,
        statistics::dot_product_f32_complex_to_f64_complex_buffer_size,
        statistics::dot_product_f32_complex_to_f64_complex_to_device_with_scratch
    );

    pub fn dot_product_to_f64_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_f32_complex_to_f64_complex_buffer_size,
            statistics::dot_product_f32_complex_to_f64_complex_to_device_with_scratch,
        )
    }
}

impl<'a> SignalPipeline<'a, Complex64> {
    impl_statistic_output_pipeline_methods!(
        Complex64,
        Complex64,
        sum_complex_buffer_size,
        sum_complex_with_scratch_into,
        statistics::sum_f64_complex_buffer_size,
        statistics::sum_f64_complex_to_device_with_scratch
    );

    pub fn sum_complex(self) -> Result<Self> {
        self.statistic_output(
            statistics::sum_f64_complex_buffer_size,
            statistics::sum_f64_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex64,
        Complex64,
        mean_complex_buffer_size,
        mean_complex_with_scratch_into,
        statistics::mean_f64_complex_buffer_size,
        statistics::mean_f64_complex_to_device_with_scratch
    );

    pub fn mean_complex(self) -> Result<Self> {
        self.statistic_output(
            statistics::mean_f64_complex_buffer_size,
            statistics::mean_f64_complex_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex64,
        f64,
        norm_inf_to_f64_buffer_size,
        norm_inf_to_f64_with_scratch_into,
        statistics::norm_inf_f64_complex_to_f64_buffer_size,
        statistics::norm_inf_f64_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_inf_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.statistic_output(
            statistics::norm_inf_f64_complex_to_f64_buffer_size,
            statistics::norm_inf_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex64,
        f64,
        norm_l1_to_f64_buffer_size,
        norm_l1_to_f64_with_scratch_into,
        statistics::norm_l1_f64_complex_to_f64_buffer_size,
        statistics::norm_l1_f64_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_l1_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.statistic_output(
            statistics::norm_l1_f64_complex_to_f64_buffer_size,
            statistics::norm_l1_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_statistic_output_pipeline_methods!(
        Complex64,
        f64,
        norm_l2_to_f64_buffer_size,
        norm_l2_to_f64_with_scratch_into,
        statistics::norm_l2_f64_complex_to_f64_buffer_size,
        statistics::norm_l2_f64_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_l2_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.statistic_output(
            statistics::norm_l2_f64_complex_to_f64_buffer_size,
            statistics::norm_l2_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex64,
        f64,
        norm_diff_inf_to_f64_buffer_size,
        norm_diff_inf_to_f64_with_scratch_into,
        statistics::norm_diff_inf_f64_complex_to_f64_buffer_size,
        statistics::norm_diff_inf_f64_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_diff_inf_to_f64(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_f64_complex_to_f64_buffer_size,
            statistics::norm_diff_inf_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex64,
        f64,
        norm_diff_l1_to_f64_buffer_size,
        norm_diff_l1_to_f64_with_scratch_into,
        statistics::norm_diff_l1_f64_complex_to_f64_buffer_size,
        statistics::norm_diff_l1_f64_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_diff_l1_to_f64(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_f64_complex_to_f64_buffer_size,
            statistics::norm_diff_l1_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex64,
        f64,
        norm_diff_l2_to_f64_buffer_size,
        norm_diff_l2_to_f64_with_scratch_into,
        statistics::norm_diff_l2_f64_complex_to_f64_buffer_size,
        statistics::norm_diff_l2_f64_complex_to_f64_to_device_with_scratch
    );

    pub fn norm_diff_l2_to_f64(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_f64_complex_to_f64_buffer_size,
            statistics::norm_diff_l2_f64_complex_to_f64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        Complex64,
        Complex64,
        dot_product_complex_buffer_size,
        dot_product_complex_with_scratch_into,
        statistics::dot_product_f64_complex_buffer_size,
        statistics::dot_product_f64_complex_to_device_with_scratch
    );

    pub fn dot_product_complex(self, other: &SignalView<'_, Complex64>) -> Result<Self> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_f64_complex_buffer_size,
            statistics::dot_product_f64_complex_to_device_with_scratch,
        )
    }
}

impl<'a> SignalPipeline<'a, f32> {
    pub fn dot_product_to_f64_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::dot_product_f32_to_f64_buffer_size(stream_context, source)
    }

    pub fn dot_product_to_f64_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, f64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::dot_product_f32_to_f64_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    pub fn dot_product_to_f64(
        self,
        other: &SignalView<'_, f32>,
    ) -> Result<SignalPipeline<'a, f64>> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_f32_to_f64_buffer_size,
            statistics::dot_product_f32_to_f64_to_device_with_scratch,
        )
    }

    pub fn dot_product_with_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::dot_product_f32_and_f32_complex_to_f32_complex_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn dot_product_with_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, Complex32>,
        destination: &mut SignalViewMut<'_, Complex32>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::dot_product_f32_and_f32_complex_to_f32_complex_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    pub fn dot_product_with_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex32>> {
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_f32_and_f32_complex_to_f32_complex_buffer_size,
            statistics::dot_product_f32_and_f32_complex_to_f32_complex_to_device_with_scratch,
        )
    }

    pub fn dot_product_with_complex_to_f64_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        statistics::dot_product_f32_and_f32_complex_to_f64_complex_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn dot_product_with_complex_to_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
    ) -> Result<usize> {
        Self::dot_product_with_complex_to_f64_complex_buffer_size(stream_context, source)
    }

    pub fn dot_product_with_complex_to_f64_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, Complex32>,
        destination: &mut SignalViewMut<'_, Complex64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::dot_product_f32_and_f32_complex_to_f64_complex_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    pub fn dot_product_with_complex_to_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f32>,
        source2: &SignalView<'_, Complex32>,
        destination: &mut SignalViewMut<'_, Complex64>,
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

    pub fn dot_product_with_complex_to_f64_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_f32_and_f32_complex_to_f64_complex_buffer_size,
            statistics::dot_product_f32_and_f32_complex_to_f64_complex_to_device_with_scratch,
        )
    }

    pub fn dot_product_with_complex_to_complex(
        self,
        other: &SignalView<'_, Complex32>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.dot_product_with_complex_to_f64_complex(other)
    }
}

impl<'a> SignalPipeline<'a, f64> {
    pub fn dot_product_with_complex_buffer_size(
        stream_context: &StreamContext,
        source: &SignalView<'_, f64>,
    ) -> Result<usize> {
        statistics::dot_product_f64_and_f64_complex_to_f64_complex_buffer_size(
            stream_context,
            source,
        )
    }

    pub fn dot_product_with_complex_with_scratch_into(
        stream_context: &StreamContext,
        source1: &SignalView<'_, f64>,
        source2: &SignalView<'_, Complex64>,
        destination: &mut SignalViewMut<'_, Complex64>,
        scratch: &mut ScratchBuffer,
    ) -> Result<()> {
        statistics::dot_product_f64_and_f64_complex_to_f64_complex_to_device_with_scratch(
            stream_context,
            source1,
            source2,
            destination,
            scratch,
        )
    }

    pub fn dot_product_with_complex(
        self,
        other: &SignalView<'_, Complex64>,
    ) -> Result<SignalPipeline<'a, Complex64>> {
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_f64_and_f64_complex_to_f64_complex_buffer_size,
            statistics::dot_product_f64_and_f64_complex_to_f64_complex_to_device_with_scratch,
        )
    }
}
