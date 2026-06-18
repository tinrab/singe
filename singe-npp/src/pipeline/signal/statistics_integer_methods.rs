use super::*;

macro_rules! impl_scaled_statistic_pipeline_methods {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            $npp_with_scratch(stream_context, source, destination, scale_factor, scratch)
        }
    };
}

macro_rules! impl_scaled_binary_statistic_output_pipeline_methods {
    (
        $source_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source1: &SignalView<'_, $source_ty>,
            source2: &SignalView<'_, $source_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            $npp_with_scratch(
                stream_context,
                source1,
                source2,
                destination,
                scale_factor,
                scratch,
            )
        }
    };
}

macro_rules! impl_heterogeneous_binary_statistic_output_pipeline_methods {
    (
        $source1_ty:ty,
        $source2_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source1_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source1: &SignalView<'_, $source1_ty>,
            source2: &SignalView<'_, $source2_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            $npp_with_scratch(stream_context, source1, source2, destination, scratch)
        }
    };
}

macro_rules! impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods {
    (
        $source1_ty:ty,
        $source2_ty:ty,
        $destination_ty:ty,
        $buffer_size_name:ident,
        $with_scratch_name:ident,
        $npp_buffer_size:path,
        $npp_with_scratch:path
    ) => {
        pub fn $buffer_size_name(
            stream_context: &StreamContext,
            source: &SignalView<'_, $source1_ty>,
        ) -> Result<usize> {
            $npp_buffer_size(stream_context, source)
        }

        pub fn $with_scratch_name(
            stream_context: &StreamContext,
            source1: &SignalView<'_, $source1_ty>,
            source2: &SignalView<'_, $source2_ty>,
            destination: &mut SignalViewMut<'_, $destination_ty>,
            scale_factor: i32,
            scratch: &mut ScratchBuffer,
        ) -> Result<()> {
            $npp_with_scratch(
                stream_context,
                source1,
                source2,
                destination,
                scale_factor,
                scratch,
            )
        }
    };
}

impl<'a> SignalPipeline<'a, i16> {
    impl_scaled_statistic_pipeline_methods!(
        i16,
        i16,
        sum_scaled_buffer_size,
        sum_scaled_with_scratch_into,
        statistics::sum_i16_scaled_buffer_size,
        statistics::sum_i16_scaled_to_device_with_scratch
    );

    pub fn sum_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_scaled_buffer_size,
            statistics::sum_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i32,
        sum_to_i32_scaled_buffer_size,
        sum_to_i32_scaled_with_scratch_into,
        statistics::sum_i16_to_i32_scaled_buffer_size,
        statistics::sum_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn sum_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_to_i32_scaled_buffer_size,
            statistics::sum_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i16,
        mean_scaled_buffer_size,
        mean_scaled_with_scratch_into,
        statistics::mean_i16_scaled_buffer_size,
        statistics::mean_i16_scaled_to_device_with_scratch
    );

    pub fn mean_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::mean_i16_scaled_buffer_size,
            statistics::mean_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i16,
        standard_deviation_scaled_buffer_size,
        standard_deviation_scaled_with_scratch_into,
        statistics::standard_deviation_i16_scaled_buffer_size,
        statistics::standard_deviation_i16_scaled_to_device_with_scratch
    );

    pub fn standard_deviation_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::standard_deviation_i16_scaled_buffer_size,
            statistics::standard_deviation_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i32,
        standard_deviation_to_i32_scaled_buffer_size,
        standard_deviation_to_i32_scaled_with_scratch_into,
        statistics::standard_deviation_i16_to_i32_scaled_buffer_size,
        statistics::standard_deviation_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn standard_deviation_to_i32_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::standard_deviation_i16_to_i32_scaled_buffer_size,
            statistics::standard_deviation_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i32,
        norm_inf_to_i32_scaled_buffer_size,
        norm_inf_to_i32_scaled_with_scratch_into,
        statistics::norm_inf_i16_to_i32_scaled_buffer_size,
        statistics::norm_inf_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn norm_inf_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_inf_i16_to_i32_scaled_buffer_size,
            statistics::norm_inf_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i32,
        norm_l1_to_i32_scaled_buffer_size,
        norm_l1_to_i32_scaled_with_scratch_into,
        statistics::norm_l1_i16_to_i32_scaled_buffer_size,
        statistics::norm_l1_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn norm_l1_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l1_i16_to_i32_scaled_buffer_size,
            statistics::norm_l1_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i64,
        norm_l1_to_i64_scaled_buffer_size,
        norm_l1_to_i64_scaled_with_scratch_into,
        statistics::norm_l1_i16_to_i64_scaled_buffer_size,
        statistics::norm_l1_i16_to_i64_scaled_to_device_with_scratch
    );

    pub fn norm_l1_to_i64_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i64>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l1_i16_to_i64_scaled_buffer_size,
            statistics::norm_l1_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i32,
        norm_l2_to_i32_scaled_buffer_size,
        norm_l2_to_i32_scaled_with_scratch_into,
        statistics::norm_l2_i16_to_i32_scaled_buffer_size,
        statistics::norm_l2_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn norm_l2_to_i32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_i16_to_i32_scaled_buffer_size,
            statistics::norm_l2_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i16,
        i64,
        norm_l2_squared_to_i64_scaled_buffer_size,
        norm_l2_squared_to_i64_scaled_with_scratch_into,
        statistics::norm_l2_squared_i16_to_i64_scaled_buffer_size,
        statistics::norm_l2_squared_i16_to_i64_scaled_to_device_with_scratch
    );

    pub fn norm_l2_squared_to_i64_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::norm_l2_squared_i16_to_i64_scaled_buffer_size,
            statistics::norm_l2_squared_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

    pub fn norm_diff_inf_to_f32(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_inf_i16_to_f32_buffer_size,
            statistics::norm_diff_inf_i16_to_f32_to_device_with_scratch,
        )
    }

    pub fn norm_diff_l1_to_f32(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l1_i16_to_f32_buffer_size,
            statistics::norm_diff_l1_i16_to_f32_to_device_with_scratch,
        )
    }

    pub fn norm_diff_l2_to_f32(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.binary_statistic_output(
            other,
            statistics::norm_diff_l2_i16_to_f32_buffer_size,
            statistics::norm_diff_l2_i16_to_f32_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i16,
        i32,
        norm_diff_inf_to_i32_scaled_buffer_size,
        norm_diff_inf_to_i32_scaled_with_scratch_into,
        statistics::norm_diff_inf_i16_to_i32_scaled_buffer_size,
        statistics::norm_diff_inf_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn norm_diff_inf_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_inf_i16_to_i32_scaled_buffer_size,
            statistics::norm_diff_inf_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i16,
        i32,
        norm_diff_l1_to_i32_scaled_buffer_size,
        norm_diff_l1_to_i32_scaled_with_scratch_into,
        statistics::norm_diff_l1_i16_to_i32_scaled_buffer_size,
        statistics::norm_diff_l1_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn norm_diff_l1_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l1_i16_to_i32_scaled_buffer_size,
            statistics::norm_diff_l1_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i16,
        i64,
        norm_diff_l1_to_i64_scaled_buffer_size,
        norm_diff_l1_to_i64_scaled_with_scratch_into,
        statistics::norm_diff_l1_i16_to_i64_scaled_buffer_size,
        statistics::norm_diff_l1_i16_to_i64_scaled_to_device_with_scratch
    );

    pub fn norm_diff_l1_to_i64_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l1_i16_to_i64_scaled_buffer_size,
            statistics::norm_diff_l1_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i16,
        i32,
        norm_diff_l2_to_i32_scaled_buffer_size,
        norm_diff_l2_to_i32_scaled_with_scratch_into,
        statistics::norm_diff_l2_i16_to_i32_scaled_buffer_size,
        statistics::norm_diff_l2_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn norm_diff_l2_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_i16_to_i32_scaled_buffer_size,
            statistics::norm_diff_l2_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i16,
        i64,
        norm_diff_l2_squared_to_i64_scaled_buffer_size,
        norm_diff_l2_squared_to_i64_scaled_with_scratch_into,
        statistics::norm_diff_l2_squared_i16_to_i64_scaled_buffer_size,
        statistics::norm_diff_l2_squared_i16_to_i64_scaled_to_device_with_scratch
    );

    pub fn norm_diff_l2_squared_to_i64_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::norm_diff_l2_squared_i16_to_i64_scaled_buffer_size,
            statistics::norm_diff_l2_squared_i16_to_i64_scaled_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        i16,
        i64,
        dot_product_to_i64_buffer_size,
        dot_product_to_i64_with_scratch_into,
        statistics::dot_product_i16_to_i64_buffer_size,
        statistics::dot_product_i16_to_i64_to_device_with_scratch
    );

    pub fn dot_product_to_i64(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_to_i64_buffer_size,
            statistics::dot_product_i16_to_i64_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        i16,
        f32,
        dot_product_to_f32_buffer_size,
        dot_product_to_f32_with_scratch_into,
        statistics::dot_product_i16_to_f32_buffer_size,
        statistics::dot_product_i16_to_f32_to_device_with_scratch
    );

    pub fn dot_product_to_f32(
        self,
        other: &SignalView<'_, i16>,
    ) -> Result<SignalPipeline<'a, f32>> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_to_f32_buffer_size,
            statistics::dot_product_i16_to_f32_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i16,
        i16,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i16_scaled_buffer_size,
        statistics::dot_product_i16_scaled_to_device_with_scratch
    );

    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_scaled_buffer_size,
            statistics::dot_product_i16_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i16,
        i32,
        dot_product_to_i32_scaled_buffer_size,
        dot_product_to_i32_scaled_with_scratch_into,
        statistics::dot_product_i16_to_i32_scaled_buffer_size,
        statistics::dot_product_i16_to_i32_scaled_to_device_with_scratch
    );

    pub fn dot_product_to_i32_scaled(
        self,
        other: &SignalView<'_, i16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_to_i32_scaled_buffer_size,
            statistics::dot_product_i16_to_i32_scaled_to_device_with_scratch,
        )
    }

    impl_heterogeneous_binary_statistic_output_pipeline_methods!(
        i16,
        ComplexI16,
        ComplexI64,
        dot_product_with_complex_to_i64_complex_buffer_size,
        dot_product_with_complex_to_i64_complex_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_to_i64_complex_buffer_size,
        statistics::dot_product_i16_and_i16_complex_to_i64_complex_to_device_with_scratch
    );

    pub fn dot_product_with_complex_to_i64_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, ComplexI64>> {
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_i16_and_i16_complex_to_i64_complex_buffer_size,
            statistics::dot_product_i16_and_i16_complex_to_i64_complex_to_device_with_scratch,
        )
    }

    impl_heterogeneous_binary_statistic_output_pipeline_methods!(
        i16,
        ComplexI16,
        Complex32,
        dot_product_with_complex_to_f32_complex_buffer_size,
        dot_product_with_complex_to_f32_complex_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_to_f32_complex_buffer_size,
        statistics::dot_product_i16_and_i16_complex_to_f32_complex_to_device_with_scratch
    );

    pub fn dot_product_with_complex_to_f32_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, Complex32>> {
        self.heterogeneous_binary_statistic_output(
            other,
            statistics::dot_product_i16_and_i16_complex_to_f32_complex_buffer_size,
            statistics::dot_product_i16_and_i16_complex_to_f32_complex_to_device_with_scratch,
        )
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        i16,
        ComplexI16,
        ComplexI16,
        dot_product_with_complex_scaled_buffer_size,
        dot_product_with_complex_scaled_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_scaled_buffer_size,
        statistics::dot_product_i16_and_i16_complex_scaled_to_device_with_scratch
    );

    pub fn dot_product_with_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI16>> {
        self.scaled_heterogeneous_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_and_i16_complex_scaled_buffer_size,
            statistics::dot_product_i16_and_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        i16,
        ComplexI16,
        ComplexI32,
        dot_product_with_complex_to_i32_complex_scaled_buffer_size,
        dot_product_with_complex_to_i32_complex_scaled_with_scratch_into,
        statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_buffer_size,
        statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_to_device_with_scratch
    );

    pub fn dot_product_with_complex_to_i32_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.scaled_heterogeneous_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_buffer_size,
            statistics::dot_product_i16_and_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
        )
    }

    pub fn dot_product_with_complex_to_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.dot_product_with_complex_to_i32_complex_scaled(other, scale_factor)
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        i16,
        i32,
        i32,
        dot_product_with_i32_to_i32_scaled_buffer_size,
        dot_product_with_i32_to_i32_scaled_with_scratch_into,
        statistics::dot_product_i16_and_i32_to_i32_scaled_buffer_size,
        statistics::dot_product_i16_and_i32_to_i32_scaled_to_device_with_scratch
    );

    pub fn dot_product_with_i32_to_i32_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.scaled_heterogeneous_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_and_i32_to_i32_scaled_buffer_size,
            statistics::dot_product_i16_and_i32_to_i32_scaled_to_device_with_scratch,
        )
    }

    pub fn dot_product_with_i32_to_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.dot_product_with_i32_to_i32_scaled(other, scale_factor)
    }

    pub fn dot_product_with_to_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.dot_product_with_i32_to_i32_scaled(other, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i32> {
    impl_scaled_statistic_pipeline_methods!(
        i32,
        i32,
        sum_scaled_buffer_size,
        sum_scaled_with_scratch_into,
        statistics::sum_i32_scaled_buffer_size,
        statistics::sum_i32_scaled_to_device_with_scratch
    );

    pub fn sum_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i32_scaled_buffer_size,
            statistics::sum_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        i32,
        i32,
        mean_scaled_buffer_size,
        mean_scaled_with_scratch_into,
        statistics::mean_i32_scaled_buffer_size,
        statistics::mean_i32_scaled_to_device_with_scratch
    );

    pub fn mean_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::mean_i32_scaled_buffer_size,
            statistics::mean_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        i32,
        i32,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i32_scaled_buffer_size,
        statistics::dot_product_i32_scaled_to_device_with_scratch
    );

    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, i32>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i32_scaled_buffer_size,
            statistics::dot_product_i32_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_heterogeneous_binary_statistic_output_pipeline_methods!(
        i32,
        ComplexI32,
        ComplexI32,
        dot_product_with_complex_scaled_buffer_size,
        dot_product_with_complex_scaled_with_scratch_into,
        statistics::dot_product_i32_and_i32_complex_scaled_buffer_size,
        statistics::dot_product_i32_and_i32_complex_scaled_to_device_with_scratch
    );

    pub fn dot_product_with_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
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
        ComplexI16,
        ComplexI16,
        sum_scaled_buffer_size,
        sum_scaled_with_scratch_into,
        statistics::sum_i16_complex_scaled_buffer_size,
        statistics::sum_i16_complex_scaled_to_device_with_scratch
    );

    pub fn sum_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_complex_scaled_buffer_size,
            statistics::sum_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        ComplexI16,
        ComplexI32,
        sum_to_i32_complex_scaled_buffer_size,
        sum_to_i32_complex_scaled_with_scratch_into,
        statistics::sum_i16_complex_to_i32_complex_scaled_buffer_size,
        statistics::sum_i16_complex_to_i32_complex_scaled_to_device_with_scratch
    );

    pub fn sum_to_i32_complex_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::sum_i16_complex_to_i32_complex_scaled_buffer_size,
            statistics::sum_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_statistic_pipeline_methods!(
        ComplexI16,
        ComplexI16,
        mean_scaled_buffer_size,
        mean_scaled_with_scratch_into,
        statistics::mean_i16_complex_scaled_buffer_size,
        statistics::mean_i16_complex_scaled_to_device_with_scratch
    );

    pub fn mean_scaled(self, scale_factor: i32) -> Result<Self> {
        self.scaled_statistic_output(
            scale_factor,
            statistics::mean_i16_complex_scaled_buffer_size,
            statistics::mean_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        ComplexI16,
        ComplexI64,
        dot_product_to_i64_complex_buffer_size,
        dot_product_to_i64_complex_with_scratch_into,
        statistics::dot_product_i16_complex_to_i64_complex_buffer_size,
        statistics::dot_product_i16_complex_to_i64_complex_to_device_with_scratch
    );

    pub fn dot_product_to_i64_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, ComplexI64>> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_complex_to_i64_complex_buffer_size,
            statistics::dot_product_i16_complex_to_i64_complex_to_device_with_scratch,
        )
    }

    impl_binary_statistic_output_pipeline_methods!(
        ComplexI16,
        Complex32,
        dot_product_to_f32_complex_buffer_size,
        dot_product_to_f32_complex_with_scratch_into,
        statistics::dot_product_i16_complex_to_f32_complex_buffer_size,
        statistics::dot_product_i16_complex_to_f32_complex_to_device_with_scratch
    );

    pub fn dot_product_to_f32_complex(
        self,
        other: &SignalView<'_, ComplexI16>,
    ) -> Result<SignalPipeline<'a, Complex32>> {
        self.binary_statistic_output(
            other,
            statistics::dot_product_i16_complex_to_f32_complex_buffer_size,
            statistics::dot_product_i16_complex_to_f32_complex_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        ComplexI16,
        ComplexI16,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i16_complex_scaled_buffer_size,
        statistics::dot_product_i16_complex_scaled_to_device_with_scratch
    );

    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_complex_scaled_buffer_size,
            statistics::dot_product_i16_complex_scaled_to_device_with_scratch,
        )
    }

    impl_scaled_binary_statistic_output_pipeline_methods!(
        ComplexI16,
        ComplexI32,
        dot_product_to_i32_complex_scaled_buffer_size,
        dot_product_to_i32_complex_scaled_with_scratch_into,
        statistics::dot_product_i16_complex_to_i32_complex_scaled_buffer_size,
        statistics::dot_product_i16_complex_to_i32_complex_scaled_to_device_with_scratch
    );

    pub fn dot_product_to_i32_complex_scaled(
        self,
        other: &SignalView<'_, ComplexI16>,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, ComplexI32>> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i16_complex_to_i32_complex_scaled_buffer_size,
            statistics::dot_product_i16_complex_to_i32_complex_scaled_to_device_with_scratch,
        )
    }
}

impl<'a> SignalPipeline<'a, ComplexI32> {
    impl_scaled_binary_statistic_output_pipeline_methods!(
        ComplexI32,
        ComplexI32,
        dot_product_scaled_buffer_size,
        dot_product_scaled_with_scratch_into,
        statistics::dot_product_i32_complex_scaled_buffer_size,
        statistics::dot_product_i32_complex_scaled_to_device_with_scratch
    );

    pub fn dot_product_scaled(
        self,
        other: &SignalView<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.scaled_binary_statistic_output(
            other,
            scale_factor,
            statistics::dot_product_i32_complex_scaled_buffer_size,
            statistics::dot_product_i32_complex_scaled_to_device_with_scratch,
        )
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: CopySignal<T> + EveryStatisticSignal<T>,
{
    pub fn min_every(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.every_statistic(other, <Self as EveryStatisticSignal<T>>::min_every_in_place)
    }

    pub fn max_every(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.every_statistic(other, <Self as EveryStatisticSignal<T>>::max_every_in_place)
    }

    fn every_statistic(
        mut self,
        other: &SignalView<'_, T>,
        statistic: fn(&StreamContext, &SignalView<'_, T>, &mut SignalViewMut<'_, T>) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut()?;
                statistic(self.stream_context, other, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self.workspace.signal::<T>(source.len())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopySignal<T>>::copy(self.stream_context, source, &mut destination_view)?;
                statistic(self.stream_context, other, &mut destination_view)?;
                self.backing = SignalBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
