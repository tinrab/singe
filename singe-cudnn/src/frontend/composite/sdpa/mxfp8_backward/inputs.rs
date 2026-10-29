use crate::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{TensorId, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaMxfp8BackwardInputs,
            support::{
                SDPA_MXFP8_BACKWARD_SCALE_D_O_SHAPE, SDPA_MXFP8_BACKWARD_SCALE_D_O_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_K_SHAPE, SDPA_MXFP8_BACKWARD_SCALE_K_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_Q_SHAPE, SDPA_MXFP8_BACKWARD_SCALE_Q_T_SHAPE,
                SDPA_MXFP8_BACKWARD_SCALE_V_SHAPE,
            },
        },
        error::Error as FrontendError,
        graph::{BlockScaleDequantizeInputs, Graph, TensorOutputConfig},
        operation::{BlockScaleDequantizeConfig, PointwiseOperation},
    },
    math::NanPropagation,
};

impl Graph {
    pub(crate) fn validate_mxfp8_backward_scale_dimensions(
        &self,
        inputs: &SdpaMxfp8BackwardInputs,
    ) -> Result<()> {
        for (values, scale, axis, name) in [
            (
                inputs.query,
                inputs.scale_query,
                3,
                SDPA_MXFP8_BACKWARD_SCALE_Q_SHAPE,
            ),
            (
                inputs.query,
                inputs.sequence_scale_query,
                2,
                SDPA_MXFP8_BACKWARD_SCALE_Q_T_SHAPE,
            ),
            (
                inputs.key,
                inputs.scale_key,
                3,
                SDPA_MXFP8_BACKWARD_SCALE_K_SHAPE,
            ),
            (
                inputs.key,
                inputs.sequence_scale_key,
                2,
                SDPA_MXFP8_BACKWARD_SCALE_K_T_SHAPE,
            ),
            (
                inputs.value,
                inputs.scale_value,
                3,
                SDPA_MXFP8_BACKWARD_SCALE_V_SHAPE,
            ),
            (
                inputs.output_gradient,
                inputs.scale_output_gradient,
                3,
                SDPA_MXFP8_BACKWARD_SCALE_D_O_SHAPE,
            ),
            (
                inputs.output_gradient,
                inputs.sequence_scale_output_gradient,
                2,
                SDPA_MXFP8_BACKWARD_SCALE_D_O_T_SHAPE,
            ),
        ] {
            self.validate_tensor_rank(scale, 4, name)?;
            let mut expected = self.tensor_spec(values)?.shape.dimensions().to_vec();
            expected[axis] = expected[axis] / 32 + i64::from(expected[axis] % 32 != 0);
            let actual = self.tensor_spec(scale)?.shape.dimensions();
            if actual[..2] != expected[..2] || actual[2] < expected[2] || actual[3] < expected[3] {
                return Err(FrontendError::TensorLayoutMismatch { tensor_id: scale, operation: name, reason: format!(
                        "expected batch/head dimensions {:?} and padded scale dimensions at least {:?}, got {actual:?}",
                        &expected[..2],
                        &expected[2..],
                    ).into() }.into());
            }
        }
        Ok(())
    }

    pub(crate) fn mxfp8_attention_scale_tensor(
        &mut self,
        attention_scale: TensorId,
        name: &'static str,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.validate_tensor_scalar_shape(attention_scale, name)?;
        if self.tensor_spec(attention_scale)?.data_type == compute_type {
            return Ok(attention_scale);
        }
        let converted = self.pointwise(
            PointwiseOperation::unary(
                PointwiseMode::Identity,
                attention_scale,
                TensorOutputConfig::new().with_data_type(compute_type),
            )
            .with_compute_type(compute_type)
            .with_nan_propagation(NanPropagation::NotPropagate),
        )?;
        self.modify_tensor(converted, TensorSpec::virtual_tensor)?;
        Ok(converted)
    }

    pub(crate) fn mxfp8_dequantize_input_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output_data_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.block_scale_dequantize(
            BlockScaleDequantizeInputs::new(input, scale),
            BlockScaleDequantizeConfig::new([1, 32])
                .with_compute_type(compute_type)
                .with_output_data_type(output_data_type),
        )
    }

    pub(crate) fn mxfp8_dequantize_transposed_input_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output_data_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let input_t = self.transpose_last_two_view(input)?;
        let scale_t = self.transpose_last_two_view(scale)?;
        self.block_scale_dequantize(
            BlockScaleDequantizeInputs::new(input_t, scale_t),
            BlockScaleDequantizeConfig::new([32, 1])
                .with_compute_type(compute_type)
                .with_output_data_type(output_data_type),
        )
    }

    pub(crate) fn mxfp8_dequantize_sequence_scaled_input_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output_data_type: DataType,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let input_t = self.transpose_last_two_view(input)?;
        let scale_t = self.transpose_last_two_view(scale)?;
        let dequantized_t = self.block_scale_dequantize(
            BlockScaleDequantizeInputs::new(input_t, scale_t),
            BlockScaleDequantizeConfig::new([1, 32])
                .with_compute_type(compute_type)
                .with_output_data_type(output_data_type),
        )?;
        self.transpose_last_two_view(dequantized_t)
    }
}
