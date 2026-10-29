use half::{bf16, f16};

use crate::{
    backend::{
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorId, TensorPlacement, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::support::sdpa_scalar_tensor,
        error::{Error as FrontendError, SdpaRequirement},
        graph::{Graph, reduce_last_axis},
        operation::{
            AttentionBackwardAuxGradients, AttentionBackwardConfig, AttentionDropoutMask,
            AttentionDropoutMode, PointwiseOperation, ReductionConfig, SdpaScoreModifiers,
        },
    },
    math::NanPropagation,
    scalar::ScalarValue,
};

impl Graph {
    /// Reduces the saved output and output-gradient product before sink and dropout corrections.
    pub(crate) fn sdpa_backward_output_dot(
        &mut self,
        output_gradient: TensorId,
        output: TensorId,
        shape: Shape,
        compute_type: DataType,
        intermediate_type: DataType,
    ) -> Result<TensorId> {
        let product = self.sdpa_binary(
            PointwiseMode::Mul,
            output_gradient,
            output,
            intermediate_type,
            compute_type,
        )?;
        let sum = self.tensor(TensorSpec::new(intermediate_type, shape).virtual_tensor());
        self.reduction_into(
            product,
            sum,
            &ReductionConfig::new(ReduceTensorOperator::Add).with_compute_type(compute_type),
        )?;
        Ok(sum)
    }

    pub(crate) fn sdpa_add_backward_sink_token_gradient(
        &mut self,
        d_sink_token: Option<TensorId>,
        modifiers: &SdpaScoreModifiers,
        aux_gradients: AttentionBackwardAuxGradients,
        stats: TensorId,
        softmax_sum: TensorId,
        scores_shape: &Shape,
        compute_type: DataType,
        intermediate_type: DataType,
    ) -> Result<()> {
        let Some(d_sink_token) = d_sink_token else {
            if aux_gradients.sink_token() {
                return Err(FrontendError::SdpaRequires {
                    requirement: SdpaRequirement::SinkToken,
                    operation: "sdpa backward sink gradient",
                }
                .into());
            }
            return Ok(());
        };
        let Some(sink_token) = modifiers.sink_token() else {
            return Err(FrontendError::SdpaRequires {
                requirement: SdpaRequirement::SinkToken,
                operation: "sdpa backward sink gradient",
            }
            .into());
        };

        let sink_minus_stats = self.sdpa_binary(
            PointwiseMode::Sub,
            sink_token,
            stats,
            intermediate_type,
            compute_type,
        )?;
        let exp_sink = self.tensor(
            TensorSpec::new(intermediate_type, reduce_last_axis(scores_shape)?).virtual_tensor(),
        );
        self.pointwise_into(PointwiseOperation::Unary {
            name: None,
            mode: PointwiseMode::Exp,
            input: sink_minus_stats,
            output: exp_sink,
            compute_type: Some(compute_type),
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            axis: None,
        })?;
        let per_token_grad = self.sdpa_binary(
            PointwiseMode::Mul,
            exp_sink,
            softmax_sum,
            intermediate_type,
            compute_type,
        )?;
        self.reduction_into(
            per_token_grad,
            d_sink_token,
            &ReductionConfig::new(ReduceTensorOperator::Add).with_compute_type(compute_type),
        )?;
        Ok(())
    }

    /// Corrects the saved-output row dot after any sink-gradient branch.
    /// Keep identity multiplication explicit for the native attention pattern matcher.
    pub(crate) fn sdpa_backward_scale_softmax_sum(
        &mut self,
        softmax_sum: TensorId,
        config: &AttentionBackwardConfig,
        compute_type: DataType,
        intermediate_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        let inverse =
            self.sdpa_backward_dropout_scale_inverse(config, compute_type, intermediate_type)?;
        self.sdpa_binary(
            PointwiseMode::Mul,
            softmax_sum,
            inverse,
            output_type,
            compute_type,
        )
    }

    fn sdpa_backward_dropout_scale_inverse(
        &mut self,
        config: &AttentionBackwardConfig,
        compute_type: DataType,
        intermediate_type: DataType,
    ) -> Result<TensorId> {
        let scale = match config.dropout_mode() {
            AttentionDropoutMode::None => {
                return Ok(self.tensor(sdpa_scalar_tensor(compute_type, 1.0)?));
            }
            AttentionDropoutMode::Philox(dropout) => {
                return Ok(self.tensor(sdpa_scalar_tensor(compute_type, dropout.inverse_scale())?));
            }
            AttentionDropoutMode::CustomMask(AttentionDropoutMask { scale, .. }) => {
                if let Some(inverse) = config.dropout_scale_inverse() {
                    return Ok(inverse);
                }
                scale
            }
        };
        let constant = match self.tensor_spec(scale)?.placement {
            TensorPlacement::Constant(ScalarValue::F32(value)) => Some(value),
            TensorPlacement::Constant(ScalarValue::F16(bits)) => {
                Some(f16::from_bits(bits).to_f32())
            }
            TensorPlacement::Constant(ScalarValue::Bf16(bits)) => {
                Some(bf16::from_bits(bits).to_f32())
            }
            _ => None,
        };
        if let Some(value) = constant {
            return Ok(self.tensor(sdpa_scalar_tensor(compute_type, value.recip())?));
        }
        let one = self.tensor(sdpa_scalar_tensor(compute_type, 1.0)?);
        self.sdpa_binary(
            PointwiseMode::Div,
            one,
            scale,
            intermediate_type,
            compute_type,
        )
    }
}
