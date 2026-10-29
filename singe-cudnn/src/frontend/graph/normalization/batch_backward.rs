use crate::{
    backend::tensor::Shape,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig, infer_normalization_channel_dimensions},
        operation::{
            BatchNormalizationBackwardConfig, BatchNormalizationBackwardInputs,
            BatchNormalizationBackwardOutputs, NormalizationOperation,
            NormalizationSavedStatistics, Operation,
        },
    },
};

impl Graph {
    pub(crate) fn validate_batch_normalization_backward(
        &self,
        inputs: BatchNormalizationBackwardInputs,
        outputs: BatchNormalizationBackwardOutputs,
        config: &BatchNormalizationBackwardConfig,
    ) -> Result<()> {
        let BatchNormalizationBackwardInputs {
            input,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            output_gradient: dy,
            scale,
        } = inputs;
        let BatchNormalizationBackwardOutputs {
            input_gradient: dx,
            scale_gradient: dscale,
            bias_gradient,
        } = outputs;
        let input_tensor = self.tensor_spec(input)?;
        let expected_channel = infer_normalization_channel_dimensions(&input_tensor.shape)?;

        self.validate_tensor_dimensions(
            dy,
            input_tensor.shape.dimensions(),
            "batch norm backward dy shape",
        )?;
        self.validate_tensor_dimensions(
            dx,
            input_tensor.shape.dimensions(),
            "batch norm backward dx shape",
        )?;
        self.validate_tensors_dimensions(
            [
                (scale, "batch norm backward scale shape"),
                (dscale, "batch norm backward dscale shape"),
                (bias_gradient, "batch norm backward bias_gradient shape"),
                (mean, "batch norm backward mean shape"),
                (inv_variance, "batch norm backward inv_variance shape"),
            ],
            &expected_channel,
        )?;
        for peer_stat in config.peer_stats() {
            self.tensor_spec(*peer_stat)?;
        }

        Ok(())
    }

    /// Adds `BatchNorm` backward with explicit gradients and optional peer statistics.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, mismatched input/output-gradient dimensions, or saved-statistic, scale, or affine-gradient dimensions that differ from the per-channel shape.
    pub fn batch_normalization_backward_into(
        &mut self,
        inputs: BatchNormalizationBackwardInputs,
        outputs: BatchNormalizationBackwardOutputs,
        config: BatchNormalizationBackwardConfig,
    ) -> Result<()> {
        self.validate_batch_normalization_backward(inputs, outputs, &config)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::BatchBackward {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates the input, scale and bias gradients for `BatchNorm` backward.
    /// Each output selects its own placement and encoding.
    /// Inferred gradients use NHWC strides independently of input layouts, matching C++.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, mismatched input/output-gradient dimensions, or saved-statistic, scale, or affine-gradient dimensions that differ from the per-channel shape. Inferred output strides must also be representable.
    pub fn batch_normalization_backward(
        &mut self,
        inputs: BatchNormalizationBackwardInputs,
        outputs: BatchNormalizationBackwardOutputs<TensorOutputConfig>,
        config: BatchNormalizationBackwardConfig,
    ) -> Result<BatchNormalizationBackwardOutputs> {
        let input = self.tensor_spec(inputs.input)?;
        let scale = self.tensor_spec(inputs.scale)?;
        let affine_shape =
            Shape::channels_last(infer_normalization_channel_dimensions(&input.shape)?)?;
        let input_shape = Shape::channels_last(input.shape.dimensions().to_vec())?;
        let input_gradient_spec = outputs.input_gradient.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input_shape,
        );
        let scale_gradient_spec = outputs.scale_gradient.into_spec(
            self.config.data_type_policy,
            scale.data_type,
            affine_shape.clone(),
        );
        let bias_gradient_spec = outputs.bias_gradient.into_spec(
            self.config.data_type_policy,
            scale.data_type,
            affine_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let outputs = BatchNormalizationBackwardOutputs {
            input_gradient: self.inferred_tensor(input_gradient_spec, config.name(), "DX"),
            scale_gradient: self.inferred_tensor(scale_gradient_spec, config.name(), "DSCALE"),
            bias_gradient: self.inferred_tensor(bias_gradient_spec, config.name(), "DBIAS"),
        };
        if let Err(error) = self.batch_normalization_backward_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
