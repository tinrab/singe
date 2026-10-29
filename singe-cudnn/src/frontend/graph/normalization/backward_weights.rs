use crate::{
    backend::tensor::Shape,
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig, infer_normalization_channel_dimensions},
        operation::{
            BatchNormalizationBackwardWeightsConfig, BatchNormalizationBackwardWeightsInputs,
            BatchNormalizationBackwardWeightsOutputs, NormalizationOperation,
            NormalizationSavedStatistics, Operation,
        },
    },
};

impl Graph {
    pub(crate) fn validate_batch_normalization_backward_weights(
        &self,
        inputs: BatchNormalizationBackwardWeightsInputs,
        outputs: BatchNormalizationBackwardWeightsOutputs,
    ) -> Result<()> {
        let BatchNormalizationBackwardWeightsInputs {
            input,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            output_gradient: dy,
            scale,
        } = inputs;
        let BatchNormalizationBackwardWeightsOutputs {
            scale_gradient: dscale,
            bias_gradient,
            equivalent_bias: eq_bias,
            equivalent_output_gradient_scale: eq_scale_dy,
            equivalent_input_scale: eq_scale_x,
        } = outputs;
        let input_tensor = self.tensor_spec(input)?;
        self.validate_tensor_dimensions(
            dy,
            input_tensor.shape.dimensions(),
            "dbn weight dy shape",
        )?;
        let channel_dimensions = infer_normalization_channel_dimensions(&input_tensor.shape)?;
        self.validate_tensors_dimensions(
            [
                (scale, "dbn weight scale shape"),
                (mean, "dbn weight mean shape"),
                (inv_variance, "dbn weight inv variance shape"),
                (dscale, "dbn weight dscale shape"),
                (bias_gradient, "dbn weight bias_gradient shape"),
                (eq_bias, "dbn weight eq bias shape"),
                (eq_scale_dy, "dbn weight eq scale dy shape"),
                (eq_scale_x, "dbn weight eq scale x shape"),
            ],
            &channel_dimensions,
        )?;
        Ok(())
    }

    /// Adds the `BatchNorm` weight-gradient helper and its equivalent affine coefficients.
    /// Math precision follows `BatchNormalizationBackwardWeightsConfig`, graph policy and then the FP32 fallback.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, an output gradient whose dimensions differ from the input, or mismatched per-channel statistics, scale, gradient, or equivalent-coefficient dimensions.
    pub fn batch_normalization_backward_weights_into(
        &mut self,
        inputs: BatchNormalizationBackwardWeightsInputs,
        outputs: BatchNormalizationBackwardWeightsOutputs,
        config: BatchNormalizationBackwardWeightsConfig,
    ) -> Result<()> {
        self.validate_batch_normalization_backward_weights(inputs, outputs)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::BatchNormalizationBackwardWeights {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates scale/bias gradients and equivalent coefficients for `BatchNorm` backward.
    /// Each result selects its placement and encoding independently, with an FP32 fallback.
    /// Compute precision is selected separately by the operation configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, an output gradient whose dimensions differ from the input, or mismatched per-channel statistics, scale, gradient, or equivalent-coefficient dimensions. Inferred output strides must also be representable.
    pub fn batch_normalization_backward_weights(
        &mut self,
        inputs: BatchNormalizationBackwardWeightsInputs,
        outputs: BatchNormalizationBackwardWeightsOutputs<TensorOutputConfig>,
        config: BatchNormalizationBackwardWeightsConfig,
    ) -> Result<BatchNormalizationBackwardWeightsOutputs> {
        let shape = Shape::channels_last(infer_normalization_channel_dimensions(
            self.shape(inputs.input)?,
        )?)?;
        let policy = self.config.data_type_policy;
        let requests = BatchNormalizationBackwardWeightsOutputs {
            scale_gradient: (outputs.scale_gradient, shape.clone(), "DSCALE"),
            bias_gradient: (outputs.bias_gradient, shape.clone(), "DBIAS"),
            equivalent_bias: (outputs.equivalent_bias, shape.clone(), "EQ_BIAS"),
            equivalent_output_gradient_scale: (
                outputs.equivalent_output_gradient_scale,
                shape.clone(),
                "EQ_SCALE_DY",
            ),
            equivalent_input_scale: (outputs.equivalent_input_scale, shape, "EQ_SCALE_X"),
        };
        let checkpoint = self.mutation_checkpoint();
        let outputs = requests.map(|(request, shape, port)| {
            self.inferred_tensor(
                request.into_spec(policy, DataType::F32, shape),
                config.name(),
                port,
            )
        });
        if let Err(error) = self.batch_normalization_backward_weights_into(inputs, outputs, config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
