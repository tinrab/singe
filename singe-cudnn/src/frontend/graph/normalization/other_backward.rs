use crate::{
    backend::tensor::Shape,
    error::Result,
    frontend::{
        graph::{
            Graph, TensorOutputConfig, infer_adaptive_layer_normalization_stats_dimensions,
            infer_instance_normalization_stats_dimensions, infer_normalization_channel_dimensions,
            validate_normalization_broadcast_shape,
        },
        operation::{
            AdaptiveLayerNormalizationBackwardInputs, AdaptiveLayerNormalizationBackwardOutputs,
            InstanceNormalizationBackwardInputs, InstanceNormalizationBackwardOutputs,
            NormalizationConfig, NormalizationOperation, Operation,
        },
    },
};

impl Graph {
    pub(crate) fn validate_instance_normalization_backward(
        &self,
        inputs: InstanceNormalizationBackwardInputs,
        outputs: InstanceNormalizationBackwardOutputs,
    ) -> Result<()> {
        let input_shape = self.shape(inputs.input)?;
        let affine_dimensions = infer_normalization_channel_dimensions(input_shape)?;
        let statistics_dimensions = infer_instance_normalization_stats_dimensions(input_shape)?;
        self.validate_tensors_dimensions(
            [
                (
                    inputs.output_gradient,
                    "instance norm output gradient shape",
                ),
                (outputs.input_gradient, "instance norm input gradient shape"),
            ],
            input_shape.dimensions(),
        )?;
        self.validate_tensors_dimensions(
            [
                (inputs.scale, "instance norm scale shape"),
                (outputs.scale_gradient, "instance norm scale gradient shape"),
                (outputs.bias_gradient, "instance norm bias gradient shape"),
            ],
            &affine_dimensions,
        )?;
        self.validate_tensors_dimensions(
            [
                (inputs.statistics.mean, "instance norm mean shape"),
                (
                    inputs.statistics.inv_variance,
                    "instance norm inverse variance shape",
                ),
            ],
            &statistics_dimensions,
        )
    }

    /// Adds `InstanceNorm` backward using statistics from its forward operation.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions.
    pub fn instance_normalization_backward_into(
        &mut self,
        inputs: InstanceNormalizationBackwardInputs,
        outputs: InstanceNormalizationBackwardOutputs,
        config: NormalizationConfig,
    ) -> Result<()> {
        self.validate_instance_normalization_backward(inputs, outputs)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::InstanceBackward {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates input, scale and bias gradients for `InstanceNorm` backward.
    /// Each output selects its own placement and encoding.
    /// Inferred gradients use NHWC strides independently of input layouts, matching C++.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions. Inferred output or statistic layouts can also fail if their strides are unrepresentable.
    pub fn instance_normalization_backward(
        &mut self,
        inputs: InstanceNormalizationBackwardInputs,
        outputs: InstanceNormalizationBackwardOutputs<TensorOutputConfig>,
        config: NormalizationConfig,
    ) -> Result<InstanceNormalizationBackwardOutputs> {
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
        let outputs = InstanceNormalizationBackwardOutputs {
            input_gradient: self.inferred_tensor(input_gradient_spec, config.name(), "DX"),
            scale_gradient: self.inferred_tensor(scale_gradient_spec, config.name(), "DSCALE"),
            bias_gradient: self.inferred_tensor(bias_gradient_spec, config.name(), "DBIAS"),
        };
        if let Err(error) = self.instance_normalization_backward_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    pub(crate) fn validate_adaptive_layer_normalization_backward(
        &self,
        inputs: AdaptiveLayerNormalizationBackwardInputs,
        outputs: AdaptiveLayerNormalizationBackwardOutputs,
    ) -> Result<()> {
        let input_shape = self.shape(inputs.input)?;
        let scale_shape = self.shape(inputs.scale)?;
        let statistics_dimensions =
            infer_adaptive_layer_normalization_stats_dimensions(input_shape, scale_shape)?;
        self.validate_tensors_dimensions(
            [
                (
                    inputs.output_gradient,
                    "adaptive layer norm output gradient shape",
                ),
                (
                    outputs.input_gradient,
                    "adaptive layer norm input gradient shape",
                ),
            ],
            input_shape.dimensions(),
        )?;
        validate_normalization_broadcast_shape(input_shape, self.shape(outputs.scale_gradient)?)?;
        if let Some(bias_gradient) = outputs.bias_gradient {
            validate_normalization_broadcast_shape(input_shape, self.shape(bias_gradient)?)?;
        }
        self.validate_tensors_dimensions(
            [
                (inputs.statistics.mean, "adaptive layer norm mean shape"),
                (
                    inputs.statistics.inv_variance,
                    "adaptive layer norm inverse variance shape",
                ),
            ],
            &statistics_dimensions,
        )
    }

    /// Adds `AdaLayerNorm` backward with an optional bias gradient.
    /// Explicit affine gradients retain independent broadcast dimensions and strides.
    /// Native support selection determines executable combinations.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions.
    pub fn adaptive_layer_normalization_backward_into(
        &mut self,
        inputs: AdaptiveLayerNormalizationBackwardInputs,
        outputs: AdaptiveLayerNormalizationBackwardOutputs,
        config: NormalizationConfig,
    ) -> Result<()> {
        self.validate_adaptive_layer_normalization_backward(inputs, outputs)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::AdaptiveLayerBackward {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates `AdaLayerNorm` gradients, including the requested optional bias gradient.
    /// Scale gradients retain the scale's dimensions; each output selects its placement and encoding.
    /// Inferred gradients use row-major strides independently of input layouts, matching C++.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions. Inferred output or statistic layouts can also fail if their strides are unrepresentable.
    pub fn adaptive_layer_normalization_backward(
        &mut self,
        inputs: AdaptiveLayerNormalizationBackwardInputs,
        outputs: AdaptiveLayerNormalizationBackwardOutputs<TensorOutputConfig>,
        config: NormalizationConfig,
    ) -> Result<AdaptiveLayerNormalizationBackwardOutputs> {
        let input = self.tensor_spec(inputs.input)?;
        let scale = self.tensor_spec(inputs.scale)?;
        let affine_shape = Shape::contiguous(scale.shape.dimensions().to_vec())?;
        let input_shape = Shape::contiguous(input.shape.dimensions().to_vec())?;
        let input_gradient_spec = outputs.input_gradient.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input_shape,
        );
        let bias_gradient_spec = outputs.bias_gradient.map(|bias| {
            bias.into_spec(
                self.config.data_type_policy,
                scale.data_type,
                affine_shape.clone(),
            )
        });
        let scale_gradient_spec = outputs.scale_gradient.into_spec(
            self.config.data_type_policy,
            scale.data_type,
            affine_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let outputs = AdaptiveLayerNormalizationBackwardOutputs {
            input_gradient: self.inferred_tensor(input_gradient_spec, config.name(), "DX"),
            scale_gradient: self.inferred_tensor(scale_gradient_spec, config.name(), "DSCALE"),
            bias_gradient: bias_gradient_spec
                .map(|bias| self.inferred_tensor(bias, config.name(), "DBIAS")),
        };
        if let Err(error) = self.adaptive_layer_normalization_backward_into(inputs, outputs, config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
