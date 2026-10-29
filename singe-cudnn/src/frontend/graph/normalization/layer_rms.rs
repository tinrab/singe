use crate::{
    backend::{normalization::BackendNormalizationForwardPhase, tensor::TensorId},
    data_type::DataType,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig, validate_normalization_broadcast_shape},
        operation::{
            LayerNormalizationAffineGradients, LayerNormalizationBackwardConfig,
            LayerNormalizationBackwardInputs, LayerNormalizationBackwardOutputs,
            LayerNormalizationConfig, LayerNormalizationInputs, LayerNormalizationOutputs,
            NormalizationOperation, Operation, RmsNormalizationAffineGradients,
            RmsNormalizationBackwardConfig, RmsNormalizationBackwardInputs,
            RmsNormalizationBackwardOutputs, RmsNormalizationConfig, RmsNormalizationInputs,
            RmsNormalizationOutputs, RmsNormalizationStatistics,
        },
    },
};

impl Graph {
    #[expect(
        clippy::too_many_lines,
        reason = "the normalization graph keeps its coupled shape and execution contracts together"
    )]
    fn normalization_statistics_dimensions(
        &self,
        input: TensorId,
        axes: Option<&[usize]>,
        scale: Option<TensorId>,
        statistics: Option<TensorId>,
        phase: BackendNormalizationForwardPhase,
    ) -> Result<Vec<i64>> {
        let input_shape = &self.tensor_spec(input)?.shape;
        let dimensions = input_shape.dimensions();
        let rank = dimensions.len();
        if rank < 2 {
            return Err(FrontendError::NormalizationAxes {
                reason: "normalization requires at least two dimensions".into(),
            }
            .into());
        }
        let scale_shape = scale
            .map(|id| self.tensor_spec(id).map(|tensor| &tensor.shape))
            .transpose()?;
        let statistics_shape = statistics
            .map(|id| self.tensor_spec(id).map(|tensor| &tensor.shape))
            .transpose()?;
        for shape in scale_shape.into_iter().chain(statistics_shape) {
            validate_normalization_broadcast_shape(input_shape, shape)?;
        }
        let normalized = if let Some(axes) = axes {
            if axes.is_empty() {
                return Err(FrontendError::NormalizationAxes {
                    reason: "normalization axes cannot be empty".into(),
                }
                .into());
            }
            let mut selected = vec![false; rank];
            for &axis in axes {
                if axis >= rank || selected[axis] {
                    return Err(FrontendError::NormalizationAxes {
                        reason: "normalization axes must be distinct and within the input rank"
                            .into(),
                    }
                    .into());
                }
                selected[axis] = true;
            }
            selected
        } else if let Some(stats) = statistics_shape {
            stats
                .dimensions()
                .iter()
                .map(|&dimension| dimension == 1)
                .collect()
        } else if let Some(scale) = scale_shape {
            scale
                .dimensions()
                .iter()
                .map(|&dimension| dimension != 1)
                .collect()
        } else {
            if dimensions[1..]
                .iter()
                .filter(|&&dimension| dimension != 1)
                .count()
                > 1
            {
                return Err(FrontendError::NormalizationAxes { reason: "normalization without scale requires explicit axes or saved-statistics dimensions for this input".into() }.into());
            }
            (0..rank).map(|axis| axis > 0).collect()
        };
        let statistics_dimensions = dimensions
            .iter()
            .zip(&normalized)
            .map(|(&dimension, &reduce)| if reduce { 1 } else { dimension })
            .collect::<Vec<_>>();
        if let Some(stats) = statistics_shape
            && stats.dimensions() != statistics_dimensions
        {
            return Err(FrontendError::NormalizationAxes {
                reason: "saved statistics do not match the normalization axes".into(),
            }
            .into());
        }
        if let Some(scale) = scale_shape {
            for ((&scale_dimension, &input_dimension), &reduce) in
                scale.dimensions().iter().zip(dimensions).zip(&normalized)
            {
                if scale_dimension != 1 && (!reduce || scale_dimension != input_dimension) {
                    return Err(FrontendError::NormalizationAxes {
                        reason: "scale dimensions must belong to normalized axes".into(),
                    }
                    .into());
                }
                if phase == BackendNormalizationForwardPhase::Inference
                    && reduce
                    && input_dimension != 1
                    && scale_dimension == 1
                {
                    return Err(FrontendError::NormalizationAxes {
                        reason: "inference axes must be expressible by the scale dimensions".into(),
                    }
                    .into());
                }
            }
        } else if phase == BackendNormalizationForwardPhase::Inference {
            let mut native_statistics = vec![1; rank];
            native_statistics[0] = dimensions[0];
            if dimensions[1..]
                .iter()
                .filter(|&&dimension| dimension != 1)
                .count()
                > 1
                || statistics_dimensions != native_statistics
            {
                return Err(FrontendError::NormalizationAxes { reason: "scale-free inference requires the native rank-based normalization axes; reshape the input or supply a scale".into() }.into());
            }
        }
        Ok(statistics_dimensions)
    }

    pub(crate) fn validate_layer_normalization(
        &self,
        inputs: LayerNormalizationInputs,
        outputs: LayerNormalizationOutputs,
        config: &LayerNormalizationConfig,
    ) -> Result<()> {
        let statistics = outputs.statistics.saved();
        let statistics_dimensions = self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.affine.scale(),
            statistics.map(|stats| stats.mean),
            outputs.statistics.phase(),
        )?;
        self.validate_tensor_dimensions(
            outputs.output,
            self.shape(inputs.input)?.dimensions(),
            "layer norm output shape",
        )?;
        if let Some(bias) = inputs.affine.bias() {
            validate_normalization_broadcast_shape(self.shape(inputs.input)?, self.shape(bias)?)?;
        }
        if let Some(statistics) = statistics {
            self.validate_tensor_dimensions(
                statistics.mean,
                &statistics_dimensions,
                "layer norm mean shape",
            )?;
            self.validate_tensor_dimensions(
                statistics.inv_variance,
                &statistics_dimensions,
                "layer norm inverse variance shape",
            )?;
        }
        self.validate_tensor_scalar_shape(config.epsilon(), "layer norm epsilon")
    }

    /// Adds `LayerNorm` with optional affine operands and phase-specific statistics.
    /// Bias broadcasts against the input independently of the scale layout.
    /// Native support selection determines executable affine combinations.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, invalid or ambiguous normalization axes, incompatible affine broadcast dimensions, mismatched output or saved-statistic dimensions, or a nonscalar epsilon tensor.
    pub fn layer_normalization_into(
        &mut self,
        inputs: LayerNormalizationInputs,
        outputs: LayerNormalizationOutputs,
        config: LayerNormalizationConfig,
    ) -> Result<()> {
        self.validate_layer_normalization(inputs, outputs, &config)?;
        self.operations
            .push(Operation::Normalization(NormalizationOperation::Layer {
                inputs,
                outputs,
                config,
            }));
        Ok(())
    }

    /// Creates the output and requested training statistics with independent placement and encoding.
    /// Affine inputs are supplied by the caller and are never synthesized.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, invalid or ambiguous normalization axes, incompatible affine broadcast dimensions, mismatched output or saved-statistic dimensions, or a nonscalar epsilon tensor. Inferred statistic strides must also be representable.
    pub fn layer_normalization(
        &mut self,
        inputs: LayerNormalizationInputs,
        outputs: LayerNormalizationOutputs<TensorOutputConfig>,
        config: LayerNormalizationConfig,
    ) -> Result<LayerNormalizationOutputs> {
        let statistics_dimensions = self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.affine.scale(),
            None,
            outputs.statistics.phase(),
        )?;
        let input = self.tensor_spec(inputs.input)?;
        let output_spec = outputs.output.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let statistics_specs = self.normalization_statistics_specs(outputs.statistics, || {
            input
                .shape
                .with_dimensions_preserving_stride_order(statistics_dimensions)
        })?;
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        let statistics = self.inferred_normalization_statistics(statistics_specs, config.name());
        let outputs = LayerNormalizationOutputs { output, statistics };
        if let Err(error) = self.layer_normalization_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    pub(crate) fn validate_rms_normalization(
        &self,
        inputs: RmsNormalizationInputs,
        outputs: RmsNormalizationOutputs,
        config: &RmsNormalizationConfig,
    ) -> Result<()> {
        let statistics = outputs.statistics.inv_variance();
        let statistics_dimensions = self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.scale,
            statistics,
            outputs.statistics.phase(),
        )?;
        self.validate_tensor_dimensions(
            outputs.output,
            self.shape(inputs.input)?.dimensions(),
            "rms norm output shape",
        )?;
        if let Some(bias) = inputs.bias {
            validate_normalization_broadcast_shape(self.shape(inputs.input)?, self.shape(bias)?)?;
        }
        if let Some(statistics) = statistics {
            self.validate_tensor_dimensions(
                statistics,
                &statistics_dimensions,
                "rms norm inverse variance shape",
            )?;
        }
        self.validate_tensor_scalar_shape(config.epsilon(), "rms norm epsilon")
    }

    /// Adds `RMSNorm` with optional scale and bias operands.
    /// Bias broadcasts against the input without selecting normalization axes or inheriting scale geometry.
    /// Native support selection determines executable affine combinations.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, invalid or ambiguous normalization axes, incompatible affine broadcast dimensions, mismatched output or saved-statistic dimensions, or a nonscalar epsilon tensor.
    pub fn rms_normalization_into(
        &mut self,
        inputs: RmsNormalizationInputs,
        outputs: RmsNormalizationOutputs,
        config: RmsNormalizationConfig,
    ) -> Result<()> {
        self.validate_rms_normalization(inputs, outputs, &config)?;
        self.operations
            .push(Operation::Normalization(NormalizationOperation::Rms {
                inputs,
                outputs,
                config,
            }));
        Ok(())
    }

    /// Creates `RMSNorm` output and requested statistics with independent placement and encoding.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, invalid or ambiguous normalization axes, incompatible affine broadcast dimensions, mismatched output or saved-statistic dimensions, or a nonscalar epsilon tensor. Inferred statistic strides must also be representable.
    pub fn rms_normalization(
        &mut self,
        inputs: RmsNormalizationInputs,
        outputs: RmsNormalizationOutputs<TensorOutputConfig>,
        config: RmsNormalizationConfig,
    ) -> Result<RmsNormalizationOutputs> {
        let statistics_dimensions = self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.scale,
            None,
            outputs.statistics.phase(),
        )?;
        let input = self.tensor_spec(inputs.input)?;
        let output_spec = outputs.output.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let statistics_spec = outputs
            .statistics
            .inv_variance()
            .map(|request| -> Result<_> {
                Ok(request.into_spec(
                    self.config.data_type_policy,
                    DataType::F32,
                    input
                        .shape
                        .with_dimensions_preserving_stride_order(statistics_dimensions)?,
                ))
            })
            .transpose()?;
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        let statistics = statistics_spec.map_or(RmsNormalizationStatistics::Inference, |spec| {
            RmsNormalizationStatistics::Training {
                inv_variance: self.inferred_tensor(spec, config.name(), "INV_VARIANCE"),
            }
        });
        let outputs = RmsNormalizationOutputs { output, statistics };
        if let Err(error) = self.rms_normalization_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    pub(crate) fn validate_layer_normalization_backward(
        &self,
        inputs: LayerNormalizationBackwardInputs,
        outputs: LayerNormalizationBackwardOutputs,
        config: &LayerNormalizationBackwardConfig,
    ) -> Result<()> {
        if inputs.scale.is_some() != outputs.affine.is_some() {
            return Err(FrontendError::NormalizationAffineGradients {
                operation: "layer norm backward",
            }
            .into());
        }
        let statistics_dimensions = self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.scale,
            Some(inputs.statistics.mean),
            BackendNormalizationForwardPhase::Training,
        )?;
        self.validate_tensor_dimensions(
            inputs.statistics.inv_variance,
            &statistics_dimensions,
            "layer norm backward inverse variance shape",
        )?;
        self.validate_tensor_dimensions(
            inputs.output_gradient,
            self.shape(inputs.input)?.dimensions(),
            "layer norm output gradient shape",
        )?;
        self.validate_tensor_dimensions(
            outputs.input_gradient,
            self.shape(inputs.input)?.dimensions(),
            "layer norm input gradient shape",
        )?;
        if let Some(affine) = outputs.affine {
            let input_shape = self.shape(inputs.input)?;
            for gradient in [affine.scale_gradient, affine.bias_gradient] {
                validate_normalization_broadcast_shape(input_shape, self.shape(gradient)?)?;
            }
        }
        Ok(())
    }

    /// Adds `LayerNorm` backward; an absent scale has no affine-gradient outputs.
    /// Explicit affine gradients retain independent broadcast dimensions and strides.
    /// Native support selection determines executable combinations.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid or ambiguous normalization axes, incompatible scale or affine-gradient broadcast dimensions, mismatched saved-statistic or input/output-gradient dimensions, or affine-gradient presence inconsistent with the scale input.
    pub fn layer_normalization_backward_into(
        &mut self,
        inputs: LayerNormalizationBackwardInputs,
        outputs: LayerNormalizationBackwardOutputs,
        config: LayerNormalizationBackwardConfig,
    ) -> Result<()> {
        self.validate_layer_normalization_backward(inputs, outputs, &config)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::LayerBackward {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates input and affine gradients with independently selected placement and encoding.
    /// A scale input requires both affine-gradient results, matching C++.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid or ambiguous normalization axes, incompatible scale or affine-gradient broadcast dimensions, mismatched saved-statistic or input/output-gradient dimensions, or affine-gradient presence inconsistent with the scale input.
    pub fn layer_normalization_backward(
        &mut self,
        inputs: LayerNormalizationBackwardInputs,
        outputs: LayerNormalizationBackwardOutputs<TensorOutputConfig>,
        config: LayerNormalizationBackwardConfig,
    ) -> Result<LayerNormalizationBackwardOutputs> {
        self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.scale,
            Some(inputs.statistics.mean),
            BackendNormalizationForwardPhase::Training,
        )?;
        let input = self.tensor_spec(inputs.input)?;
        let input_gradient_spec = outputs.input_gradient.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let affine_specs = match (inputs.scale, outputs.affine) {
            (None, None) => None,
            (Some(scale), Some(affine)) => {
                let scale = self.tensor_spec(scale)?;
                let data_type = scale.data_type;
                Some(LayerNormalizationAffineGradients {
                    scale_gradient: affine.scale_gradient.into_spec(
                        self.config.data_type_policy,
                        data_type,
                        scale.shape.clone(),
                    ),
                    bias_gradient: affine.bias_gradient.into_spec(
                        self.config.data_type_policy,
                        data_type,
                        scale.shape.clone(),
                    ),
                })
            }
            _ => {
                return Err(FrontendError::NormalizationAffineGradients {
                    operation: "layer norm backward",
                }
                .into());
            }
        };
        let checkpoint = self.mutation_checkpoint();
        let input_gradient = self.inferred_tensor(input_gradient_spec, config.name(), "DX");
        let affine = affine_specs.map(|affine| LayerNormalizationAffineGradients {
            scale_gradient: self.inferred_tensor(affine.scale_gradient, config.name(), "DSCALE"),
            bias_gradient: self.inferred_tensor(affine.bias_gradient, config.name(), "DBIAS"),
        });
        let outputs = LayerNormalizationBackwardOutputs {
            input_gradient,
            affine,
        };
        if let Err(error) = self.layer_normalization_backward_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    pub(crate) fn validate_rms_normalization_backward(
        &self,
        inputs: RmsNormalizationBackwardInputs,
        outputs: RmsNormalizationBackwardOutputs,
        config: &RmsNormalizationBackwardConfig,
    ) -> Result<()> {
        if inputs.scale.is_some() != outputs.affine.is_some() {
            return Err(FrontendError::NormalizationAffineGradients {
                operation: "rms norm backward",
            }
            .into());
        }
        self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.scale,
            Some(inputs.inv_variance),
            BackendNormalizationForwardPhase::Training,
        )?;
        self.validate_tensor_dimensions(
            inputs.output_gradient,
            self.shape(inputs.input)?.dimensions(),
            "rms norm output gradient shape",
        )?;
        self.validate_tensor_dimensions(
            outputs.input_gradient,
            self.shape(inputs.input)?.dimensions(),
            "rms norm input gradient shape",
        )?;
        if let Some(affine) = outputs.affine {
            let input_shape = self.shape(inputs.input)?;
            for gradient in [Some(affine.scale_gradient), affine.bias_gradient]
                .into_iter()
                .flatten()
            {
                validate_normalization_broadcast_shape(input_shape, self.shape(gradient)?)?;
            }
        }
        Ok(())
    }

    /// Adds `RMSNorm` backward with optional scale and affine gradients.
    /// Explicit affine gradients retain independent broadcast dimensions and strides.
    /// Native support selection determines executable combinations.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid or ambiguous normalization axes, incompatible scale or affine-gradient broadcast dimensions, mismatched saved-statistic or input/output-gradient dimensions, or affine-gradient presence inconsistent with the scale input.
    pub fn rms_normalization_backward_into(
        &mut self,
        inputs: RmsNormalizationBackwardInputs,
        outputs: RmsNormalizationBackwardOutputs,
        config: RmsNormalizationBackwardConfig,
    ) -> Result<()> {
        self.validate_rms_normalization_backward(inputs, outputs, &config)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::RmsBackward {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates input and affine gradients with independently selected placement and encoding.
    /// The output request selects the optional bias gradient without a separate configuration flag.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, invalid or ambiguous normalization axes, incompatible scale or affine-gradient broadcast dimensions, mismatched saved-statistic or input/output-gradient dimensions, or affine-gradient presence inconsistent with the scale input.
    pub fn rms_normalization_backward(
        &mut self,
        inputs: RmsNormalizationBackwardInputs,
        outputs: RmsNormalizationBackwardOutputs<TensorOutputConfig>,
        config: RmsNormalizationBackwardConfig,
    ) -> Result<RmsNormalizationBackwardOutputs> {
        self.normalization_statistics_dimensions(
            inputs.input,
            config.axes(),
            inputs.scale,
            Some(inputs.inv_variance),
            BackendNormalizationForwardPhase::Training,
        )?;
        let input = self.tensor_spec(inputs.input)?;
        let input_gradient_spec = outputs.input_gradient.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let affine_specs = match (inputs.scale, outputs.affine) {
            (None, None) => None,
            (Some(scale), Some(affine)) => {
                let scale = self.tensor_spec(scale)?;
                let data_type = scale.data_type;
                Some(RmsNormalizationAffineGradients {
                    scale_gradient: affine.scale_gradient.into_spec(
                        self.config.data_type_policy,
                        data_type,
                        scale.shape.clone(),
                    ),
                    bias_gradient: affine.bias_gradient.map(|bias| {
                        bias.into_spec(self.config.data_type_policy, data_type, scale.shape.clone())
                    }),
                })
            }
            _ => {
                return Err(FrontendError::NormalizationAffineGradients {
                    operation: "rms norm backward",
                }
                .into());
            }
        };
        let checkpoint = self.mutation_checkpoint();
        let input_gradient = self.inferred_tensor(input_gradient_spec, config.name(), "DX");
        let affine = affine_specs.map(|affine| RmsNormalizationAffineGradients {
            scale_gradient: self.inferred_tensor(affine.scale_gradient, config.name(), "Dscale"),
            bias_gradient: affine
                .bias_gradient
                .map(|bias| self.inferred_tensor(bias, config.name(), "Dbias")),
        });
        let outputs = RmsNormalizationBackwardOutputs {
            input_gradient,
            affine,
        };
        if let Err(error) = self.rms_normalization_backward_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
