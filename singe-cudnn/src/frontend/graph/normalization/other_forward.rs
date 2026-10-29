use crate::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{
            Graph, TensorOutputConfig, infer_adaptive_layer_normalization_stats_dimensions,
            infer_instance_normalization_stats_dimensions, infer_normalization_channel_dimensions,
            validate_normalization_broadcast_shape,
        },
        operation::{
            AdaptiveLayerNormalizationConfig, AdaptiveLayerNormalizationInputs,
            AdaptiveLayerNormalizationOutputs, BatchNormalizationRunningStatistics,
            InstanceNormalizationConfig, InstanceNormalizationInputs, InstanceNormalizationOutputs,
            NormalizationOperation, NormalizationSavedStatistics, NormalizationStatistics,
            Operation,
        },
    },
};

impl Graph {
    pub(crate) fn inferred_batch_normalization_running_statistics(
        &mut self,
        outputs: BatchNormalizationRunningStatistics<TensorOutputConfig>,
        shape: Shape,
        scope: Option<&str>,
    ) -> BatchNormalizationRunningStatistics {
        BatchNormalizationRunningStatistics {
            mean: self.inferred_tensor(
                outputs
                    .mean
                    .into_spec(self.config.data_type_policy, DataType::F32, shape.clone()),
                scope,
                "NEXT_RUNNING_MEAN",
            ),
            variance: self.inferred_tensor(
                outputs
                    .variance
                    .into_spec(self.config.data_type_policy, DataType::F32, shape),
                scope,
                "NEXT_RUNNING_VAR",
            ),
        }
    }

    pub(crate) fn normalization_saved_statistics_specs(
        &self,
        outputs: NormalizationSavedStatistics<TensorOutputConfig>,
        shape: Shape,
    ) -> NormalizationSavedStatistics<TensorSpec> {
        NormalizationSavedStatistics {
            mean: outputs.mean.into_spec(
                self.config.data_type_policy,
                DataType::F32,
                shape.clone(),
            ),
            inv_variance: outputs.inv_variance.into_spec(
                self.config.data_type_policy,
                DataType::F32,
                shape,
            ),
        }
    }

    /// Resolves only requested training layouts, before any inferred tensors are inserted.
    pub(crate) fn normalization_statistics_specs(
        &self,
        outputs: NormalizationStatistics<TensorOutputConfig>,
        shape: impl FnOnce() -> Result<Shape>,
    ) -> Result<NormalizationStatistics<TensorSpec>> {
        Ok(match outputs {
            NormalizationStatistics::Inference => NormalizationStatistics::Inference,
            NormalizationStatistics::Training(statistics) => NormalizationStatistics::Training(
                self.normalization_saved_statistics_specs(statistics, shape()?),
            ),
        })
    }

    pub(crate) fn inferred_normalization_saved_statistics(
        &mut self,
        outputs: NormalizationSavedStatistics<TensorSpec>,
        scope: Option<&str>,
    ) -> NormalizationSavedStatistics {
        NormalizationSavedStatistics {
            mean: self.inferred_tensor(outputs.mean, scope, "MEAN"),
            inv_variance: self.inferred_tensor(outputs.inv_variance, scope, "INV_VARIANCE"),
        }
    }

    pub(crate) fn inferred_normalization_statistics(
        &mut self,
        outputs: NormalizationStatistics<TensorSpec>,
        scope: Option<&str>,
    ) -> NormalizationStatistics {
        match outputs {
            NormalizationStatistics::Inference => NormalizationStatistics::Inference,
            NormalizationStatistics::Training(statistics) => NormalizationStatistics::Training(
                self.inferred_normalization_saved_statistics(statistics, scope),
            ),
        }
    }

    pub(crate) fn validate_instance_normalization(
        &self,
        inputs: InstanceNormalizationInputs,
        outputs: InstanceNormalizationOutputs,
        config: &InstanceNormalizationConfig,
    ) -> Result<()> {
        let input_shape = self.shape(inputs.input)?;
        let affine_dimensions = infer_normalization_channel_dimensions(input_shape)?;
        self.validate_tensor_dimensions(
            outputs.output,
            input_shape.dimensions(),
            "instance norm output shape",
        )?;
        self.validate_tensors_dimensions(
            [
                (inputs.scale, "instance norm scale shape"),
                (inputs.bias, "instance norm bias shape"),
            ],
            &affine_dimensions,
        )?;
        if let Some(stats) = outputs.statistics.saved() {
            let statistics_dimensions = infer_instance_normalization_stats_dimensions(input_shape)?;
            self.validate_tensors_dimensions(
                [
                    (stats.mean, "instance norm mean shape"),
                    (stats.inv_variance, "instance norm inverse variance shape"),
                ],
                &statistics_dimensions,
            )?;
        }
        self.validate_tensor_scalar_shape(config.epsilon(), "instance norm epsilon")
    }

    /// Adds `InstanceNorm` with statistics only in the training phase.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions. Also returns an error if epsilon is not a logical scalar.
    pub fn instance_normalization_into(
        &mut self,
        inputs: InstanceNormalizationInputs,
        outputs: InstanceNormalizationOutputs,
        config: InstanceNormalizationConfig,
    ) -> Result<()> {
        self.validate_instance_normalization(inputs, outputs, &config)?;
        self.operations
            .push(Operation::Normalization(NormalizationOperation::Instance {
                inputs,
                outputs,
                config,
            }));
        Ok(())
    }

    /// Creates the output and requested statistics with independent placement and encoding.
    /// Affine inputs are supplied by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions. Also returns an error if epsilon is not a logical scalar. Inferred output or statistic layouts can also fail if their strides are unrepresentable.
    pub fn instance_normalization(
        &mut self,
        inputs: InstanceNormalizationInputs,
        outputs: InstanceNormalizationOutputs<TensorOutputConfig>,
        config: InstanceNormalizationConfig,
    ) -> Result<InstanceNormalizationOutputs> {
        let input = self.tensor_spec(inputs.input)?;
        let statistics_specs = self.normalization_statistics_specs(outputs.statistics, || {
            Shape::channels_last(infer_instance_normalization_stats_dimensions(&input.shape)?)
        })?;
        let output_spec = outputs.output.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let checkpoint = self.mutation_checkpoint();
        let outputs = InstanceNormalizationOutputs {
            output: self.inferred_tensor(output_spec, config.name(), "Y"),
            statistics: self.inferred_normalization_statistics(statistics_specs, config.name()),
        };
        if let Err(error) = self.instance_normalization_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    pub(crate) fn validate_adaptive_layer_normalization(
        &self,
        inputs: AdaptiveLayerNormalizationInputs,
        outputs: AdaptiveLayerNormalizationOutputs,
        config: &AdaptiveLayerNormalizationConfig,
    ) -> Result<()> {
        let input_shape = self.shape(inputs.input)?;
        let statistics_dimensions = infer_adaptive_layer_normalization_stats_dimensions(
            input_shape,
            self.shape(inputs.scale)?,
        )?;
        self.validate_tensor_dimensions(
            outputs.output,
            input_shape.dimensions(),
            "adaptive layer norm output shape",
        )?;
        if let Some(bias) = inputs.bias {
            validate_normalization_broadcast_shape(input_shape, self.shape(bias)?)?;
        }
        if let Some(stats) = outputs.statistics.saved() {
            self.validate_tensors_dimensions(
                [
                    (stats.mean, "adaptive layer norm mean shape"),
                    (
                        stats.inv_variance,
                        "adaptive layer norm inverse variance shape",
                    ),
                ],
                &statistics_dimensions,
            )?;
        }
        self.validate_tensor_scalar_shape(config.epsilon(), "adaptive layer norm epsilon")
    }

    /// Adds `AdaLayerNorm` with an optional bias and phase-specific saved statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions. Also returns an error if epsilon is not a logical scalar.
    pub fn adaptive_layer_normalization_into(
        &mut self,
        inputs: AdaptiveLayerNormalizationInputs,
        outputs: AdaptiveLayerNormalizationOutputs,
        config: AdaptiveLayerNormalizationConfig,
    ) -> Result<()> {
        self.validate_adaptive_layer_normalization(inputs, outputs, &config)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::AdaptiveLayer {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates the output and requested statistics with independent placement and encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor identities are unknown, input rank or affine/statistic dimensions are incompatible, or activation or gradient destinations have invalid dimensions. Also returns an error if epsilon is not a logical scalar. Inferred output or statistic layouts can also fail if their strides are unrepresentable.
    pub fn adaptive_layer_normalization(
        &mut self,
        inputs: AdaptiveLayerNormalizationInputs,
        outputs: AdaptiveLayerNormalizationOutputs<TensorOutputConfig>,
        config: AdaptiveLayerNormalizationConfig,
    ) -> Result<AdaptiveLayerNormalizationOutputs> {
        let input = self.tensor_spec(inputs.input)?;
        let statistics_dimensions = infer_adaptive_layer_normalization_stats_dimensions(
            &input.shape,
            self.shape(inputs.scale)?,
        )?;
        let statistics_specs = self.normalization_statistics_specs(outputs.statistics, || {
            input
                .shape
                .with_dimensions_preserving_stride_order(statistics_dimensions)
        })?;
        let output_spec = outputs.output.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let checkpoint = self.mutation_checkpoint();
        let outputs = AdaptiveLayerNormalizationOutputs {
            output: self.inferred_tensor(output_spec, config.name(), "Y"),
            statistics: self.inferred_normalization_statistics(statistics_specs, config.name()),
        };
        if let Err(error) = self.adaptive_layer_normalization_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
