use crate::{
    backend::tensor::{Shape, TensorId},
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig, infer_normalization_channel_dimensions},
        operation::{
            BatchNormalizationConfig, BatchNormalizationInferenceInputs, BatchNormalizationInputs,
            BatchNormalizationOutputs, NormalizationConfig, NormalizationOperation,
            NormalizationSavedStatistics, Operation,
        },
    },
};

impl Graph {
    pub(crate) fn validate_batch_normalization_inference(
        &self,
        inputs: BatchNormalizationInferenceInputs,
        output: TensorId,
    ) -> Result<()> {
        let BatchNormalizationInferenceInputs {
            input,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            scale,
            bias,
        } = inputs;
        let input_tensor = self.tensor_spec(input)?;
        let channel_dimensions = infer_normalization_channel_dimensions(&input_tensor.shape)?;
        self.validate_tensors_dimensions(
            [
                (mean, "batch norm inference mean shape"),
                (inv_variance, "batch norm inference inv variance shape"),
                (scale, "batch norm inference scale shape"),
                (bias, "batch norm inference bias shape"),
            ],
            &channel_dimensions,
        )?;
        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "batch norm inference output shape",
        )?;
        Ok(())
    }

    /// Applies `scale * (input - mean) * inv_variance + bias` using saved statistics.
    /// The saved inverse standard deviation already includes the forward epsilon.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, saved-statistic or affine dimensions that differ from the per-channel shape, or output dimensions that differ from the input.
    pub fn batch_normalization_inference_into(
        &mut self,
        inputs: BatchNormalizationInferenceInputs,
        output: TensorId,
        config: NormalizationConfig,
    ) -> Result<()> {
        self.validate_batch_normalization_inference(inputs, output)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::BatchInference {
                inputs,
                output,
                config,
            },
        ));
        Ok(())
    }

    /// Creates only the `BatchNorm` inference output with the requested placement and encoding.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, saved-statistic or affine dimensions that differ from the per-channel shape, or output dimensions that differ from the input.
    pub fn batch_normalization_inference(
        &mut self,
        inputs: BatchNormalizationInferenceInputs,
        output: TensorOutputConfig,
        config: NormalizationConfig,
    ) -> Result<TensorId> {
        let input = self.tensor_spec(inputs.input)?;
        let output_spec = output.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        if let Err(error) = self.batch_normalization_inference_into(inputs, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn validate_batch_normalization(
        &self,
        inputs: BatchNormalizationInputs,
        outputs: BatchNormalizationOutputs,
        config: &BatchNormalizationConfig,
    ) -> Result<()> {
        let BatchNormalizationInputs { input, scale, bias } = inputs;
        let BatchNormalizationOutputs {
            output,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            running,
        } = outputs;
        let input_tensor = self.tensor_spec(input)?;
        let expected_channel = infer_normalization_channel_dimensions(&input_tensor.shape)?;

        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "batch norm output shape",
        )?;
        self.validate_tensors_dimensions(
            [
                (scale, "batch norm scale shape"),
                (bias, "batch norm bias shape"),
                (mean, "batch norm mean shape"),
                (inv_variance, "batch norm inv_variance shape"),
            ],
            &expected_channel,
        )?;
        self.validate_tensor_scalar_shape(config.epsilon(), "batch norm epsilon")?;
        match (config.running(), running) {
            (Some(previous), Some(next)) => {
                self.validate_tensor_scalar_shape(previous.momentum, "batch norm momentum")?;
                self.validate_tensors_dimensions(
                    [
                        (previous.previous.mean, "batch norm previous mean shape"),
                        (
                            previous.previous.variance,
                            "batch norm previous variance shape",
                        ),
                        (next.mean, "batch norm next mean shape"),
                        (next.variance, "batch norm next variance shape"),
                    ],
                    &expected_channel,
                )?;
            }
            (None, None) => {}
            _ => return Err(FrontendError::BatchNormRunningStatsIncomplete.into()),
        }
        for peer_stat in config.peer_stats() {
            self.tensor_spec(*peer_stat)?;
        }

        Ok(())
    }

    /// Adds `BatchNorm` training with explicit output and saved statistics.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, mismatched output, affine, saved-statistic or running-statistic dimensions, nonscalar epsilon or momentum, or incomplete running-statistic inputs and outputs.
    pub fn batch_normalization_into(
        &mut self,
        inputs: BatchNormalizationInputs,
        outputs: BatchNormalizationOutputs,
        config: BatchNormalizationConfig,
    ) -> Result<()> {
        self.validate_batch_normalization(inputs, outputs, &config)?;
        self.operations
            .push(Operation::Normalization(NormalizationOperation::Batch {
                inputs,
                outputs,
                config,
            }));
        Ok(())
    }

    /// Creates the output and saved statistics with independently selected placement and encoding.
    /// Affine and previous running-statistics operands retain their caller-declared specifications.
    /// Next running mean and variance have their own placement and encoding requests.
    /// A scope names all inferred results using C++ output-port suffixes.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, mismatched output, affine, saved-statistic or running-statistic dimensions, nonscalar epsilon or momentum, or incomplete running-statistic inputs and outputs. Inferred per-channel strides must also be representable.
    pub fn batch_normalization(
        &mut self,
        inputs: BatchNormalizationInputs,
        outputs: BatchNormalizationOutputs<TensorOutputConfig>,
        config: BatchNormalizationConfig,
    ) -> Result<BatchNormalizationOutputs> {
        if config.running().is_some() != outputs.running.is_some() {
            return Err(FrontendError::BatchNormRunningStatsIncomplete.into());
        }
        let input = self.tensor_spec(inputs.input)?;
        let channel_shape =
            Shape::channels_last(infer_normalization_channel_dimensions(&input.shape)?)?;
        let output_spec = outputs.output.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let statistics_specs =
            self.normalization_saved_statistics_specs(outputs.statistics, channel_shape.clone());
        let checkpoint = self.mutation_checkpoint();
        let outputs = BatchNormalizationOutputs {
            output: self.inferred_tensor(output_spec, config.name(), "Y"),
            statistics: self
                .inferred_normalization_saved_statistics(statistics_specs, config.name()),
            running: outputs.running.map(|running| {
                self.inferred_batch_normalization_running_statistics(
                    running,
                    channel_shape,
                    config.name(),
                )
            }),
        };
        if let Err(error) = self.batch_normalization_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
