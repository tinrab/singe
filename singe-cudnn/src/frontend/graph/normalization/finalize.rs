use crate::{
    backend::tensor::Shape,
    data_type::DataType,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig},
        operation::{
            BatchNormalizationFinalizeConfig, BatchNormalizationFinalizeInputs,
            BatchNormalizationFinalizeOutputs, NormalizationOperation,
            NormalizationSavedStatistics, Operation,
        },
    },
};

impl Graph {
    pub(crate) fn validate_batch_normalization_finalize(
        &self,
        inputs: BatchNormalizationFinalizeInputs,
        outputs: BatchNormalizationFinalizeOutputs,
        config: &BatchNormalizationFinalizeConfig,
    ) -> Result<()> {
        let BatchNormalizationFinalizeInputs {
            sum,
            squared_sum: sq_sum,
            scale,
            bias,
            accumulation_count: accum_count,
        } = inputs;
        let BatchNormalizationFinalizeOutputs {
            equivalent_scale: eq_scale,
            equivalent_bias: eq_bias,
            statistics:
                NormalizationSavedStatistics {
                    mean: saved_mean,
                    inv_variance: saved_inv_variance,
                },
            ..
        } = outputs;
        let shape = self.shape(sum)?;
        self.validate_tensors_dimensions(
            [
                (sq_sum, "bn finalize sq_sum shape"),
                (scale, "bn finalize scale shape"),
                (bias, "bn finalize bias shape"),
                (saved_mean, "bn finalize saved mean shape"),
                (saved_inv_variance, "bn finalize saved inv variance shape"),
                (eq_scale, "bn finalize eq scale shape"),
                (eq_bias, "bn finalize eq bias shape"),
            ],
            shape.dimensions(),
        )?;
        self.validate_tensor_scalar_shape(accum_count, "bn finalize accum_count")?;
        self.validate_tensor_scalar_shape(config.epsilon(), "bn finalize epsilon")?;
        if let Some(momentum) = config.momentum() {
            self.validate_tensor_scalar_shape(momentum, "bn finalize momentum")?;
        }
        match (config.running(), outputs.running) {
            (Some(previous), Some(next)) => {
                self.validate_tensors_dimensions(
                    [
                        (previous.previous.mean, "bn finalize previous mean shape"),
                        (
                            previous.previous.variance,
                            "bn finalize previous variance shape",
                        ),
                        (next.mean, "bn finalize next mean shape"),
                        (next.variance, "bn finalize next variance shape"),
                    ],
                    shape.dimensions(),
                )?;
            }
            (None, None) => {}
            _ => return Err(FrontendError::BatchNormRunningStatsIncomplete.into()),
        }
        Ok(())
    }

    /// Finalizes accumulated `BatchNorm` statistics and optional running statistics.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, mismatched per-channel input or output dimensions, nonscalar accumulation count, epsilon or momentum, or incomplete running-statistic inputs and outputs.
    pub fn batch_normalization_finalize_into(
        &mut self,
        inputs: BatchNormalizationFinalizeInputs,
        outputs: BatchNormalizationFinalizeOutputs,
        config: BatchNormalizationFinalizeConfig,
    ) -> Result<()> {
        self.validate_batch_normalization_finalize(inputs, outputs, &config)?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::BatchFinalize {
                inputs,
                outputs,
                config,
            },
        ));
        Ok(())
    }

    /// Creates equivalent coefficients, saved statistics and requested running-statistics outputs.
    /// Outputs use the sum tensor's dimensions with NHWC strides, matching C++ inference.
    /// Each output selects its placement and encoding independently, with an FP32 fallback.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, mismatched per-channel input or output dimensions, nonscalar accumulation count, epsilon or momentum, or incomplete running-statistic inputs and outputs. Inferred output strides must also be representable.
    pub fn batch_normalization_finalize(
        &mut self,
        inputs: BatchNormalizationFinalizeInputs,
        outputs: BatchNormalizationFinalizeOutputs<TensorOutputConfig>,
        config: BatchNormalizationFinalizeConfig,
    ) -> Result<BatchNormalizationFinalizeOutputs> {
        if config.running().is_some() != outputs.running.is_some() {
            return Err(FrontendError::BatchNormRunningStatsIncomplete.into());
        }
        let shape = Shape::channels_last(self.shape(inputs.sum)?.dimensions().to_vec())?;
        let policy = self.config.data_type_policy;
        let statistics_specs =
            self.normalization_saved_statistics_specs(outputs.statistics, shape.clone());
        let checkpoint = self.mutation_checkpoint();
        let outputs = BatchNormalizationFinalizeOutputs {
            equivalent_scale: self.inferred_tensor(
                outputs
                    .equivalent_scale
                    .into_spec(policy, DataType::F32, shape.clone()),
                config.name(),
                "EQ_SCALE",
            ),
            equivalent_bias: self.inferred_tensor(
                outputs
                    .equivalent_bias
                    .into_spec(policy, DataType::F32, shape.clone()),
                config.name(),
                "EQ_BIAS",
            ),
            statistics: self
                .inferred_normalization_saved_statistics(statistics_specs, config.name()),
            running: outputs.running.map(|running| {
                self.inferred_batch_normalization_running_statistics(running, shape, config.name())
            }),
        };
        if let Err(error) = self.batch_normalization_finalize_into(inputs, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
