use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::{
            infer_adaptive_layer_norm_scale_bias_shape, infer_adaptive_layer_norm_stats_shape,
            infer_batch_norm_channel_shape, infer_instance_norm_scale_bias_shape,
            infer_instance_norm_stats_shape, infer_layer_norm_scale_bias_shape,
            infer_layer_norm_stats_shape,
        },
        operation::{
            AdaptiveLayerNormalizationBackwardConfig, AdaptiveLayerNormalizationConfig,
            BatchNormalizationBackwardConfig, BatchNormalizationConfig,
            BatchNormalizationFinalizeConfig, BatchNormalizationInferenceConfig, DbnWeightConfig,
            InstanceNormalizationBackwardConfig, InstanceNormalizationConfig,
            LayerNormalizationBackwardConfig, LayerNormalizationConfig, NormalizationOperation,
            Operation, RmsNormalizationBackwardConfig, RmsNormalizationConfig,
        },
        support,
    },
    tensor::{Shape, TensorId, TensorSpec},
    version,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerNormalizationOutputs {
    pub output: TensorId,
    pub mean: TensorId,
    pub inv_variance: TensorId,
    pub scale: TensorId,
    pub bias: TensorId,
}

impl LayerNormalizationOutputs {
    pub fn new(
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        scale: TensorId,
        bias: TensorId,
    ) -> Self {
        Self {
            output,
            mean,
            inv_variance,
            scale,
            bias,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormalizationOutputs {
    pub output: TensorId,
    pub inv_variance: TensorId,
    pub scale: TensorId,
}

impl RmsNormalizationOutputs {
    pub fn new(output: TensorId, inv_variance: TensorId, scale: TensorId) -> Self {
        Self {
            output,
            inv_variance,
            scale,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerNormalizationBackwardOutputs {
    pub dx: TensorId,
    pub dscale: TensorId,
    pub bias_gradient: TensorId,
}

impl LayerNormalizationBackwardOutputs {
    pub fn new(dx: TensorId, dscale: TensorId, bias_gradient: TensorId) -> Self {
        Self {
            dx,
            dscale,
            bias_gradient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormalizationBackwardOutputs {
    pub dx: TensorId,
    pub dscale: TensorId,
    pub bias_gradient: Option<TensorId>,
}

impl RmsNormalizationBackwardOutputs {
    pub fn new(dx: TensorId, dscale: TensorId, bias_gradient: Option<TensorId>) -> Self {
        Self {
            dx,
            dscale,
            bias_gradient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceNormalizationOutputs {
    pub output: TensorId,
    pub mean: TensorId,
    pub inv_variance: TensorId,
    pub scale: TensorId,
    pub bias: TensorId,
}

impl InstanceNormalizationOutputs {
    pub fn new(
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        scale: TensorId,
        bias: TensorId,
    ) -> Self {
        Self {
            output,
            mean,
            inv_variance,
            scale,
            bias,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceNormalizationBackwardOutputs {
    pub dx: TensorId,
    pub dscale: TensorId,
    pub bias_gradient: TensorId,
}

impl InstanceNormalizationBackwardOutputs {
    pub fn new(dx: TensorId, dscale: TensorId, bias_gradient: TensorId) -> Self {
        Self {
            dx,
            dscale,
            bias_gradient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchNormalizationOutputs {
    pub output: TensorId,
    pub mean: TensorId,
    pub inv_variance: TensorId,
    pub scale: TensorId,
    pub bias: TensorId,
}

impl BatchNormalizationOutputs {
    pub fn new(
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        scale: TensorId,
        bias: TensorId,
    ) -> Self {
        Self {
            output,
            mean,
            inv_variance,
            scale,
            bias,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchNormalizationBackwardOutputs {
    pub dx: TensorId,
    pub dscale: TensorId,
    pub bias_gradient: TensorId,
}

impl BatchNormalizationBackwardOutputs {
    pub fn new(dx: TensorId, dscale: TensorId, bias_gradient: TensorId) -> Self {
        Self {
            dx,
            dscale,
            bias_gradient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchNormalizationFinalizeOutputs {
    pub eq_scale: TensorId,
    pub eq_bias: TensorId,
    pub saved_mean: TensorId,
    pub saved_inv_variance: TensorId,
    pub next_running_mean: Option<TensorId>,
    pub next_running_var: Option<TensorId>,
}

impl BatchNormalizationFinalizeOutputs {
    pub fn new(
        eq_scale: TensorId,
        eq_bias: TensorId,
        saved_mean: TensorId,
        saved_inv_variance: TensorId,
        next_running_mean: Option<TensorId>,
        next_running_var: Option<TensorId>,
    ) -> Self {
        Self {
            eq_scale,
            eq_bias,
            saved_mean,
            saved_inv_variance,
            next_running_mean,
            next_running_var,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DbnWeightOutputs {
    pub dscale: TensorId,
    pub bias_gradient: TensorId,
    pub eq_bias: TensorId,
    pub eq_scale_dy: TensorId,
    pub eq_scale_x: TensorId,
}

impl DbnWeightOutputs {
    pub fn new(
        dscale: TensorId,
        bias_gradient: TensorId,
        eq_bias: TensorId,
        eq_scale_dy: TensorId,
        eq_scale_x: TensorId,
    ) -> Self {
        Self {
            dscale,
            bias_gradient,
            eq_bias,
            eq_scale_dy,
            eq_scale_x,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveLayerNormalizationOutputs {
    pub output: TensorId,
    pub mean: TensorId,
    pub inv_variance: TensorId,
    pub scale: TensorId,
}

impl AdaptiveLayerNormalizationOutputs {
    pub fn new(output: TensorId, mean: TensorId, inv_variance: TensorId, scale: TensorId) -> Self {
        Self {
            output,
            mean,
            inv_variance,
            scale,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveLayerNormalizationBackwardOutputs {
    pub dx: TensorId,
    pub dscale: TensorId,
    pub bias_gradient: Option<TensorId>,
}

impl AdaptiveLayerNormalizationBackwardOutputs {
    pub fn new(dx: TensorId, dscale: TensorId, bias_gradient: Option<TensorId>) -> Self {
        Self {
            dx,
            dscale,
            bias_gradient,
        }
    }
}

impl Graph {
    /// Adds a batch normalization inference operation with an explicit output.
    ///
    /// Batchnorm inference applies the normalization expression using supplied
    /// mean, inverse variance, scale, bias, and epsilon tensors.
    pub fn batch_normalization_inference(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        config: BatchNormalizationInferenceConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let channel_shape = infer_batch_norm_channel_shape(&input_tensor.shape)?;
        let mean_tensor = self.tensor_config(mean)?.clone();
        let expected_parameter_data_type = mean_tensor.data_type;

        self.validate_tensors_data_type(
            [
                (inv_variance, "batch norm inference inv variance data type"),
                (scale, "batch norm inference scale data type"),
                (bias, "batch norm inference bias data type"),
                (config.epsilon(), "batch norm inference epsilon data type"),
            ],
            expected_parameter_data_type,
        )?;
        self.validate_tensor_data_type(
            output,
            self.effective_io_data_type(input_tensor.data_type),
            "batch norm inference output data type",
        )?;
        self.validate_tensors_dimensions(
            [
                (mean, "batch norm inference mean shape"),
                (inv_variance, "batch norm inference inv variance shape"),
                (scale, "batch norm inference scale shape"),
                (bias, "batch norm inference bias shape"),
            ],
            channel_shape.dimensions(),
        )?;
        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "batch norm inference output shape",
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "batch norm inference epsilon")?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::BatchInference {
                input,
                mean,
                inv_variance,
                scale,
                bias,
                output,
                config,
            },
        ));
        Ok(())
    }

    /// Adds batch normalization inference and creates the output tensor.
    pub fn batch_normalization_inference_infer(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        scale: TensorId,
        bias: TensorId,
        config: BatchNormalizationInferenceConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_config(input)?.clone();
        let output_data_type = self.effective_io_data_type(input_tensor.data_type);
        let output = self.tensor(TensorSpec::new(
            output_data_type,
            input_tensor.shape.clone(),
        ));
        self.batch_normalization_inference(input, mean, inv_variance, scale, bias, output, config)?;
        Ok(output)
    }

    /// Adds a batch normalization finalize operation.
    ///
    /// `bn_finalize` calculates equivalent scale/bias and saved statistics from
    /// GenStats outputs, with optional next running mean/variance outputs.
    pub fn batch_normalization_finalize(
        &mut self,
        sum: TensorId,
        sq_sum: TensorId,
        scale: TensorId,
        bias: TensorId,
        next_running_mean: Option<TensorId>,
        next_running_var: Option<TensorId>,
        saved_mean: TensorId,
        saved_inv_variance: TensorId,
        eq_scale: TensorId,
        eq_bias: TensorId,
        accum_count: TensorId,
        config: BatchNormalizationFinalizeConfig,
    ) -> Result<()> {
        let shape = self.tensor_config(sum)?.shape.clone();
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
        self.validate_tensor_element_count(accum_count, 1, "bn finalize accum_count")?;
        self.validate_tensor_element_count(config.epsilon(), 1, "bn finalize epsilon")?;
        if let Some(momentum) = config.momentum() {
            self.validate_tensor_element_count(momentum, 1, "bn finalize momentum")?;
        }
        match (
            config.prev_running_mean(),
            config.prev_running_var(),
            config.momentum(),
            next_running_mean,
            next_running_var,
        ) {
            (
                Some(prev_running_mean),
                Some(prev_running_var),
                Some(_),
                Some(next_running_mean),
                Some(next_running_var),
            ) => {
                self.validate_tensors_dimensions(
                    [
                        (prev_running_mean, "bn finalize prev mean shape"),
                        (prev_running_var, "bn finalize prev variance shape"),
                        (next_running_mean, "bn finalize next mean shape"),
                        (next_running_var, "bn finalize next variance shape"),
                    ],
                    shape.dimensions(),
                )?;
            }
            (None, None, None, None, None) => {}
            _ => {
                return Err(Error::FrontendBatchNormFinalizeRunningStatsIncomplete);
            }
        }
        self.operations.push(Operation::Normalization(
            NormalizationOperation::BatchFinalize {
                sum,
                sq_sum,
                scale,
                bias,
                next_running_mean,
                next_running_var,
                saved_mean,
                saved_inv_variance,
                eq_scale,
                eq_bias,
                accum_count,
                config,
            },
        ));
        Ok(())
    }

    /// Adds batch normalization finalize and creates its output tensors.
    pub fn batch_normalization_finalize_infer(
        &mut self,
        sum: TensorId,
        sq_sum: TensorId,
        scale: TensorId,
        bias: TensorId,
        accum_count: TensorId,
        config: BatchNormalizationFinalizeConfig,
    ) -> Result<BatchNormalizationFinalizeOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let output_type = self.tensor_config(scale)?.data_type;
        let shape = self.tensor_config(sum)?.shape.clone();
        let eq_scale = self.tensor(TensorSpec::new(output_type, shape.clone()));
        let eq_bias = self.tensor(TensorSpec::new(output_type, shape.clone()));
        let saved_mean = self.tensor(TensorSpec::new(output_type, shape.clone()));
        let saved_inv_variance = self.tensor(TensorSpec::new(output_type, shape.clone()));
        let next_running_mean = config
            .has_running_stats()
            .then(|| self.tensor(TensorSpec::new(output_type, shape.clone())));
        let next_running_var = config
            .has_running_stats()
            .then(|| self.tensor(TensorSpec::new(output_type, shape)));
        if let Err(error) = self.batch_normalization_finalize(
            sum,
            sq_sum,
            scale,
            bias,
            next_running_mean,
            next_running_var,
            saved_mean,
            saved_inv_variance,
            eq_scale,
            eq_bias,
            accum_count,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(BatchNormalizationFinalizeOutputs::new(
            eq_scale,
            eq_bias,
            saved_mean,
            saved_inv_variance,
            next_running_mean,
            next_running_var,
        ))
    }

    fn batch_normalization_peer_stat_matches(
        input_shape: &Shape,
        channel_shape: &Shape,
        peer_stat_shape: &Shape,
    ) -> bool {
        if peer_stat_shape.dimensions() == channel_shape.dimensions() {
            return true;
        }

        let input_dims = input_shape.dimensions();
        let channel_dims = channel_shape.dimensions();
        if input_dims.len() != 4 || channel_dims.len() != 4 {
            return false;
        }
        let peer_stat_dims = peer_stat_shape.dimensions();
        peer_stat_dims
            == [
                2,
                input_dims[0].checked_mul(channel_dims[1]).unwrap_or(-1),
                channel_dims[2],
                channel_dims[3],
            ]
            || peer_stat_dims
                == [
                    2,
                    channel_dims[1].checked_mul(4).unwrap_or(-1),
                    channel_dims[2],
                    channel_dims[3],
                ]
    }

    /// Adds a DBN weight-gradient helper operation.
    ///
    /// DBN is the cuDNN frontend batchnorm backward family; this helper records
    /// the weight-gradient portion used by fused normalization patterns.
    pub fn dbn_weight(
        &mut self,
        dy: TensorId,
        input: TensorId,
        scale: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        eq_bias: TensorId,
        eq_scale_dy: TensorId,
        eq_scale_x: TensorId,
        config: DbnWeightConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        self.validate_tensor_dimensions(
            dy,
            input_tensor.shape.dimensions(),
            "dbn weight dy shape",
        )?;
        let channel_shape = infer_batch_norm_channel_shape(&input_tensor.shape)?;
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
            channel_shape.dimensions(),
        )?;
        self.operations.push(Operation::Normalization(
            NormalizationOperation::DbnWeight {
                dy,
                input,
                scale,
                mean,
                inv_variance,
                dscale,
                bias_gradient,
                eq_bias,
                eq_scale_dy,
                eq_scale_x,
                config,
            },
        ));
        Ok(())
    }

    /// Adds DBN weight-gradient and creates scale/bias_gradient tensors.
    pub fn dbn_weight_infer(
        &mut self,
        dy: TensorId,
        input: TensorId,
        scale: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: DbnWeightConfig,
    ) -> Result<DbnWeightOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let shape = infer_batch_norm_channel_shape(&input_tensor.shape)?;
        let output_data_type = self.effective_io_data_type(DataType::F32);
        let dscale = self.tensor(TensorSpec::new(output_data_type, shape.clone()));
        let bias_gradient = self.tensor(TensorSpec::new(output_data_type, shape.clone()));
        let eq_bias = self.tensor(TensorSpec::new(output_data_type, shape.clone()));
        let eq_scale_dy = self.tensor(TensorSpec::new(output_data_type, shape.clone()));
        let eq_scale_x = self.tensor(TensorSpec::new(output_data_type, shape));
        if let Err(error) = self.dbn_weight(
            dy,
            input,
            scale,
            mean,
            inv_variance,
            dscale,
            bias_gradient,
            eq_bias,
            eq_scale_dy,
            eq_scale_x,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(DbnWeightOutputs::new(
            dscale,
            bias_gradient,
            eq_bias,
            eq_scale_dy,
            eq_scale_x,
        ))
    }

    /// Adds a layer normalization forward operation with explicit outputs.
    ///
    /// Layernorm normalizes across features independently for each sample and
    /// can emit mean and inverse-variance statistics for training.
    pub fn layer_normalization(
        &mut self,
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: LayerNormalizationConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "layer norm output shape",
        )?;
        let expected_scale = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        self.validate_tensors_dimensions(
            [
                (scale, "layer norm scale shape"),
                (bias, "layer norm bias shape"),
            ],
            expected_scale.dimensions(),
        )?;
        let expected_stats = infer_layer_norm_stats_shape(&input_tensor.shape, &expected_scale)?;
        self.validate_tensors_dimensions(
            [
                (mean, "layer norm mean shape"),
                (inv_variance, "layer norm inv_variance shape"),
            ],
            expected_stats.dimensions(),
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "layer norm epsilon")?;

        self.operations
            .push(Operation::Normalization(NormalizationOperation::Layer {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            }));
        Ok(())
    }

    /// Adds layer normalization forward and creates output/stat tensors.
    pub fn layer_normalization_infer(
        &mut self,
        input: TensorId,
        config: LayerNormalizationConfig,
    ) -> Result<LayerNormalizationOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_shape = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let stats_shape = infer_layer_norm_stats_shape(&input_tensor.shape, &scale_shape)?;
        let io_data_type = self.effective_io_data_type(input_tensor.data_type);

        let scale = self.tensor(TensorSpec::new(io_data_type, scale_shape.clone()));
        let bias = self.tensor(TensorSpec::new(io_data_type, scale_shape));
        let output = self.tensor(TensorSpec::new(io_data_type, input_tensor.shape.clone()));
        let stats_type = self.effective_intermediate_data_type(DataType::F32);
        let mean = self.tensor(TensorSpec::new(stats_type, stats_shape.clone()).virtual_tensor());
        let inv_variance = self.tensor(TensorSpec::new(stats_type, stats_shape).virtual_tensor());

        if let Err(error) =
            self.layer_normalization(input, scale, bias, output, mean, inv_variance, config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(LayerNormalizationOutputs::new(
            output,
            mean,
            inv_variance,
            scale,
            bias,
        ))
    }

    /// Adds an RMS normalization forward operation with explicit outputs.
    pub fn rms_normalization(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output: TensorId,
        inv_variance: TensorId,
        config: RmsNormalizationConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "rms norm output shape",
        )?;
        let expected_scale = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        self.validate_tensor_dimensions(
            scale,
            expected_scale.dimensions(),
            "rms norm scale shape",
        )?;
        let expected_stats = infer_layer_norm_stats_shape(&input_tensor.shape, &expected_scale)?;
        self.validate_tensor_dimensions(
            inv_variance,
            expected_stats.dimensions(),
            "rms norm inv_variance shape",
        )?;
        if let Some(bias) = config.bias() {
            self.validate_tensor_dimensions(
                bias,
                expected_scale.dimensions(),
                "rms norm bias shape",
            )?;
        }
        self.validate_tensor_element_count(config.epsilon(), 1, "rms norm epsilon")?;

        self.operations
            .push(Operation::Normalization(NormalizationOperation::Rms {
                input,
                scale,
                bias: config.bias(),
                output,
                inv_variance,
                config,
            }));
        Ok(())
    }

    /// Adds RMS normalization forward and creates output/stat tensors.
    pub fn rms_normalization_infer(
        &mut self,
        input: TensorId,
        config: RmsNormalizationConfig,
    ) -> Result<RmsNormalizationOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_shape = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let stats_shape = infer_layer_norm_stats_shape(&input_tensor.shape, &scale_shape)?;
        let io_data_type = self.effective_io_data_type(input_tensor.data_type);

        let scale = self.tensor(TensorSpec::new(io_data_type, scale_shape));
        let output = self.tensor(TensorSpec::new(io_data_type, input_tensor.shape.clone()));
        let stats_type = self.effective_intermediate_data_type(DataType::F32);
        let inv_variance = self.tensor(TensorSpec::new(stats_type, stats_shape).virtual_tensor());

        if let Err(error) = self.rms_normalization(input, scale, output, inv_variance, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(RmsNormalizationOutputs::new(output, inv_variance, scale))
    }

    /// Adds a layer normalization backward operation with explicit outputs.
    ///
    /// DLN computes input, scale, and bias_gradients during layernorm
    /// backpropagation.
    pub fn layer_normalization_backward(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: LayerNormalizationBackwardConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();
        let expected_scale = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let expected_stats =
            infer_layer_norm_stats_shape(&input_tensor.shape, &scale_tensor.shape)?;

        self.validate_tensor_dimensions(
            dy,
            input_tensor.shape.dimensions(),
            "layer norm backward dy shape",
        )?;
        self.validate_tensor_dimensions(
            dx,
            input_tensor.shape.dimensions(),
            "layer norm backward dx shape",
        )?;
        self.validate_tensors_dimensions(
            [
                (scale, "layer norm backward scale shape"),
                (dscale, "layer norm backward dscale shape"),
                (bias_gradient, "layer norm backward bias_gradient shape"),
            ],
            expected_scale.dimensions(),
        )?;
        self.validate_tensors_dimensions(
            [
                (mean, "layer norm backward mean shape"),
                (inv_variance, "layer norm backward inv_variance shape"),
            ],
            expected_stats.dimensions(),
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "layer norm backward epsilon")?;

        self.operations.push(Operation::Normalization(
            NormalizationOperation::LayerBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            },
        ));
        Ok(())
    }

    /// Adds layer normalization backward and creates gradient tensors.
    pub fn layer_normalization_backward_infer(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        config: LayerNormalizationBackwardConfig,
    ) -> Result<LayerNormalizationBackwardOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();
        let scale_shape = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let dx_data_type = self.effective_io_data_type(input_tensor.data_type);
        let scale_data_type = self.effective_io_data_type(scale_tensor.data_type);
        let dx = self.tensor(TensorSpec::new(dx_data_type, input_tensor.shape.clone()));
        let dscale = self.tensor(TensorSpec::new(scale_data_type, scale_shape.clone()));
        let bias_gradient = self.tensor(TensorSpec::new(scale_data_type, scale_shape));

        if let Err(error) = self.layer_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(LayerNormalizationBackwardOutputs::new(
            dx,
            dscale,
            bias_gradient,
        ))
    }

    /// Adds an RMS normalization backward operation with explicit outputs.
    pub fn rms_normalization_backward(
        &mut self,
        input: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: Option<TensorId>,
        dx: TensorId,
        config: RmsNormalizationBackwardConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();

        let expected_scale = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let expected_stats =
            infer_layer_norm_stats_shape(&input_tensor.shape, &scale_tensor.shape)?;

        self.validate_tensors_dimensions(
            [
                (dy, "rms norm backward dy shape"),
                (dx, "rms norm backward dx shape"),
            ],
            input_tensor.shape.dimensions(),
        )?;
        self.validate_tensors_dimensions(
            [
                (scale, "rms norm backward scale shape"),
                (dscale, "rms norm backward dscale shape"),
            ],
            expected_scale.dimensions(),
        )?;
        self.validate_tensor_dimensions(
            inv_variance,
            expected_stats.dimensions(),
            "rms norm backward inv_variance shape",
        )?;
        if let Some(bias_gradient) = bias_gradient {
            self.validate_tensor_dimensions(
                bias_gradient,
                expected_scale.dimensions(),
                "rms norm backward bias_gradient shape",
            )?;
        }
        self.operations.push(Operation::Normalization(
            NormalizationOperation::RmsBackward {
                input,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            },
        ));
        Ok(())
    }

    /// Adds RMS normalization backward and creates gradient tensors.
    pub fn rms_normalization_backward_infer(
        &mut self,
        input: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        config: RmsNormalizationBackwardConfig,
    ) -> Result<RmsNormalizationBackwardOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();
        let scale_shape = infer_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let dx_data_type = self.effective_io_data_type(input_tensor.data_type);
        let scale_data_type = self.effective_io_data_type(scale_tensor.data_type);
        let dx = self.tensor(TensorSpec::new(dx_data_type, input_tensor.shape.clone()));
        let dscale = self.tensor(TensorSpec::new(scale_data_type, scale_shape.clone()));
        let bias_gradient = if config.has_bias_gradient() {
            Some(self.tensor(TensorSpec::new(scale_data_type, scale_shape)))
        } else {
            None
        };

        if let Err(error) = self.rms_normalization_backward(
            input,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(RmsNormalizationBackwardOutputs::new(
            dx,
            dscale,
            bias_gradient,
        ))
    }

    /// Adds an instance normalization forward operation with explicit outputs.
    ///
    /// Instancenorm computes the standard normalization expression across each
    /// sample and can emit mean and inverse-variance statistics.
    pub fn instance_normalization(
        &mut self,
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: InstanceNormalizationConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let expected_scale = infer_instance_norm_scale_bias_shape(&input_tensor.shape)?;
        let expected_stats = infer_instance_norm_stats_shape(&input_tensor.shape)?;

        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "instance norm output shape",
        )?;
        self.validate_tensors_dimensions(
            [
                (scale, "instance norm scale shape"),
                (bias, "instance norm bias shape"),
            ],
            expected_scale.dimensions(),
        )?;
        self.validate_tensors_dimensions(
            [
                (mean, "instance norm mean shape"),
                (inv_variance, "instance norm inv_variance shape"),
            ],
            expected_stats.dimensions(),
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "instance norm epsilon")?;

        self.operations
            .push(Operation::Normalization(NormalizationOperation::Instance {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            }));
        Ok(())
    }

    /// Adds instance normalization forward and creates output/stat tensors.
    pub fn instance_normalization_infer(
        &mut self,
        input: TensorId,
        config: InstanceNormalizationConfig,
    ) -> Result<InstanceNormalizationOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_shape = infer_instance_norm_scale_bias_shape(&input_tensor.shape)?;
        let stats_shape = infer_instance_norm_stats_shape(&input_tensor.shape)?;
        let io_data_type = self.effective_io_data_type(input_tensor.data_type);

        let scale = self.tensor(TensorSpec::new(io_data_type, scale_shape.clone()));
        let bias = self.tensor(TensorSpec::new(io_data_type, scale_shape));
        let output = self.tensor(TensorSpec::new(io_data_type, input_tensor.shape.clone()));
        let stats_type = self.effective_intermediate_data_type(DataType::F32);
        let mean = self.tensor(TensorSpec::new(stats_type, stats_shape.clone()));
        let inv_variance = self.tensor(TensorSpec::new(stats_type, stats_shape));

        if let Err(error) =
            self.instance_normalization(input, scale, bias, output, mean, inv_variance, config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(InstanceNormalizationOutputs::new(
            output,
            mean,
            inv_variance,
            scale,
            bias,
        ))
    }

    /// Adds an instance normalization backward operation with explicit outputs.
    pub fn instance_normalization_backward(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: InstanceNormalizationBackwardConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let expected_scale = infer_instance_norm_scale_bias_shape(&input_tensor.shape)?;
        let expected_stats = infer_instance_norm_stats_shape(&input_tensor.shape)?;

        self.validate_tensor_dimensions(
            dy,
            input_tensor.shape.dimensions(),
            "instance norm backward dy shape",
        )?;
        self.validate_tensor_dimensions(
            dx,
            input_tensor.shape.dimensions(),
            "instance norm backward dx shape",
        )?;
        self.validate_tensors_dimensions(
            [
                (scale, "instance norm backward scale shape"),
                (dscale, "instance norm backward dscale shape"),
                (bias_gradient, "instance norm backward bias_gradient shape"),
            ],
            expected_scale.dimensions(),
        )?;
        self.validate_tensors_dimensions(
            [
                (mean, "instance norm backward mean shape"),
                (inv_variance, "instance norm backward inv_variance shape"),
            ],
            expected_stats.dimensions(),
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "instance norm backward epsilon")?;

        self.operations.push(Operation::Normalization(
            NormalizationOperation::InstanceBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            },
        ));
        Ok(())
    }

    /// Adds instance normalization backward and creates gradient tensors.
    pub fn instance_normalization_backward_infer(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        config: InstanceNormalizationBackwardConfig,
    ) -> Result<InstanceNormalizationBackwardOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();
        let scale_shape = infer_instance_norm_scale_bias_shape(&input_tensor.shape)?;
        let dx_data_type = self.effective_io_data_type(input_tensor.data_type);
        let scale_data_type = self.effective_io_data_type(scale_tensor.data_type);
        let dx = self.tensor(TensorSpec::new(dx_data_type, input_tensor.shape.clone()));
        let dscale = self.tensor(TensorSpec::new(scale_data_type, scale_shape.clone()));
        let bias_gradient = self.tensor(TensorSpec::new(scale_data_type, scale_shape));

        if let Err(error) = self.instance_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(InstanceNormalizationBackwardOutputs::new(
            dx,
            dscale,
            bias_gradient,
        ))
    }

    /// Adds a batch normalization forward operation with explicit outputs.
    ///
    /// Batchnorm forward may also compute saved statistics and next running
    /// statistics.
    pub fn batch_normalization(
        &mut self,
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: BatchNormalizationConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let expected_channel = infer_batch_norm_channel_shape(&input_tensor.shape)?;

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
            expected_channel.dimensions(),
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "batch norm epsilon")?;
        if let Some(running) = config.running() {
            let momentum = running.momentum;
            self.validate_tensor_element_count(momentum, 1, "batch norm momentum")?;
            self.validate_tensors_dimensions(
                [
                    (running.prev_mean, "batch norm prev_mean shape"),
                    (running.prev_var, "batch norm prev_var shape"),
                    (running.next_mean, "batch norm next_mean shape"),
                    (running.next_var, "batch norm next_var shape"),
                ],
                expected_channel.dimensions(),
            )?;
        }
        for peer_stat in config.peer_stats() {
            let peer_stat_tensor = self.tensor_config(*peer_stat)?.clone();
            if !Self::batch_normalization_peer_stat_matches(
                &input_tensor.shape,
                &expected_channel,
                &peer_stat_tensor.shape,
            ) {
                return Err(Error::ShapeMismatch {
                    name: support::BATCH_NORM_PEER_STAT_SHAPE.into(),
                    expected: expected_channel.dimensions().to_vec(),
                    actual: peer_stat_tensor.shape.dimensions().to_vec(),
                });
            }
        }

        self.operations
            .push(Operation::Normalization(NormalizationOperation::Batch {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            }));
        Ok(())
    }

    /// Adds batch normalization forward and creates output/stat tensors.
    pub fn batch_normalization_infer(
        &mut self,
        input: TensorId,
        config: BatchNormalizationConfig,
    ) -> Result<BatchNormalizationOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let channel_shape = infer_batch_norm_channel_shape(&input_tensor.shape)?;
        let io_data_type = self.effective_io_data_type(input_tensor.data_type);

        let scale = self.tensor(TensorSpec::new(io_data_type, channel_shape.clone()));
        let bias = self.tensor(TensorSpec::new(io_data_type, channel_shape.clone()));
        let output = self.tensor(TensorSpec::new(io_data_type, input_tensor.shape.clone()));
        let stats_type = self.effective_intermediate_data_type(DataType::F32);
        let mean = self.tensor(TensorSpec::new(stats_type, channel_shape.clone()).virtual_tensor());
        let inv_variance = self.tensor(TensorSpec::new(stats_type, channel_shape).virtual_tensor());

        if let Err(error) =
            self.batch_normalization(input, scale, bias, output, mean, inv_variance, config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(BatchNormalizationOutputs::new(
            output,
            mean,
            inv_variance,
            scale,
            bias,
        ))
    }

    /// Adds a batch normalization backward operation with explicit outputs.
    ///
    /// DBN computes input, scale, and bias_gradients during batchnorm
    /// backpropagation.
    pub fn batch_normalization_backward(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: BatchNormalizationBackwardConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let expected_channel = infer_batch_norm_channel_shape(&input_tensor.shape)?;

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
            expected_channel.dimensions(),
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "batch norm backward epsilon")?;
        for peer_stat in config.peer_stats() {
            let peer_stat_tensor = self.tensor_config(*peer_stat)?.clone();
            if !Self::batch_normalization_peer_stat_matches(
                &input_tensor.shape,
                &expected_channel,
                &peer_stat_tensor.shape,
            ) {
                return Err(Error::ShapeMismatch {
                    name: "batch norm backward peer_stat shape".into(),
                    expected: expected_channel.dimensions().to_vec(),
                    actual: peer_stat_tensor.shape.dimensions().to_vec(),
                });
            }
        }

        self.operations.push(Operation::Normalization(
            NormalizationOperation::BatchBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            },
        ));
        Ok(())
    }

    /// Adds batch normalization backward and creates gradient tensors.
    pub fn batch_normalization_backward_infer(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        config: BatchNormalizationBackwardConfig,
    ) -> Result<BatchNormalizationBackwardOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();
        let channel_shape = infer_batch_norm_channel_shape(&input_tensor.shape)?;
        let dx_data_type = self.effective_io_data_type(input_tensor.data_type);
        let scale_data_type = self.effective_io_data_type(scale_tensor.data_type);
        let dx = self.tensor(TensorSpec::new(dx_data_type, input_tensor.shape.clone()));
        let dscale = self.tensor(TensorSpec::new(scale_data_type, channel_shape.clone()));
        let bias_gradient = self.tensor(TensorSpec::new(scale_data_type, channel_shape));

        if let Err(error) = self.batch_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(BatchNormalizationBackwardOutputs::new(
            dx,
            dscale,
            bias_gradient,
        ))
    }

    /// Adds an adaptive layer normalization forward operation with explicit outputs.
    ///
    /// Adaptive layernorm uses scale and bias values that vary across samples in
    /// a batch.
    pub fn adaptive_layer_normalization(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: AdaptiveLayerNormalizationConfig,
    ) -> Result<()> {
        self.validate_adaptive_layer_normalization_support_surface_for_version(version()?.raw())?;

        let input_tensor = self.tensor_config(input)?.clone();
        let expected_scale = infer_adaptive_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let expected_stats = infer_adaptive_layer_norm_stats_shape(&input_tensor.shape)?;

        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "adaptive layer norm output shape",
        )?;
        self.validate_tensor_dimensions(
            scale,
            expected_scale.dimensions(),
            "adaptive layer norm scale shape",
        )?;
        if let Some(bias) = config.bias() {
            self.validate_tensor_dimensions(
                bias,
                expected_scale.dimensions(),
                "adaptive layer norm bias shape",
            )?;
        }
        self.validate_tensor_dimensions(
            mean,
            expected_stats.dimensions(),
            "adaptive layer norm mean shape",
        )?;
        self.validate_tensor_dimensions(
            inv_variance,
            expected_stats.dimensions(),
            "adaptive layer norm inv_variance shape",
        )?;
        self.validate_tensor_element_count(config.epsilon(), 1, "adaptive layer norm epsilon")?;

        self.operations.push(Operation::Normalization(
            NormalizationOperation::AdaptiveLayer {
                input,
                scale,
                bias: config.bias(),
                output,
                mean,
                inv_variance,
                config,
            },
        ));
        Ok(())
    }

    /// Adds adaptive layer normalization forward and creates output/stat tensors.
    pub fn adaptive_layer_normalization_infer(
        &mut self,
        input: TensorId,
        config: AdaptiveLayerNormalizationConfig,
    ) -> Result<AdaptiveLayerNormalizationOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_shape = infer_adaptive_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let stats_shape = infer_adaptive_layer_norm_stats_shape(&input_tensor.shape)?;
        let io_data_type = self.effective_io_data_type(input_tensor.data_type);

        let scale = self.tensor(TensorSpec::new(io_data_type, scale_shape));
        let output = self.tensor(TensorSpec::new(io_data_type, input_tensor.shape.clone()));
        let stats_type = self.effective_intermediate_data_type(DataType::F32);
        let mean = self.tensor(TensorSpec::new(stats_type, stats_shape.clone()).virtual_tensor());
        let inv_variance = self.tensor(TensorSpec::new(stats_type, stats_shape).virtual_tensor());

        if let Err(error) =
            self.adaptive_layer_normalization(input, scale, output, mean, inv_variance, config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(AdaptiveLayerNormalizationOutputs::new(
            output,
            mean,
            inv_variance,
            scale,
        ))
    }

    /// Adds an adaptive layer normalization backward operation with explicit outputs.
    pub fn adaptive_layer_normalization_backward(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: Option<TensorId>,
        dx: TensorId,
        config: AdaptiveLayerNormalizationBackwardConfig,
    ) -> Result<()> {
        self.validate_adaptive_layer_normalization_support_surface_for_version(version()?.raw())?;

        let input_tensor = self.tensor_config(input)?.clone();
        let expected_scale = infer_adaptive_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let expected_stats = infer_adaptive_layer_norm_stats_shape(&input_tensor.shape)?;

        self.validate_tensors_dimensions(
            [
                (dy, "adaptive layer norm backward dy shape"),
                (dx, "adaptive layer norm backward dx shape"),
            ],
            input_tensor.shape.dimensions(),
        )?;
        self.validate_tensors_dimensions(
            [
                (scale, "adaptive layer norm backward scale shape"),
                (dscale, "adaptive layer norm backward dscale shape"),
            ],
            expected_scale.dimensions(),
        )?;
        if let Some(bias_gradient) = bias_gradient {
            self.validate_tensor_dimensions(
                bias_gradient,
                expected_scale.dimensions(),
                "adaptive layer norm backward bias_gradient shape",
            )?;
        }
        self.validate_tensors_dimensions(
            [
                (mean, "adaptive layer norm backward mean shape"),
                (
                    inv_variance,
                    "adaptive layer norm backward inv_variance shape",
                ),
            ],
            expected_stats.dimensions(),
        )?;

        self.operations.push(Operation::Normalization(
            NormalizationOperation::AdaptiveLayerBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            },
        ));
        Ok(())
    }

    /// Adds adaptive layer normalization backward and creates gradient tensors.
    pub fn adaptive_layer_normalization_backward_infer(
        &mut self,
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        config: AdaptiveLayerNormalizationBackwardConfig,
    ) -> Result<AdaptiveLayerNormalizationBackwardOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let scale_tensor = self.tensor_config(scale)?.clone();
        let scale_shape = infer_adaptive_layer_norm_scale_bias_shape(&input_tensor.shape)?;
        let dx_data_type = self.effective_io_data_type(input_tensor.data_type);
        let scale_data_type = self.effective_io_data_type(scale_tensor.data_type);
        let dx = self.tensor(TensorSpec::new(dx_data_type, input_tensor.shape.clone()));
        let dscale = self.tensor(TensorSpec::new(scale_data_type, scale_shape.clone()));
        let bias_gradient = if config.has_bias_gradient() {
            Some(self.tensor(TensorSpec::new(scale_data_type, scale_shape)))
        } else {
            None
        };

        if let Err(error) = self.adaptive_layer_normalization_backward(
            input,
            mean,
            inv_variance,
            dy,
            scale,
            dscale,
            bias_gradient,
            dx,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(AdaptiveLayerNormalizationBackwardOutputs::new(
            dx,
            dscale,
            bias_gradient,
        ))
    }

    pub(crate) fn validate_adaptive_layer_normalization_support_surface_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        support::ADAPTIVE_LAYER_NORM.require_frontend_feature(cudnn_version)
    }
}
