use crate::{
    backend::registry::{
        BatchNormalizationAttributes, LayerNormalizationAttributes, Operation, OperationAttributes,
    },
    dispatch::call_one,
    error::Result,
    nn::{Layer, ModelState, StateKey},
    tensor::Tensor,
};

/// Evaluation batch normalization.
/// Running statistics are persistent buffers; forward neither updates them nor changes the model's state revision.
#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    weight: Option<StateKey>,
    bias: Option<StateKey>,
    running_mean: StateKey,
    running_var: StateKey,
    eps: f32,
}

impl BatchNorm2d {
    /// Stores affine keys, running-statistic keys, and numerical epsilon for evaluation batch normalization.
    ///
    /// Tensor resolution and parameter validation occur when the layer is applied.
    #[must_use]
    pub fn new(
        weight: Option<StateKey>,
        bias: Option<StateKey>,
        running_mean: StateKey,
        running_var: StateKey,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps,
        }
    }
}

impl Layer for BatchNorm2d {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let weight = self
            .weight
            .as_ref()
            .map(|key| state.tensor(key))
            .transpose()?;
        let bias = self
            .bias
            .as_ref()
            .map(|key| state.tensor(key))
            .transpose()?;
        let mean = state.tensor(&self.running_mean)?;
        let variance = state.tensor(&self.running_var)?;
        let mut inputs = vec![input];
        inputs.extend(weight);
        inputs.extend(bias);
        inputs.extend([mean, variance]);
        call_one(
            Operation::BatchNorm,
            OperationAttributes::BatchNormalization(BatchNormalizationAttributes {
                has_weight: weight.is_some(),
                has_bias: bias.is_some(),
                has_running_mean: true,
                has_running_var: true,
                training: false,
                momentum: 0.1,
                eps: self.eps,
            }),
            &inputs,
        )
    }
}

/// A trailing-dimension layer-normalization layer with optional named affine weight and bias.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    shape: Vec<i64>,
    weight: Option<StateKey>,
    bias: Option<StateKey>,
    eps: f32,
}

impl LayerNorm {
    /// Stores the trailing normalized shape, optional affine state keys, and numerical epsilon.
    ///
    /// Shape and parameter validation occur when the layer is applied.
    #[must_use]
    pub fn new(
        shape: Vec<i64>,
        weight: Option<StateKey>,
        bias: Option<StateKey>,
        eps: f32,
    ) -> Self {
        Self {
            shape,
            weight,
            bias,
            eps,
        }
    }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        normalization(
            state,
            input,
            Operation::LayerNorm,
            &self.shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }
}

/// A trailing-dimension root-mean-square normalization layer with optional named weight.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    shape: Vec<i64>,
    weight: Option<StateKey>,
    eps: f32,
}

impl RmsNorm {
    /// Stores the trailing normalized shape, optional weight key, and numerical epsilon without validation.
    #[must_use]
    pub fn new(shape: Vec<i64>, weight: Option<StateKey>, eps: f32) -> Self {
        Self { shape, weight, eps }
    }
}

impl Layer for RmsNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        normalization(
            state,
            input,
            Operation::RmsNorm,
            &self.shape,
            self.weight.as_ref(),
            None,
            self.eps,
        )
    }
}

// RMS normalization scaled by `1 + weight`, as Gemma writes it: an all-zero weight leaves the norm alone.
#[derive(Debug, Clone)]
pub struct OffsetRmsNorm {
    norm: RmsNorm,
    weight: StateKey,
}

impl OffsetRmsNorm {
    #[must_use]
    pub fn new(shape: Vec<i64>, weight: StateKey, eps: f32) -> Self {
        Self {
            norm: RmsNorm::new(shape, None, eps),
            weight,
        }
    }
}

impl Layer for OffsetRmsNorm {
    // Computes half-precision inputs in `F32` and rounds once, as Gemma and Qwen3.5 scale the normed values by one plus the weight in `F32`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let weight = state.tensor(&self.weight)?;
        input.rounding_once(|wide| {
            let normed = self.norm.forward(wide, state)?;
            normed.add(&normed.mul(&weight.cast(wide.data_type())?)?)
        })
    }
}

fn normalization(
    state: &ModelState,
    input: &Tensor,
    operation: Operation,
    shape: &[i64],
    weight: Option<&StateKey>,
    bias: Option<&StateKey>,
    eps: f32,
) -> Result<Tensor> {
    let weight = weight.map(|key| state.tensor(key)).transpose()?;
    let bias = bias.map(|key| state.tensor(key)).transpose()?;
    let mut inputs = vec![input];
    inputs.extend(weight);
    inputs.extend(bias);
    call_one(
        operation,
        OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
            normalized_shape: shape.to_vec(),
            has_weight: weight.is_some(),
            has_bias: bias.is_some(),
            eps,
        }),
        &inputs,
    )
}

// A learned scale per feature, `[W]`, by which a residual branch's output is multiplied before it is added, as DINOv2 and CaiT scale theirs.
#[derive(Debug, Clone)]
pub struct LayerScale {
    scale: StateKey,
}

impl LayerScale {
    #[must_use]
    pub const fn new(scale: StateKey) -> Self {
        Self { scale }
    }
}

impl Layer for LayerScale {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        input.mul(state.tensor(&self.scale)?)
    }
}
