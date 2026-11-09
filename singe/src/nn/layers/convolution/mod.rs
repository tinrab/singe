// Convolutions over spatial axes.

use crate::{
    backend::registry::{Conv2dAttributes, Operation, OperationAttributes},
    dispatch::call_one,
    error::{Error, Result},
    nn::{Layer, ModelState, StateKey},
    tensor::Tensor,
};

/// Grouped two-dimensional convolution options.
///
/// Defaults use unit stride and dilation, zero padding, and one group.
/// Options are validated when the layer is applied.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    /// Vertical and horizontal stride.
    pub stride: (i64, i64),
    /// Padding above, below, left, and right of the input.
    pub padding: (i64, i64, i64, i64),
    /// Vertical and horizontal spacing between filter elements.
    pub dilation: (i64, i64),
    /// Number of independent channel groups.
    pub groups: i64,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

/// A convolution layer resolving filter weights and optional bias from named state.
///
/// Input axes are batch, channel, height, and width.
/// Filter axes are output channel, input channel per group, height, and width.
#[derive(Debug, Clone)]
pub struct Conv2d {
    weight: StateKey,
    bias: Option<StateKey>,
    config: Conv2dConfig,
}

impl Conv2d {
    /// Stores convolution state keys and options without resolving tensors or validating shapes.
    #[must_use]
    pub fn new(weight: StateKey, bias: Option<StateKey>, config: Conv2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }
}

impl Layer for Conv2d {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let config = self.config;
        let bias = self
            .bias
            .as_ref()
            .map(|key| state.tensor(key))
            .transpose()?;
        let mut inputs = vec![input, state.tensor(&self.weight)?];
        inputs.extend(bias);
        call_one(
            Operation::Conv2d,
            OperationAttributes::Conv2d(Conv2dAttributes {
                has_bias: bias.is_some(),
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
                groups: config.groups,
            }),
            &inputs,
        )
    }
}

// One-dimensional convolution options: stride, padding before and after, dilation, and groups.
#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    pub stride: i64,
    pub padding: (i64, i64),
    pub dilation: i64,
    pub groups: i64,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            stride: 1,
            padding: (0, 0),
            dilation: 1,
            groups: 1,
        }
    }
}

// A convolution over `[B, C, L]` with weights `[O, C / groups, K]`, as `torch.nn.Conv1d` computes it, run as a two-dimensional one of height one.
#[derive(Debug, Clone)]
pub struct Conv1d {
    weight: StateKey,
    bias: Option<StateKey>,
    config: Conv1dConfig,
}

impl Conv1d {
    #[must_use]
    pub const fn new(weight: StateKey, bias: Option<StateKey>, config: Conv1dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }
}

impl Layer for Conv1d {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let config = self.config;
        let bias = self
            .bias
            .as_ref()
            .map(|key| state.tensor(key))
            .transpose()?;
        let input = input.unsqueeze(2)?;
        let weight = state.tensor(&self.weight)?.unsqueeze(2)?;
        let mut inputs = vec![&input, &weight];
        inputs.extend(bias);
        call_one(
            Operation::Conv2d,
            OperationAttributes::Conv2d(Conv2dAttributes {
                has_bias: bias.is_some(),
                stride: (1, config.stride),
                padding: (0, 0, config.padding.0, config.padding.1),
                dilation: (1, config.dilation),
                groups: config.groups,
            }),
            &inputs,
        )?
        .squeeze(2)
    }
}

// Cuts images `[B, C, H, W]` into square patches and projects each to a token, `[B, H / P · W / P, D]`, in row-major patch order, as vision transformers embed images.
// The projection is a convolution whose stride is its kernel, `P`.
#[derive(Debug, Clone)]
pub struct PatchEmbedding {
    projection: Conv2d,
}

impl PatchEmbedding {
    #[must_use]
    pub const fn new(projection: Conv2d) -> Self {
        Self { projection }
    }
}

impl Layer for PatchEmbedding {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let patches = input.apply(&self.projection, state)?;
        if patches.shape().dimensions().len() != 4 {
            return Err(Error::invalid_argument(
                "patch embedding requires [batch, channels, height, width]",
            ));
        }
        patches.flatten(2, 3)?.transpose(1, 2)
    }
}
