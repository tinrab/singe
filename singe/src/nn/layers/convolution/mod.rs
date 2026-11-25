// Convolutions over spatial axes.

use crate::{
    backend::registry::{Conv2dAttributes, Operation, OperationAttributes},
    dispatch::call_one,
    error::{Error, Result},
    nn::{Layer, ModelState, StateKey},
    shape::spatial::{Padding, Padding2d, Spatial2d, Spatial2dI64, Spatial3dI64},
    tensor::Tensor,
};

/// Grouped two-dimensional convolution options.
///
/// Defaults use unit stride and dilation, zero padding, and one group.
/// Options are validated when the layer is applied.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    /// Vertical and horizontal stride.
    pub stride: Spatial2dI64,
    /// Padding of the input's height and width.
    pub padding: Padding2d,
    /// Vertical and horizontal spacing between filter elements.
    pub dilation: Spatial2dI64,
    /// Number of independent channel groups.
    pub groups: i64,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            stride: Spatial2d::square(1),
            padding: Padding2d::default(),
            dilation: Spatial2d::square(1),
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
    pub padding: Padding,
    pub dilation: i64,
    pub groups: i64,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            stride: 1,
            padding: Padding::ZERO,
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
                stride: Spatial2d::new(1, config.stride),
                padding: Padding2d::new(Padding::ZERO, config.padding),
                dilation: Spatial2d::new(1, config.dilation),
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

// Transposed convolution options along each spatial axis: stride, padding taken off each side, output padding added after, dilation, and groups, as `torch.nn.ConvTranspose2d` takes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTransposeConfig<T> {
    pub stride: T,
    pub padding: T,
    pub output_padding: T,
    pub dilation: T,
    pub groups: i64,
}

pub type ConvTranspose1dConfig = ConvTransposeConfig<i64>;

pub type ConvTranspose2dConfig = ConvTransposeConfig<Spatial2dI64>;

impl Default for ConvTranspose1dConfig {
    fn default() -> Self {
        Self {
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        }
    }
}

impl Default for ConvTranspose2dConfig {
    fn default() -> Self {
        Self {
            stride: Spatial2d::square(1),
            padding: Spatial2d::square(0),
            output_padding: Spatial2d::square(0),
            dilation: Spatial2d::square(1),
            groups: 1,
        }
    }
}

// A transposed convolution over `[B, C, L]` with weights `[C, O / groups, K]`, as `torch.nn.ConvTranspose1d` computes it.
#[derive(Debug, Clone)]
pub struct ConvTranspose1d {
    weight: StateKey,
    bias: Option<StateKey>,
    config: ConvTranspose1dConfig,
}

impl ConvTranspose1d {
    #[must_use]
    pub const fn new(
        weight: StateKey,
        bias: Option<StateKey>,
        config: ConvTranspose1dConfig,
    ) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }
}

impl Layer for ConvTranspose1d {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let config = self.config;
        let bias = self
            .bias
            .as_ref()
            .map(|key| state.tensor(key))
            .transpose()?;
        input.convolution_transpose1d(
            state.tensor(&self.weight)?,
            bias,
            config.stride,
            config.padding,
            config.output_padding,
            config.dilation,
            config.groups,
        )
    }
}

// A transposed convolution over `[B, C, H, W]` with weights `[C, O / groups, KH, KW]`, as `torch.nn.ConvTranspose2d` computes it.
#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    weight: StateKey,
    bias: Option<StateKey>,
    config: ConvTranspose2dConfig,
}

impl ConvTranspose2d {
    #[must_use]
    pub const fn new(
        weight: StateKey,
        bias: Option<StateKey>,
        config: ConvTranspose2dConfig,
    ) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }
}

impl Layer for ConvTranspose2d {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let config = self.config;
        let bias = self
            .bias
            .as_ref()
            .map(|key| state.tensor(key))
            .transpose()?;
        input.convolution_transpose2d(
            state.tensor(&self.weight)?,
            bias,
            config.stride,
            config.padding,
            config.output_padding,
            config.dilation,
            config.groups,
        )
    }
}

// Three-dimensional convolution options: stride, padding on both sides, and dilation along depth, height, and width, and groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub stride: Spatial3dI64,
    pub padding: Spatial3dI64,
    pub dilation: Spatial3dI64,
    pub groups: i64,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            stride: Spatial3dI64::cube(1),
            padding: Spatial3dI64::cube(0),
            dilation: Spatial3dI64::cube(1),
            groups: 1,
        }
    }
}

// A convolution over `[B, C, D, H, W]` with weights `[O, C / groups, KD, KH, KW]`, as `torch.nn.Conv3d` computes it, such as a video model's tubelet embedding.
#[derive(Debug, Clone)]
pub struct Conv3d {
    weight: StateKey,
    bias: Option<StateKey>,
    config: Conv3dConfig,
}

impl Conv3d {
    #[must_use]
    pub const fn new(weight: StateKey, bias: Option<StateKey>, config: Conv3dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }
}

impl Layer for Conv3d {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let config = self.config;
        let bias = self
            .bias
            .as_ref()
            .map(|key| state.tensor(key))
            .transpose()?;
        input.convolution3d(
            state.tensor(&self.weight)?,
            bias,
            config.stride,
            config.padding,
            config.dilation,
            config.groups,
        )
    }
}
