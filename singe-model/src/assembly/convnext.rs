//! `ConvNeXt`, as `transformers`' `ConvNextModel` and `ConvNextV2Model` lay it out: a patchifying stem, then stages of blocks that filter each channel with a 7 × 7 kernel and widen and narrow the channels in between, each stage after the first opening with a strided convolution.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, ModelState, StateKey},
    shape::spatial::Spatial2d,
    tensor::NormOrder,
};

use crate::{
    assembly::{channel_norm, same_2d},
    format::RuleSet,
    schema::Schema,
};

/// The epsilon of every layer norm inside the network, which the config's `layer_norm_eps` does not set.
const NORM_EPS: f32 = 1e-6;

/// The depthwise kernel of every block.
const KERNEL: i64 = 7;

/// How a block reweights its widened channels before narrowing them, or its output before the residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvNextScaling {
    /// `ConvNeXt`'s per-channel layer scale of the output.
    LayerScale,
    /// `ConvNeXt` V2's global response normalization of the widened channels.
    ResponseNorm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvNextConfig {
    /// Each stage's blocks.
    pub depths: Vec<i64>,
    /// Each stage's channels.
    pub widths: Vec<i64>,
    pub patch: i64,
    pub activation: Activation,
    pub scaling: ConvNextScaling,
}

/// A block's [`ConvNextScaling`] with its parameters.
#[derive(Debug, Clone)]
enum BlockScaling {
    /// The layer scale, `[C]`.
    Layer(StateKey),
    /// The response norm's weight and bias, `[1, 1, 1, 4C]` each.
    Response { weight: StateKey, bias: StateKey },
}

/// A `ConvNeXt` block: a depthwise 7 × 7 convolution, a layer norm, the channels widened fourfold and narrowed back, and the input added.
#[derive(Debug, Clone)]
pub struct ConvNextBlock {
    depthwise: Conv2d,
    norm: LayerNorm,
    up: Linear,
    down: Linear,
    activation: Activation,
    scaling: BlockScaling,
}

impl ConvNextBlock {
    /// Registers `depthwise`, `norm`, `up`, `down`, and `scale` or `response_norm.{weight,bias}`.
    pub fn create(
        schema: &mut Schema<'_>,
        width: i64,
        activation: Activation,
        scaling: ConvNextScaling,
    ) -> SingeResult<Self> {
        let inner = 4 * width;
        let scaling = match scaling {
            ConvNextScaling::LayerScale => BlockScaling::Layer(schema.parameter("scale", [width])?),
            ConvNextScaling::ResponseNorm => BlockScaling::Response {
                weight: schema.parameter("response_norm.weight", [1, 1, 1, inner])?,
                bias: schema.parameter("response_norm.bias", [1, 1, 1, inner])?,
            },
        };
        Ok(Self {
            depthwise: schema.conv2d(
                "depthwise",
                width,
                width,
                KERNEL,
                Conv2dConfig {
                    groups: width,
                    ..same_2d(KERNEL, 1)
                },
                true,
            )?,
            norm: schema.layer_norm("norm", width, NORM_EPS)?,
            up: schema.linear("up", width, inner, true)?,
            down: schema.linear("down", inner, width, true)?,
            activation,
            scaling,
        })
    }
}

impl Layer for ConvNextBlock {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input
            .apply(&self.depthwise, state)?
            .permute(&[0, 2, 3, 1])?
            .apply(&self.norm, state)?
            .apply(&self.up, state)?
            .apply(&self.activation, state)?;
        if let BlockScaling::Response { weight, bias } = &self.scaling {
            // Each channel's L2 norm over the map, relative to the mean of those norms.
            let norms = hidden.vector_norm(NormOrder::Finite(2.0), &[1, 2], true)?;
            let relative = norms.div(&norms.mean(&[3], true)?.add_scalar(1e-6)?)?;
            hidden = hidden
                .mul(&relative)?
                .mul(state.tensor(weight)?)?
                .add(state.tensor(bias)?)?
                .add(&hidden)?;
        }

        let mut output = hidden.apply(&self.down, state)?;
        if let BlockScaling::Layer(scale) = &self.scaling {
            output = output.mul(state.tensor(scale)?)?;
        }
        output.permute(&[0, 3, 1, 2])?.add(input)
    }
}

#[derive(Debug, Clone)]
struct Stage {
    downsample: Option<(LayerNorm, Conv2d)>,
    blocks: Vec<ConvNextBlock>,
}

/// The network from images to each stage's map, `[B, C, H, W]`.
#[derive(Debug, Clone)]
pub struct ConvNext {
    patches: Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<Stage>,
}

impl ConvNext {
    /// Registers `patches`, `patch_norm`, and `stages.N` with `downsample.{norm,convolution}` after the first and `blocks.M`.
    pub fn create(schema: &mut Schema<'_>, config: &ConvNextConfig) -> SingeResult<Self> {
        let first = config.widths[0];
        let patches = schema.conv2d(
            "patches",
            3,
            first,
            config.patch,
            Conv2dConfig {
                stride: Spatial2d::square(config.patch),
                ..Conv2dConfig::default()
            },
            true,
        )?;
        let patch_norm = schema.layer_norm("patch_norm", first, NORM_EPS)?;
        let mut stages = Vec::with_capacity(config.widths.len());
        let mut input = first;
        for (index, (&width, &depth)) in config.widths.iter().zip(&config.depths).enumerate() {
            let mut schema = schema.scope(format!("stages.{index}"));
            let downsample = (index > 0)
                .then(|| -> SingeResult<_> {
                    Ok((
                        schema.layer_norm("downsample.norm", input, NORM_EPS)?,
                        schema.conv2d(
                            "downsample.convolution",
                            input,
                            width,
                            2,
                            Conv2dConfig {
                                stride: Spatial2d::square(2),
                                ..Conv2dConfig::default()
                            },
                            true,
                        )?,
                    ))
                })
                .transpose()?;
            let blocks = (0..depth)
                .map(|block| {
                    ConvNextBlock::create(
                        &mut schema.scope(format!("blocks.{block}")),
                        width,
                        config.activation,
                        config.scaling,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?;
            stages.push(Stage { downsample, blocks });
            input = width;
        }
        Ok(Self {
            patches,
            patch_norm,
            stages,
        })
    }

    /// Returns each stage's map, the first at a quarter of the image's size and each after it at half the one before.
    pub fn stages(&self, input: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let mut hidden =
            channel_norm(&input.apply(&self.patches, state)?, &self.patch_norm, state)?;
        let mut maps = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            if let Some((norm, convolution)) = &stage.downsample {
                hidden = channel_norm(&hidden, norm, state)?.apply(convolution, state)?;
            }

            for block in &stage.blocks {
                hidden = hidden.apply(block, state)?;
            }

            maps.push(hidden.clone());
        }
        Ok(maps)
    }
}

/// Imports a `ConvNeXt` stored as `transformers` names it under `from` to the schema's names under `to`.
#[must_use]
pub fn convnext_rules(from: &str, to: &str) -> RuleSet {
    RuleSet::new()
        .rename(
            &format!("{from}.embeddings.patch_embeddings"),
            &format!("{to}.patches"),
        )
        .rename(
            &format!("{from}.embeddings.layernorm"),
            &format!("{to}.patch_norm"),
        )
        .rename_each(
            &format!("{from}.encoder.stages.*"),
            &format!("{to}.stages.*"),
            &[
                ("downsampling_layer.0", "downsample.norm"),
                ("downsampling_layer.1", "downsample.convolution"),
            ],
        )
        .rename_each(
            &format!("{from}.encoder.stages.*.layers.*"),
            &format!("{to}.stages.*.blocks.*"),
            &[
                ("dwconv", "depthwise"),
                ("layernorm", "norm"),
                ("pwconv1", "up"),
                ("pwconv2", "down"),
                ("layer_scale_parameter", "scale"),
                ("grn", "response_norm"),
            ],
        )
}
