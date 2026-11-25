//! The Conformer, as `NeMo`'s `FastConformer` and `ESPnet` write it, under Singe's names: a convolutional subsampling of log-mel features, then blocks of a half-weighted feed-forward, self-attention over relative positions, a convolution module, and another half-weighted feed-forward, each normed before, and a norm after.
//!
//! The subsampling's state is `subsampling.input`, `subsampling.depthwise.N` and `subsampling.pointwise.N` (or `subsampling.convolutions.N` when it strides with full convolutions), and `subsampling.output`.
//! A block's is `feed_forward_in_norm`, `feed_forward_in` (`up`, `down`), `attention_norm`, `attention` (`query`, `key`, `value`, `output`, `positions`, `content_bias`, `position_bias`), `convolution_norm`, `convolution` (`pointwise_in`, `depthwise`, `norm`, `pointwise_out`), `feed_forward_out_norm`, `feed_forward_out`, and `norm`.

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    graph::{DimensionExpr, capture},
    nn::{
        Activation, BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Layer, LayerNorm,
        Linear, LocalAttention, Mlp, ModelState, RelativeAttention, RelativeAttentionInput,
        TensorSource,
    },
};

use crate::{
    assembly::{PYTORCH_NORM_EPS, relative_sinusoids},
    format::{Rules, join},
    schema::Schema,
};

/// How a subsampling strides: `NeMo`'s `dw_striding`, a full convolution and then pairs of a depthwise and a pointwise one, or `striding`, full convolutions only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsamplingKind {
    DepthwiseStriding,
    Striding,
}

/// The most bytes of the first subsampling convolution's output that one span of frames holds.
const SUBSAMPLING_BUDGET: f64 = (256 << 20) as f64;

/// A convolutional subsampling of `[B, T, mels]` features by `factor` in time and frequency, a power of two, through convolutions of `kernel` and stride two with `channels` channels, each followed by `ReLU`, or by a pointwise convolution and then `ReLU` for a depthwise one, then a linear to `width`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubsamplingConfig {
    pub kind: SubsamplingKind,
    pub mels: i64,
    pub factor: i64,
    pub channels: i64,
    pub kernel: i64,
    pub width: i64,
}

impl SubsamplingConfig {
    /// Returns the number of strided convolutions, `log2(factor)`.
    #[must_use]
    pub const fn strides(&self) -> i64 {
        self.factor.trailing_zeros() as i64
    }

    /// Returns what one strided convolution leaves of `length`.
    #[must_use]
    pub const fn strided(&self, length: i64) -> i64 {
        let padding = (self.kernel - 1) / 2;
        (length + 2 * padding - self.kernel).div_euclid(2) + 1
    }

    /// Returns what the subsampling leaves of `length` steps.
    #[must_use]
    pub const fn subsampled(&self, length: i64) -> i64 {
        let mut length = length;
        let mut stride = 0;
        while stride < self.strides() {
            length = self.strided(length);
            stride += 1;
        }
        length
    }

    const fn convolution(&self, groups: i64) -> Conv2dConfig {
        let padding = (self.kernel - 1) / 2;
        Conv2dConfig {
            stride: (2, 2),
            padding: (padding, padding, padding, padding),
            dilation: (1, 1),
            groups,
        }
    }
}

#[derive(Debug, Clone)]
enum Stage {
    Strided(Conv2d),
    Depthwise {
        depthwise: Conv2d,
        pointwise: Conv2d,
    },
}

#[derive(Debug, Clone)]
pub struct ConvSubsampling {
    config: SubsamplingConfig,
    input: Conv2d,
    stages: Vec<Stage>,
    output: Linear,
}

impl ConvSubsampling {
    pub fn create(schema: &mut Schema<'_>, config: &SubsamplingConfig) -> SingeResult<Self> {
        let strides = config.strides();
        if config.factor < 2 || config.factor.count_ones() != 1 {
            return Err(SingeError::invalid_argument(
                "a subsampling factor other than a power of two",
            ));
        }

        let channels = config.channels;
        let kernel = config.kernel;
        let input = schema.conv2d("input", 1, channels, kernel, config.convolution(1), true)?;
        let stages = (0..strides - 1)
            .map(|index| match config.kind {
                SubsamplingKind::DepthwiseStriding => Ok(Stage::Depthwise {
                    depthwise: schema.conv2d(
                        &format!("depthwise.{index}"),
                        channels,
                        channels,
                        kernel,
                        config.convolution(channels),
                        true,
                    )?,
                    pointwise: schema.conv2d(
                        &format!("pointwise.{index}"),
                        channels,
                        channels,
                        1,
                        Conv2dConfig::default(),
                        true,
                    )?,
                }),
                SubsamplingKind::Striding => Ok(Stage::Strided(schema.conv2d(
                    &format!("convolutions.{index}"),
                    channels,
                    channels,
                    kernel,
                    config.convolution(1),
                    true,
                )?)),
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let mut frequencies = config.mels;
        for _ in 0..strides {
            frequencies = config.strided(frequencies);
        }

        let output = schema.linear("output", channels * frequencies, config.width, true)?;
        Ok(Self {
            config: *config,
            input,
            stages,
            output,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &SubsamplingConfig {
        &self.config
    }

    /// Subsamples features `[B, T, mels]` to `[B, T', width]`; with `lengths`, an `I64` `[B]`, each strided convolution's steps past a sequence's length are zeroed, and the lengths it leaves are returned.
    /// When the first convolution's output would pass `SUBSAMPLING_BUDGET`, the frames run in spans, each with its neighbors' frames on both sides, and each span keeps only its own rows, which are exactly the whole clip's.
    pub fn subsample(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Option<Tensor>)> {
        let &[batch, frames, _] = features.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "a subsampling reads features, [batch, frames, mels]",
            ));
        };
        let Some(span) = self.span(batch, frames, features.data_type())? else {
            return self.subsample_span(features, lengths, state);
        };

        let factor = self.config.factor;
        let overlap = self.overlap();
        let rows = self.config.subsampled(frames);
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < frames {
            let from = (start - overlap).max(0);
            let to = (start + span + overlap).min(frames);
            let local = lengths
                .map(|lengths| {
                    lengths
                        .cast(DataType::F32)?
                        .sub_scalar(from as f64)?
                        .clamp(0.0, (to - from) as f32)?
                        .cast(DataType::I64)
                })
                .transpose()?;
            let (hidden, _) =
                self.subsample_span(&features.narrow(1, from, to - from)?, local.as_ref(), state)?;
            let first = start / factor;
            let last = ((start + span) / factor).min(rows);
            pieces.push(hidden.narrow(1, first - from / factor, last - first)?);
            start += span;
        }

        let lengths = lengths
            .map(|lengths| {
                (0..self.config.strides())
                    .try_fold(lengths.clone(), |lengths, _| self.strided_lengths(&lengths))
            })
            .transpose()?;
        Ok((Tensor::concat(&pieces, 1)?, lengths))
    }

    /// Returns the frames a span of the subsampling reads from each neighbor: enough that the convolutions' zero padding at a span's ends changes none of its own rows, a multiple of the factor so spans start on a row.
    const fn overlap(&self) -> i64 {
        self.config.factor * (self.config.kernel - 1)
    }

    /// Returns the frames of each span, a multiple of the factor, or `None` when the first convolution's output over every frame fits `SUBSAMPLING_BUDGET`, one span would hold every frame, or a program with run-time axes records the subsampling.
    fn span(&self, batch: i64, frames: i64, data_type: DataType) -> SingeResult<Option<i64>> {
        if capture::has_run_time_axes()? {
            return Ok(None);
        }

        let config = &self.config;
        // The first convolution's output per input frame, which halves the frames.
        let per_frame = (batch * config.channels * config.strided(config.mels)) as f64
            * data_type.size_bytes() as f64
            / 2.0;
        if per_frame * frames as f64 <= SUBSAMPLING_BUDGET {
            return Ok(None);
        }

        let fitting = (SUBSAMPLING_BUDGET / per_frame) as i64 - 2 * self.overlap();
        let span = (fitting / config.factor * config.factor).max(8 * config.factor);
        Ok((span < frames).then_some(span))
    }

    fn subsample_span(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Option<Tensor>)> {
        let mut lengths = lengths.cloned();
        let mut hidden = features.unsqueeze(1)?;
        let strided = |hidden: Tensor, lengths: &mut Option<Tensor>| -> SingeResult<Tensor> {
            let Some(current) = lengths.as_ref() else {
                return Ok(hidden);
            };
            let next = self.strided_lengths(current)?;
            let masked = mask_steps(&hidden, &next, 2)?;
            *lengths = Some(next);
            Ok(masked)
        };
        hidden = strided(hidden.apply(&self.input, state)?, &mut lengths)?.relu()?;
        for stage in &self.stages {
            hidden = match stage {
                Stage::Strided(convolution) => {
                    strided(hidden.apply(convolution, state)?, &mut lengths)?.relu()?
                }
                Stage::Depthwise {
                    depthwise,
                    pointwise,
                } => {
                    let hidden = strided(hidden.apply(depthwise, state)?, &mut lengths)?;
                    let hidden = hidden.apply(pointwise, state)?;
                    match &lengths {
                        Some(lengths) => mask_steps(&hidden, lengths, 2)?,
                        None => hidden,
                    }
                    .relu()?
                }
            };
        }

        if hidden.shape().dimensions().len() != 4 {
            return Err(SingeError::invalid_argument("a subsampling's convolutions"));
        }

        let hidden = hidden
            .transpose(1, 2)?
            .contiguous()?
            .flatten(2, 3)?
            .apply(&self.output, state)?;
        Ok((hidden, lengths))
    }

    /// Returns what one strided convolution leaves of each length, `floor((length + 2p - k) / 2) + 1`.
    fn strided_lengths(&self, lengths: &Tensor) -> SingeResult<Tensor> {
        let padding = (self.config.kernel - 1) / 2;
        lengths
            .cast(DataType::F32)?
            .add_scalar((2 * padding - self.config.kernel) as f64)?
            .div_scalar(2.0)?
            .floor()?
            .add_scalar(1.0)?
            .cast(DataType::I64)
    }
}

/// Zeroes the steps of `hidden` along `axis` from each sequence's length on, `lengths` an `I64` `[B]`.
fn mask_steps(hidden: &Tensor, lengths: &Tensor, axis: usize) -> SingeResult<Tensor> {
    let mut shape = vec![DimensionExpr::from(1_i64); hidden.shape().dimensions().len()];
    shape[0] = hidden.dimension(0)?;
    shape[axis] = hidden.dimension(axis)?;
    let positions = Tensor::range(hidden.device(), 0_i64, shape[axis].clone())?.unsqueeze(0)?;
    let kept = positions
        .less(&lengths.cast(DataType::I64)?.unsqueeze(1)?)?
        .cast(hidden.data_type())?
        .reshape_with(&shape)?;
    hidden.mul(&kept)
}

/// A Conformer block's sizes and choices.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConformerConfig {
    pub width: i64,
    pub heads: i64,
    pub inner: i64,
    /// The depthwise convolution's kernel, odd.
    pub kernel: i64,
    pub activation: Activation,
    /// Whether the feed-forwards' and attention's linears have biases.
    pub projection_biases: bool,
    pub convolution_biases: bool,
    pub eps: f32,
    pub local: Option<LocalAttention>,
}

/// A Conformer's convolution module: a pointwise convolution to twice the width, GLU over channels, a depthwise convolution, batch norm, the activation, and a pointwise convolution back.
#[derive(Debug, Clone)]
pub struct ConvolutionModule {
    pointwise_in: Conv1d,
    depthwise: Conv1d,
    norm: BatchNorm,
    activation: Activation,
    pointwise_out: Conv1d,
}

impl ConvolutionModule {
    pub fn create(schema: &mut Schema<'_>, config: &ConformerConfig) -> SingeResult<Self> {
        let width = config.width;
        let bias = config.convolution_biases;
        let padding = (config.kernel - 1) / 2;
        Ok(Self {
            pointwise_in: schema.conv1d(
                "pointwise_in",
                width,
                2 * width,
                1,
                Conv1dConfig::default(),
                bias,
            )?,
            depthwise: schema.conv1d(
                "depthwise",
                width,
                width,
                config.kernel,
                Conv1dConfig {
                    padding: (padding, padding),
                    groups: width,
                    ..Conv1dConfig::default()
                },
                bias,
            )?,
            norm: schema.batch_norm("norm", width, PYTORCH_NORM_EPS)?,
            activation: config.activation,
            pointwise_out: schema.conv1d(
                "pointwise_out",
                width,
                width,
                1,
                Conv1dConfig::default(),
                bias,
            )?,
        })
    }

    /// Convolves `[B, T, W]`; with `lengths`, steps past each sequence's length are zeroed before the depthwise convolution reads them.
    pub fn convolve(
        &self,
        hidden: &Tensor,
        lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let hidden = hidden
            .transpose(1, 2)?
            .apply(&self.pointwise_in, state)?
            .glu(1)?;
        let hidden = match lengths {
            Some(lengths) => mask_steps(&hidden, lengths, 2)?,
            None => hidden,
        };
        hidden
            .apply(&self.depthwise, state)?
            .apply(&self.norm, state)?
            .apply(&self.activation, state)?
            .apply(&self.pointwise_out, state)?
            .transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
pub struct ConformerBlock {
    feed_forward_in_norm: LayerNorm,
    feed_forward_in: Mlp,
    attention_norm: LayerNorm,
    attention: RelativeAttention,
    convolution_norm: LayerNorm,
    convolution: ConvolutionModule,
    feed_forward_out_norm: LayerNorm,
    feed_forward_out: Mlp,
    norm: LayerNorm,
}

impl ConformerBlock {
    pub fn create(schema: &mut Schema<'_>, config: &ConformerConfig) -> SingeResult<Self> {
        let width = config.width;
        let eps = config.eps;
        let feed_forward = |schema: &mut Schema<'_>, name: &str| {
            schema.mlp(
                name,
                [width, config.inner, width],
                config.activation,
                config.projection_biases,
            )
        };
        let mut attention = schema.scope("attention").relative_attention(
            width,
            config.heads,
            [config.projection_biases; 4],
        )?;
        if let Some(local) = config.local {
            attention = attention.local(local)?;
        }

        Ok(Self {
            feed_forward_in_norm: schema.layer_norm("feed_forward_in_norm", width, eps)?,
            feed_forward_in: feed_forward(schema, "feed_forward_in")?,
            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
            attention,
            convolution_norm: schema.layer_norm("convolution_norm", width, eps)?,
            convolution: ConvolutionModule::create(&mut schema.scope("convolution"), config)?,
            feed_forward_out_norm: schema.layer_norm("feed_forward_out_norm", width, eps)?,
            feed_forward_out: feed_forward(schema, "feed_forward_out")?,
            norm: schema.layer_norm("norm", width, eps)?,
        })
    }

    /// Returns the largest distance its attention reads over `steps` rows.
    pub fn reach(&self, steps: i64) -> SingeResult<i64> {
        self.attention.reach(steps)
    }

    /// Runs the block over `[B, T, W]` with the embeddings of its distances, `[2R + 1, W]` for `R`, its reach.
    pub fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let half = |branch: Tensor| branch.mul_scalar(0.5);
        let hidden = hidden.add(&half(
            hidden
                .apply(&self.feed_forward_in_norm, state)?
                .apply(&self.feed_forward_in, state)?,
        )?)?;
        let normed = hidden.apply(&self.attention_norm, state)?;
        let hidden = hidden.add(&self.attention.forward(
            &RelativeAttentionInput {
                hidden: &normed,
                positions,
                key_lengths: lengths,
            },
            state,
        )?)?;
        let hidden = hidden.add(&self.convolution.convolve(
            &hidden.apply(&self.convolution_norm, state)?,
            lengths,
            state,
        )?)?;
        let hidden = hidden.add(&half(
            hidden
                .apply(&self.feed_forward_out_norm, state)?
                .apply(&self.feed_forward_out, state)?,
        )?)?;
        hidden.apply(&self.norm, state)
    }
}

/// A Conformer encoder: its subsampling, its blocks, and how it scales the subsampled input.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConformerEncoderConfig {
    pub subsampling: SubsamplingConfig,
    pub block: ConformerConfig,
    pub layers: i64,
    /// Whether the subsampled input is scaled by the root of the width, `NeMo`'s `xscaling`.
    pub scale_input: bool,
}

#[derive(Debug, Clone)]
pub struct ConformerEncoder {
    config: ConformerEncoderConfig,
    subsampling: ConvSubsampling,
    blocks: Vec<ConformerBlock>,
}

impl ConformerEncoder {
    /// Registers the encoder's state, `subsampling` and `blocks.N`, under `schema`.
    pub fn create(schema: &mut Schema<'_>, config: &ConformerEncoderConfig) -> SingeResult<Self> {
        Ok(Self {
            config: *config,
            subsampling: ConvSubsampling::create(
                &mut schema.scope("subsampling"),
                &config.subsampling,
            )?,
            blocks: (0..config.layers)
                .map(|layer| {
                    ConformerBlock::create(
                        &mut schema.scope(format!("blocks.{layer}")),
                        &config.block,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &ConformerEncoderConfig {
        &self.config
    }

    /// Encodes log-mel features, `[B, mels, T]`, to `[B, T', W]`; with `lengths`, an `I64` `[B]` of each clip's own frames, each clip attends to and convolves only its own steps, and the lengths it leaves are returned.
    pub fn encode(
        &self,
        features: &Tensor,
        lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Option<Tensor>)> {
        let (mut hidden, lengths) =
            self.subsampling
                .subsample(&features.transpose(1, 2)?, lengths, state)?;
        let width = self.config.block.width;
        if self.config.scale_input {
            hidden = hidden.mul_scalar((width as f64).sqrt())?;
        }

        let steps = hidden.shape().dimensions()[1];
        let reach = match self.blocks.first() {
            Some(block) => block.reach(steps)?,
            None => steps - 1,
        };
        // Reaching every other row follows the rows a program runs, and a local window stays as it is.
        let reach = if reach == steps - 1 {
            hidden.dimension(1)? - 1
        } else {
            reach.into()
        };
        let positions =
            relative_sinusoids(hidden.device(), reach, width)?.cast(hidden.data_type())?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &positions, lengths.as_ref(), state)?;
        }
        Ok((hidden, lengths))
    }
}

impl Layer for ConformerEncoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        Ok(self.encode(input, None, state)?.0)
    }
}

/// Imports the Conformer encoder `transformers` stores for `NeMo`'s models, such as Parakeet's `encoder`, from `from` into `to`.
/// It reads `dw_striding` subsampling, whose layers `transformers` keeps at the indices of `NeMo`'s `nn.Sequential`: a convolution at 0, then a depthwise one at `2 + 3n` and a pointwise one at `3 + 3n`.
#[must_use]
pub fn conformer_rules(from: &str, to: &str) -> Rules {
    let layers = join(from, "subsampling.layers");
    let subsampling = join(to, "subsampling");
    let target = subsampling.clone();
    Rules::new()
        .rename(&join(&layers, "0"), &join(&subsampling, "input"))
        .custom(&join(&layers, "*"), move |found| {
            let index = found.number(0)?;
            let (kind, stage) = match (index - 2).rem_euclid(3) {
                0 => ("depthwise", (index - 2) / 3),
                1 => ("pointwise", (index - 3) / 3),
                _ => {
                    return Err(SingeError::invalid_argument(format!(
                        "`{}` is no subsampling convolution",
                        found.name
                    )));
                }
            };
            Ok(vec![(
                format!("{target}.{kind}.{stage}{}", found.rest),
                TensorSource::Stored(found.name.to_owned()),
            )])
        })
        .rename(
            &join(from, "subsampling.linear"),
            &join(&subsampling, "output"),
        )
        // A batch norm's count of the batches it saw in training, which evaluation never reads.
        .skip(&join(from, "layers.*.conv.norm.num_batches_tracked"))
        .rename_each(
            &join(from, "layers.*"),
            &join(to, "blocks.*"),
            &[
                ("norm_feed_forward1", "feed_forward_in_norm"),
                ("feed_forward1.linear1", "feed_forward_in.up"),
                ("feed_forward1.linear2", "feed_forward_in.down"),
                ("norm_self_att", "attention_norm"),
                ("self_attn.q_proj", "attention.query"),
                ("self_attn.k_proj", "attention.key"),
                ("self_attn.v_proj", "attention.value"),
                ("self_attn.o_proj", "attention.output"),
                ("self_attn.relative_k_proj", "attention.positions"),
                ("self_attn.bias_u", "attention.content_bias"),
                ("self_attn.bias_v", "attention.position_bias"),
                ("norm_conv", "convolution_norm"),
                ("conv.pointwise_conv1", "convolution.pointwise_in"),
                ("conv.depthwise_conv", "convolution.depthwise"),
                ("conv.norm", "convolution.norm"),
                ("conv.pointwise_conv2", "convolution.pointwise_out"),
                ("norm_feed_forward2", "feed_forward_out_norm"),
                ("feed_forward2.linear1", "feed_forward_out.up"),
                ("feed_forward2.linear2", "feed_forward_out.down"),
                ("norm_out", "norm"),
            ],
        )
}
