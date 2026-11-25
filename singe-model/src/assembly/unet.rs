//! The blocks `diffusers`' `UNet` and VAE models build from: residual blocks of group norms and convolutions that read a timestep's embedding, strided and nearest-neighbor resamplers, and self-attention over an image's positions.
//!
//! A residual block's state is `norm1`, `conv1`, `time` where it reads a timestep, `norm2`, `conv2`, and `shortcut` where its widths differ; an attention's is `norm`, `query`, `key`, `value`, and `output`; a resampler's is its convolution.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, GroupNorm, Layer, Linear, Mlp, ModelState, Upsample},
    shape::spatial::{Padding, Padding2d, Spatial2d, Spatial2dI64},
    tensor::InterpolateMode,
};

use crate::{
    assembly::{
        Attention, AttentionSpec, SinusoidArithmetic, SinusoidOrder, same_2d, sinusoids, spatial,
        to_map, to_tokens,
    },
    format::Rules,
    head::upload,
    schema::Schema,
};

/// How a `UNet`'s or a VAE's group norms are made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Norms {
    pub groups: i64,
    pub eps: f32,
}

/// `ResnetBlock2D` with the default time scale shift: the timestep's embedding, activated and projected, is added between the two convolutions.
#[derive(Debug, Clone)]
pub struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    time: Option<Linear>,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
    activation: Activation,
    /// What the sum of the block and its shortcut is divided by.
    output_scale: f64,
}

impl ResnetBlock {
    pub fn create(
        schema: &mut Schema<'_>,
        [input, output]: [i64; 2],
        time: Option<i64>,
        norms: Norms,
        output_scale: f64,
    ) -> SingeResult<Self> {
        Ok(Self {
            norm1: schema.group_norm("norm1", input, norms.groups, norms.eps)?,
            conv1: schema.conv2d("conv1", input, output, 3, same_2d(3, 1), true)?,
            time: time
                .map(|width| schema.linear("time", width, output, true))
                .transpose()?,
            norm2: schema.group_norm("norm2", output, norms.groups, norms.eps)?,
            conv2: schema.conv2d("conv2", output, output, 3, same_2d(3, 1), true)?,
            shortcut: (input != output)
                .then(|| schema.conv2d("shortcut", input, output, 1, Conv2dConfig::default(), true))
                .transpose()?,
            activation: Activation::Silu,
            output_scale,
        })
    }

    /// Runs `input`, `[B, C, H, W]`, through the block, reading `time`, the timestep's embedding, `[B, T]`, where the block has a projection of it.
    pub fn forward(
        &self,
        input: &Tensor,
        time: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = input
            .apply(&self.norm1, state)?
            .apply(&self.activation, state)?
            .apply(&self.conv1, state)?;
        if let (Some(projection), Some(time)) = (&self.time, time) {
            let time = time
                .apply(&self.activation, state)?
                .apply(projection, state)?
                .unsqueeze(2)?
                .unsqueeze(3)?;
            hidden = hidden.add(&time)?;
        }

        let hidden = hidden
            .apply(&self.norm2, state)?
            .apply(&self.activation, state)?
            .apply(&self.conv2, state)?;
        let shortcut = match &self.shortcut {
            Some(shortcut) => input.apply(shortcut, state)?,
            None => input.clone(),
        };
        let sum = shortcut.add(&hidden)?;
        if (self.output_scale - 1.0).abs() < f64::EPSILON {
            Ok(sum)
        } else {
            sum.div_scalar(self.output_scale)
        }
    }
}

pub fn resnet_stack(
    schema: &mut Schema<'_>,
    count: usize,
    [input, output]: [i64; 2],
    time: Option<i64>,
    norms: Norms,
) -> SingeResult<Vec<ResnetBlock>> {
    (0..count)
        .map(|layer| {
            let from = if layer == 0 { input } else { output };
            ResnetBlock::create(
                &mut schema.scope(format!("resnets.{layer}")),
                [from, output],
                time,
                norms,
                1.0,
            )
        })
        .collect()
}

/// Registers the convolution of `Downsample2D` as `name`: a 3 × 3 kernel of stride two, padded by `padding` on every side, or, at zero, only below and to the right, as `diffusers` pads then.
pub fn downsample(
    schema: &mut Schema<'_>,
    name: &str,
    channels: i64,
    padding: i64,
) -> SingeResult<Conv2d> {
    let padding = if padding == 0 {
        Padding2d::new(Padding::new(0, 1), Padding::new(0, 1))
    } else {
        Padding2d::symmetric(padding, padding)
    };
    schema.conv2d(
        name,
        channels,
        channels,
        3,
        Conv2dConfig {
            stride: Spatial2d::square(2),
            padding,
            ..Conv2dConfig::default()
        },
        true,
    )
}

/// `Upsample2D` with a convolution: each pixel repeated two by two, then a 3 × 3 convolution.
#[derive(Debug, Clone)]
pub struct UpsampleBlock {
    conv: Conv2d,
}

impl UpsampleBlock {
    pub fn create(schema: &mut Schema<'_>, name: &str, channels: i64) -> SingeResult<Self> {
        Ok(Self {
            conv: schema.conv2d(name, channels, channels, 3, same_2d(3, 1), true)?,
        })
    }

    /// Upsamples `input` to `size`, as `diffusers` does to meet a skip connection the doubled size would miss.
    pub fn forward_to(
        &self,
        input: &Tensor,
        size: Spatial2dI64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        input
            .interpolate(size, InterpolateMode::Nearest)?
            .apply(&self.conv, state)
    }
}

impl Layer for UpsampleBlock {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&Upsample::nearest(2), state)?
            .apply(&self.conv, state)
    }
}

/// `diffusers`' `Attention` over an image's positions: group-normed, attended with a residual connection, and divided by `rescale`.
#[derive(Debug, Clone)]
pub struct SpatialAttention {
    norm: GroupNorm,
    attention: Attention,
    rescale: f64,
}

impl SpatialAttention {
    pub fn create(
        schema: &mut Schema<'_>,
        channels: i64,
        heads: i64,
        norms: Norms,
        rescale: f64,
    ) -> SingeResult<Self> {
        Ok(Self {
            norm: schema.group_norm("norm", channels, norms.groups, norms.eps)?,
            attention: Attention::create(schema, AttentionSpec::square(channels, heads))?,
            rescale,
        })
    }
}

impl Layer for SpatialAttention {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let size = spatial(input)?;
        let hidden = to_tokens(&input.apply(&self.norm, state)?)?;
        let attended = self.attention.forward(&hidden, &hidden, &hidden, state)?;
        let sum = to_map(&attended, size)?.add(input)?;
        if (self.rescale - 1.0).abs() < f64::EPSILON {
            Ok(sum)
        } else {
            sum.div_scalar(self.rescale)
        }
    }
}

/// `UNetMidBlock2D`: a residual block, an attention of type `A`, and another residual block, as `resnets.0`, `attentions.0`, and `resnets.1`.
#[derive(Debug, Clone)]
pub struct MiddleBlock<A> {
    first: ResnetBlock,
    attention: A,
    second: ResnetBlock,
}

impl<A> MiddleBlock<A> {
    pub fn create(
        schema: &mut Schema<'_>,
        width: i64,
        time: Option<i64>,
        norms: Norms,
        scale: f64,
        attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let first = ResnetBlock::create(
            &mut schema.scope("resnets.0"),
            [width, width],
            time,
            norms,
            scale,
        )?;
        let attention = attention(&mut schema.scope("attentions.0"))?;
        Ok(Self {
            first,
            attention,
            second: ResnetBlock::create(
                &mut schema.scope("resnets.1"),
                [width, width],
                time,
                norms,
                scale,
            )?,
        })
    }

    pub fn forward_with(
        &self,
        input: &Tensor,
        time: Option<&Tensor>,
        state: &ModelState,
        attend: impl FnOnce(Tensor, &A) -> SingeResult<Tensor>,
    ) -> SingeResult<Tensor> {
        let hidden = attend(self.first.forward(input, time, state)?, &self.attention)?;
        self.second.forward(&hidden, time, state)
    }
}

impl<A: Layer> Layer for MiddleBlock<A> {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.forward_with(input, None, state, |hidden, attention| {
            hidden.apply(attention, state)
        })
    }
}

/// Returns `get_timestep_embedding`'s `width` features of `timestep` as the `[1, width]` tensor a time embedding reads, in `state`'s floating type on its device.
pub fn timestep_input(
    timestep: f32,
    width: i64,
    order: SinusoidOrder,
    shift: f32,
    state: &ModelState,
) -> SingeResult<Tensor> {
    let count = usize::try_from(width).map_err(|_| SingeError::out_of_range("width"))?;
    upload(
        state,
        [1, width],
        sinusoids(
            f64::from(timestep),
            count,
            order,
            SinusoidArithmetic::Reciprocal { shift },
        ),
    )
}

/// Where in a `UNet` a block sits: in a down stage, the middle, or an up stage, by the stage's index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnetSite {
    Down(usize),
    Middle,
    Up(usize),
}

/// What `UNet2DModel` and `UNet2DConditionModel` share.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnetConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub sample_size: i64,
    pub widths: Vec<i64>,
    /// Whether each down block attends.
    pub down_attention: Vec<bool>,
    /// Whether each up block attends.
    pub up_attention: Vec<bool>,
    pub layers_per_block: usize,
    pub groups: i64,
    pub eps: f32,
    pub downsample_padding: i64,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f32,
    /// What the middle block's residual blocks divide their sums by.
    pub middle_scale: f64,
}

impl UnetConfig {
    #[must_use]
    pub const fn norms(&self) -> Norms {
        Norms {
            groups: self.groups,
            eps: self.eps,
        }
    }

    pub fn time_input(&self, timestep: f32, state: &ModelState) -> SingeResult<Tensor> {
        timestep_input(
            timestep,
            self.widths[0],
            if self.flip_sin_to_cos {
                SinusoidOrder::CosinesFirst
            } else {
                SinusoidOrder::SinesFirst
            },
            self.freq_shift,
            state,
        )
    }

    fn time_width(&self) -> i64 {
        self.widths[0] * 4
    }
}

#[derive(Debug, Clone)]
struct Stage<A, R> {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<A>,
    resample: Option<R>,
}

fn attentions<A>(
    stage: &mut Schema<'_>,
    attends: bool,
    count: usize,
    mut attention: impl FnMut(&mut Schema<'_>) -> SingeResult<A>,
) -> SingeResult<Vec<A>> {
    if !attends {
        return Ok(Vec::new());
    }

    (0..count)
        .map(|layer| attention(&mut stage.scope(format!("attentions.{layer}"))))
        .collect()
}

impl<A> Stage<A, Conv2d> {
    fn down(
        schema: &mut Schema<'_>,
        config: &UnetConfig,
        index: usize,
        attention: &mut impl FnMut(&mut Schema<'_>, UnetSite, i64) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let widths = &config.widths;
        let output = widths[index];
        let mut stage = schema.scope(format!("down.{index}"));
        Ok(Self {
            resnets: resnet_stack(
                &mut stage,
                config.layers_per_block,
                [widths[index.saturating_sub(1)], output],
                Some(config.time_width()),
                config.norms(),
            )?,
            attentions: attentions(
                &mut stage,
                config.down_attention[index],
                config.layers_per_block,
                |schema| attention(schema, UnetSite::Down(index), output),
            )?,
            resample: (index + 1 < widths.len())
                .then(|| downsample(&mut stage, "downsample", output, config.downsample_padding))
                .transpose()?,
        })
    }
}

impl<A> Stage<A, UpsampleBlock> {
    fn up(
        schema: &mut Schema<'_>,
        config: &UnetConfig,
        index: usize,
        attention: &mut impl FnMut(&mut Schema<'_>, UnetSite, i64) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let reversed = config.widths.iter().rev().copied().collect::<Vec<_>>();
        let last = reversed.len() - 1;
        let previous = reversed[index.saturating_sub(1)];
        let output = reversed[index];
        let input = reversed[(index + 1).min(last)];
        let layers = config.layers_per_block + 1;
        let mut stage = schema.scope(format!("up.{index}"));
        let resnets = (0..layers)
            .map(|layer| {
                let skip = if layer == layers - 1 { input } else { output };
                let from = if layer == 0 { previous } else { output };
                ResnetBlock::create(
                    &mut stage.scope(format!("resnets.{layer}")),
                    [from + skip, output],
                    Some(config.time_width()),
                    config.norms(),
                    1.0,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            resnets,
            attentions: attentions(&mut stage, config.up_attention[index], layers, |schema| {
                attention(schema, UnetSite::Up(index), output)
            })?,
            resample: (index < last)
                .then(|| UpsampleBlock::create(&mut stage, "upsample", output))
                .transpose()?,
        })
    }
}

/// A `UNet`'s `input` convolution, `time` embedding, `down.N`, `middle`, and `up.N` stages, closing `norm`, and `output` convolution: residual blocks reading the timestep, each followed by an attention of type `A` where its stage attends, down to a middle of a block, an attention, and a block, and back up, each up block reading the down blocks' outputs in reverse.
#[derive(Debug, Clone)]
pub struct Unet<A> {
    input: Conv2d,
    time: Mlp,
    down: Vec<Stage<A, Conv2d>>,
    middle: MiddleBlock<A>,
    up: Vec<Stage<A, UpsampleBlock>>,
    norm: GroupNorm,
    output: Conv2d,
}

impl<A> Unet<A> {
    /// Registers the `UNet` of `config`, with `attention` building the attention at a site in its scope, for the stage's width.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &UnetConfig,
        mut attention: impl FnMut(&mut Schema<'_>, UnetSite, i64) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let widths = &config.widths;
        let (first, last) = (widths[0], widths[widths.len() - 1]);
        let time_width = config.time_width();
        let norms = config.norms();
        Ok(Self {
            input: schema.conv2d("input", config.in_channels, first, 3, same_2d(3, 1), true)?,
            time: schema.mlp(
                "time",
                [first, time_width, time_width],
                Activation::Silu,
                true,
            )?,
            down: (0..widths.len())
                .map(|index| Stage::down(schema, config, index, &mut attention))
                .collect::<SingeResult<Vec<_>>>()?,
            middle: MiddleBlock::create(
                &mut schema.scope("middle"),
                last,
                Some(time_width),
                norms,
                config.middle_scale,
                |schema| attention(schema, UnetSite::Middle, last),
            )?,
            up: (0..widths.len())
                .map(|index| Stage::up(schema, config, index, &mut attention))
                .collect::<SingeResult<Vec<_>>>()?,
            norm: schema.group_norm("norm", first, norms.groups, norms.eps)?,
            output: schema.conv2d("output", first, config.out_channels, 3, same_2d(3, 1), true)?,
        })
    }

    pub fn embed(&self, sample: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        sample.apply(&self.input, state)
    }

    /// Runs `input`, the embedded sample, through the stages at the timestep whose features `time` holds, `[1, F]`, each residual block reading the timestep's embedding, and `block` after each residual block but the middle's second, given the block's attention where its stage attends, its site, and its index in the stage.
    pub fn denoise(
        &self,
        input: Tensor,
        time: &Tensor,
        state: &ModelState,
        mut block: impl FnMut(Tensor, Option<&A>, UnetSite, usize) -> SingeResult<Tensor>,
    ) -> SingeResult<Tensor> {
        let time = time.apply(&self.time, state)?;
        let mut hidden = input;
        let mut skips = vec![hidden.clone()];
        for (number, stage) in self.down.iter().enumerate() {
            for (index, resnet) in stage.resnets.iter().enumerate() {
                hidden = resnet.forward(&hidden, Some(&time), state)?;
                hidden = block(
                    hidden,
                    stage.attentions.get(index),
                    UnetSite::Down(number),
                    index,
                )?;
                skips.push(hidden.clone());
            }
            if let Some(downsample) = &stage.resample {
                hidden = hidden.apply(downsample, state)?;
                skips.push(hidden.clone());
            }
        }

        hidden = self
            .middle
            .forward_with(&hidden, Some(&time), state, |hidden, attention| {
                block(hidden, Some(attention), UnetSite::Middle, 0)
            })?;
        for (number, stage) in self.up.iter().enumerate() {
            for (index, resnet) in stage.resnets.iter().enumerate() {
                let skip = skips
                    .pop()
                    .ok_or_else(|| SingeError::missing_input("a skip connection"))?;
                hidden =
                    resnet.forward(&Tensor::concat(&[hidden, skip], 1)?, Some(&time), state)?;
                hidden = block(
                    hidden,
                    stage.attentions.get(index),
                    UnetSite::Up(number),
                    index,
                )?;
            }
            if let Some(upsample) = &stage.resample {
                // The next skip's size, which doubling misses after an odd size was halved.
                let skip = skips
                    .last()
                    .ok_or_else(|| SingeError::missing_input("a skip connection"))?;
                hidden = upsample.forward_to(&hidden, spatial(skip)?, state)?;
            }
        }
        hidden
            .apply(&self.norm, state)?
            .apply(&Activation::Silu, state)?
            .apply(&self.output, state)
    }
}

/// Imports the residual blocks, attentions, and resamplers of `diffusers`' block `from` (such as `down_blocks.*`) into `to` (such as `down.*`), under the attention's names of before and after `diffusers` renamed them.
#[must_use]
pub fn unet_block_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("resnets.*.norm1", "resnets.*.norm1"),
            ("resnets.*.conv1", "resnets.*.conv1"),
            ("resnets.*.time_emb_proj", "resnets.*.time"),
            ("resnets.*.norm2", "resnets.*.norm2"),
            ("resnets.*.conv2", "resnets.*.conv2"),
            ("resnets.*.conv_shortcut", "resnets.*.shortcut"),
            ("attentions.*.group_norm", "attentions.*.norm"),
            ("attentions.*.query", "attentions.*.query"),
            ("attentions.*.to_q", "attentions.*.query"),
            ("attentions.*.key", "attentions.*.key"),
            ("attentions.*.to_k", "attentions.*.key"),
            ("attentions.*.value", "attentions.*.value"),
            ("attentions.*.to_v", "attentions.*.value"),
            ("attentions.*.proj_attn", "attentions.*.output"),
            ("attentions.*.to_out.0", "attentions.*.output"),
            ("downsamplers.0.conv", "downsample"),
            ("upsamplers.0.conv", "upsample"),
        ],
    )
}
