// The blocks `diffusers`' UNets and VAEs build from: residual blocks of group norms and convolutions that read a timestep's embedding, strided and nearest-neighbor resamplers, and self-attention over an image's positions.
//
// A residual block's state is `norm1`, `conv1`, `time` where it reads a timestep, `norm2`, `conv2`, and `shortcut` where its widths differ; an attention's is `norm`, `query`, `key`, `value`, and `output`; a resampler's is its convolution.

use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, GroupNorm, Layer, Linear, ModelState, Upsample},
    tensor::{AttentionOptions, InterpolateMode},
};

use crate::{
    error::{Error, Result},
    format::Rules,
    schema::Schema,
};

// How a UNet's or a VAE's group norms are made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Norms {
    pub groups: i64,
    pub eps: f32,
}

// A 3 × 3 convolution with a bias, padded by one on every side to keep the size.
pub fn conv3x3(
    schema: &mut Schema<'_>,
    name: &str,
    input: i64,
    output: i64,
) -> SingeResult<Conv2d> {
    schema.conv2d(
        name,
        input,
        output,
        3,
        Conv2dConfig {
            padding: (1, 1, 1, 1),
            ..Conv2dConfig::default()
        },
        true,
    )
}

// `ResnetBlock2D` with the default time scale shift: the timestep's embedding, activated and projected, is added between the two convolutions.
#[derive(Debug, Clone)]
pub struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    time: Option<Linear>,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
    activation: Activation,
    // What the sum of the block and its shortcut is divided by.
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
            conv1: conv3x3(schema, "conv1", input, output)?,
            time: time
                .map(|width| schema.linear("time", width, output, true))
                .transpose()?,
            norm2: schema.group_norm("norm2", output, norms.groups, norms.eps)?,
            conv2: conv3x3(schema, "conv2", output, output)?,
            shortcut: (input != output)
                .then(|| schema.conv2d("shortcut", input, output, 1, Conv2dConfig::default(), true))
                .transpose()?,
            activation: Activation::Silu,
            output_scale,
        })
    }

    // `input`, `[B, C, H, W]`, through the block, reading `time`, the timestep's embedding, `[B, T]`, where the block has a projection of it.
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

// `Downsample2D` with a convolution: stride two, padded by `padding` on every side, or, at zero, only below and to the right, as `diffusers` pads then.
pub fn downsample(
    schema: &mut Schema<'_>,
    name: &str,
    channels: i64,
    padding: i64,
) -> SingeResult<Conv2d> {
    let padding = if padding == 0 {
        (0, 1, 0, 1)
    } else {
        (padding, padding, padding, padding)
    };
    schema.conv2d(
        name,
        channels,
        channels,
        3,
        Conv2dConfig {
            stride: (2, 2),
            padding,
            ..Conv2dConfig::default()
        },
        true,
    )
}

// `Upsample2D` with a convolution: each pixel repeated two by two, then a 3 × 3 convolution.
#[derive(Debug, Clone)]
pub struct UpsampleBlock {
    conv: Conv2d,
}

impl UpsampleBlock {
    pub fn create(schema: &mut Schema<'_>, name: &str, channels: i64) -> SingeResult<Self> {
        Ok(Self {
            conv: conv3x3(schema, name, channels, channels)?,
        })
    }

    // Upsamples `input` to `height × width`, as `diffusers` does to meet a skip connection the doubled size would miss.
    pub fn forward_to(
        &self,
        input: &Tensor,
        height: i64,
        width: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        input
            .interpolate((height, width), InterpolateMode::Nearest)?
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

// `diffusers`' `Attention` over an image's positions: group-normed, attended with a residual connection, and divided by `rescale`.
#[derive(Debug, Clone)]
pub struct SpatialAttention {
    norm: GroupNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
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
            query: schema.linear("query", channels, channels, true)?,
            key: schema.linear("key", channels, channels, true)?,
            value: schema.linear("value", channels, channels, true)?,
            output: schema.linear("output", channels, channels, true)?,
            heads,
            rescale,
        })
    }
}

impl Layer for SpatialAttention {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = input.shape().dimensions();
        let (batch, channels, height, width) =
            (dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
        let positions = height * width;
        let hidden = input
            .apply(&self.norm, state)?
            .reshape([batch, channels, positions])?
            .transpose(1, 2)?;
        let heads = |projection: &Linear| -> SingeResult<Tensor> {
            hidden
                .apply(projection, state)?
                .reshape([batch, positions, self.heads, channels / self.heads])?
                .permute(&[0, 2, 1, 3])
        };
        let attended = heads(&self.query)?
            .attend(
                &heads(&self.key)?,
                &heads(&self.value)?,
                None,
                None,
                AttentionOptions::default(),
            )?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([batch, positions, channels])?
            .apply(&self.output, state)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape([batch, channels, height, width])?;
        let sum = attended.add(input)?;
        if (self.rescale - 1.0).abs() < f64::EPSILON {
            Ok(sum)
        } else {
            sum.div_scalar(self.rescale)
        }
    }
}

// `TimestepEmbedding`: a timestep's sinusoidal features through a linear layer, an activation, and another.
#[derive(Debug, Clone)]
pub struct TimeEmbedding {
    input: Linear,
    output: Linear,
}

impl TimeEmbedding {
    pub fn create(
        schema: &mut Schema<'_>,
        name: &str,
        features: i64,
        width: i64,
    ) -> SingeResult<Self> {
        let mut schema = schema.scope(name);
        Ok(Self {
            input: schema.linear("input", features, width, true)?,
            output: schema.linear("output", width, width, true)?,
        })
    }
}

impl Layer for TimeEmbedding {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.input, state)?
            .apply(&Activation::Silu, state)?
            .apply(&self.output, state)
    }
}

// `get_timestep_embedding`'s `width` features of `timestep`: the sines and then the cosines, or the cosines first when `flip`, of the timestep at frequencies falling geometrically from one, shifted by `shift`.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "timesteps and widths are small")]
fn timestep_features(timestep: f32, width: usize, flip: bool, shift: f32) -> Vec<f32> {
    let half = width / 2;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "PyTorch scales an `F32` tensor by the `F64` logarithm in `F32`"
    )]
    let log = -(10_000_f64.ln() as f32);
    // The CUDA kernel divides by a host scalar through its reciprocal.
    let reciprocal = 1.0 / (half as f32 - shift);
    let angles = (0..half)
        .map(|index| timestep * (log * index as f32 * reciprocal).exp())
        .collect::<Vec<f32>>();
    let (sines, cosines) = (
        angles.iter().map(|angle| angle.sin()),
        angles.iter().map(|angle| angle.cos()),
    );
    if flip {
        cosines.chain(sines).collect()
    } else {
        sines.chain(cosines).collect()
    }
}

// `timestep_features` as the `[1, width]` tensor a time embedding reads, in `state`'s floating type on its device.
pub fn timestep_input(
    timestep: f32,
    width: i64,
    flip: bool,
    shift: f32,
    state: &ModelState,
) -> SingeResult<Tensor> {
    let count = usize::try_from(width).map_err(|_| singe::Error::out_of_range("width"))?;
    Tensor::from_vec(
        state.device()?,
        Shape::create([1, width])?,
        timestep_features(timestep, count, flip, shift),
    )?
    .cast(state.floating_type()?)
}

// Whether each of a UNet's blocks attends, from its types, refusing a type other than `plain` and `attending`.
pub fn attends(types: &[String], plain: &str, attending: &str) -> Result<Vec<bool>> {
    types
        .iter()
        .map(|kind| match kind.as_str() {
            kind if kind == plain => Ok(false),
            kind if kind == attending => Ok(true),
            other => Err(Error::UnsupportedConfig(format!("UNet block {other}"))),
        })
        .collect()
}

// Imports the residual blocks, attentions, and resamplers of `diffusers`' block `from` (such as `down_blocks.*`) into `to` (such as `down.*`), under the attention's names of before and after `diffusers` renamed them.
#[must_use]
pub fn unet_block_rules(from: &str, to: &str) -> Rules {
    [
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
    ]
    .iter()
    .fold(Rules::new(), |rules, (stored, target)| {
        rules.rename(&format!("{from}.{stored}"), &format!("{to}.{target}"))
    })
}
