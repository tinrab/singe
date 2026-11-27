//! `SEANet`'s convolutional encoder and decoder, as `EnCodec` and Mimi build them: strided convolutions and residual blocks, an optional LSTM, and transposed convolutions mirroring them.
//! Every convolution pads from its input's length so its last window is whole, causally or on both sides, and every transposed convolution trims its overhang, as `EncodecConv1d` and `EncodecConvTranspose1d` do.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{
        Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Layer, ModelState, Recurrent,
        RecurrentCell, RecurrentConfig,
    },
    shape::spatial::Padding,
};

use crate::{
    assembly::recurrent_rules,
    format::{RuleSet, join},
    schema::Schema,
};

/// How a convolution pads its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PadMode {
    Constant,
    Reflect,
    Replicate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeanetConfig {
    pub channels: i64,
    pub filters: i64,
    /// The decoder's upsampling ratios, which the encoder downsamples by in reverse.
    pub ratios: Vec<i64>,
    pub residual_layers: i64,
    pub residual_kernel: i64,
    pub dilation_growth: i64,
    pub compress: i64,
    pub kernel: i64,
    pub last_kernel: i64,
    pub hidden: i64,
    /// The LSTM's layers after the encoder's last downsampling and before the decoder's first upsampling, or none at zero.
    pub lstm_layers: i64,
    pub conv_shortcut: bool,
    pub causal: bool,
    pub pad: PadMode,
    /// The share of a transposed convolution's overhang trimmed from the right, when causal.
    pub trim_right: f64,
}

impl SeanetConfig {
    /// Returns the index of the encoder's LSTM among its layers, after the first convolution and each ratio's residual blocks, ELU, and strided convolution.
    #[must_use]
    pub fn encoder_lstm(&self) -> i64 {
        1 + i64::try_from(self.ratios.len()).unwrap_or(i64::MAX) * (self.residual_layers + 2)
    }
}

/// Returns the length of a tensor's last axis, its time.
fn time_length(input: &Tensor) -> SingeResult<i64> {
    input
        .shape()
        .dimensions()
        .last()
        .copied()
        .ok_or_else(|| SingeError::invalid_argument("a SEANet input without a time axis"))
}

/// A convolution padded from its input's length so its last window is whole, as `EncodecConv1d` pads it.
#[derive(Debug, Clone)]
pub struct PaddedConv {
    convolution: Conv1d,
    kernel: i64,
    stride: i64,
    causal: bool,
    pad: PadMode,
}

impl PaddedConv {
    /// Registers `conv`, from `input` to `out` channels with `kernel`, `stride`, and `dilation`.
    pub fn create(
        schema: &mut Schema<'_>,
        (causal, pad): (bool, PadMode),
        [input, out, kernel, stride, dilation]: [i64; 5],
        bias: bool,
    ) -> SingeResult<Self> {
        Ok(Self {
            convolution: schema.conv1d(
                "conv",
                input,
                out,
                kernel,
                Conv1dConfig {
                    stride,
                    dilation,
                    ..Conv1dConfig::default()
                },
                bias,
            )?,
            kernel: (kernel - 1) * dilation + 1,
            stride,
            causal,
            pad,
        })
    }

    /// Returns `input` padded `before` and `after` along time, reflecting with zeros added first when it is no longer than a side, as `_pad1d` does.
    fn padded(&self, input: &Tensor, before: i64, after: i64) -> SingeResult<Tensor> {
        let zeros = |input: &Tensor, before: i64, after: i64| {
            input.pad(
                &[Padding::ZERO, Padding::ZERO, Padding::new(before, after)],
                0.0,
            )
        };
        let length = time_length(input)?;
        match self.pad {
            PadMode::Constant => zeros(input, before, after),
            PadMode::Replicate => {
                let mut pieces = vec![input.narrow(2, 0, 1)?; usize::try_from(before).unwrap_or(0)];
                pieces.push(input.clone());
                pieces.extend(vec![
                    input.narrow(2, length - 1, 1)?;
                    usize::try_from(after).unwrap_or(0)
                ]);
                Tensor::concat(&pieces, 2)
            }
            PadMode::Reflect => {
                let extra = (before.max(after) - length + 1).max(0);
                let input = if extra > 0 {
                    zeros(input, 0, extra)?
                } else {
                    input.clone()
                };
                let padded = input.pad_reflect(2, before, after)?;
                padded.narrow(2, 0, time_length(&padded)? - extra)
            }
        }
    }
}

impl Layer for PaddedConv {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let length = time_length(input)?;
        let total = self.kernel - self.stride;
        let frames =
            ((length - self.kernel + total) as f64 / self.stride as f64 + 1.0).ceil() as i64 - 1;
        let extra = frames * self.stride + self.kernel - total - length;
        let (before, after) = if self.causal {
            (total, extra)
        } else {
            (total - total / 2, total / 2 + extra)
        };
        self.padded(input, before, after)?
            .apply(&self.convolution, state)
    }
}

/// A transposed convolution trimmed of its overhang, from the right when causal, as `EncodecConvTranspose1d` trims it.
#[derive(Debug, Clone)]
pub struct TrimmedTranspose {
    convolution: ConvTranspose1d,
    overhang: i64,
    right: i64,
}

impl TrimmedTranspose {
    /// Registers `conv`, from `input` to `out` channels in `groups` with `kernel` and `stride`.
    pub fn create(
        schema: &mut Schema<'_>,
        (causal, trim_right): (bool, f64),
        [input, out, kernel, stride, groups]: [i64; 5],
        bias: bool,
    ) -> SingeResult<Self> {
        let overhang = kernel - stride;
        let right = if causal {
            (overhang as f64 * trim_right).ceil() as i64
        } else {
            overhang / 2
        };
        Ok(Self {
            convolution: schema.conv_transpose1d(
                "conv",
                input,
                out,
                kernel,
                ConvTranspose1dConfig {
                    stride,
                    groups,
                    ..ConvTranspose1dConfig::default()
                },
                bias,
            )?,
            overhang,
            right,
        })
    }
}

impl Layer for TrimmedTranspose {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let output = input.apply(&self.convolution, state)?;
        let left = self.overhang - self.right;
        output.narrow(2, left, time_length(&output)? - left - self.right)
    }
}

/// A residual block: ELU, a convolution narrowing the channels, ELU, a 1 × 1 convolution widening them back, added to the input or its 1 × 1 projection.
#[derive(Debug, Clone)]
struct ResidualBlock {
    narrow: PaddedConv,
    widen: PaddedConv,
    shortcut: Option<PaddedConv>,
}

impl Layer for ResidualBlock {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = input
            .elu(1.0)?
            .apply(&self.narrow, state)?
            .elu(1.0)?
            .apply(&self.widen, state)?;
        let shortcut = match &self.shortcut {
            Some(shortcut) => input.apply(shortcut, state)?,
            None => input.clone(),
        };
        shortcut.add(&hidden)
    }
}

/// One of a stack's layers, in the order `transformers` numbers them.
#[derive(Debug, Clone)]
enum SeanetLayer {
    Conv(PaddedConv),
    Transpose(TrimmedTranspose),
    Residual(Box<ResidualBlock>),
    Lstm(Box<Recurrent>),
    Elu,
}

impl Layer for SeanetLayer {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Conv(layer) => layer.forward(input, state),
            Self::Transpose(layer) => layer.forward(input, state),
            Self::Residual(layer) => layer.forward(input, state),
            // The LSTM reads time-major sequences and adds its input back, as `EncodecLSTM` does.
            Self::Lstm(lstm) => {
                let sequence = input.transpose(1, 2)?;
                let (output, _) = lstm.forward_sequence(&sequence, None, None, state)?;
                output.add(&sequence)?.transpose(1, 2)
            }
            Self::Elu => input.elu(1.0),
        }
    }
}

/// A `SEANet` encoder, from samples, `[B, C, N]`, to latents, `[B, D, T]`, or decoder, back from latents to samples.
#[derive(Debug, Clone)]
pub struct Seanet {
    layers: Vec<SeanetLayer>,
}

/// The layers of a stack, registered as `layers.N` in `transformers`' order.
struct Builder<'a, 'b> {
    schema: &'a mut Schema<'b>,
    config: &'a SeanetConfig,
    layers: Vec<SeanetLayer>,
}

impl Builder<'_, '_> {
    fn scope(&mut self) -> Schema<'_> {
        self.schema.scope(format!("layers.{}", self.layers.len()))
    }

    fn padding(&self) -> (bool, PadMode) {
        (self.config.causal, self.config.pad)
    }

    fn conv(&mut self, shape: [i64; 5]) -> SingeResult<()> {
        let padding = self.padding();
        let layer = PaddedConv::create(&mut self.scope(), padding, shape, true)?;
        self.layers.push(SeanetLayer::Conv(layer));
        Ok(())
    }

    fn transpose(&mut self, [input, out, kernel, stride]: [i64; 4]) -> SingeResult<()> {
        let trim = (self.config.causal, self.config.trim_right);
        let layer = TrimmedTranspose::create(
            &mut self.scope(),
            trim,
            [input, out, kernel, stride, 1],
            true,
        )?;
        self.layers.push(SeanetLayer::Transpose(layer));
        Ok(())
    }

    fn residuals(&mut self, width: i64) -> SingeResult<()> {
        let config = self.config;
        let padding = self.padding();
        for index in 0..config.residual_layers {
            let dilation = config.dilation_growth.pow(
                u32::try_from(index).map_err(|_| SingeError::out_of_range("residual layers"))?,
            );
            let hidden = width / config.compress;
            let mut scope = self.scope();
            let block = ResidualBlock {
                narrow: PaddedConv::create(
                    &mut scope.scope("block.1"),
                    padding,
                    [width, hidden, config.residual_kernel, 1, dilation],
                    true,
                )?,
                widen: PaddedConv::create(
                    &mut scope.scope("block.3"),
                    padding,
                    [hidden, width, 1, 1, 1],
                    true,
                )?,
                shortcut: config
                    .conv_shortcut
                    .then(|| {
                        PaddedConv::create(
                            &mut scope.scope("shortcut"),
                            padding,
                            [width, width, 1, 1, 1],
                            true,
                        )
                    })
                    .transpose()?,
            };
            self.layers.push(SeanetLayer::Residual(Box::new(block)));
        }
        Ok(())
    }

    fn lstm(&mut self, width: i64) -> SingeResult<()> {
        let layers = self.config.lstm_layers;
        if layers == 0 {
            return Ok(());
        }

        let lstm = self.scope().recurrent(
            "lstm",
            &RecurrentConfig {
                cell: RecurrentCell::Lstm,
                input: width,
                hidden: width,
                layers,
                bidirectional: false,
                biases: true,
                projection: None,
            },
        )?;
        self.layers.push(SeanetLayer::Lstm(Box::new(lstm)));
        Ok(())
    }
}

impl Seanet {
    /// Registers an encoder as `EncodecEncoder` and `MimiEncoder` build theirs.
    pub fn encoder(schema: &mut Schema<'_>, config: &SeanetConfig) -> SingeResult<Self> {
        let mut builder = Builder {
            schema,
            config,
            layers: Vec::new(),
        };
        builder.conv([config.channels, config.filters, config.kernel, 1, 1])?;
        let mut width = config.filters;
        for &ratio in config.ratios.iter().rev() {
            builder.residuals(width)?;
            builder.layers.push(SeanetLayer::Elu);
            builder.conv([width, 2 * width, 2 * ratio, ratio, 1])?;
            width *= 2;
        }

        builder.lstm(width)?;
        builder.layers.push(SeanetLayer::Elu);
        builder.conv([width, config.hidden, config.last_kernel, 1, 1])?;
        Ok(Self {
            layers: builder.layers,
        })
    }

    /// Registers a decoder as `EncodecDecoder` and `MimiDecoder` build theirs.
    pub fn decoder(schema: &mut Schema<'_>, config: &SeanetConfig) -> SingeResult<Self> {
        let mut builder = Builder {
            schema,
            config,
            layers: Vec::new(),
        };
        let mut width = config.filters
            << u32::try_from(config.ratios.len())
                .map_err(|_| SingeError::out_of_range("ratios"))?;
        builder.conv([config.hidden, width, config.kernel, 1, 1])?;
        builder.lstm(width)?;
        for &ratio in &config.ratios {
            builder.layers.push(SeanetLayer::Elu);
            builder.transpose([width, width / 2, 2 * ratio, ratio])?;
            builder.residuals(width / 2)?;
            width /= 2;
        }

        builder.layers.push(SeanetLayer::Elu);
        builder.conv([config.filters, config.channels, config.last_kernel, 1, 1])?;
        Ok(Self {
            layers: builder.layers,
        })
    }
}

impl Layer for Seanet {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for layer in &self.layers {
            hidden = hidden.apply(layer, state)?;
        }
        Ok(hidden)
    }
}

/// Imports a `SEANet` encoder and decoder stored as `transformers` names them under `encoder` and `decoder`, to [`Seanet`]'s names under the same parts, folding weight norms when `weight_norm`.
#[must_use]
pub fn seanet_rules(
    config: &SeanetConfig,
    [encoder, decoder]: [&str; 2],
    weight_norm: bool,
) -> RuleSet {
    let mut rules = RuleSet::new();
    if weight_norm {
        for part in [encoder, decoder] {
            for convolution in ["*.conv", "*.block.*.conv", "*.shortcut.conv"] {
                let name = join(part, &format!("layers.{convolution}"));
                rules = rules.weight_norm(&name, &format!("{name}.weight"), 0);
            }
        }
    }

    if config.lstm_layers > 0 {
        for lstm in [
            join(encoder, &format!("layers.{}.lstm", config.encoder_lstm())),
            join(decoder, "layers.1.lstm"),
        ] {
            rules = rules.then(recurrent_rules(&lstm, &lstm));
        }
    }
    // The convolutions' weights and biases, under the same names.
    rules.rename(encoder, encoder).rename(decoder, decoder)
}
