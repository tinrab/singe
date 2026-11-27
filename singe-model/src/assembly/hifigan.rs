//! HiFi-GAN: a convolution, then upsampling stages, each a leaky `ReLU` and a transposed convolution followed by the mean of residual blocks of several kernels, then a leaky `ReLU`, a convolution to one channel, and tanh, as VITS's decoder and `FastSpeech2ConformerHifiGan` run it.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, Conv1d, ConvTranspose1d, ConvTranspose1dConfig, Layer, ModelState},
};

use crate::{
    assembly::same_1d,
    format::{Rules, join},
    schema::Schema,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HifiGanConfig {
    /// The channels of what it reads, such as a spectrogram's mel bins.
    pub input: i64,
    pub initial: i64,
    pub upsample_rates: Vec<i64>,
    pub upsample_kernels: Vec<i64>,
    pub resblock_kernels: Vec<i64>,
    pub resblock_dilations: Vec<Vec<i64>>,
    pub leaky_slope: f64,
    /// Whether the last convolution has a bias, as `FastSpeech2Conformer`'s does and VITS's does not.
    pub post_bias: bool,
}

/// A residual block: for each dilation, a leaky `ReLU`, a dilated convolution, a leaky `ReLU`, and a convolution, added back.
#[derive(Debug, Clone)]
struct ResidualBlock {
    first: Vec<Conv1d>,
    second: Vec<Conv1d>,
}

impl ResidualBlock {
    fn create(
        schema: &mut Schema<'_>,
        channels: i64,
        kernel: i64,
        dilations: &[i64],
    ) -> SingeResult<Self> {
        let conv = |schema: &mut Schema<'_>, name: String, dilation: i64| {
            schema.conv1d(
                &name,
                channels,
                channels,
                kernel,
                same_1d(kernel, dilation),
                true,
            )
        };
        Ok(Self {
            first: dilations
                .iter()
                .enumerate()
                .map(|(layer, &dilation)| conv(schema, format!("first.{layer}"), dilation))
                .collect::<SingeResult<Vec<_>>>()?,
            second: (0..dilations.len())
                .map(|layer| conv(schema, format!("second.{layer}"), 1))
                .collect::<SingeResult<Vec<_>>>()?,
        })
    }

    fn forward(&self, input: &Tensor, slope: f64, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for (first, second) in self.first.iter().zip(&self.second) {
            let inner = hidden
                .leaky_relu(slope)?
                .apply(first, state)?
                .leaky_relu(slope)?
                .apply(second, state)?;
            hidden = hidden.add(&inner)?;
        }
        Ok(hidden)
    }
}

/// HiFi-GAN: from features, `[B, C, T]`, to samples, `[B, 1, T · Π rates]`.
#[derive(Debug, Clone)]
pub struct HifiGan {
    pre: Conv1d,
    upsamplers: Vec<ConvTranspose1d>,
    resblocks: Vec<ResidualBlock>,
    post: Conv1d,
    kernels: usize,
    slope: f64,
}

impl HifiGan {
    /// Registers `pre`, `upsamplers.N`, `resblocks.N` with `first.N` and `second.N`, and `post`.
    pub fn create(schema: &mut Schema<'_>, config: &HifiGanConfig) -> SingeResult<Self> {
        let kernels = config.resblock_kernels.len();
        if kernels == 0 {
            return Err(SingeError::invalid_argument(
                "a HiFi-GAN without residual blocks",
            ));
        }

        let pre = schema.conv1d("pre", config.input, config.initial, 7, same_1d(7, 1), true)?;
        let mut channels = config.initial;
        let mut upsamplers = Vec::with_capacity(config.upsample_rates.len());
        let mut resblocks = Vec::with_capacity(config.upsample_rates.len() * kernels);
        for (index, (&rate, &kernel)) in config
            .upsample_rates
            .iter()
            .zip(&config.upsample_kernels)
            .enumerate()
        {
            upsamplers.push(schema.conv_transpose1d(
                &format!("upsamplers.{index}"),
                channels,
                channels / 2,
                kernel,
                ConvTranspose1dConfig {
                    stride: rate,
                    padding: (kernel - rate) / 2,
                    ..ConvTranspose1dConfig::default()
                },
                true,
            )?);
            channels /= 2;
            for (&kernel, dilations) in config
                .resblock_kernels
                .iter()
                .zip(&config.resblock_dilations)
            {
                let block = resblocks.len();
                resblocks.push(ResidualBlock::create(
                    &mut schema.scope(format!("resblocks.{block}")),
                    channels,
                    kernel,
                    dilations,
                )?);
            }
        }

        Ok(Self {
            pre,
            upsamplers,
            resblocks,
            post: schema.conv1d("post", channels, 1, 7, same_1d(7, 1), config.post_bias)?,
            kernels,
            slope: config.leaky_slope,
        })
    }
}

impl Layer for HifiGan {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let slope = self.slope;
        let mut hidden = input.apply(&self.pre, state)?;
        for (upsampler, blocks) in self
            .upsamplers
            .iter()
            .zip(self.resblocks.chunks(self.kernels))
        {
            hidden = hidden.leaky_relu(slope)?.apply(upsampler, state)?;
            let mut outputs = blocks
                .iter()
                .map(|block| block.forward(&hidden, slope, state));
            let first = outputs
                .next()
                .ok_or_else(|| SingeError::invalid_argument("no residual blocks"))??;
            let sum = outputs.try_fold(first, |sum, output| sum.add(&output?))?;
            let count = blocks.len() as f64;
            hidden = sum.mul_scalar(1.0 / count)?;
        }

        // The last activation keeps `leaky_relu`'s default slope.
        hidden
            .leaky_relu(0.01)?
            .apply(&self.post, state)?
            .apply(&Activation::Tanh, state)
    }
}

/// Imports a HiFi-GAN stored as `transformers` names it under `from` to [`HifiGan`]'s names under `to`.
#[must_use]
pub fn hifigan_rules(from: &str, to: &str) -> Rules {
    Rules::new()
        .rename(&join(from, "conv_pre"), &join(to, "pre"))
        .rename(&join(from, "conv_post"), &join(to, "post"))
        .rename(&join(from, "upsampler"), &join(to, "upsamplers"))
        .rename(
            &join(from, "resblocks.*.convs1"),
            &join(to, "resblocks.*.first"),
        )
        .rename(
            &join(from, "resblocks.*.convs2"),
            &join(to, "resblocks.*.second"),
        )
}
