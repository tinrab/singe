// VITS, as MMS-TTS has it: a text encoder of relative-position attention gives each character a prior and a hidden state, a stochastic duration predictor run backward through its flows says how many frames each character lasts, the priors repeated that long run backward through coupling flows, and HiFi-GAN turns the latents into samples.

mod spline;
mod transformers;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTransposeConfig, Embedding,
        LayerNorm, Linear, Model, ModelState, StateBuilder, StateKey,
    },
};

use crate::{
    error::Result,
    format::Loader,
    head::{HeadLoader, Speech, SpeechLayers, TextToSpeech},
    model::ModelEntry,
    processor::host_values,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub ffn_kernel: i64,
    pub window: i64,
    pub eps: f32,
    pub flow_size: i64,
    pub duration_flows: i64,
    pub duration_bins: i64,
    pub tail_bound: f32,
    pub dds_layers: i64,
    pub dds_kernel: i64,
    pub prior_flows: i64,
    pub wavenet_layers: i64,
    pub wavenet_kernel: i64,
    pub upsample_initial: i64,
    pub upsample_rates: Vec<i64>,
    pub upsample_kernels: Vec<i64>,
    pub resblock_kernels: Vec<i64>,
    pub resblock_dilations: Vec<Vec<i64>>,
    pub leaky_slope: f64,
    pub sample_rate: i64,
    pub noise_scale: f32,
    pub noise_scale_duration: f32,
    pub speaking_rate: f32,
}

fn same(kernel: i64, dilation: i64) -> Conv1dConfig {
    let padding = (kernel * dilation - dilation) / 2;
    Conv1dConfig {
        padding: (padding, padding),
        dilation,
        ..Conv1dConfig::default()
    }
}

// A layer norm over the channels of `[B, C, T]`.
fn channel_norm(input: &Tensor, norm: &LayerNorm, state: &ModelState) -> SingeResult<Tensor> {
    input.transpose(1, 2)?.apply(norm, state)?.transpose(1, 2)
}

#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    relative_keys: StateKey,
    relative_values: StateKey,
    heads: i64,
    window: i64,
}

impl Attention {
    // Each pair of positions' relative embedding of `table`, `[1, window · 2 + 1, head]`, `[T, T, head]`, zero beyond the window.
    fn relative(&self, table: &Tensor, length: i64) -> SingeResult<Tensor> {
        let head = table.shape().dimensions()[2];
        let window = self.window;
        let (index, inside): (Vec<i64>, Vec<f32>) = (0..length)
            .flat_map(|query| (0..length).map(move |key| key - query))
            .map(|offset| {
                if offset.abs() <= window {
                    (offset + window, 1.0)
                } else {
                    (0, 0.0)
                }
            })
            .unzip();
        let device = table.device();
        let count = length * length;
        let index = Tensor::from_vec(device, Shape::create([count, 1])?, index)?
            .expand(Shape::create([count, head])?)?;
        let inside = Tensor::from_vec(device, Shape::create([count, 1])?, inside)?
            .cast(table.data_type())?;
        table
            .squeeze(0)?
            .gather(0, &index)?
            .mul(&inside)?
            .reshape([length, length, head])
    }

    fn forward(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let (length, width) = (dimensions[1], dimensions[2]);
        let head = width / self.heads;
        let split = |tensor: Tensor| tensor.reshape([length, self.heads, head])?.transpose(0, 1);
        #[expect(clippy::cast_precision_loss, reason = "head widths are small")]
        let scale = (head as f64).powf(-0.5);
        let query = split(hidden.apply(&self.query, state)?.mul_scalar(scale)?)?;
        let key = split(hidden.apply(&self.key, state)?)?;
        let value = split(hidden.apply(&self.value, state)?)?;
        // Each query's dot product with each key, and with the embedding of their offset.
        let relative_keys = self.relative(state.tensor(&self.relative_keys)?, length)?;
        let scores = query.matmul(&key.transpose(1, 2)?)?.add(
            &query
                .unsqueeze(2)?
                .mul(&relative_keys.unsqueeze(0)?)?
                .sum(&[-1], false)?,
        )?;
        let probabilities = scores.softmax(-1)?;
        let relative_values = self.relative(state.tensor(&self.relative_values)?, length)?;
        let attended = probabilities.matmul(&value)?.add(
            &probabilities
                .unsqueeze(3)?
                .mul(&relative_values.unsqueeze(0)?)?
                .sum(&[2], false)?,
        )?;
        attended
            .transpose(0, 1)?
            .reshape([1, length, width])?
            .apply(&self.output, state)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    attention: Attention,
    attention_norm: LayerNorm,
    up: Conv1d,
    down: Conv1d,
    feed_forward_norm: LayerNorm,
}

// A stack of dilated depthwise and pointwise convolutions, each normed and activated, with residuals.
#[derive(Debug, Clone)]
struct DepthSeparable {
    dilated: Vec<Conv1d>,
    pointwise: Vec<Conv1d>,
    first_norms: Vec<LayerNorm>,
    second_norms: Vec<LayerNorm>,
}

impl DepthSeparable {
    fn forward(
        &self,
        input: &Tensor,
        condition: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = match condition {
            Some(condition) => input.add(condition)?,
            None => input.clone(),
        };
        for index in 0..self.dilated.len() {
            let inner = channel_norm(
                &hidden.apply(&self.dilated[index], state)?,
                &self.first_norms[index],
                state,
            )?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.pointwise[index], state)?;
            let inner = channel_norm(&inner, &self.second_norms[index], state)?
                .apply(&Activation::GeluExact, state)?;
            hidden = hidden.add(&inner)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone)]
struct ConvFlow {
    pre: Conv1d,
    dds: DepthSeparable,
    projection: Conv1d,
}

#[derive(Debug, Clone)]
struct WaveNet {
    inputs: Vec<Conv1d>,
    residual_skips: Vec<Conv1d>,
    width: i64,
}

impl WaveNet {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        let mut output: Option<Tensor> = None;
        let last = self.inputs.len() - 1;
        for (index, (layer, residual_skip)) in
            self.inputs.iter().zip(&self.residual_skips).enumerate()
        {
            let gates = hidden.apply(layer, state)?;
            let acts = gates
                .narrow(1, 0, self.width)?
                .apply(&Activation::Tanh, state)?
                .mul(
                    &gates
                        .narrow(1, self.width, self.width)?
                        .apply(&Activation::Sigmoid, state)?,
                )?;
            let out = acts.apply(residual_skip, state)?;
            let skip = if index < last {
                hidden = hidden.add(&out.narrow(1, 0, self.width)?)?;
                out.narrow(1, self.width, self.width)?
            } else {
                out
            };
            output = Some(match output {
                Some(sum) => sum.add(&skip)?,
                None => skip,
            });
        }
        output.ok_or_else(|| singe::Error::invalid_argument("a WaveNet without layers"))
    }
}

#[derive(Debug, Clone)]
struct Coupling {
    pre: Conv1d,
    wavenet: WaveNet,
    post: Conv1d,
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    first: Vec<Conv1d>,
    second: Vec<Conv1d>,
}

#[derive(Debug, Clone)]
pub struct VitsLayers {
    tokens: Embedding,
    layers: Vec<EncoderLayer>,
    prior: Conv1d,
    duration_pre: Conv1d,
    duration_dds: DepthSeparable,
    duration_projection: Conv1d,
    // The affine flow's shift and log scale, then the convolutional flows, as the predictor stores them.
    affine: (StateKey, StateKey),
    duration_flows: Vec<ConvFlow>,
    couplings: Vec<Coupling>,
    decoder_pre: Conv1d,
    upsamplers: Vec<ConvTranspose1d>,
    resblocks: Vec<ResidualBlock>,
    decoder_post: Conv1d,
    config: VitsConfig,
}

// Reverses the order of `[1, C, T]`'s channels.
fn flip_channels(input: &Tensor) -> SingeResult<Tensor> {
    let channels = input.shape().dimensions()[1];
    let parts = (0..channels)
        .rev()
        .map(|channel| input.narrow(1, channel, 1))
        .collect::<SingeResult<Vec<_>>>()?;
    Tensor::concat(&parts, 1)
}

impl VitsLayers {
    // The duration predictor run backward from `noise`, `[1, 2, T]`, conditioned on the text's states: each character's log duration.
    fn log_durations(
        &self,
        hidden: &Tensor,
        noise: Tensor,
        state: &ModelState,
    ) -> SingeResult<Vec<f32>> {
        let condition = self
            .duration_dds
            .forward(&hidden.apply(&self.duration_pre, state)?, None, state)?
            .apply(&self.duration_projection, state)?;
        let length = hidden.shape().dimensions()[2];
        let bins = usize::try_from(self.config.duration_bins)
            .map_err(|_| singe::Error::out_of_range("bins"))?;
        #[expect(clippy::cast_precision_loss, reason = "widths are small")]
        let scale = (self.config.width as f32).sqrt();
        let mut latents = noise;
        // The convolutional flows backward, but the first, which inference skips, as `transformers` does.
        for flow in self.duration_flows.iter().skip(1).rev() {
            latents = flip_channels(&latents)?;
            let first = latents.narrow(1, 0, 1)?;
            let second = host_values(&latents.narrow(1, 1, 1)?)?;
            let parameters = flow
                .dds
                .forward(&first.apply(&flow.pre, state)?, Some(&condition), state)?
                .apply(&flow.projection, state)?;
            let parameters = host_values(&parameters.squeeze(0)?.transpose(0, 1)?.contiguous()?)?;
            let per_point = parameters.len() / second.len();
            let inverted = second
                .iter()
                .enumerate()
                .map(|(point, &value)| {
                    let row = &parameters[point * per_point..(point + 1) * per_point];
                    let widths = row[..bins]
                        .iter()
                        .map(|value| value / scale)
                        .collect::<Vec<_>>();
                    let heights = row[bins..2 * bins]
                        .iter()
                        .map(|value| value / scale)
                        .collect::<Vec<_>>();
                    spline::inverse(
                        value,
                        &widths,
                        &heights,
                        &row[2 * bins..],
                        self.config.tail_bound,
                    )
                })
                .collect::<Vec<_>>();
            let second =
                Tensor::from_vec(latents.device(), Shape::create([1, 1, length])?, inverted)?
                    .cast(latents.data_type())?;
            latents = Tensor::concat(&[first, second], 1)?;
        }

        latents = flip_channels(&latents)?;
        let (shift, log_scale) = (state.tensor(&self.affine.0)?, state.tensor(&self.affine.1)?);
        latents = latents.sub(shift)?.mul(&log_scale.neg()?.exp()?)?;
        host_values(&latents.narrow(1, 0, 1)?)
    }
}

impl SpeechLayers for VitsLayers {
    fn speak(
        &self,
        tokens: &[i64],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        scales: (f32, f32),
        state: &ModelState,
    ) -> SingeResult<Vec<f32>> {
        let config = &self.config;
        let device = state.device()?;
        let length =
            i64::try_from(tokens.len()).map_err(|_| singe::Error::out_of_range("tokens"))?;
        let tokens = Tensor::from_vec(device, Shape::create([1, length])?, tokens.to_vec())?;
        #[expect(clippy::cast_precision_loss, reason = "widths are small")]
        let mut hidden = tokens
            .apply(&self.tokens, state)?
            .mul_scalar((config.width as f64).sqrt())?;
        for layer in &self.layers {
            hidden = hidden
                .add(&layer.attention.forward(&hidden, state)?)?
                .apply(&layer.attention_norm, state)?;
            let inner = hidden
                .transpose(1, 2)?
                .apply(&layer.up, state)?
                .apply(&Activation::Relu, state)?
                .apply(&layer.down, state)?
                .transpose(1, 2)?;
            hidden = hidden.add(&inner)?.apply(&layer.feed_forward_norm, state)?;
        }

        let hidden = hidden.transpose(1, 2)?;
        let moments = hidden.apply(&self.prior, state)?;
        let (means, log_variances) = (
            moments.narrow(1, 0, config.flow_size)?,
            moments.narrow(1, config.flow_size, config.flow_size)?,
        );

        let (noise_scale, duration_scale) = scales;
        let count = usize::try_from(length).map_err(|_| singe::Error::out_of_range("tokens"))?;
        let duration_noise = noise(2 * count)
            .into_iter()
            .map(|value| value * duration_scale)
            .collect();
        let duration_noise =
            Tensor::from_vec(device, Shape::create([1, 2, length])?, duration_noise)?
                .cast(hidden.data_type())?;
        let log_durations = self.log_durations(&hidden, duration_noise, state)?;
        let length_scale = 1.0 / config.speaking_rate;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "durations are small whole numbers"
        )]
        let durations = log_durations
            .iter()
            .map(|log_duration| (log_duration.exp() * length_scale).ceil().max(0.0) as usize)
            .collect::<Vec<_>>();
        let frames = durations.iter().sum::<usize>().max(1);
        // Each character's prior repeated over its frames.
        let mut index = Vec::with_capacity(frames);
        for (character, &duration) in durations.iter().enumerate() {
            let character =
                i64::try_from(character).map_err(|_| singe::Error::out_of_range("tokens"))?;
            index.extend(iter::repeat_n(character, duration));
        }

        index.resize(frames, 0);
        let frames = i64::try_from(frames).map_err(|_| singe::Error::out_of_range("frames"))?;
        let index = Tensor::from_vec(device, Shape::create([frames])?, index)?;
        let expand = |tensor: &Tensor| tensor.squeeze(0)?.index_select(1, &index)?.unsqueeze(0);
        let (means, log_variances) = (expand(&means)?, expand(&log_variances)?);
        let count = usize::try_from(frames * config.flow_size)
            .map_err(|_| singe::Error::out_of_range("frames"))?;
        let prior_noise = Tensor::from_vec(
            device,
            Shape::create([1, config.flow_size, frames])?,
            noise(count),
        )?
        .cast(means.data_type())?;
        let mut latents = means.add(
            &prior_noise
                .mul(&log_variances.exp()?)?
                .mul_scalar(f64::from(noise_scale))?,
        )?;

        let half = config.flow_size / 2;
        for coupling in self.couplings.iter().rev() {
            latents = flip_channels(&latents)?;
            let first = latents.narrow(1, 0, half)?;
            let mean = coupling
                .wavenet
                .forward(&first.apply(&coupling.pre, state)?, state)?
                .apply(&coupling.post, state)?;
            latents = Tensor::concat(&[first, latents.narrow(1, half, half)?.sub(&mean)?], 1)?;
        }

        let slope = config.leaky_slope;
        let kernels = self.resblocks.len() / self.upsamplers.len().max(1);
        let mut hidden = latents.apply(&self.decoder_pre, state)?;
        for (index, upsampler) in self.upsamplers.iter().enumerate() {
            hidden = hidden.leaky_relu(slope)?.apply(upsampler, state)?;
            let mut sum: Option<Tensor> = None;
            for block in &self.resblocks[index * kernels..(index + 1) * kernels] {
                let mut state_block = hidden.clone();
                for (first, second) in block.first.iter().zip(&block.second) {
                    let inner = state_block
                        .leaky_relu(slope)?
                        .apply(first, state)?
                        .leaky_relu(slope)?
                        .apply(second, state)?;
                    state_block = state_block.add(&inner)?;
                }

                sum = Some(match sum {
                    Some(sum) => sum.add(&state_block)?,
                    None => state_block,
                });
            }

            #[expect(clippy::cast_precision_loss, reason = "kernel counts are small")]
            let blocks = kernels as f64;
            hidden = sum
                .ok_or_else(|| singe::Error::invalid_argument("no residual blocks"))?
                .mul_scalar(1.0 / blocks)?;
        }

        // The last activation keeps `leaky_relu`'s default slope.
        let waveform = hidden
            .leaky_relu(0.01)?
            .apply(&self.decoder_post, state)?
            .apply(&Activation::Tanh, state)?;
        host_values(&waveform)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "vits",
    formats: &[(Format::Transformers, &["vits"])],
    heads: &[HeadLoader::TextToSpeech(load_speech)],
};

fn depth_separable(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<DepthSeparable> {
    let width = config.width;
    let mut dds = DepthSeparable {
        dilated: Vec::new(),
        pointwise: Vec::new(),
        first_norms: Vec::new(),
        second_norms: Vec::new(),
    };
    for index in 0..config.dds_layers {
        let dilation = config
            .dds_kernel
            .pow(u32::try_from(index).map_err(|_| singe::Error::out_of_range("layers"))?);
        let mut layer = same(config.dds_kernel, dilation);
        layer.groups = width;
        dds.dilated.push(schema.conv1d(
            &format!("dilated.{index}"),
            width,
            width,
            config.dds_kernel,
            layer,
            true,
        )?);
        dds.pointwise.push(schema.conv1d(
            &format!("pointwise.{index}"),
            width,
            width,
            1,
            Conv1dConfig::default(),
            true,
        )?);
        dds.first_norms
            .push(schema.layer_norm(&format!("first_norms.{index}"), width, 1e-5)?);
        dds.second_norms
            .push(schema.layer_norm(&format!("second_norms.{index}"), width, 1e-5)?);
    }
    Ok(dds)
}

#[expect(
    clippy::too_many_lines,
    reason = "the schema lists every part of the model once"
)]
pub fn load_speech(loader: &mut Loader<'_>) -> Result<TextToSpeech> {
    let config: VitsConfig = loader.config(transformers::config)?;
    let width = config.width;
    let head = width / config.heads;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let tokens = Embedding::new(schema.parameter("tokens.weight", [config.vocab_size, width])?);
        let pad_left = (config.ffn_kernel - 1) / 2;
        let ffn = Conv1dConfig {
            padding: (pad_left, config.ffn_kernel / 2),
            ..Conv1dConfig::default()
        };
        let layers = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("layers.{index}"));
                Ok(EncoderLayer {
                    attention: Attention {
                        query: schema.linear("query", width, width, true)?,
                        key: schema.linear("key", width, width, true)?,
                        value: schema.linear("value", width, width, true)?,
                        output: schema.linear("output", width, width, true)?,
                        relative_keys: schema
                            .parameter("relative_keys", [1, 2 * config.window + 1, head])?,
                        relative_values: schema
                            .parameter("relative_values", [1, 2 * config.window + 1, head])?,
                        heads: config.heads,
                        window: config.window,
                    },
                    attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
                    up: schema.conv1d("up", width, config.inner, config.ffn_kernel, ffn, true)?,
                    down: schema.conv1d(
                        "down",
                        config.inner,
                        width,
                        config.ffn_kernel,
                        ffn,
                        true,
                    )?,
                    feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let prior = schema.conv1d(
            "prior",
            width,
            2 * config.flow_size,
            1,
            Conv1dConfig::default(),
            true,
        )?;
        let (duration_pre, duration_dds, duration_projection) = {
            let mut schema = schema.scope("duration");
            (
                schema.conv1d("pre", width, width, 1, Conv1dConfig::default(), true)?,
                depth_separable(&mut schema.scope("dds"), &config)?,
                schema.conv1d("projection", width, width, 1, Conv1dConfig::default(), true)?,
            )
        };
        let affine = (
            schema.parameter("duration.affine.shift", [2, 1])?,
            schema.parameter("duration.affine.log_scale", [2, 1])?,
        );
        let parameters = 3 * config.duration_bins - 1;
        let duration_flows = (0..config.duration_flows)
            .map(|index| {
                let mut schema = schema.scope(format!("duration.flows.{index}"));
                Ok(ConvFlow {
                    pre: schema.conv1d("pre", 1, width, 1, Conv1dConfig::default(), true)?,
                    dds: depth_separable(&mut schema.scope("dds"), &config)?,
                    projection: schema.conv1d(
                        "projection",
                        width,
                        parameters,
                        1,
                        Conv1dConfig::default(),
                        true,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let half = config.flow_size / 2;
        let couplings = (0..config.prior_flows)
            .map(|index| {
                let mut schema = schema.scope(format!("couplings.{index}"));
                let wavenet = WaveNet {
                    inputs: (0..config.wavenet_layers)
                        .map(|layer| {
                            schema.conv1d(
                                &format!("wavenet.inputs.{layer}"),
                                width,
                                2 * width,
                                config.wavenet_kernel,
                                same(config.wavenet_kernel, 1),
                                true,
                            )
                        })
                        .collect::<SingeResult<Vec<_>>>()?,
                    residual_skips: (0..config.wavenet_layers)
                        .map(|layer| {
                            let out = if layer + 1 < config.wavenet_layers {
                                2 * width
                            } else {
                                width
                            };
                            schema.conv1d(
                                &format!("wavenet.residual_skips.{layer}"),
                                width,
                                out,
                                1,
                                Conv1dConfig::default(),
                                true,
                            )
                        })
                        .collect::<SingeResult<Vec<_>>>()?,
                    width,
                };
                Ok(Coupling {
                    pre: schema.conv1d("pre", half, width, 1, Conv1dConfig::default(), true)?,
                    wavenet,
                    post: schema.conv1d("post", width, half, 1, Conv1dConfig::default(), true)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoder_pre = schema.conv1d(
            "decoder.pre",
            config.flow_size,
            config.upsample_initial,
            7,
            same(7, 1),
            true,
        )?;
        let mut channels = config.upsample_initial;
        let mut upsamplers = Vec::new();
        let mut resblocks = Vec::new();
        for (index, (&rate, &kernel)) in config
            .upsample_rates
            .iter()
            .zip(&config.upsample_kernels)
            .enumerate()
        {
            upsamplers.push(schema.conv_transpose1d(
                &format!("decoder.upsamplers.{index}"),
                channels,
                channels / 2,
                kernel,
                ConvTransposeConfig {
                    stride: [rate],
                    padding: [(kernel - rate) / 2],
                    ..ConvTransposeConfig::default()
                },
                true,
            )?);
            channels /= 2;
            for (kernel, dilations) in config
                .resblock_kernels
                .iter()
                .zip(&config.resblock_dilations)
            {
                let block = resblocks.len();
                let mut schema = schema.scope(format!("decoder.resblocks.{block}"));
                resblocks.push(ResidualBlock {
                    first: dilations
                        .iter()
                        .enumerate()
                        .map(|(layer, &dilation)| {
                            schema.conv1d(
                                &format!("first.{layer}"),
                                channels,
                                channels,
                                *kernel,
                                same(*kernel, dilation),
                                true,
                            )
                        })
                        .collect::<SingeResult<Vec<_>>>()?,
                    second: (0..dilations.len())
                        .map(|layer| {
                            schema.conv1d(
                                &format!("second.{layer}"),
                                channels,
                                channels,
                                *kernel,
                                same(*kernel, 1),
                                true,
                            )
                        })
                        .collect::<SingeResult<Vec<_>>>()?,
                });
            }
        }

        let decoder_post = schema.conv1d("decoder.post", channels, 1, 7, same(7, 1), false)?;
        VitsLayers {
            tokens,
            layers,
            prior,
            duration_pre,
            duration_dds,
            duration_projection,
            affine,
            duration_flows,
            couplings,
            decoder_pre,
            upsamplers,
            resblocks,
            decoder_post,
            config: config.clone(),
        }
    };
    let layers: Box<dyn SpeechLayers> = Box::new(layers);
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(TextToSpeech::new(
        model,
        Speech {
            sample_rate: config.sample_rate,
            noise_scale: config.noise_scale,
            noise_scale_duration: config.noise_scale_duration,
        },
    ))
}
