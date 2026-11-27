//! VITS, as MMS-TTS has it: a text encoder of relative-position attention gives each character a prior and a hidden state, a stochastic duration predictor run backward through its flows says how many frames each character lasts, latents sampled from the priors repeated that long run backward through coupling flows, and HiFi-GAN turns the latents into samples.

mod spline;
mod transformers;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{
        Activation, Conv1d, Conv1dConfig, Embedding, Layer, LayerNorm, Linear, ModelState, StateKey,
    },
    shape::spatial::Padding,
};

use crate::{
    assembly::{HifiGan, HifiGanConfig, PYTORCH_NORM_EPS, channel_norm, same_1d},
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, Speech, SpeechLayers, TextToSpeech},
    model::ModelEntry,
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
    pub vocoder: HifiGanConfig,
    pub sample_rate: i64,
    pub noise_scale: f32,
    pub noise_scale_duration: f32,
    pub speaking_rate: f32,
}

/// Reverses the order of the channels of `input`, `[B, C, T]`.
fn flip_channels(input: &Tensor) -> SingeResult<Tensor> {
    let &[_, channels, _] = input.shape().dimensions() else {
        return Err(SingeError::invalid_argument("VITS flow latents"));
    };
    let parts = (0..channels)
        .rev()
        .map(|channel| input.narrow(1, channel, 1))
        .collect::<SingeResult<Vec<_>>>()?;
    Tensor::concat(&parts, 1)
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
    fn create(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<Self> {
        let [query, key, value, output] = schema.attention_projections(config.width, [true; 4])?;
        let relative = [1, 2 * config.window + 1, config.width / config.heads];
        Ok(Self {
            query,
            key,
            value,
            output,
            relative_keys: schema.parameter("relative_keys", relative)?,
            relative_values: schema.parameter("relative_values", relative)?,
            heads: config.heads,
            window: config.window,
        })
    }

    /// Returns each pair of positions' relative embedding from `table`, `[1, window · 2 + 1, head]`, as `[T, T, head]`, zero beyond the window.
    fn relative(&self, table: &Tensor, length: i64) -> SingeResult<Tensor> {
        let &[_, _, head] = table.shape().dimensions() else {
            return Err(SingeError::invalid_argument("VITS relative embeddings"));
        };
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
        let &[_, length, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("VITS attention input"));
        };
        let head = width / self.heads;
        let split = |tensor: Tensor| tensor.reshape([length, self.heads, head])?.transpose(0, 1);
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

impl EncoderLayer {
    fn create(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<Self> {
        let width = config.width;
        let kernel = config.ffn_kernel;
        let feed_forward = Conv1dConfig {
            padding: Padding::new((kernel - 1) / 2, kernel / 2),
            ..Conv1dConfig::default()
        };
        Ok(Self {
            attention: Attention::create(&mut schema.scope("attention"), config)?,
            attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
            up: schema.conv1d(
                "feed_forward.up",
                width,
                config.inner,
                kernel,
                feed_forward,
                true,
            )?,
            down: schema.conv1d(
                "feed_forward.down",
                config.inner,
                width,
                kernel,
                feed_forward,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
        })
    }

    fn forward(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = hidden
            .add(&self.attention.forward(hidden, state)?)?
            .apply(&self.attention_norm, state)?;
        let inner = hidden
            .transpose(1, 2)?
            .apply(&self.up, state)?
            .apply(&Activation::Relu, state)?
            .apply(&self.down, state)?
            .transpose(1, 2)?;
        hidden.add(&inner)?.apply(&self.feed_forward_norm, state)
    }
}

#[derive(Debug, Clone)]
struct DepthSeparableLayer {
    dilated: Conv1d,
    pointwise: Conv1d,
    first_norm: LayerNorm,
    second_norm: LayerNorm,
}

/// A stack of dilated depthwise and pointwise convolutions, each normed and activated, with residuals, over its input plus the condition when one is given.
#[derive(Debug, Clone)]
struct DepthSeparable {
    layers: Vec<DepthSeparableLayer>,
}

impl DepthSeparable {
    fn create(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<Self> {
        let (width, kernel) = (config.width, config.dds_kernel);
        let layers = (0..config.dds_layers)
            .map(|index| {
                let power =
                    u32::try_from(index).map_err(|_| SingeError::out_of_range("dds_layers"))?;
                Ok(DepthSeparableLayer {
                    dilated: schema.conv1d(
                        &format!("dilated.{index}"),
                        width,
                        width,
                        kernel,
                        Conv1dConfig {
                            groups: width,
                            ..same_1d(kernel, kernel.pow(power))
                        },
                        true,
                    )?,
                    pointwise: schema.conv1d(
                        &format!("pointwise.{index}"),
                        width,
                        width,
                        1,
                        Conv1dConfig::default(),
                        true,
                    )?,
                    first_norm: schema.layer_norm(
                        &format!("first_norms.{index}"),
                        width,
                        PYTORCH_NORM_EPS,
                    )?,
                    second_norm: schema.layer_norm(
                        &format!("second_norms.{index}"),
                        width,
                        PYTORCH_NORM_EPS,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self { layers })
    }

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
        for layer in &self.layers {
            let inner = channel_norm(
                &hidden.apply(&layer.dilated, state)?,
                &layer.first_norm,
                state,
            )?
            .apply(&Activation::GeluExact, state)?
            .apply(&layer.pointwise, state)?;
            let inner = channel_norm(&inner, &layer.second_norm, state)?
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

impl ConvFlow {
    fn create(
        schema: &mut Schema<'_>,
        config: &VitsConfig,
        input: i64,
        output: i64,
    ) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            pre: schema.conv1d("pre", input, width, 1, Conv1dConfig::default(), true)?,
            dds: DepthSeparable::create(&mut schema.scope("dds"), config)?,
            projection: schema.conv1d(
                "projection",
                width,
                output,
                1,
                Conv1dConfig::default(),
                true,
            )?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        condition: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.dds
            .forward(&input.apply(&self.pre, state)?, condition, state)?
            .apply(&self.projection, state)
    }
}

#[derive(Debug, Clone)]
struct DurationPredictor {
    condition: ConvFlow,
    shift: StateKey,
    log_scale: StateKey,
    flows: Vec<ConvFlow>,
    bins: usize,
    tail_bound: f32,
    scale: f32,
}

impl DurationPredictor {
    fn create(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<Self> {
        let width = config.width;
        let parameters = 3 * config.duration_bins - 1;
        let scale = (width as f32).sqrt();
        Ok(Self {
            condition: ConvFlow::create(schema, config, width, width)?,
            shift: schema.parameter("affine.shift", [2, 1])?,
            log_scale: schema.parameter("affine.log_scale", [2, 1])?,
            flows: (0..config.duration_flows)
                .map(|index| {
                    ConvFlow::create(
                        &mut schema.scope(format!("flows.{index}")),
                        config,
                        1,
                        parameters,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            bins: usize::try_from(config.duration_bins)
                .map_err(|_| SingeError::out_of_range("duration_bins"))?,
            tail_bound: config.tail_bound,
            scale,
        })
    }

    /// Returns each character's log duration: the duration predictor run backward from `noise`, `[1, 2, T]`, conditioned on the text's states.
    fn log_durations(
        &self,
        hidden: &Tensor,
        noise: Tensor,
        state: &ModelState,
    ) -> SingeResult<Vec<f32>> {
        let condition = self.condition.forward(hidden, None, state)?;
        let &[_, _, length] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("VITS text states"));
        };
        let (bins, scale) = (self.bins, self.scale);
        let mut latents = noise;
        // The convolutional flows backward, but the first, which inference skips, as `transformers` does.
        for flow in self.flows.iter().skip(1).rev() {
            latents = flip_channels(&latents)?;
            let first = latents.narrow(1, 0, 1)?;
            let second = latents
                .narrow(1, 1, 1)?
                .cast(DataType::F32)?
                .to_vec::<f32>()?;
            let parameters = flow
                .forward(&first, Some(&condition), state)?
                .squeeze(0)?
                .transpose(0, 1)?
                .cast(DataType::F32)?
                .to_vec::<f32>()?;
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
                    spline::inverse(value, &widths, &heights, &row[2 * bins..], self.tail_bound)
                })
                .collect::<Vec<_>>();
            let second =
                Tensor::from_vec(latents.device(), Shape::create([1, 1, length])?, inverted)?
                    .cast(latents.data_type())?;
            latents = Tensor::concat(&[first, second], 1)?;
        }

        let (shift, log_scale) = (state.tensor(&self.shift)?, state.tensor(&self.log_scale)?);
        flip_channels(&latents)?
            .sub(shift)?
            .mul(&log_scale.neg()?.exp()?)?
            .narrow(1, 0, 1)?
            .cast(DataType::F32)?
            .to_vec::<f32>()
    }
}

#[derive(Debug, Clone)]
struct WaveNetLayer {
    input: Conv1d,
    residual_skip: Conv1d,
}

impl WaveNetLayer {
    fn forward(&self, hidden: &Tensor, width: i64, state: &ModelState) -> SingeResult<Tensor> {
        let gates = hidden.apply(&self.input, state)?;
        gates
            .narrow(1, 0, width)?
            .apply(&Activation::Tanh, state)?
            .mul(
                &gates
                    .narrow(1, width, width)?
                    .apply(&Activation::Sigmoid, state)?,
            )?
            .apply(&self.residual_skip, state)
    }
}

#[derive(Debug, Clone)]
struct WaveNet {
    layers: Vec<WaveNetLayer>,
    width: i64,
}

impl WaveNet {
    fn create(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<Self> {
        let (width, kernel, count) = (config.width, config.wavenet_kernel, config.wavenet_layers);
        let layers = (0..count)
            .map(|layer| {
                let skips = if layer + 1 < count { 2 * width } else { width };
                Ok(WaveNetLayer {
                    input: schema.conv1d(
                        &format!("inputs.{layer}"),
                        width,
                        2 * width,
                        kernel,
                        same_1d(kernel, 1),
                        true,
                    )?,
                    residual_skip: schema.conv1d(
                        &format!("residual_skips.{layer}"),
                        width,
                        skips,
                        1,
                        Conv1dConfig::default(),
                        true,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self { layers, width })
    }

    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let width = self.width;
        let last = self.layers.len().saturating_sub(1);
        let mut hidden = input.clone();
        let mut skips = self.layers.iter().enumerate().map(|(index, layer)| {
            let output = layer.forward(&hidden, width, state)?;
            if index == last {
                return Ok(output);
            }

            hidden = hidden.add(&output.narrow(1, 0, width)?)?;
            output.narrow(1, width, width)
        });
        let first = skips
            .next()
            .ok_or_else(|| SingeError::invalid_argument("a WaveNet without layers"))??;
        skips.try_fold(first, |sum, skip| sum.add(&skip?))
    }
}

#[derive(Debug, Clone)]
struct Coupling {
    pre: Conv1d,
    wavenet: WaveNet,
    post: Conv1d,
    half: i64,
}

impl Coupling {
    fn create(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<Self> {
        let (width, half) = (config.width, config.flow_size / 2);
        Ok(Self {
            pre: schema.conv1d("pre", half, width, 1, Conv1dConfig::default(), true)?,
            wavenet: WaveNet::create(&mut schema.scope("wavenet"), config)?,
            post: schema.conv1d("post", width, half, 1, Conv1dConfig::default(), true)?,
            half,
        })
    }

    fn forward(&self, latents: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let latents = flip_channels(latents)?;
        let first = latents.narrow(1, 0, self.half)?;
        let mean = self
            .wavenet
            .forward(&first.apply(&self.pre, state)?, state)?
            .apply(&self.post, state)?;
        let second = latents.narrow(1, self.half, self.half)?.sub(&mean)?;
        Tensor::concat(&[first, second], 1)
    }
}

#[derive(Debug, Clone)]
pub struct VitsLayers {
    tokens: Embedding,
    blocks: Vec<EncoderLayer>,
    prior: Conv1d,
    duration: DurationPredictor,
    couplings: Vec<Coupling>,
    vocoder: HifiGan,
    config: VitsConfig,
}

impl VitsLayers {
    pub fn create(schema: &mut Schema<'_>, config: &VitsConfig) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            tokens: Embedding::new(schema.parameter("tokens.weight", [config.vocab_size, width])?),
            blocks: (0..config.layers)
                .map(|index| {
                    EncoderLayer::create(&mut schema.scope(format!("blocks.{index}")), config)
                })
                .collect::<SingeResult<Vec<_>>>()?,
            prior: schema.conv1d(
                "prior",
                width,
                2 * config.flow_size,
                1,
                Conv1dConfig::default(),
                true,
            )?,
            duration: DurationPredictor::create(&mut schema.scope("duration"), config)?,
            couplings: (0..config.prior_flows)
                .map(|index| {
                    Coupling::create(&mut schema.scope(format!("couplings.{index}")), config)
                })
                .collect::<SingeResult<Vec<_>>>()?,
            vocoder: HifiGan::create(&mut schema.scope("decoder"), &config.vocoder)?,
            config: config.clone(),
        })
    }

    fn encode_text(&self, tokens: &[i64], state: &ModelState) -> SingeResult<Tensor> {
        let length = i64::try_from(tokens.len()).map_err(|_| SingeError::out_of_range("tokens"))?;
        let tokens = Tensor::from_vec(
            state.device()?,
            Shape::create([1, length])?,
            tokens.to_vec(),
        )?;
        let mut hidden = tokens
            .apply(&self.tokens, state)?
            .mul_scalar((self.config.width as f64).sqrt())?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, state)?;
        }

        hidden.transpose(1, 2)
    }

    fn frame_characters(&self, log_durations: &[f32]) -> SingeResult<Vec<i64>> {
        let length_scale = 1.0 / self.config.speaking_rate;
        let durations = log_durations
            .iter()
            .map(|log_duration| (log_duration.exp() * length_scale).ceil().max(0.0) as usize)
            .collect::<Vec<_>>();
        let frames = durations.iter().sum::<usize>().max(1);
        let mut index = Vec::with_capacity(frames);
        for (character, &duration) in durations.iter().enumerate() {
            let character =
                i64::try_from(character).map_err(|_| SingeError::out_of_range("tokens"))?;
            index.extend(iter::repeat_n(character, duration));
        }

        index.resize(frames, 0);
        Ok(index)
    }

    fn spoken(
        &self,
        tokens: &[i64],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        scales: (f32, f32),
        state: &ModelState,
    ) -> SingeResult<Vec<f32>> {
        let flow_size = self.config.flow_size;
        let device = state.device()?;
        let (noise_scale, duration_scale) = scales;
        let hidden = self.encode_text(tokens, state)?;
        let moments = hidden.apply(&self.prior, state)?;

        let length = i64::try_from(tokens.len()).map_err(|_| SingeError::out_of_range("tokens"))?;
        let duration_noise = noise(2 * tokens.len())
            .into_iter()
            .map(|value| value * duration_scale)
            .collect();
        let duration_noise =
            Tensor::from_vec(device, Shape::create([1, 2, length])?, duration_noise)?
                .cast(hidden.data_type())?;
        let index = self.frame_characters(&self.duration.log_durations(
            &hidden,
            duration_noise,
            state,
        )?)?;

        // Each character's prior repeated over its frames.
        let frames = i64::try_from(index.len()).map_err(|_| SingeError::out_of_range("frames"))?;
        let index = Tensor::from_vec(device, Shape::create([frames])?, index)?;
        let expand = |tensor: Tensor| tensor.squeeze(0)?.index_select(1, &index)?.unsqueeze(0);
        let means = expand(moments.narrow(1, 0, flow_size)?)?;
        let log_variances = expand(moments.narrow(1, flow_size, flow_size)?)?;
        let count =
            usize::try_from(frames * flow_size).map_err(|_| SingeError::out_of_range("frames"))?;
        let prior_noise =
            Tensor::from_vec(device, Shape::create([1, flow_size, frames])?, noise(count))?
                .cast(means.data_type())?;
        let latents = means.add(
            &prior_noise
                .mul(&log_variances.exp()?)?
                .mul_scalar(f64::from(noise_scale))?,
        )?;

        let latents = self
            .couplings
            .iter()
            .rev()
            .try_fold(latents, |latents, coupling| {
                coupling.forward(&latents, state)
            })?;
        self.vocoder
            .forward(&latents, state)?
            .cast(DataType::F32)?
            .to_vec::<f32>()
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
        // TF32 convolutions take the clip from 86 dB of SNR against `transformers`' to 34 dB: full precision.
        with_convolution_precision(ComputePrecision::Full, || {
            self.spoken(tokens, noise, scales, state)
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "vits",
    formats: &[(Format::Transformers, &["vits"])],
    loaders: &[HeadLoader::TextToSpeech(load_speech)],
};

pub fn load_speech(loader: &mut Loader<'_>) -> Result<TextToSpeech> {
    let config: VitsConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        VitsLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn SpeechLayers>(Box::new(layers)))
    })?;
    Ok(TextToSpeech::new(
        loader.load_whole(&model, &transformers::rules(&config))?,
        Speech {
            sample_rate: config.sample_rate,
            noise_scale: config.noise_scale,
            noise_scale_duration: config.noise_scale_duration,
        },
    ))
}
