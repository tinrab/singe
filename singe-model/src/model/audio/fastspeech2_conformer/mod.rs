//! `FastSpeech2Conformer`: phonemes through a post-norm Conformer encoder, pitch and energy predicted and embedded back, each phoneme repeated for its predicted frames, a Conformer decoder over the frames, a mel projection refined by a convolutional postnet, and HiFi-GAN.

mod transformers;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{Activation, BatchNorm, Conv1d, Embedding, Layer, LayerNorm, Linear, ModelState},
};

use crate::{
    assembly::{
        ConformerBlock, ConformerConfig, HifiGan, HifiGanConfig, PYTORCH_NORM_EPS,
        relative_sinusoids, same_1d,
    },
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, Speech, SpeechLayers, TextToSpeech},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The sizes of a stack of convolutions, such as a variance predictor's or the postnet's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvStack {
    pub layers: i64,
    pub channels: i64,
    pub kernel: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FastSpeech2Config {
    pub vocab_size: i64,
    pub width: i64,
    pub mels: i64,
    pub encoder: ConformerConfig,
    pub encoder_layers: i64,
    pub decoder: ConformerConfig,
    pub decoder_layers: i64,
    pub duration: ConvStack,
    pub pitch: ConvStack,
    pub pitch_embed_kernel: i64,
    pub energy: ConvStack,
    pub energy_embed_kernel: i64,
    pub postnet: ConvStack,
    pub vocoder: HifiGanConfig,
    pub sample_rate: i64,
}

/// A variance predictor: convolutions, each with `ReLU` and a layer norm over channels, then a projection to one value per phoneme.
#[derive(Debug, Clone)]
struct Predictor {
    layers: Vec<(Conv1d, LayerNorm)>,
    output: Linear,
}

impl Predictor {
    fn create(schema: &mut Schema<'_>, width: i64, stack: ConvStack) -> SingeResult<Self> {
        Ok(Self {
            layers: (0..stack.layers)
                .map(|index| {
                    let input = if index == 0 { width } else { stack.channels };
                    let mut layer = schema.scope(format!("layers.{index}"));
                    Ok((
                        layer.conv1d(
                            "conv",
                            input,
                            stack.channels,
                            stack.kernel,
                            same_1d(stack.kernel, 1),
                            true,
                        )?,
                        layer.layer_norm("norm", stack.channels, PYTORCH_NORM_EPS)?,
                    ))
                })
                .collect::<SingeResult<Vec<_>>>()?,
            output: schema.linear("output", stack.channels, 1, true)?,
        })
    }
}

impl Layer for Predictor {
    /// Returns one value per phoneme, `[B, T, 1]`, of the encoder's states, `[B, T, W]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.transpose(1, 2)?;
        for (conv, norm) in &self.layers {
            hidden = hidden
                .apply(conv, state)?
                .apply(&Activation::Relu, state)?
                .transpose(1, 2)?
                .apply(norm, state)?
                .transpose(1, 2)?;
        }
        hidden.transpose(1, 2)?.apply(&self.output, state)
    }
}

/// A predicted variance's embedding: a convolution from its one channel to the width.
#[derive(Debug, Clone)]
struct VarianceEmbedding(Conv1d);

impl Layer for VarianceEmbedding {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .transpose(1, 2)?
            .apply(&self.0, state)?
            .transpose(1, 2)
    }
}

/// A Conformer stack over `[B, T, W]`, its input scaled by the root of the width, with the relative positions of its length.
#[derive(Debug, Clone)]
struct Stack {
    blocks: Vec<ConformerBlock>,
    width: i64,
}

impl Stack {
    fn create(schema: &mut Schema<'_>, config: &ConformerConfig, layers: i64) -> SingeResult<Self> {
        Ok(Self {
            blocks: (0..layers)
                .map(|index| {
                    ConformerBlock::create(&mut schema.scope(format!("blocks.{index}")), config)
                })
                .collect::<SingeResult<Vec<_>>>()?,
            width: config.width,
        })
    }
}

impl Layer for Stack {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let steps = input.shape().dimensions()[1];
        let positions = relative_sinusoids(input.device(), (steps - 1).into(), self.width)?
            .cast(input.data_type())?;
        let mut hidden = input.mul_scalar((self.width as f64).sqrt())?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &positions, None, state)?;
        }
        Ok(hidden)
    }
}

/// The postnet's layers: a convolution without bias and a batch norm, with tanh on all but the last.
#[derive(Debug, Clone)]
struct Postnet {
    layers: Vec<(Conv1d, BatchNorm)>,
}

impl Postnet {
    fn create(schema: &mut Schema<'_>, mels: i64, stack: ConvStack) -> SingeResult<Self> {
        Ok(Self {
            layers: (0..stack.layers)
                .map(|index| {
                    let input = if index == 0 { mels } else { stack.channels };
                    let output = if index + 1 == stack.layers {
                        mels
                    } else {
                        stack.channels
                    };
                    let mut layer = schema.scope(format!("layers.{index}"));
                    Ok((
                        layer.conv1d(
                            "conv",
                            input,
                            output,
                            stack.kernel,
                            same_1d(stack.kernel, 1),
                            false,
                        )?,
                        layer.batch_norm("norm", output, PYTORCH_NORM_EPS)?,
                    ))
                })
                .collect::<SingeResult<Vec<_>>>()?,
        })
    }
}

impl Layer for Postnet {
    /// Returns the postnet's refinement, `[B, M, T]`, of mels, `[B, M, T]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for (index, (conv, norm)) in self.layers.iter().enumerate() {
            hidden = hidden.apply(conv, state)?.apply(norm, state)?;
            if index + 1 < self.layers.len() {
                hidden = hidden.apply(&Activation::Tanh, state)?;
            }
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone)]
pub struct FastSpeech2Layers {
    tokens: Embedding,
    encoder: Stack,
    duration: Predictor,
    pitch: Predictor,
    pitch_embed: VarianceEmbedding,
    energy: Predictor,
    energy_embed: VarianceEmbedding,
    decoder: Stack,
    mels: Linear,
    postnet: Postnet,
    vocoder: HifiGan,
}

impl FastSpeech2Layers {
    pub fn create(schema: &mut Schema<'_>, config: &FastSpeech2Config) -> SingeResult<Self> {
        let width = config.width;
        let embedding = |schema: &mut Schema<'_>, name: &str, kernel: i64| {
            schema
                .conv1d(name, 1, width, kernel, same_1d(kernel, 1), true)
                .map(VarianceEmbedding)
        };
        Ok(Self {
            tokens: Embedding::new(schema.parameter("tokens.weight", [config.vocab_size, width])?),
            encoder: Stack::create(
                &mut schema.scope("encoder"),
                &config.encoder,
                config.encoder_layers,
            )?,
            duration: Predictor::create(&mut schema.scope("duration"), width, config.duration)?,
            pitch: Predictor::create(&mut schema.scope("pitch"), width, config.pitch)?,
            pitch_embed: embedding(schema, "pitch_embed", config.pitch_embed_kernel)?,
            energy: Predictor::create(&mut schema.scope("energy"), width, config.energy)?,
            energy_embed: embedding(schema, "energy_embed", config.energy_embed_kernel)?,
            decoder: Stack::create(
                &mut schema.scope("decoder"),
                &config.decoder,
                config.decoder_layers,
            )?,
            mels: schema.linear("mels", width, config.mels, true)?,
            postnet: Postnet::create(&mut schema.scope("postnet"), config.mels, config.postnet)?,
            vocoder: HifiGan::create(&mut schema.scope("vocoder"), &config.vocoder)?,
        })
    }

    fn spoken(&self, tokens: &[i64], state: &ModelState) -> SingeResult<Vec<f32>> {
        let device = state.device()?;
        let length = i64::try_from(tokens.len()).map_err(|_| SingeError::out_of_range("tokens"))?;
        let tokens = Tensor::from_vec(device, Shape::create([1, length])?, tokens.to_vec())?;
        let encoded = tokens
            .apply(&self.tokens, state)?
            .apply(&self.encoder, state)?;
        let log_durations = encoded
            .apply(&self.duration, state)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        let varied = encoded
            .add(
                &encoded
                    .apply(&self.energy, state)?
                    .apply(&self.energy_embed, state)?,
            )?
            .add(
                &encoded
                    .apply(&self.pitch, state)?
                    .apply(&self.pitch_embed, state)?,
            )?;
        let frames = frame_phonemes(&log_durations);
        let count = i64::try_from(frames.len()).map_err(|_| SingeError::out_of_range("frames"))?;
        let frames = Tensor::from_vec(device, Shape::create([count])?, frames)?;
        let mels = varied
            .squeeze(0)?
            .index_select(0, &frames)?
            .unsqueeze(0)?
            .apply(&self.decoder, state)?
            .apply(&self.mels, state)?
            .transpose(1, 2)?;
        let refined = mels.add(&mels.apply(&self.postnet, state)?)?;
        refined
            .apply(&self.vocoder, state)?
            .cast(DataType::F32)?
            .to_vec::<f32>()
    }
}

/// Returns each frame's phoneme from the duration predictor's log durations, as `length_regulator` repeats them: each phoneme for `round(exp(d) − 1)` frames, at least none, and each for one frame when none would have any.
fn frame_phonemes(log_durations: &[f32]) -> Vec<i64> {
    let mut durations = log_durations
        .iter()
        .map(|log_duration| (log_duration.exp() - 1.0).round_ties_even().max(0.0) as usize)
        .collect::<Vec<_>>();
    if durations.iter().all(|&duration| duration == 0) {
        durations.fill(1);
    }

    durations
        .iter()
        .zip(0_i64..)
        .flat_map(|(&duration, phoneme)| iter::repeat_n(phoneme, duration))
        .collect()
}

impl SpeechLayers for FastSpeech2Layers {
    /// Speaks without noise, so neither `noise` nor `scales` changes what it returns.
    fn speak(
        &self,
        tokens: &[i64],
        _: &mut dyn FnMut(usize) -> Vec<f32>,
        _: (f32, f32),
        state: &ModelState,
    ) -> SingeResult<Vec<f32>> {
        // TF32 convolutions take the clip from 88 dB of SNR against `transformers`' to 36 dB: full precision.
        with_convolution_precision(ComputePrecision::Full, || self.spoken(tokens, state))
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "fastspeech2_conformer",
    formats: &[(
        Format::Transformers,
        &["fastspeech2_conformer_with_hifigan"],
    )],
    loaders: &[HeadLoader::TextToSpeech(load_speech)],
};

pub fn load_speech(loader: &mut Loader<'_>) -> Result<TextToSpeech> {
    let config: FastSpeech2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        FastSpeech2Layers::create(schema, &config)
            .map(|layers| Boxed::<dyn SpeechLayers>(Box::new(layers)))
    })?;
    Ok(TextToSpeech::new(
        loader.load_whole(&model, &transformers::rules())?,
        Speech {
            sample_rate: config.sample_rate,
            noise_scale: 0.0,
            noise_scale_duration: 0.0,
        },
    ))
}
