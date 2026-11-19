// Heads of models that hear or speak clips: a CTC recognizer's scores of each frame's characters, an audio classifier's of its labels, a codec's reconstructions, a zero-shot classifier's comparisons of clips and texts, and a text-to-speech model's samples.

use std::{f64::consts::PI, fmt::Debug};

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState},
};

use super::{Classifier, ContrastiveEncoder};
use crate::{
    error::Result,
    processor::fbank::{ast_features, clap_features},
};

// Layers that read raw samples, `[B, N]`.
#[derive(Debug)]
pub struct AudioLayers(pub Box<dyn Layer>);

impl Layer for AudioLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.0.forward(input, state)
    }
}

// A CTC recognizer, as `*ForCTC` heads are: its layers give each frame's scores of its vocabulary, `[B, T, V]`, from samples normalized when `normalize` is set.
#[derive(Debug)]
pub struct CtcRecognizer {
    model: Model<AudioLayers>,
    sample_rate: i64,
    normalize: bool,
}

impl CtcRecognizer {
    #[must_use]
    pub const fn new(model: Model<AudioLayers>, sample_rate: i64, normalize: bool) -> Self {
        Self {
            model,
            sample_rate,
            normalize,
        }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<AudioLayers> {
        &self.model
    }

    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.sample_rate
    }

    // Each frame's most likely token of one clip's samples.
    #[expect(
        clippy::cast_precision_loss,
        reason = "clips are far shorter than 2^24 samples"
    )]
    pub fn best_tokens(&self, samples: &[f32]) -> Result<Vec<i64>> {
        let samples = if self.normalize {
            // Zero mean and unit variance, as `Wav2Vec2FeatureExtractor` normalizes a clip.
            let count = samples.len() as f32;
            let mean = samples.iter().sum::<f32>() / count;
            let variance = samples
                .iter()
                .map(|sample| (sample - mean).powi(2))
                .sum::<f32>()
                / count;
            let deviation = (variance + 1e-7).sqrt();
            samples
                .iter()
                .map(|sample| (sample - mean) / deviation)
                .collect()
        } else {
            samples.to_vec()
        };
        let count = i64::try_from(samples.len())
            .map_err(|_| crate::Error::unsupported("a clip this long"))?;
        let device = self.model.state().device()?;
        let samples = Tensor::from_vec(device, singe::Shape::create([1, count])?, samples)?
            .cast(self.model.state().floating_type()?)?;
        let (_, best) = self.model.forward(&samples)?.topk(1, -1, true, false)?;
        Ok(best.contiguous()?.to_vec::<i64>()?)
    }
}

// The log mel filter banks an audio classifier reads, as AST's feature extractor computes them: `frames` of `mels` from clips at `sample_rate`, normalized by `mean` and twice `std`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FilterBankOptions {
    pub sample_rate: i64,
    pub mels: usize,
    pub frames: usize,
    pub mean: f64,
    pub std: f64,
}

// An audio classifier, as `*ForAudioClassification` heads are: its layers read a clip's filter banks, `[B, frames, mels]`.
#[derive(Debug)]
pub struct AudioClassifier {
    pub classifier: Classifier,
    pub features: FilterBankOptions,
}

impl AudioClassifier {
    // Each label's score for one clip's samples.
    pub fn scores(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let options = self.features;
        #[expect(clippy::cast_precision_loss, reason = "sample rates are small")]
        let features = ast_features(
            samples,
            options.sample_rate as f64,
            options.mels,
            options.frames,
            options.mean,
            options.std,
        );
        let state = self.classifier.model().state();
        let shape = singe::Shape::create([
            1,
            i64::try_from(options.frames)
                .map_err(|_| crate::Error::unsupported("this many frames"))?,
            i64::try_from(options.mels).map_err(|_| crate::Error::unsupported("this many mels"))?,
        ])?;
        let features =
            Tensor::from_vec(state.device()?, shape, features)?.cast(state.floating_type()?)?;
        Ok(self.classifier.scores(&features)?.remove(0))
    }
}

// An audio codec, as DAC is: its layers encode a batch of clips, `[B, N]`, to codes and decode them back to samples, `[B, N]`.
#[derive(Debug)]
pub struct AudioCodec {
    model: Model<AudioLayers>,
    sample_rate: i64,
}

impl AudioCodec {
    #[must_use]
    pub const fn new(model: Model<AudioLayers>, sample_rate: i64) -> Self {
        Self { model, sample_rate }
    }

    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.sample_rate
    }

    // One clip's samples through the codec's codes and back.
    pub fn reconstruct(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let count = i64::try_from(samples.len())
            .map_err(|_| crate::Error::unsupported("a clip this long"))?;
        let state = self.model.state();
        let samples = Tensor::from_vec(
            state.device()?,
            singe::Shape::create([1, count])?,
            samples.to_vec(),
        )?
        .cast(state.floating_type()?)?;
        Ok(self
            .model
            .forward(&samples)?
            .cast(singe::DataType::F32)?
            .to_vec::<f32>()?)
    }
}

// The spectrogram a contrastive audio model reads, as CLAP's feature extractor computes it: clips at `sample_rate` repeated to `samples`, frames of `fft` every `hop`, and `mels` Slaney filters between `low` and `high` Hz.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectrogramOptions {
    pub sample_rate: i64,
    pub samples: usize,
    pub fft: usize,
    pub hop: usize,
    pub mels: usize,
    pub low: f64,
    pub high: f64,
}

// A zero-shot audio classifier, as CLAP's `*Model` is: a contrastive model of clips' spectrograms and texts.
#[derive(Debug)]
pub struct ZeroShotAudioClassifier {
    pub encoder: ContrastiveEncoder,
    pub spectrogram: SpectrogramOptions,
}

impl ZeroShotAudioClassifier {
    // One clip's spectrogram, `[1, 1, frames, mels]`.
    pub fn features(&self, samples: &[f32]) -> Result<Tensor> {
        let options = self.spectrogram;
        #[expect(clippy::cast_precision_loss, reason = "sample rates are small")]
        let features = clap_features(
            samples,
            options.sample_rate as f64,
            options.samples,
            options.fft,
            options.hop,
            options.mels,
            options.low,
            options.high,
        );
        let mels =
            i64::try_from(options.mels).map_err(|_| crate::Error::unsupported("this many mels"))?;
        let frames = i64::try_from(features.len())
            .map_err(|_| crate::Error::unsupported("this many frames"))?
            / mels;
        let state = self.encoder.inputs().state();
        Ok(Tensor::from_vec(
            state.device()?,
            singe::Shape::create([1, 1, frames, mels])?,
            features,
        )?
        .cast(state.floating_type()?)?)
    }
}

// What a text-to-speech model's layers give: the samples of `tokens` spoken, drawing standard normal noise of a given length from `noise`, scaled by `scales`, the latents' and the durations'.
pub trait SpeechLayers: Debug + Send + Sync {
    fn speak(
        &self,
        tokens: &[i64],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        scales: (f32, f32),
        state: &ModelState,
    ) -> SingeResult<Vec<f32>>;
}

// How a text-to-speech model speaks by default: its sample rate and its noise scales.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Speech {
    pub sample_rate: i64,
    pub noise_scale: f32,
    pub noise_scale_duration: f32,
}

#[derive(Debug)]
pub struct TextToSpeech {
    model: Model<Box<dyn SpeechLayers>>,
    speech: Speech,
}

// Standard normal values from a seed, by SplitMix64 and the Box–Muller transform.
// `transformers` draws VITS's duration noise from torch's CPU generator and its latents' noise on the model's device, so a sample matches `transformers`' only without noise.
fn normal_noise(seed: u64) -> impl FnMut(usize) -> Vec<f32> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        #[expect(clippy::cast_precision_loss, reason = "53 bits of a uniform value")]
        let uniform = ((value >> 11) as f64 + 0.5) / (1_u64 << 53) as f64;
        uniform
    };
    move |count| {
        (0..count)
            .map(|_| {
                let (first, second) = (next(), next());
                #[expect(clippy::cast_possible_truncation, reason = "noise is `F32`")]
                let value = ((-2.0 * first.ln()).sqrt() * (2.0 * PI * second).cos()) as f32;
                value
            })
            .collect()
    }
}

impl TextToSpeech {
    #[must_use]
    pub const fn new(model: Model<Box<dyn SpeechLayers>>, speech: Speech) -> Self {
        Self { model, speech }
    }

    #[must_use]
    pub const fn speech(&self) -> &Speech {
        &self.speech
    }

    // Speaks `tokens`, with the noise scales given or the model's, from noise seeded by `seed`.
    pub fn speak(
        &self,
        tokens: &[i64],
        scales: (Option<f32>, Option<f32>),
        seed: u64,
    ) -> Result<Vec<f32>> {
        let scales = (
            scales.0.unwrap_or(self.speech.noise_scale),
            scales.1.unwrap_or(self.speech.noise_scale_duration),
        );
        let mut noise = normal_noise(seed);
        Ok(self
            .model
            .layer()
            .speak(tokens, &mut noise, scales, self.model.state())?)
    }
}
