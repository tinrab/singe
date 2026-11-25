//! Heads of models that hear or speak clips: a CTC recognizer's scores of each frame's tokens, an audio classifier's of its labels, a codec's reconstructions, a zero-shot classifier's comparisons of clips and texts, and a text-to-speech model's samples.

use std::{f64::consts::PI, fmt::Debug};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Layer, Model, ModelState},
    signal::LogMelSpectrogram,
};

use crate::{
    error::Result,
    head::{Boxed, Classifier, ContrastiveEncoder, upload},
    processor::fbank::{FilterBankOptions, SpectrogramOptions, ast_features, clap_features},
};

/// What a CTC recognizer's layers read of a clip.
#[derive(Debug, Clone)]
pub enum CtcInput {
    /// The clip's samples, `[1, N]`, normalized when `normalize` is set, as wav2vec2's layers read them.
    Samples { normalize: bool },
    /// The clip's log-mel features, `[1, mels, F]`, as Parakeet's layers read them.
    LogMel(LogMelSpectrogram),
}

/// A CTC recognizer, as `*ForCTC` heads are: its layers give each frame's scores of its vocabulary, `[B, T, V]`, from what `input` makes of a clip.
#[derive(Debug)]
pub struct CtcRecognizer {
    model: Model<Boxed<dyn Layer>>,
    sample_rate: i64,
    input: CtcInput,
}

impl CtcRecognizer {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>, sample_rate: i64, input: CtcInput) -> Self {
        Self {
            model,
            sample_rate,
            input,
        }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn Layer>> {
        &self.model
    }

    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.sample_rate
    }

    #[must_use]
    pub const fn input(&self) -> &CtcInput {
        &self.input
    }

    /// Returns each frame's most likely token for one clip's samples.
    pub fn best_tokens(&self, samples: &[f32]) -> Result<Vec<i64>> {
        let count =
            i64::try_from(samples.len()).map_err(|_| SingeError::out_of_range("clip samples"))?;
        let input = match &self.input {
            CtcInput::Samples { normalize } => {
                let samples = if *normalize {
                    normalized(samples)
                } else {
                    samples.to_vec()
                };
                upload(self.model.state(), [1, count], samples)?
            }
            CtcInput::LogMel(front_end) => {
                let frames = front_end.own_frames(count);
                if frames == 0 {
                    return Ok(Vec::new());
                }

                let samples = Tensor::from_vec(
                    self.model.state().device()?,
                    Shape::create([1, count])?,
                    samples.to_vec(),
                )?;
                // A clip alone needs no mask past its own frames, as a transducer's does not.
                let (features, _) = front_end.features(&samples, None)?;
                features
                    .narrow(2, 0, frames)?
                    .cast(self.model.state().floating_type()?)?
            }
        };
        let (_, best) = self.model.forward(&input)?.topk(1, -1, true, false)?;
        Ok(best.contiguous()?.to_vec::<i64>()?)
    }
}

/// Returns `samples` at zero mean and unit variance, as `transformers`' `Wav2Vec2FeatureExtractor` normalizes a clip.
fn normalized(samples: &[f32]) -> Vec<f32> {
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
}

/// An audio classifier, as `*ForAudioClassification` heads are: its layers read a clip's filter banks, `[B, frames, mels]`.
#[derive(Debug)]
pub struct AudioClassifier {
    pub classifier: Classifier,
    pub features: FilterBankOptions,
}

impl AudioClassifier {
    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.features.sample_rate
    }

    /// Returns each label's score for one clip's samples, before any score function.
    pub fn scores(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let options = self.features;
        let features = upload(
            self.classifier.model().state(),
            [
                1,
                i64::try_from(options.frames).map_err(|_| SingeError::out_of_range("frames"))?,
                i64::try_from(options.mels).map_err(|_| SingeError::out_of_range("mels"))?,
            ],
            ast_features(samples, &options),
        )?;
        Ok(self.classifier.scores(&features)?.remove(0))
    }
}

/// An audio codec, as DAC is: its layers encode a batch of clips, `[B, N]`, to codes and decode them back to samples, `[B, M]`, with `M` at most `N`.
#[derive(Debug)]
pub struct AudioCodec {
    model: Model<Boxed<dyn Layer>>,
    sample_rate: i64,
}

impl AudioCodec {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>, sample_rate: i64) -> Self {
        Self { model, sample_rate }
    }

    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.sample_rate
    }

    /// Returns one clip's samples encoded to the codec's codes and decoded back.
    pub fn reconstruct(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let count =
            i64::try_from(samples.len()).map_err(|_| SingeError::out_of_range("clip samples"))?;
        let samples = upload(self.model.state(), [1, count], samples.to_vec())?;
        Ok(self
            .model
            .forward(&samples)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?)
    }
}

/// A zero-shot audio classifier, as CLAP's `*Model` is: a contrastive model of clips' spectrograms and texts.
#[derive(Debug)]
pub struct ZeroShotAudioClassifier {
    pub encoder: ContrastiveEncoder,
    pub spectrogram: SpectrogramOptions,
}

impl ZeroShotAudioClassifier {
    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.spectrogram.sample_rate
    }

    /// Returns one clip's spectrogram, `[1, 1, frames, mels]`.
    pub fn features(&self, samples: &[f32]) -> Result<Tensor> {
        let options = self.spectrogram;
        let features = clap_features(samples, &options);
        let mels = i64::try_from(options.mels).map_err(|_| SingeError::out_of_range("mels"))?;
        let frames = i64::try_from(features.len())
            .map_err(|_| SingeError::out_of_range("spectrogram features"))?
            / mels;
        Ok(upload(
            self.encoder.inputs().state(),
            [1, 1, frames, mels],
            features,
        )?)
    }
}

/// A text-to-speech model's layers.
pub trait SpeechLayers: Debug + Send + Sync {
    /// Returns the samples of `tokens` spoken, drawing standard normal noise of a given length from `noise`, scaled by `scales`, the latents' and then the durations'.
    fn speak(
        &self,
        tokens: &[i64],
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        scales: (f32, f32),
        state: &ModelState,
    ) -> SingeResult<Vec<f32>>;
}

/// How a text-to-speech model speaks by default: its sample rate and its noise scales.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Speech {
    pub sample_rate: i64,
    pub noise_scale: f32,
    pub noise_scale_duration: f32,
}

#[derive(Debug)]
pub struct TextToSpeech {
    model: Model<Boxed<dyn SpeechLayers>>,
    speech: Speech,
}

/// Returns a source of standard normal values from `seed`, by `SplitMix64` and the Box–Muller transform.
/// `transformers` draws VITS's duration noise from torch's CPU generator and its latents' noise on the model's device, so a sample matches `transformers`' only without noise.
fn normal_noise(seed: u64) -> impl FnMut(usize) -> Vec<f32> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;

        ((value >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    move |count| {
        (0..count)
            .map(|_| {
                let (first, second) = (next(), next());

                ((-2.0 * first.ln()).sqrt() * (2.0 * PI * second).cos()) as f32
            })
            .collect()
    }
}

impl TextToSpeech {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn SpeechLayers>>, speech: Speech) -> Self {
        Self { model, speech }
    }

    #[must_use]
    pub const fn speech(&self) -> &Speech {
        &self.speech
    }

    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.speech.sample_rate
    }

    /// Returns the samples of `tokens` spoken, with the noise scales given or else the model's, from noise seeded by `seed`.
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
