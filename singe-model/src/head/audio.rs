//! Heads of models that hear or speak clips: a CTC recognizer's scores of each frame's tokens, an audio classifier's of its labels, a codec's reconstructions, a zero-shot classifier's comparisons of clips and texts, and a text-to-speech model's samples.

use std::{f64::consts::PI, fmt::Debug, sync::Arc};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Layer, Model, ModelState},
    signal::LogMelSpectrogram,
};

use crate::{
    error::{Error, Result},
    format::transformers::SampleExtractor,
    function::{
        TensorFunction,
        forward::{External, ExternalHead, Forward},
    },
    head::{Boxed, Classifier, ContrastiveEncoder, upload},
    model::audio::parakeet,
    processor::fbank::{FilterBankOptions, SpectrogramOptions, ast_features, clap_features},
    source::Source,
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
    model: Forward<Boxed<dyn Layer>>,
    sample_rate: i64,
    input: CtcInput,
    convolutions: Vec<Convolution>,
}

#[derive(Debug, Clone, Copy)]
struct Convolution {
    kernel: i64,
    stride: i64,
}

impl CtcRecognizer {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>, sample_rate: i64, input: CtcInput) -> Self {
        Self {
            model: Forward::Native(model),
            sample_rate,
            input,
            convolutions: Vec::new(),
        }
    }

    pub fn get_model(&self) -> Result<&Model<Boxed<dyn Layer>>> {
        self.model.get_native()
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
                self.model.upload([1, count], samples)?
            }
            CtcInput::LogMel(front_end) => {
                let frames = front_end.own_frames(count);
                if frames == 0 {
                    return Ok(Vec::new());
                }

                // An external function moves and casts its inputs itself.
                let (device, data_type) = match &self.model {
                    Forward::Native(model) => {
                        (model.state().device()?, model.state().floating_type()?)
                    }
                    Forward::External(_) => (Device::Cpu, DataType::F32),
                };
                let samples =
                    Tensor::from_vec(device, Shape::create([1, count])?, samples.to_vec())?;
                // A clip alone needs no mask past its own frames, as a transducer's does not.
                let (features, _) = front_end.features(&samples, None)?;
                let features = features.narrow(2, 0, frames)?.cast(data_type)?;
                match &self.model {
                    Forward::Native(_) => features,
                    // `transformers` reads `input_features` as `[B, frames, mels]`.
                    Forward::External(_) => features.transpose(1, 2)?.contiguous()?,
                }
            }
        };
        let mut scores = self.model.forward(&input)?;
        if let (Forward::External(_), CtcInput::Samples { .. }) = (&self.model, &self.input) {
            let frames = self.convolutions.iter().fold(count, |length, convolution| {
                ((length - convolution.kernel).div_euclid(convolution.stride) + 1).max(0)
            });
            scores = scores.narrow(1, 0, frames)?;
        }

        let (_, best) = scores.topk(1, -1, true, false)?;
        Ok(best.contiguous()?.to_vec::<i64>()?)
    }
}

#[derive(Deserialize)]
struct CtcFields {
    conv_kernel: Option<Vec<i64>>,
    conv_stride: Option<Vec<i64>>,
}

impl ExternalHead for CtcRecognizer {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        let mut model = External::create(function, &["logits"])?;
        // Parakeet's export reads log-mel features, whose frames a step holds exactly, since its attention reads every frame.
        if model.input("input_features").is_some() {
            let features = parakeet::features(source)?;
            return Ok(Self {
                model: Forward::External(model),
                sample_rate: features.sample_rate,
                input: CtcInput::LogMel(LogMelSpectrogram::create(features.log_mel())?),
                convolutions: Vec::new(),
            });
        }

        let fields: CtcFields = source.json("config.json")?;
        let (Some(kernels), Some(strides)) = (fields.conv_kernel, fields.conv_stride) else {
            return Err(Error::unsupported(
                "a CTC recognizer from an exported program or an ONNX file without `conv_kernel` and `conv_stride` in its config, as models that read log-mel features have",
            ));
        };
        if kernels.len() != strides.len() {
            return Err(Error::invalid(
                "a config whose `conv_kernel` and `conv_stride` differ in length",
            ));
        }

        let extractor: SampleExtractor = source.json("preprocessor_config.json")?;
        // Clips pad to a step along the samples' named axis, and their frames are cut back here, as `transformers` pads clips for models without an attention mask.
        let samples = model
            .input("input_values")
            .and_then(|spec| spec.axes.get(1).cloned().flatten());
        if let Some(samples) = samples {
            model = model.narrowing(&samples);
        }
        Ok(Self {
            model: Forward::External(model),
            sample_rate: extractor.sampling_rate,
            input: CtcInput::Samples {
                normalize: extractor.do_normalize,
            },
            convolutions: kernels
                .into_iter()
                .zip(strides)
                .map(|(kernel, stride)| Convolution { kernel, stride })
                .collect(),
        })
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
        let features = self.classifier.upload(
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

impl ExternalHead for AudioClassifier {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        let (model, config) = External::classifier(source, function, 1)?;
        Ok(Self {
            classifier: Classifier::external(model, config),
            features: FilterBankOptions::load(source)?,
        })
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
