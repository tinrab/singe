//! `text-to-audio`: text to speech, from VITS, as `transformers`' text-to-speech pipeline gives it, text to music, from `MusicGen`, as `MusicgenForConditionalGeneration.generate` gives it, or speech generated as codec frames, from CSM, of text or of a conversation whose earlier turns' audio gives the voices.
//! Generators sample or not as their generation configs say unless the parameters ask otherwise.
//! VITS's noise is Singe's own, seeded, so it matches `transformers`' only with the noise scales at zero.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    head::{AudioGenerator, Head, SamplingRequest, SpeechPrompt, TextToMusic, TextToSpeech},
    model,
    processor::text::{SpeechTokenizer, TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, outputs::GeneratedAudio},
};

#[derive(Debug)]
enum AudioModel {
    Speech {
        model: TextToSpeech,
        tokenizer: SpeechTokenizer,
    },
    Music {
        model: Box<TextToMusic>,
        tokenizer: Box<TextTokenizer>,
    },
    Frames(Box<dyn AudioGenerator>),
}

#[derive(Debug)]
pub struct TextToAudioPipeline {
    model: AudioModel,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct TextToAudioParameters {
    /// How much noise varies the voice, or the model's own amount when `None`.
    pub noise_scale: Option<f32>,
    /// How much noise varies the durations, or the model's own amount when `None`.
    pub noise_scale_duration: Option<f32>,
    pub seed: u64,
    /// A music generator's decoder steps or a speech generator's frames, or the generation config's when `None`.
    pub max_new_tokens: Option<usize>,
    /// Whether a generator samples its codes, or the generation config's choice when `None`.
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
}

impl TextToAudioParameters {
    const fn sampling(&self) -> SamplingRequest {
        SamplingRequest {
            do_sample: self.do_sample,
            temperature: self.temperature,
            top_k: self.top_k,
            seed: self.seed,
        }
    }
}

impl Pipeline for TextToAudioPipeline {
    type Input = SpeechPrompt;

    type Parameters = TextToAudioParameters;

    type Output = GeneratedAudio;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let head = model::default_head(source)?;
        let model = if head == Head::TextToMusic {
            AudioModel::Music {
                model: Box::new(model::load(source, options)?),
                tokenizer: Box::new(TextTokenizer::load(source)?),
            }
        } else if head == Head::AudioGeneration {
            AudioModel::Frames(model::load(source, options)?)
        } else {
            AudioModel::Speech {
                model: model::load(source, options)?,
                tokenizer: SpeechTokenizer::load(source)?,
            }
        };
        Ok(Self { model, options })
    }

    fn run_one(&self, prompt: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let text = || match prompt {
            SpeechPrompt::Text(text) => Ok(text),
            SpeechPrompt::Conversation(_) => Err(Error::unsupported(
                "a conversation for a model that speaks text alone",
            )),
        };
        match &self.model {
            AudioModel::Speech { model, tokenizer } => Ok(GeneratedAudio {
                audio: model.speak(
                    &tokenizer.encode(text()?),
                    (parameters.noise_scale, parameters.noise_scale_duration),
                    parameters.seed,
                )?,
                sampling_rate: model.sample_rate(),
            }),
            AudioModel::Music { model, tokenizer } => {
                let batch =
                    tokenizer.encode_batch(&[text()?.as_str()], None, self.options.device)?;
                Ok(GeneratedAudio {
                    audio: model.generate(
                        &batch,
                        parameters.max_new_tokens,
                        &parameters.sampling(),
                    )?,
                    sampling_rate: model.sample_rate(),
                })
            }
            AudioModel::Frames(model) => Ok(GeneratedAudio {
                audio: model.generate_audio(
                    prompt,
                    parameters.max_new_tokens,
                    &parameters.sampling(),
                )?,
                sampling_rate: model.sample_rate(),
            }),
        }
    }
}
