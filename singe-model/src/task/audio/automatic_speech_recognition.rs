// `automatic-speech-recognition`: a clip of speech to the text it says.

use serde::Deserialize;
use singe::{Device, Shape, Tensor, generation::GenerationOptions};

use crate::{
    BatchOptions,
    error::Result,
    head::{CtcRecognizer, Head, SpeechToText, Transcription},
    model,
    processor::{
        audio::Audio,
        text::{CtcVocabulary, TextTokenizer},
    },
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

// A model that writes text, or one whose frames' characters CTC decoding reads.
#[derive(Debug)]
enum Recognizer {
    Generative {
        model: Box<dyn SpeechToText>,
        tokenizer: Box<TextTokenizer>,
    },
    Ctc {
        model: CtcRecognizer,
        vocabulary: CtcVocabulary,
    },
}

#[derive(Debug)]
pub struct AutomaticSpeechRecognitionPipeline {
    model: Recognizer,
    device: Device,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct AutomaticSpeechRecognitionParameters {
    pub max_new_tokens: usize,
    // The clip's language, which a model that names it itself otherwise detects.
    pub language: Option<String>,
    pub compile: bool,
}

impl Default for AutomaticSpeechRecognitionParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            language: None,
            compile: true,
        }
    }
}

impl Pipeline for AutomaticSpeechRecognitionPipeline {
    const TASK: Task = Task::AutomaticSpeechRecognition;

    type Input = Audio;

    type Parameters = AutomaticSpeechRecognitionParameters;

    type Output = Transcription;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model = if model::head(source)? == Head::Ctc {
            Recognizer::Ctc {
                model: model::load(source, options)?,
                vocabulary: CtcVocabulary::load(source.directory())?,
            }
        } else {
            Recognizer::Generative {
                model: model::load(source, options)?,
                tokenizer: Box::new(TextTokenizer::load(source.directory())?),
            }
        };
        Ok(Self {
            model,
            device: options.device,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let options = BatchOptions::new(GenerationOptions {
            max_new_tokens: parameters.max_new_tokens,
            ..GenerationOptions::default()
        })
        .with_compile(parameters.compile);
        inputs
            .iter()
            .map(|clip| match &self.model {
                Recognizer::Generative { model, tokenizer } => {
                    clip.expect_rate(model.sample_rate())?;
                    let count = i64::try_from(clip.samples.len())
                        .map_err(|_| crate::Error::unsupported("a clip this long"))?;
                    let samples = Tensor::from_vec(
                        self.device,
                        Shape::create([count])?,
                        clip.samples.clone(),
                    )?;
                    model.transcribe(
                        tokenizer,
                        &samples,
                        parameters.language.as_deref(),
                        &options,
                    )
                }
                Recognizer::Ctc { model, vocabulary } => {
                    clip.expect_rate(model.sample_rate())?;
                    Ok(Transcription {
                        language: None,
                        text: vocabulary.decode(&model.best_tokens(&clip.samples)?)?,
                    })
                }
            })
            .collect()
    }
}
