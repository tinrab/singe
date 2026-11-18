// `zero-shot-audio-classification`: a clip and candidate labels to each label's score, best first: the softmax of the clip's scaled cosines with each label's hypothesis, as `transformers`' pipeline scores them with CLAP.

use serde::Deserialize;

use crate::{
    error::Result,
    head::{ScoreFunction, ZeroShotAudioClassifier},
    model,
    processing::{audio::Audio, text::TextTokenizer},
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task, hypotheses,
        vision::image_classification::{LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct ZeroShotAudioClassificationPipeline {
    model: ZeroShotAudioClassifier,
    tokenizer: TextTokenizer,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ZeroShotAudioInput {
    pub audio: Audio,
    pub candidate_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ZeroShotAudioClassificationParameters {
    // Where a label goes in its hypothesis.
    pub hypothesis_template: String,
}

impl Default for ZeroShotAudioClassificationParameters {
    fn default() -> Self {
        Self {
            hypothesis_template: "This is a sound of {}.".to_owned(),
        }
    }
}

impl Pipeline for ZeroShotAudioClassificationPipeline {
    const TASK: Task = Task::ZeroShotAudioClassification;

    type Input = ZeroShotAudioInput;

    type Parameters = ZeroShotAudioClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let device = self.model.encoder.inputs().state().device()?;
        inputs
            .iter()
            .map(|input| {
                let hypotheses =
                    hypotheses(&parameters.hypothesis_template, &input.candidate_labels)?;
                input
                    .audio
                    .expect_rate(self.model.spectrogram.sample_rate)?;
                let texts = hypotheses.iter().map(String::as_str).collect::<Vec<_>>();
                let tokens = self.tokenizer.encode_batch(&texts, None, device)?;
                let features = self.model.features(&input.audio.samples)?;
                let logits = self.model.encoder.logits(&features, &tokens)?;
                Ok(ranked(
                    &logits[0],
                    &input.candidate_labels,
                    ScoreFunction::Softmax,
                    usize::MAX,
                ))
            })
            .collect()
    }
}
