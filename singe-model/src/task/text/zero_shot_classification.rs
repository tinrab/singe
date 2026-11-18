// `zero-shot-classification`: a text and candidate labels to each label's score, from a natural language inference model, as `transformers` scores them.
// Each label becomes a hypothesis the text is a premise of, and a label scores as the model finds the text entails its hypothesis.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    head::{ScoreFunction, SequenceClassifier},
    model,
    processing::text::{EncodeOptions, TextInput, TextTokenizer, Truncation},
    source::{LoadOptions, Source},
    task::{Pipeline, Task, hypotheses},
};

#[derive(Debug)]
pub struct ZeroShotClassificationPipeline {
    model: SequenceClassifier,
    tokenizer: TextTokenizer,
    // The labels of the model's entailment and contradiction scores.
    entailment: usize,
    contradiction: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ZeroShotClassificationInput {
    pub sequences: String,
    pub candidate_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ZeroShotClassificationParameters {
    // Where a label goes in its hypothesis.
    pub hypothesis_template: String,
    // Scores each label alone, entailment against contradiction, instead of the labels against each other.
    pub multi_label: bool,
    // Truncates each pair to this many tokens, cutting the text, not its hypothesis.
    pub max_length: Option<usize>,
}

impl Default for ZeroShotClassificationParameters {
    fn default() -> Self {
        Self {
            hypothesis_template: "This example is {}.".to_owned(),
            multi_label: false,
            max_length: None,
        }
    }
}

// The labels from the best, and their scores.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ZeroShotClassificationOutput {
    pub sequence: String,
    pub labels: Vec<String>,
    pub scores: Vec<f32>,
}

impl Pipeline for ZeroShotClassificationPipeline {
    const TASK: Task = Task::ZeroShotClassification;

    type Input = ZeroShotClassificationInput;

    type Parameters = ZeroShotClassificationParameters;

    type Output = ZeroShotClassificationOutput;

    // The entailment label is the first whose name starts with "entail", and the contradiction label the last, or the first when entailment is the last, as `transformers` finds them.
    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model: SequenceClassifier = model::load(source, options)?;
        let labels = &model.config().labels;
        let entailment = labels
            .iter()
            .position(|label| label.to_lowercase().starts_with("entail"))
            .ok_or_else(|| Error::unsupported("an inference model without an entailment label"))?;
        let contradiction = if entailment == 0 { labels.len() - 1 } else { 0 };
        Ok(Self {
            model,
            tokenizer: TextTokenizer::load(source.directory())?,
            entailment,
            contradiction,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let device = self.model.model().state().device()?;
        inputs
            .iter()
            .map(|input| {
                let hypotheses =
                    hypotheses(&parameters.hypothesis_template, &input.candidate_labels)?;
                let pairs = hypotheses
                    .iter()
                    .map(|hypothesis| TextInput::Pair(&input.sequences, hypothesis))
                    .collect::<Vec<_>>();
                let options = EncodeOptions {
                    max_length: parameters.max_length,
                    truncation: Truncation::OnlyFirst,
                    stride: None,
                };
                let encoded = self.tokenizer.encode_inputs(&pairs, &options, device)?;
                let scores = self.model.scores(&encoded.batch)?;
                let scores = if parameters.multi_label || scores.len() == 1 {
                    scores
                        .iter()
                        .map(|row| {
                            ScoreFunction::Softmax
                                .apply(&[row[self.contradiction], row[self.entailment]])[1]
                        })
                        .collect::<Vec<_>>()
                } else {
                    let entailments = scores
                        .iter()
                        .map(|row| row[self.entailment])
                        .collect::<Vec<_>>();
                    ScoreFunction::Softmax.apply(&entailments)
                };
                let mut order = (0..scores.len()).collect::<Vec<_>>();
                order.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
                Ok(ZeroShotClassificationOutput {
                    sequence: input.sequences.clone(),
                    labels: order
                        .iter()
                        .map(|&index| input.candidate_labels[index].clone())
                        .collect(),
                    scores: order.iter().map(|&index| scores[index]).collect(),
                })
            })
            .collect()
    }
}
