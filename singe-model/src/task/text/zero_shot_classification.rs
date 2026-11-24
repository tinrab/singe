//! `zero-shot-classification`: a text and candidate labels to each label's score, from a natural language inference model, as `transformers` scores them.
//! Each label becomes a hypothesis the text is a premise of, and a label scores as the model finds the text entails its hypothesis.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    head::{ScoreFunction, SequenceClassifier},
    host, model,
    processor::text::{EncodeOptions, TextInput, TextTokenizer, Truncation},
    source::{LoadOptions, Source},
    task::{Pipeline, hypotheses},
};

#[derive(Debug)]
pub struct ZeroShotClassificationPipeline {
    model: SequenceClassifier,
    tokenizer: TextTokenizer,
    /// The index of the model's entailment label.
    entailment: usize,
    /// The index of the model's contradiction label.
    contradiction: usize,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ZeroShotClassificationInput {
    pub sequences: String,
    pub candidate_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ZeroShotClassificationParameters {
    /// The hypothesis a label goes in, in place of its `{}`; `This example is {}.` by default.
    pub hypothesis_template: String,
    /// Whether to score each label alone, entailment against contradiction, instead of the labels against each other.
    pub multi_label: bool,
    /// The tokens each pair is truncated to, cutting the text, not its hypothesis, with no truncation for `None`.
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

/// The input text, and the labels from the best with their scores.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ZeroShotClassificationOutput {
    pub sequence: String,
    pub labels: Vec<String>,
    pub scores: Vec<f32>,
}

impl Pipeline for ZeroShotClassificationPipeline {
    type Input = ZeroShotClassificationInput;

    type Parameters = ZeroShotClassificationParameters;

    type Output = ZeroShotClassificationOutput;

    /// Loads the model and tokenizer, taking as the entailment label the first whose name starts with `entail`, and as the contradiction label the first, or the last when entailment is the first, as `transformers` finds them.
    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model: SequenceClassifier = model::load(source, options)?;
        let labels = model.config().labels();
        let entailment = labels
            .iter()
            .position(|label| label.to_lowercase().starts_with("entail"))
            .ok_or_else(|| Error::unsupported("an inference model without an entailment label"))?;
        let contradiction = if entailment == 0 { labels.len() - 1 } else { 0 };
        Ok(Self {
            model,
            tokenizer: TextTokenizer::load(source)?,
            entailment,
            contradiction,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let hypotheses = hypotheses(&parameters.hypothesis_template, &input.candidate_labels)?;
        let pairs = hypotheses
            .iter()
            .map(|hypothesis| TextInput::Pair(&input.sequences, hypothesis))
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            max_length: parameters.max_length,
            truncation: Truncation::OnlyFirst,
            stride: None,
        };
        let encoded = self
            .tokenizer
            .encode_inputs(&pairs, &options, self.options.device)?;
        let scores = self.model.scores(&encoded.batch)?;
        let scores = if parameters.multi_label || scores.len() == 1 {
            scores
                .iter()
                .map(|row| {
                    ScoreFunction::Softmax.apply(&[row[self.contradiction], row[self.entailment]])
                        [1]
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
        host::best_first(&mut order, usize::MAX, |&index| scores[index]);
        Ok(ZeroShotClassificationOutput {
            sequence: input.sequences.clone(),
            labels: order
                .iter()
                .map(|&index| input.candidate_labels[index].clone())
                .collect(),
            scores: order.iter().map(|&index| scores[index]).collect(),
        })
    }
}
