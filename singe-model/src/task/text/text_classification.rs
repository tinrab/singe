// `text-classification`: a text, or a pair of texts, to its labels' scores, from a sequence classification head.

use std::result::Result as StdResult;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    error::{Error, Result},
    head::{ScoreFunction, SequenceClassifier},
    model,
    processing::text::{EncodeOptions, TextInput, TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct TextClassificationPipeline {
    model: SequenceClassifier,
    tokenizer: TextTokenizer,
}

// A text, or a pair of texts, as `transformers` takes `{"text", "text_pair"}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum TextClassificationInput {
    Text(String),
    Pair { text: String, text_pair: String },
}

impl TextClassificationInput {
    fn text(&self) -> TextInput<'_> {
        match self {
            Self::Text(text) => TextInput::Single(text),
            Self::Pair { text, text_pair } => TextInput::Pair(text, text_pair),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct TextClassificationParameters {
    pub top_k: TopK,
    // How scores become what the output reports, or the model's own choice when absent.
    pub function_to_apply: Option<ScoreFunction>,
    // Truncates each input to this many tokens, keeping its special tokens.
    pub max_length: Option<usize>,
}

// Which labels a text's output lists, as `transformers`' `top_k` asks: the best alone when absent, that many from the best for a number, and every one from the best for `null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TopK {
    #[default]
    Best,
    Labels(usize),
    All,
}

impl<'de> Deserialize<'de> for TopK {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        Option::<usize>::deserialize(deserializer)
            .map(|top_k| top_k.map_or(Self::All, Self::Labels))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

// The best label of a text, or its labels from the best when `top_k` is given.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum TextClassificationOutput {
    Best(LabelScore),
    Ranked(Vec<LabelScore>),
}

impl Pipeline for TextClassificationPipeline {
    const TASK: Task = Task::TextClassification;

    type Input = TextClassificationInput;

    type Parameters = TextClassificationParameters;

    type Output = TextClassificationOutput;

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
        let texts = inputs
            .iter()
            .map(TextClassificationInput::text)
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            max_length: parameters.max_length,
            ..EncodeOptions::default()
        };
        let encoded =
            self.tokenizer
                .encode_inputs(&texts, &options, self.model.model().state().device()?)?;
        let config = self.model.config();
        let function = parameters.function_to_apply.unwrap_or(config.function);
        self.model
            .scores(&encoded.batch)?
            .iter()
            .map(|row| {
                let mut ranked = function
                    .apply(row)
                    .into_iter()
                    .zip(&config.labels)
                    .map(|(score, label)| LabelScore {
                        label: label.clone(),
                        score,
                    })
                    .collect::<Vec<_>>();
                ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
                Ok(match parameters.top_k {
                    TopK::Best => TextClassificationOutput::Best(
                        ranked
                            .into_iter()
                            .next()
                            .ok_or_else(|| Error::unsupported("a classifier of no labels"))?,
                    ),
                    TopK::Labels(count) => {
                        ranked.truncate(count);
                        TextClassificationOutput::Ranked(ranked)
                    }
                    TopK::All => TextClassificationOutput::Ranked(ranked),
                })
            })
            .collect()
    }
}
