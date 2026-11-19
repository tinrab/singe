// `text-classification`: a text, or a pair of texts, to its labels' scores, from a sequence classification head.

use std::result::Result as StdResult;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    error::{Error, Result},
    head::{ScoreFunction, SequenceClassifier},
    model,
    processor::text::{EncodeOptions, TextInput, TextTokenizer},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        ranking::{LabelScore, ranked},
        run_alone,
    },
};

#[derive(Debug)]
pub struct TextClassificationPipeline {
    model: SequenceClassifier,
    tokenizer: TextTokenizer,
    options: LoadOptions,
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

// The best label of a text, or its labels from the best when `top_k` is given.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum TextClassificationOutput {
    Best(LabelScore),
    Ranked(Vec<LabelScore>),
}

impl Pipeline for TextClassificationPipeline {
    type Input = TextClassificationInput;

    type Parameters = TextClassificationParameters;

    type Output = TextClassificationOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        run_alone(self, input, parameters)
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
        let encoded = self
            .tokenizer
            .encode_inputs(&texts, &options, self.options.device)?;
        let config = self.model.config();
        let function = parameters
            .function_to_apply
            .unwrap_or_else(|| config.function());
        let count = match parameters.top_k {
            TopK::Best => 1,
            TopK::Labels(count) => count,
            TopK::All => usize::MAX,
        };
        self.model
            .scores(&encoded.batch)?
            .iter()
            .map(|row| {
                let ranked = ranked(row, config.labels(), function, count);
                Ok(match parameters.top_k {
                    TopK::Best => TextClassificationOutput::Best(
                        ranked
                            .into_iter()
                            .next()
                            .ok_or_else(|| Error::unsupported("a classifier of no labels"))?,
                    ),
                    TopK::Labels(_) | TopK::All => TextClassificationOutput::Ranked(ranked),
                })
            })
            .collect()
    }
}
