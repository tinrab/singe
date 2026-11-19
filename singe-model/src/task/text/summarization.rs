// `summarization`: text to a shorter text that keeps its content, from an encoder-decoder such as T5, after the prefix its config names for the task, as `transformers`' summarization pipeline gave it before `transformers` 5.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::Result,
    head::{Seq2SeqInput, Seq2SeqLm},
    model,
    processor::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct SummarizationPipeline {
    model: Seq2SeqLm,
    tokenizer: TextTokenizer,
    prefix: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct Seq2SeqParameters {
    // The most tokens to generate, or as many as the snapshot's generation config allows.
    pub max_new_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Summary {
    pub summary_text: String,
}

// `config.json`'s `task_specific_params`, which give each task's settings, such as T5's prefix `summarize: `.
pub(crate) fn task_parameters(source: &Source) -> Result<Value> {
    let config: Value = source.json("config.json")?;
    Ok(config["task_specific_params"].clone())
}

// The prefix `parameters` give task `task`, or none.
pub(crate) fn task_prefix(parameters: &Value, task: &str) -> String {
    parameters[task]["prefix"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

// Generates from `input` with `model`, after the decoder's start token, and returns the generated text without special tokens.
pub(crate) fn generate_from(
    model: &Seq2SeqLm,
    tokenizer: &TextTokenizer,
    input: Seq2SeqInput<'_>,
    max_new_tokens: Option<usize>,
) -> Result<String> {
    let encoded = model.encode(input)?;
    let generated = model.generate(&encoded, &[model.decoder_start()?], max_new_tokens)?;
    tokenizer.decode(&generated, true)
}

// Generates from `text`, encoded with its special tokens, as `generate_from` does.
pub(crate) fn generate_text(
    model: &Seq2SeqLm,
    tokenizer: &TextTokenizer,
    text: &str,
    max_new_tokens: Option<usize>,
) -> Result<String> {
    let device = model.model().state().device()?;
    let batch = tokenizer.encode_batch(&[text], None, device)?;
    generate_from(
        model,
        tokenizer,
        Seq2SeqInput::Tokens(&batch),
        max_new_tokens,
    )
}

impl Pipeline for SummarizationPipeline {
    const TASK: Task = Task::Summarization;

    type Input = String;

    type Parameters = Seq2SeqParameters;

    type Output = Summary;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
            prefix: task_prefix(&task_parameters(source)?, "summarization"),
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|text| {
                Ok(Summary {
                    summary_text: generate_text(
                        &self.model,
                        &self.tokenizer,
                        &format!("{}{text}", self.prefix),
                        parameters.max_new_tokens,
                    )?,
                })
            })
            .collect()
    }
}
