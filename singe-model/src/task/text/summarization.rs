// `summarization`: text to a shorter text that keeps its content, from an encoder-decoder such as T5, after the prefix its config names for the task, as `transformers`' summarization pipeline gave it before `transformers` 5.

use serde::Serialize;

use crate::{
    error::Result,
    head::Seq2SeqLm,
    model,
    processor::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        seq2seq::{Seq2SeqParameters, TaskPrefixes},
    },
};

#[derive(Debug)]
pub struct SummarizationPipeline {
    model: Seq2SeqLm,
    tokenizer: TextTokenizer,
    prefix: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Summary {
    pub summary_text: String,
}

impl Pipeline for SummarizationPipeline {
    type Input = String;

    type Parameters = Seq2SeqParameters;

    type Output = Summary;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source)?,
            prefix: TaskPrefixes::load(source)?
                .prefix("summarization")
                .to_owned(),
        })
    }

    fn run_one(&self, text: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        Ok(Summary {
            summary_text: self.model.generate_from_text(
                &self.tokenizer,
                &format!("{}{text}", self.prefix),
                parameters.max_new_tokens,
                parameters.compile,
            )?,
        })
    }
}
