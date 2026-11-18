// `translation`: text in one language to text in another, from an encoder-decoder such as T5, after the prefix its config names for the pair of languages, as `transformers`' `translation_{src}_to_{tgt}` pipelines gave it before `transformers` 5.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::{Error, Result},
    head::Seq2SeqLm,
    model,
    processing::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        text::summarization::{generate_text, task_parameters, task_prefix},
    },
};

#[derive(Debug)]
pub struct TranslationPipeline {
    model: Seq2SeqLm,
    tokenizer: TextTokenizer,
    // `config.json`'s `task_specific_params`, which give each pair of languages its prefix.
    task_parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct TranslationParameters {
    // The languages translated from and to, as the config's `translation_{src_lang}_to_{tgt_lang}` names them.
    pub src_lang: String,
    pub tgt_lang: String,
    pub max_new_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Translation {
    pub translation_text: String,
}

impl Pipeline for TranslationPipeline {
    const TASK: Task = Task::Translation;

    type Input = String;

    type Parameters = TranslationParameters;

    type Output = Translation;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
            task_parameters: task_parameters(source)?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        if parameters.src_lang.is_empty() || parameters.tgt_lang.is_empty() {
            return Err(Error::unsupported(
                "a translation without src_lang and tgt_lang",
            ));
        }

        let task = format!(
            "translation_{}_to_{}",
            parameters.src_lang, parameters.tgt_lang
        );
        let prefix = task_prefix(&self.task_parameters, &task);
        inputs
            .iter()
            .map(|text| {
                Ok(Translation {
                    translation_text: generate_text(
                        &self.model,
                        &self.tokenizer,
                        &format!("{prefix}{text}"),
                        parameters.max_new_tokens,
                    )?,
                })
            })
            .collect()
    }
}
