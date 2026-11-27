//! `translation`: text in one language to text in another, from an encoder-decoder such as T5, after the prefix its config names for the pair of languages, as `transformers`' `translation_{src}_to_{tgt}` pipelines gave it before `transformers` 5.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    head::Seq2SeqLm,
    model,
    processor::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        seq2seq::{Seq2SeqParameters, TaskPrefixTable},
    },
};

#[derive(Debug)]
pub struct TranslationPipeline {
    model: Seq2SeqLm,
    tokenizer: TextTokenizer,
    prefixes: TaskPrefixTable,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct TranslationParameters {
    /// The language translated from, as the config's `translation_{src_lang}_to_{tgt_lang}` names it.
    pub src_lang: String,
    /// The language translated to, as the config's `translation_{src_lang}_to_{tgt_lang}` names it.
    pub tgt_lang: String,
    #[serde(flatten)]
    pub generation: Seq2SeqParameters,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Translation {
    pub translation_text: String,
}

impl Pipeline for TranslationPipeline {
    type Input = String;

    type Parameters = TranslationParameters;

    type Output = Translation;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source)?,
            prefixes: TaskPrefixTable::load(source)?,
        })
    }

    fn run_one(&self, text: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        if parameters.src_lang.is_empty() || parameters.tgt_lang.is_empty() {
            return Err(Error::invalid(
                "a translation without src_lang and tgt_lang",
            ));
        }

        let prefix = self.prefixes.prefix(&format!(
            "translation_{}_to_{}",
            parameters.src_lang, parameters.tgt_lang
        ));
        Ok(Translation {
            translation_text: self.model.generate_from_text(
                &self.tokenizer,
                &format!("{prefix}{text}"),
                parameters.generation.max_new_tokens,
                parameters.generation.compile,
            )?,
        })
    }
}
