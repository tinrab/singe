//! `sentence-similarity`: a source sentence and others to how alike each is to it, the cosine of their embeddings.
//! The model is a `sentence-transformers` one: a text encoder, and a pooling its `modules.json` describes.

use std::iter;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use singe::{Error as SingeError, Tensor, nn::Model};

use crate::{
    error::Result,
    head::{SentenceEmbedder, SentenceLayers},
    processor::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct SentenceSimilarityPipeline {
    model: SentenceEmbedder,
    tokenizer: TextTokenizer,
    /// The most tokens of a sentence unless a caller names another: `sentence_bert_config.json`'s `max_seq_length`, or else the tokenizer's longest input capped at the model's positions, as `sentence-transformers` truncates.
    max_length: Option<usize>,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SentenceSimilarityInput {
    pub source_sentence: String,
    pub sentences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct SentenceSimilarityParameters {
    /// The tokens each sentence is truncated to, or the snapshot's limit when `None`.
    pub max_length: Option<usize>,
}

/// Each sentence's similarity to the source sentence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SentenceSimilarityOutput {
    pub scores: Vec<f32>,
}

impl SentenceSimilarityPipeline {
    pub fn get_model(&self) -> Result<&Model<SentenceLayers>> {
        self.model.get_model()
    }

    /// Returns each text's vector, `[B, W]` of `F32`, truncated to `max_length` tokens, or to the snapshot's limit for `None`.
    pub fn embed(&self, texts: &[&str], max_length: Option<usize>) -> Result<Tensor> {
        let batch = self.tokenizer.encode_batch(
            texts,
            max_length.or(self.max_length),
            self.options.device,
        )?;
        self.model.embed(&batch)
    }
}

impl Pipeline for SentenceSimilarityPipeline {
    type Input = SentenceSimilarityInput;

    type Parameters = SentenceSimilarityParameters;

    type Output = SentenceSimilarityOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model = SentenceEmbedder::load(source, options)?;
        let tokenizer = TextTokenizer::load(source)?;
        let length = |file: &str, key: &str| -> Result<Option<usize>> {
            Ok(source
                .optional_json::<Value>(file)?
                .and_then(|config| config[key].as_u64())
                .and_then(|length| usize::try_from(length).ok()))
        };
        // The tokenizer's limit is capped at the model's positions, as `sentence-transformers` caps it.
        let positions = length("config.json", "max_position_embeddings")?;
        let capped = match (tokenizer.max_length(), positions) {
            (Some(tokenizer), Some(positions)) => Some(tokenizer.min(positions)),
            (tokenizer, positions) => tokenizer.or(positions),
        };
        let max_length = length("sentence_bert_config.json", "max_seq_length")?.or(capped);
        Ok(Self {
            model,
            tokenizer,
            max_length,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let texts = iter::once(input.source_sentence.as_str())
            .chain(input.sentences.iter().map(String::as_str))
            .collect::<Vec<_>>();
        let vectors = self.embed(&texts, parameters.max_length)?;
        let &[_, width] = vectors.shape().dimensions() else {
            return Err(SingeError::invalid_argument("sentence embeddings, [B, W]").into());
        };
        let width =
            usize::try_from(width).map_err(|_| SingeError::out_of_range("embedding width"))?;
        let values = vectors.to_vec::<f32>()?;
        let (source, others) = values.split_at(width);
        Ok(SentenceSimilarityOutput {
            scores: others
                .chunks(width)
                .map(|other| cosine(source, other))
                .collect(),
        })
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    dot(left, right) / (dot(left, left).sqrt() * dot(right, right).sqrt()).max(f32::EPSILON)
}
