// `sentence-similarity`: a source sentence and others to how alike each is to it, the cosine of their embeddings.
// The model is a `sentence-transformers` one: a text encoder, and a pooling its `modules.json` describes.

use std::{iter, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState, Pooling, PoolingInput, PoolingMode, TokenBatch},
};

use crate::{
    error::{Error, Result},
    head::{TextEncoder, TextEncoderLayers},
    model,
    processing::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

// A text encoder and the pooling that turns its hidden states into each sentence's vector, `[B, W]`.
#[derive(Debug)]
pub struct SentenceLayers {
    encoder: TextEncoderLayers,
    pooling: Pooling,
}

impl Layer<TokenBatch> for SentenceLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self.encoder.forward(input, state)?;
        self.pooling.forward(
            &PoolingInput {
                hidden: &hidden,
                padding: input.padding(),
            },
            state,
        )
    }
}

pub type SentenceEmbedder = Model<SentenceLayers>;

#[derive(Debug)]
pub struct SentenceSimilarityPipeline {
    model: SentenceEmbedder,
    tokenizer: TextTokenizer,
    // The most tokens of a sentence unless a caller names another: `sentence_bert_config.json`'s `max_seq_length`, or else the tokenizer's longest input capped at the model's positions, as `sentence-transformers` truncates.
    max_length: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SentenceSimilarityInput {
    pub source_sentence: String,
    pub sentences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct SentenceSimilarityParameters {
    // Truncates each sentence to this many tokens, or to the snapshot's limit when absent.
    pub max_length: Option<usize>,
}

// Each sentence's similarity to the source sentence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SentenceSimilarityOutput {
    pub scores: Vec<f32>,
}

impl SentenceSimilarityPipeline {
    #[must_use]
    pub const fn model(&self) -> &SentenceEmbedder {
        &self.model
    }

    // Returns each text's vector, `[B, W]` of `F32`.
    pub fn embed(&self, texts: &[&str], max_length: Option<usize>) -> Result<Tensor> {
        let device = self.model.state().device()?;
        let batch = self
            .tokenizer
            .encode_batch(texts, max_length.or(self.max_length), device)?;
        Ok(self.model.forward_group(&batch)?.cast(DataType::F32)?)
    }
}

impl Pipeline for SentenceSimilarityPipeline {
    const TASK: Task = Task::SentenceSimilarity;

    type Input = SentenceSimilarityInput;

    type Parameters = SentenceSimilarityParameters;

    type Output = SentenceSimilarityOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let pooling = pooling(source)?;
        let encoder: TextEncoder = model::load(source, options)?;
        let layers = SentenceLayers {
            encoder: TextEncoderLayers(Box::new(EncoderLayer(encoder.clone()))),
            pooling,
        };
        let tokenizer = TextTokenizer::load(source.directory())?;
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
            model: Model::new(layers, encoder.state().clone()),
            tokenizer,
            max_length,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| {
                let texts = iter::once(input.source_sentence.as_str())
                    .chain(input.sentences.iter().map(String::as_str))
                    .collect::<Vec<_>>();
                let vectors = self.embed(&texts, parameters.max_length)?;
                let width = usize::try_from(vectors.shape().dimensions()[1])
                    .map_err(|_| Error::unsupported("this width"))?;
                let values = vectors.to_vec::<f32>()?;
                let (source, others) = values.split_at(width);
                Ok(SentenceSimilarityOutput {
                    scores: others
                        .chunks(width)
                        .map(|other| cosine(source, other))
                        .collect(),
                })
            })
            .collect()
    }
}

// A loaded encoder used as a layer, whose state is the embedder's own.
#[derive(Debug)]
struct EncoderLayer(TextEncoder);

impl Layer<TokenBatch> for EncoderLayer {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.0.layer().forward(input, state)
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    dot(left, right) / (dot(left, left).sqrt() * dot(right, right).sqrt()).max(f32::EPSILON)
}

// One entry of `modules.json`.
#[derive(Debug, Deserialize)]
struct Module {
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

// The modes of a `Pooling` module's `config.json`, of which one may be set.
#[derive(Debug, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "sentence-transformers stores each pooling mode as its own flag"
)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
}

const TRANSFORMER: &str = "sentence_transformers.models.Transformer";
const POOLING: &str = "sentence_transformers.models.Pooling";
const NORMALIZE: &str = "sentence_transformers.models.Normalize";

// Reads the pooling `modules.json` describes: a transformer, one pooling mode, and an optional normalization.
fn pooling(source: &Source) -> Result<Pooling> {
    let modules: Vec<Module> = source.json("modules.json")?;
    let Some((first, rest)) = modules.split_first() else {
        return Err(Error::unsupported(
            "sentence-transformers modules without a transformer",
        ));
    };
    if first.kind != TRANSFORMER {
        return Err(Error::unsupported(
            "sentence-transformers modules without a transformer",
        ));
    }

    let mut mode = None;
    let mut normalize = false;
    for module in rest {
        match module.kind.as_str() {
            POOLING if mode.is_none() => {
                let config: PoolingConfig =
                    source.json(Path::new(&module.path).join("config.json"))?;
                let modes = [
                    (config.pooling_mode_cls_token, PoolingMode::Cls),
                    (config.pooling_mode_mean_tokens, PoolingMode::Mean),
                    (config.pooling_mode_max_tokens, PoolingMode::Max),
                    (
                        config.pooling_mode_mean_sqrt_len_tokens,
                        PoolingMode::MeanSqrtLength,
                    ),
                    (config.pooling_mode_lasttoken, PoolingMode::LastToken),
                ]
                .into_iter()
                .filter_map(|(set, mode)| set.then_some(mode))
                .collect::<Vec<_>>();
                let ([chosen], false) = (modes.as_slice(), config.pooling_mode_weightedmean_tokens)
                else {
                    return Err(Error::unsupported(
                        "pooling other than one of the plain modes",
                    ));
                };
                mode = Some(*chosen);
            }
            NORMALIZE if mode.is_some() && !normalize => normalize = true,
            other => {
                return Err(Error::UnsupportedConfig(format!(
                    "sentence-transformers module {other} in this place"
                )));
            }
        }
    }

    let mode =
        mode.ok_or_else(|| Error::unsupported("sentence-transformers modules without pooling"))?;
    Ok(Pooling::new(mode, normalize))
}
