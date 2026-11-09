// Models that embed a sentence as one vector to compare with others, Hugging Face's `sentence-similarity`, each under its `model_type`.
// A snapshot is a `sentence-transformers` one, whose `modules.json` says how token states pool into the sentence's vector.

pub mod bert;

use std::path::Path;

use serde::Deserialize;
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{Layer, MeanPooling, Model, ModelState, PoolingInput, TokenBatch},
};

use crate::{
    Error, Result,
    config::{Snapshot, unsupported},
};

// The encoder a snapshot's `model_type` names, returning each token's hidden state, `[B, T, W]`.
#[derive(Debug, Clone)]
pub enum SentenceEncoder {
    Bert(bert::BertLayers),
}

impl Layer<TokenBatch> for SentenceEncoder {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Bert(layers) => layers.forward(input, state),
        }
    }
}

// An encoder and the pooling that turns its hidden states into each sentence's vector, `[B, W]`.
#[derive(Debug, Clone)]
pub struct SentenceLayers {
    encoder: SentenceEncoder,
    pooling: MeanPooling,
}

impl SentenceLayers {
    #[must_use]
    pub const fn encoder(&self) -> &SentenceEncoder {
        &self.encoder
    }
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

// One entry of `modules.json`.
#[derive(Debug, Deserialize)]
struct Module {
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

// The modes of a `Pooling` module's `config.json`, of which this reads mean pooling alone.
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

// Reads the pooling `modules.json` describes: a transformer, mean pooling, and an optional normalization.
fn pooling(snapshot: &Snapshot) -> Result<MeanPooling> {
    let modules: Vec<Module> = snapshot.json("modules.json")?;
    let Some((first, rest)) = modules.split_first() else {
        return Err(unsupported(
            "sentence-transformers modules without a transformer",
        ));
    };
    if first.kind != TRANSFORMER {
        return Err(unsupported(
            "sentence-transformers modules without a transformer",
        ));
    }

    let mut pooled = false;
    let mut normalize = false;
    for module in rest {
        match module.kind.as_str() {
            POOLING if !pooled => {
                let config: PoolingConfig =
                    snapshot.json(Path::new(&module.path).join("config.json"))?;
                let others = config.pooling_mode_cls_token
                    || config.pooling_mode_max_tokens
                    || config.pooling_mode_mean_sqrt_len_tokens
                    || config.pooling_mode_weightedmean_tokens
                    || config.pooling_mode_lasttoken;
                if !config.pooling_mode_mean_tokens || others {
                    return Err(unsupported("pooling other than the mean of tokens"));
                }

                pooled = true;
            }
            NORMALIZE if pooled && !normalize => normalize = true,
            other => {
                return Err(Error::UnsupportedConfig(format!(
                    "sentence-transformers module {other} in this place"
                )));
            }
        }
    }

    if !pooled {
        return Err(unsupported("sentence-transformers modules without pooling"));
    }
    Ok(MeanPooling::new(normalize))
}

// Loads the encoder a snapshot's `model_type` names, with the pooling its `modules.json` describes.
pub fn load_sentence_embedder(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<SentenceEmbedder> {
    let snapshot = snapshot.as_ref();
    let files = Snapshot::new(snapshot);
    let pooling = pooling(&files)?;
    let (encoder, state) = match files.model_type()?.as_str() {
        "bert" => {
            let model = bert::load(snapshot, device, data_type)?;
            (
                SentenceEncoder::Bert(model.layer().clone()),
                model.state().clone(),
            )
        }
        other => return Err(Error::UnsupportedConfig(format!("model type {other}"))),
    };
    Ok(Model::new(SentenceLayers { encoder, pooling }, state))
}
