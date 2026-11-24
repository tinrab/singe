use std::path::Path;

use serde::Deserialize;
use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState, Pooling, PoolingInput, PoolingMode, TokenBatch},
};

use crate::{
    error::{Error, Result},
    head::{Boxed, TextEncoder},
    model,
    source::{LoadOptions, Source},
};

/// A text encoder and the pooling that turns its hidden states into each sentence's vector, `[B, W]`.
#[derive(Debug)]
pub struct SentenceLayers {
    encoder: Boxed<dyn Layer<TokenBatch>>,
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

#[derive(Debug)]
pub struct SentenceEmbedder {
    model: Model<SentenceLayers>,
}

impl SentenceEmbedder {
    pub fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let pooling = pooling(source)?;
        let encoder: TextEncoder = model::load(source, options)?;
        let layers = SentenceLayers {
            encoder: Boxed::<dyn Layer<TokenBatch>>(Box::new(EncoderLayer(encoder.clone()))),
            pooling,
        };
        Ok(Self {
            model: Model::new(layers, encoder.state().clone()),
        })
    }

    #[must_use]
    pub const fn model(&self) -> &Model<SentenceLayers> {
        &self.model
    }

    pub fn embed(&self, batch: &TokenBatch) -> Result<Tensor> {
        Ok(self.model.forward_group(batch)?.cast(DataType::F32)?)
    }
}

/// A loaded encoder used as a layer, whose state is the embedder's own.
#[derive(Debug)]
struct EncoderLayer(TextEncoder);

impl Layer<TokenBatch> for EncoderLayer {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.0.layer().forward(input, state)
    }
}

/// One entry of a `sentence-transformers` snapshot's `modules.json`.
#[derive(Debug, Deserialize)]
struct Module {
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

/// The mode flags of a `sentence-transformers` `Pooling` module's `config.json`, of which [`pooling`] accepts exactly one set.
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

/// Reads the pooling `modules.json` describes: a transformer first, then one pooling mode, then an optional normalization.
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
                return Err(Error::unsupported(format!(
                    "sentence-transformers module {other} outside transformer, pooling, normalize"
                )));
            }
        }
    }

    let mode =
        mode.ok_or_else(|| Error::unsupported("sentence-transformers modules without pooling"))?;
    Ok(Pooling::new(mode, normalize))
}
