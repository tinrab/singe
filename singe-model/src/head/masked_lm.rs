//! Masked language modeling, as `transformers`' `*ForMaskedLM` heads: every token's hidden state through a projection, its activation, and a norm, to the vocabulary's scores, `[B, T, vocab]`.
//! It is written once for every model; its state is `masked_lm.projection`, `masked_lm.norm`, and `masked_lm.output`, whose weight is the token embedding table when the model ties them.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, Layer, LayerNorm, Linear, Model, ModelState, TokenBatch},
};

use crate::{
    assembly::PostNormEncoder,
    error::Result,
    function::{
        TensorFunction,
        forward::{External, ExternalHead, Forward},
    },
    head::{EncoderHead, HeadLayouts},
    schema::Schema,
    source::Source,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskedLmConfig {
    /// Whether the output's weight is the token embedding table.
    pub tied: bool,
}

#[derive(Debug)]
pub struct MaskedLmLayers {
    encoder: Box<dyn Layer<TokenBatch>>,
    projection: Linear,
    activation: Activation,
    norm: LayerNorm,
    output: Linear,
}

impl Layer<TokenBatch> for MaskedLmLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder
            .forward(input, state)?
            .apply(&self.projection, state)?
            .apply(&self.activation, state)?
            .apply(&self.norm, state)?
            .apply(&self.output, state)
    }
}

#[derive(Debug)]
pub struct MaskedLm {
    model: Forward<MaskedLmLayers>,
}

impl MaskedLm {
    pub fn get_model(&self) -> Result<&Model<MaskedLmLayers>> {
        self.model.get_native()
    }

    /// Returns the vocabulary's scores at each of `positions`, a row and a token of `batch`.
    pub fn scores_at(
        &self,
        batch: &TokenBatch,
        positions: &[(usize, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        let scores = self.model.forward_group(batch)?;
        positions
            .iter()
            .map(|&(row, token)| {
                let index = |value: usize| {
                    i64::try_from(value).map_err(|_| SingeError::out_of_range("batch position"))
                };
                Ok(scores
                    .narrow(0, index(row)?, 1)?
                    .narrow(1, index(token)?, 1)?
                    .cast(DataType::F32)?
                    .to_vec::<f32>()?)
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct TieField {
    tie_word_embeddings: Option<bool>,
}

impl EncoderHead for MaskedLm {
    type Config = MaskedLmConfig;

    type Layers = MaskedLmLayers;

    fn transformers_config(source: &Source) -> Result<MaskedLmConfig> {
        let field: TieField = source.json("config.json")?;
        // `transformers` ties the output to the token embeddings unless the config says otherwise.
        Ok(MaskedLmConfig {
            tied: field.tie_word_embeddings.unwrap_or(true),
        })
    }

    fn layers(
        schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        config: &MaskedLmConfig,
        layouts: &HeadLayouts,
    ) -> SingeResult<MaskedLmLayers> {
        let encoder_config = encoder.config();
        let (width, vocab_size) = (encoder_config.width, encoder_config.vocab_size);
        let mut schema = schema.scope("masked_lm");
        let projection = schema.linear("projection", width, width, true)?;
        let norm = schema.layer_norm("norm", width, encoder_config.eps)?;
        let weight = if config.tied {
            schema.tie("output.weight", encoder.tokens())?
        } else {
            schema.parameter("output.weight", [vocab_size, width])?
        };
        let bias = schema.parameter("output.bias", [vocab_size])?;
        Ok(MaskedLmLayers {
            activation: layouts
                .masked_lm_activation
                .unwrap_or(encoder_config.activation),
            encoder: Box::new(encoder),
            projection,
            norm,
            output: Linear::new(weight, Some(bias)),
        })
    }

    fn new(model: Model<MaskedLmLayers>, _config: MaskedLmConfig) -> Self {
        Self {
            model: Forward::Native(model),
        }
    }
}

impl ExternalHead for MaskedLm {
    fn external(_source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        Ok(Self {
            model: Forward::External(External::create(function, &["logits"])?),
        })
    }
}
