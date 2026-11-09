use std::path::Path;

use serde::Deserialize;
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        Activation, DenseAttentionInput, Embedding, EncoderBlock, Layer, Model, ModelState,
        SelfAttention, StateBuilder, TokenBatch, TokenEmbeddings,
    },
};

use crate::{
    assembly::load_weights,
    config::{Snapshot, unsupported},
    error::Result,
    schema::Schema,
};

fn absolute() -> String {
    "absolute".to_owned()
}

// The fields of a Hugging Face BERT `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BertConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub hidden_act: Activation,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    pub layer_norm_eps: f32,
    #[serde(default = "absolute")]
    pub position_embedding_type: String,
    #[serde(default)]
    pub is_decoder: bool,
}

impl BertConfig {
    // A checkpoint using one of these would load and then compute something else.
    fn validate(&self) -> Result<()> {
        if self.position_embedding_type != "absolute" {
            return Err(unsupported("relative position embeddings"));
        }

        if self.is_decoder {
            return Err(unsupported("a BERT decoder"));
        }
        Ok(())
    }
}

// BERT's encoder: word, position, and token type embeddings, then blocks that norm after each residual.
// It returns the last hidden states, `[B, T, W]`, of which padded rows hold what attending to the real tokens gives.
#[derive(Debug, Clone)]
pub struct BertLayers {
    embeddings: TokenEmbeddings,
    blocks: Vec<EncoderBlock>,
}

pub type Bert = Model<BertLayers>;

impl Layer<TokenBatch> for BertLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = self.embeddings.forward(input, state)?;
        let key_lengths = input.key_lengths()?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: Some(&key_lengths),
                },
                state,
            )?;
        }
        Ok(hidden)
    }
}

fn block(schema: &mut Schema<'_>, config: &BertConfig) -> SingeResult<EncoderBlock> {
    let width = config.hidden_size;
    let eps = config.layer_norm_eps;
    let inner = config.intermediate_size;
    let (attention, attention_norm) = {
        let mut schema = schema.scope("attention");
        let [query, key, value] = {
            let mut schema = schema.scope("self");
            [
                schema.linear("query", width, width, true)?,
                schema.linear("key", width, width, true)?,
                schema.linear("value", width, width, true)?,
            ]
        };
        let mut output = schema.scope("output");
        let attention = SelfAttention::create(
            query,
            key,
            value,
            output.linear("dense", width, width, true)?,
            config.num_attention_heads,
            width,
        )?;
        (attention, output.layer_norm("LayerNorm", width, eps)?)
    };
    let expand = schema
        .scope("intermediate")
        .linear("dense", width, inner, true)?;
    let (contract, feed_forward_norm) = {
        let mut schema = schema.scope("output");
        (
            schema.linear("dense", inner, width, true)?,
            schema.layer_norm("LayerNorm", width, eps)?,
        )
    };
    Ok(EncoderBlock::new(
        attention,
        attention_norm,
        expand,
        contract,
        feed_forward_norm,
    )
    .with_activation(config.hidden_act))
}

fn schema(config: &BertConfig, data_type: DataType) -> Result<Bert> {
    let width = config.hidden_size;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut model = Schema::root(&mut builder, data_type);
        let embeddings = {
            let mut schema = model.scope("embeddings");
            let words = schema.parameter("word_embeddings.weight", [config.vocab_size, width])?;
            let positions = schema.parameter(
                "position_embeddings.weight",
                [config.max_position_embeddings, width],
            )?;
            let token_types = schema.parameter(
                "token_type_embeddings.weight",
                [config.type_vocab_size, width],
            )?;
            let norm = schema.layer_norm("LayerNorm", width, config.layer_norm_eps)?;
            TokenEmbeddings::new(Embedding::new(words))
                .with_positions(positions)
                .with_token_types(Embedding::new(token_types))
                .with_norm(norm)
        };
        let blocks = (0..config.num_hidden_layers)
            .map(|layer| block(&mut model.scope(format!("encoder.layer.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        BertLayers { embeddings, blocks }
    };
    Ok(Model::new(layers, builder.finish()?))
}

// Checkpoints of a model with a task head store the encoder under `bert.`, and older ones name LayerNorm parameters `gamma` and `beta`.
// The pooler and task heads serve other tasks, and `position_ids` is what this definition computes itself.
fn state_name(name: &str) -> Option<String> {
    let name = name.strip_prefix("bert.").unwrap_or(name);
    if name.starts_with("pooler.")
        || name.starts_with("cls.")
        || name == "embeddings.position_ids"
        || name == "embeddings.token_type_ids"
    {
        return None;
    }

    if let Some(norm) = name.strip_suffix(".gamma") {
        return Some(format!("{norm}.weight"));
    }

    if let Some(norm) = name.strip_suffix(".beta") {
        return Some(format!("{norm}.bias"));
    }
    Some(name.to_owned())
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Bert> {
    let config: BertConfig = Snapshot::new(&snapshot).config()?;
    config.validate()?;
    load_weights(
        &schema(&config, data_type)?,
        snapshot,
        device,
        Some(&state_name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_model_names_map_to_the_encoder_and_heads_are_ignored() {
        assert_eq!(
            state_name("bert.encoder.layer.0.output.dense.weight").unwrap(),
            "encoder.layer.0.output.dense.weight"
        );
        assert_eq!(
            state_name("embeddings.LayerNorm.gamma").unwrap(),
            "embeddings.LayerNorm.weight"
        );
        assert_eq!(
            state_name("bert.embeddings.LayerNorm.beta").unwrap(),
            "embeddings.LayerNorm.bias"
        );
        assert_eq!(state_name("pooler.dense.weight"), None);
        assert_eq!(state_name("cls.predictions.bias"), None);
        assert_eq!(state_name("embeddings.position_ids"), None);
    }
}
