// Reads a `transformers` BART config and imports its checkpoint, whose output reads the shared embedding.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{BartStackConfig, PYTORCH_NORM_EPS, bart_stack_rules, embedding_scale},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, default_true},
    },
    model::text::bart::BartConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vocab_size: i64,
    d_model: i64,
    encoder_layers: i64,
    decoder_layers: i64,
    encoder_attention_heads: i64,
    decoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    decoder_ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    max_position_embeddings: i64,
    #[serde(default)]
    scale_embedding: bool,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

pub(crate) fn config(source: &Source) -> Result<BartConfig> {
    let fields: Fields = source.json("config.json")?;
    // The output reads the shared embedding, and the rules skip a stored `lm_head`.
    if !fields.tie_word_embeddings {
        return Err(Error::unsupported(
            "a BART output untied from its embedding",
        ));
    }

    let stack = |layers, heads, inner| BartStackConfig {
        vocab_size: fields.vocab_size,
        width: fields.d_model,
        layers,
        heads,
        inner,
        activation: fields.activation_function,
        max_positions: fields.max_position_embeddings,
        embedding_scale: embedding_scale(fields.scale_embedding, fields.d_model),
        embedding_norm: true,
        pre_norm: false,
        cross_width: fields.d_model,
        eps: PYTORCH_NORM_EPS,
    };
    Ok(BartConfig {
        encoder: stack(
            fields.encoder_layers,
            fields.encoder_attention_heads,
            fields.encoder_ffn_dim,
        ),
        decoder: stack(
            fields.decoder_layers,
            fields.decoder_attention_heads,
            fields.decoder_ffn_dim,
        ),
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("model.shared", "tokens")
        .rename("final_logits_bias", "logits_bias")
        // Copies of the shared embedding, which the output reads too.
        .skip("model.encoder.embed_tokens")
        .skip("model.decoder.embed_tokens")
        .skip("lm_head")
        .then(bart_stack_rules("model.encoder", "encoder"))
        .then(bart_stack_rules("model.decoder", "decoder"))
}
