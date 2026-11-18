// Reads a `transformers` Whisper snapshot: its config, and the rules that import its encoder's and decoder's tensors.

use serde::Deserialize;
use singe::nn::Activation;

use super::WhisperConfig;
use crate::{
    assembly::pre_norm_encoder_rules,
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct Fields {
    vocab_size: i64,
    num_mel_bins: i64,
    d_model: i64,
    encoder_layers: i64,
    encoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    decoder_layers: i64,
    decoder_attention_heads: i64,
    decoder_ffn_dim: i64,
    max_source_positions: i64,
    max_target_positions: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    #[serde(default)]
    scale_embedding: bool,
}

pub(crate) fn config(source: &Source) -> Result<WhisperConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.scale_embedding {
        return Err(Error::unsupported("scaled embeddings"));
    }
    Ok(WhisperConfig {
        vocab_size: fields.vocab_size,
        mels: fields.num_mel_bins,
        width: fields.d_model,
        encoder_layers: fields.encoder_layers,
        encoder_heads: fields.encoder_attention_heads,
        encoder_inner: fields.encoder_ffn_dim,
        decoder_layers: fields.decoder_layers,
        decoder_heads: fields.decoder_attention_heads,
        decoder_inner: fields.decoder_ffn_dim,
        source_positions: fields.max_source_positions,
        target_positions: fields.max_target_positions,
        activation: fields.activation_function,
    })
}

pub(crate) fn encoder_rules() -> Rules {
    let encoder = |name: &str| format!("model.encoder.{name}");
    Rules::new()
        .rename(&encoder("conv1"), "convolutions.0")
        .rename(&encoder("conv2"), "convolutions.1")
        .rename(&encoder("embed_positions"), "embeddings.positions")
        .rename(&encoder("layer_norm"), "norm")
        .then(pre_norm_encoder_rules(&encoder("layers"), "blocks"))
}

// A decoder layer's self-attention and feed-forward layer are stored as an encoder layer's are, and its cross-attention as `encoder_attn`.
pub(crate) fn decoder_rules() -> Rules {
    let decoder = |name: &str| format!("model.decoder.{name}");
    let layer = |name: &str| decoder(&format!("layers.*.{name}"));
    Rules::new()
        .rename(&decoder("embed_tokens"), "embeddings.tokens")
        .rename(&decoder("embed_positions"), "embeddings.positions")
        .rename(
            &layer("encoder_attn_layer_norm"),
            "blocks.*.cross_attention_norm",
        )
        .rename(
            &layer("encoder_attn.q_proj"),
            "blocks.*.cross_attention.query",
        )
        .rename(
            &layer("encoder_attn.k_proj"),
            "blocks.*.cross_attention.key",
        )
        .rename(
            &layer("encoder_attn.v_proj"),
            "blocks.*.cross_attention.value",
        )
        .rename(
            &layer("encoder_attn.out_proj"),
            "blocks.*.cross_attention.output",
        )
        .then(pre_norm_encoder_rules(&decoder("layers"), "blocks"))
        .rename(&decoder("layer_norm"), "norm")
        // The head is tied to the token embeddings; a checkpoint that stores it too has it checked against them.
        .rename("proj_out", "head")
}
