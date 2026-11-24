//! Reads a `transformers` Whisper snapshot: its config, and the rules that import its encoder's and decoder's tensors.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{attention_rules, pre_norm_encoder_rules},
    error::{Error, Result},
    format::{Rules, join, transformers::activation_field},
    model::audio::whisper::WhisperConfig,
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
    let encoder = "model.encoder";
    Rules::new()
        .rename_each(
            encoder,
            "encoder",
            &[
                ("conv1", "convolutions.0"),
                ("conv2", "convolutions.1"),
                ("embed_positions", "embeddings.positions"),
                ("layer_norm", "norm"),
            ],
        )
        .then(pre_norm_encoder_rules(
            &join(encoder, "layers"),
            "encoder.blocks",
        ))
}

/// Returns the rules that import the decoder's tensors.
/// A decoder layer's self-attention and feed-forward layer are stored as an encoder layer's are, and its cross-attention as `encoder_attn`.
pub(crate) fn decoder_rules() -> Rules {
    let decoder = "model.decoder";
    let layers = join(decoder, "layers");
    Rules::new()
        .rename_each(
            decoder,
            "decoder",
            &[
                ("embed_tokens", "embeddings.tokens"),
                ("embed_positions", "embeddings.positions"),
            ],
        )
        .rename(
            &join(&layers, "*.encoder_attn_layer_norm"),
            "decoder.blocks.*.cross_attention_norm",
        )
        .then(attention_rules(
            &join(&layers, "*.encoder_attn"),
            "decoder.blocks.*.cross_attention",
        ))
        .then(pre_norm_encoder_rules(&layers, "decoder.blocks"))
        .rename(&join(decoder, "layer_norm"), "decoder.norm")
        // The head is tied to the token embeddings; a checkpoint that stores it too has it checked against them.
        .rename("proj_out", "decoder.head")
}
