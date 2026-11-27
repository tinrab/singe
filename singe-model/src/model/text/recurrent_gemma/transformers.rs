//! Reads a `transformers` `RecurrentGemma` config and imports its checkpoint.

use serde::Deserialize;
use singe::{
    data_type::bf16,
    nn::{Activation, RgLruSpec},
};

use crate::{
    assembly::DecoderConfig,
    error::{Error, Result},
    format::{
        RuleSet,
        transformers::{DecoderDefaults, DecoderFields, activation_field, default_true},
    },
    model::text::recurrent_gemma::RecurrentGemmaConfig,
    source::Source,
};

const fn default_activation() -> Activation {
    Activation::Gelu
}

const fn default_rotated_fraction() -> f64 {
    0.5
}

fn default_block_types() -> Vec<BlockType> {
    vec![
        BlockType::Recurrent,
        BlockType::Recurrent,
        BlockType::Attention,
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BlockType {
    Recurrent,
    Attention,
}

/// The config's fields; those it leaves out take the defaults of `transformers`' `RecurrentGemmaConfig`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    #[serde(default = "default_activation", deserialize_with = "activation_field")]
    hidden_activation: Activation,
    lru_width: Option<i64>,
    conv1d_width: i64,
    attention_window_size: i64,
    logits_soft_cap: f32,
    #[serde(default = "default_rotated_fraction")]
    partial_rotary_factor: f64,
    #[serde(default = "default_block_types")]
    block_types: Vec<BlockType>,
    #[serde(default = "default_true")]
    embeddings_scale_by_sqrt_dim: bool,
}

pub(crate) fn config(source: &Source) -> Result<RecurrentGemmaConfig> {
    let fields: Fields = source.json("config.json")?;
    if !fields.embeddings_scale_by_sqrt_dim {
        return Err(Error::unsupported("unscaled embeddings"));
    }

    if fields.block_types.is_empty() {
        return Err(Error::unsupported("no block types"));
    }

    let mut decoder = DecoderConfig {
        activation: fields.hidden_activation,
        mlp_bias: true,
        ..fields.decoder.config(DecoderDefaults {
            tied: true,
            head_dim: None,
        })?
    };
    decoder.inner = decoder.inner.map(|inner| inner / 2);
    decoder.rotary_dim = (decoder.head_dim as f64 * fields.partial_rotary_factor) as i64;
    let attention_layers = fields
        .block_types
        .iter()
        .cycle()
        .take(usize::try_from(decoder.layers).unwrap_or_default())
        .enumerate()
        .filter_map(|(index, &kind)| (kind == BlockType::Attention).then_some(index))
        .collect();
    let width = fields.lru_width.unwrap_or(decoder.width);
    Ok(RecurrentGemmaConfig {
        recurrent: RgLruSpec {
            width,
            blocks: decoder.heads,
            kernel: fields.conv1d_width,
        },
        attention_layers,
        window: fields.attention_window_size,
        embedding_scale: bf16::from_f32((decoder.width as f32).sqrt()).to_f32(),
        logit_cap: fields.logits_soft_cap,
        decoder,
    })
}

pub(crate) fn rules() -> RuleSet {
    let layers = "model.layers.*";
    RuleSet::new()
        .rename("model.embed_tokens", "embeddings.tokens")
        .rename("model.final_norm", "norm")
        .rename_each(
            layers,
            "blocks.*",
            &[
                ("temporal_pre_norm", "attention_norm"),
                ("channel_pre_norm", "feed_forward_norm"),
                ("mlp_block.gate_proj", "feed_forward.gate"),
                ("mlp_block.up_proj", "feed_forward.up"),
                ("mlp_block.down_proj", "feed_forward.down"),
                ("temporal_block.q_proj", "attention.query"),
                ("temporal_block.k_proj", "attention.key"),
                ("temporal_block.v_proj", "attention.value"),
                ("temporal_block.o_proj", "attention.output"),
                ("temporal_block.linear_x", "recurrent.input"),
                ("temporal_block.linear_y", "recurrent.gate"),
                ("temporal_block.linear_out", "recurrent.output"),
                ("temporal_block.conv_1d", "recurrent.convolution"),
                ("temporal_block.rg_lru.recurrent_param", "recurrent.rate"),
                (
                    "temporal_block.rg_lru.input_gate_weight",
                    "recurrent.input_gate.weight",
                ),
                (
                    "temporal_block.rg_lru.input_gate_bias",
                    "recurrent.input_gate.bias",
                ),
                (
                    "temporal_block.rg_lru.recurrent_gate_weight",
                    "recurrent.recurrence_gate.weight",
                ),
                (
                    "temporal_block.rg_lru.recurrent_gate_bias",
                    "recurrent.recurrence_gate.bias",
                ),
            ],
        )
}
