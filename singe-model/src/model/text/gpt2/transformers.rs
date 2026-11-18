// Reads a `transformers` GPT-2 config, and imports its checkpoint, whose `Conv1D` weights are stored `[in, out]` and whose attention fuses its query, key, and value projections in `c_attn`.

use serde::Deserialize;
use singe::nn::Activation;

use super::Gpt2Config;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    source::Source,
};

const fn default_scale_attention() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vocab_size: i64,
    n_embd: i64,
    n_head: i64,
    n_layer: i64,
    n_positions: i64,
    #[serde(default)]
    n_inner: Option<i64>,
    layer_norm_epsilon: f32,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    #[serde(flatten)]
    attention: AttentionFields,
}

// How GPT-2's attention scales its scores, which models with its blocks share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(crate) struct AttentionFields {
    #[serde(default = "default_scale_attention")]
    scale_attn_weights: bool,
    #[serde(default)]
    scale_attn_by_inverse_layer_idx: bool,
    #[serde(default)]
    reorder_and_upcast_attn: bool,
}

impl AttentionFields {
    // Refuses scaling other than by the root of the head width, which a checkpoint would load with and then compute something else.
    pub(crate) fn refuse_other_scaling(self) -> Result<()> {
        if !self.scale_attn_weights
            || self.scale_attn_by_inverse_layer_idx
            || self.reorder_and_upcast_attn
        {
            return Err(Error::unsupported("non-default attention scaling"));
        }
        Ok(())
    }
}

pub(crate) fn config(source: &Source) -> Result<Gpt2Config> {
    let fields: Fields = source.json("config.json")?;
    fields.attention.refuse_other_scaling()?;
    Ok(Gpt2Config {
        vocab_size: fields.vocab_size,
        width: fields.n_embd,
        heads: fields.n_head,
        layers: fields.n_layer,
        positions: fields.n_positions,
        inner: fields.n_inner.unwrap_or(4 * fields.n_embd),
        eps: fields.layer_norm_epsilon,
        activation: fields.activation_function,
    })
}

// Older checkpoints store blocks without `transformer.`.
pub(crate) fn rules() -> Rules {
    model_rules("transformer.")
        .then(model_rules(""))
        .rename("lm_head", "head")
}

// The rules for a model whose names start with `prefix`.
pub(crate) fn model_rules(prefix: &str) -> Rules {
    let model = |name: &str| format!("{prefix}{name}");
    let layer = |name: &str| format!("{prefix}h.*.{name}");
    Rules::new()
        .rename(&model("wte"), "embeddings.tokens")
        .rename(&model("wpe"), "embeddings.positions")
        // Older checkpoints also store attention masks, which this definition computes itself.
        .skip(&layer("attn.bias"))
        .skip(&layer("attn.masked_bias"))
        .rename(&layer("ln_1"), "blocks.*.attention_norm")
        .transpose_split(
            &layer("attn.c_attn.weight"),
            &[
                "blocks.*.attention.query.weight",
                "blocks.*.attention.key.weight",
                "blocks.*.attention.value.weight",
            ],
        )
        .split(
            &layer("attn.c_attn.bias"),
            0,
            &[
                "blocks.*.attention.query.bias",
                "blocks.*.attention.key.bias",
                "blocks.*.attention.value.bias",
            ],
        )
        .transpose(
            &layer("attn.c_proj.weight"),
            "blocks.*.attention.output.weight",
        )
        .rename(&layer("attn.c_proj.bias"), "blocks.*.attention.output.bias")
        .rename(&layer("ln_2"), "blocks.*.feed_forward_norm")
        .transpose(&layer("mlp.c_fc.weight"), "blocks.*.feed_forward.up.weight")
        .rename(&layer("mlp.c_fc.bias"), "blocks.*.feed_forward.up.bias")
        .transpose(
            &layer("mlp.c_proj.weight"),
            "blocks.*.feed_forward.down.weight",
        )
        .rename(&layer("mlp.c_proj.bias"), "blocks.*.feed_forward.down.bias")
        .rename(&model("ln_f"), "norm")
}
