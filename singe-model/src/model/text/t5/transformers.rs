//! Reads a `transformers` T5 config and imports its checkpoint, whose blocks number their parts `layer.0` to `layer.2` and whose first block holds its stack's position bias.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules, join,
        transformers::{activation, default_true},
    },
    model::text::t5::{StackRole, T5Config},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vocab_size: i64,
    d_model: i64,
    d_kv: i64,
    d_ff: i64,
    num_heads: i64,
    num_layers: i64,
    num_decoder_layers: Option<i64>,
    relative_attention_num_buckets: i64,
    #[serde(default = "max_distance")]
    relative_attention_max_distance: i64,
    layer_norm_epsilon: f32,
    #[serde(default = "relu")]
    feed_forward_proj: String,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

const fn max_distance() -> i64 {
    128
}

/// Returns `T5Config`'s default `feed_forward_proj`, which the original T5 checkpoints leave to it.
fn relu() -> String {
    "relu".to_owned()
}

pub(crate) fn config(source: &Source) -> Result<T5Config> {
    let fields: Fields = source.json("config.json")?;
    // `gated-gelu` names a gated feed-forward and its activation, as T5 v1.1 has it.
    let (gated, name) = match fields.feed_forward_proj.strip_prefix("gated-") {
        Some(name) => (true, name),
        None => (false, fields.feed_forward_proj.as_str()),
    };
    let activation = match name {
        // `transformers` maps T5's `gelu` to its tanh approximation.
        "gelu" if gated => Activation::Gelu,
        other => activation(other)?,
    };
    if fields.relative_attention_num_buckets < 2 {
        return Err(Error::unsupported("fewer than two position buckets"));
    }

    Ok(T5Config {
        vocab_size: fields.vocab_size,
        width: fields.d_model,
        head_dim: fields.d_kv,
        heads: fields.num_heads,
        inner: fields.d_ff,
        encoder_layers: fields.num_layers,
        decoder_layers: fields.num_decoder_layers.unwrap_or(fields.num_layers),
        buckets: fields.relative_attention_num_buckets,
        max_distance: fields.relative_attention_max_distance,
        eps: fields.layer_norm_epsilon,
        activation,
        gated,
        tied: fields.tie_word_embeddings,
    })
}

fn projection_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("q", "query"),
            ("k", "key"),
            ("v", "value"),
            ("o", "output"),
        ],
    )
}

fn stack_rules(stack: &str, role: StackRole) -> Rules {
    let blocks = format!("{stack}.blocks.*");
    let part = |index: usize| format!("{stack}.block.*.layer.{index}");
    // A decoder block's cross-attention is its part 1, which moves its feed-forward layer to part 2.
    let (cross, feed_forward) = match role {
        StackRole::Encoder => (Rules::new(), part(1)),
        StackRole::Decoder => (
            projection_rules(
                &join(&part(1), "EncDecAttention"),
                &join(&blocks, "cross_attention"),
            )
            .rename(
                &join(&part(1), "layer_norm"),
                &join(&blocks, "cross_attention_norm"),
            ),
            part(2),
        ),
    };
    Rules::new()
        .rename(
            &format!("{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            &format!("{stack}.position_bias"),
        )
        .then(projection_rules(
            &join(&part(0), "SelfAttention"),
            &join(&blocks, "attention"),
        ))
        .rename(
            &join(&part(0), "layer_norm"),
            &join(&blocks, "attention_norm"),
        )
        .then(cross)
        .rename_each(
            &feed_forward,
            &blocks,
            &[
                ("layer_norm", "feed_forward_norm"),
                ("DenseReluDense.wi", "feed_forward.up"),
                ("DenseReluDense.wi_0", "feed_forward.gate"),
                ("DenseReluDense.wi_1", "feed_forward.up"),
                ("DenseReluDense.wo", "feed_forward.down"),
            ],
        )
        .rename(
            &format!("{stack}.final_layer_norm"),
            &format!("{stack}.norm"),
        )
}

pub(crate) fn rules(tied: bool) -> Rules {
    let rules = Rules::new()
        .rename("shared", "tokens")
        // Each stack's copy of the shared embedding.
        .skip("encoder.embed_tokens")
        .skip("decoder.embed_tokens")
        // A cross-attention position bias the original checkpoints store and `transformers` never reads.
        .skip("decoder.block.0.layer.1.EncDecAttention.relative_attention_bias");
    let rules = if tied {
        rules.skip("lm_head")
    } else {
        rules.rename("lm_head", "head")
    };
    rules
        .then(stack_rules("encoder", StackRole::Encoder))
        .then(stack_rules("decoder", StackRole::Decoder))
}
