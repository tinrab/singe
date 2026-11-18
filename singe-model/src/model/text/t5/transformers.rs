// Reads a `transformers` T5 config and imports its checkpoint, whose blocks number their parts `layer.0` to `layer.2` and whose first block holds its stack's position bias.

use serde::Deserialize;
use singe::nn::Activation;

use super::T5Config;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation},
    source::Source,
};

const fn enabled() -> bool {
    true
}

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
    #[serde(default = "enabled")]
    tie_word_embeddings: bool,
}

const fn max_distance() -> i64 {
    128
}

// `T5Config`'s default, which the original T5 checkpoints leave to it.
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
        head: fields.d_kv,
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

fn attention(rules: Rules, from: &str, to: &str) -> Rules {
    rules
        .rename(&format!("{from}.q"), &format!("{to}.query"))
        .rename(&format!("{from}.k"), &format!("{to}.key"))
        .rename(&format!("{from}.v"), &format!("{to}.value"))
        .rename(&format!("{from}.o"), &format!("{to}.output"))
}

pub(crate) fn rules(tied: bool) -> Rules {
    let mut rules = Rules::new()
        .rename("shared", "tokens")
        // Each stack's copy of the shared embedding.
        .skip("encoder.embed_tokens")
        .skip("decoder.embed_tokens")
        // A cross-attention position bias the original checkpoints store and `transformers` never reads.
        .skip("decoder.block.0.layer.1.EncDecAttention.relative_attention_bias");
    rules = if tied {
        rules.skip("lm_head")
    } else {
        rules.rename("lm_head", "head")
    };
    for (stack, decoder) in [("encoder", false), ("decoder", true)] {
        // A decoder block's cross-attention is its part 1, which moves its feed-forward layer to part 2.
        let feed_forward = if decoder { 2 } else { 1 };
        let block = |part: usize, name: &str| format!("{stack}.block.*.layer.{part}.{name}");
        let target = |name: &str| format!("{stack}.blocks.*.{name}");
        rules = rules.rename(
            &format!("{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            &format!("{stack}.position_bias"),
        );
        rules = attention(rules, &block(0, "SelfAttention"), &target("attention"));
        rules = rules.rename(&block(0, "layer_norm"), &target("attention_norm"));
        if decoder {
            rules = attention(
                rules,
                &block(1, "EncDecAttention"),
                &target("cross_attention"),
            );
            rules = rules.rename(&block(1, "layer_norm"), &target("cross_norm"));
        }

        rules = rules
            .rename(
                &block(feed_forward, "layer_norm"),
                &target("feed_forward_norm"),
            )
            .rename(
                &block(feed_forward, "DenseReluDense.wi"),
                &target("feed_forward.up"),
            )
            .rename(
                &block(feed_forward, "DenseReluDense.wi_0"),
                &target("feed_forward.up"),
            )
            .rename(
                &block(feed_forward, "DenseReluDense.wi_1"),
                &target("feed_forward.gate"),
            )
            .rename(
                &block(feed_forward, "DenseReluDense.wo"),
                &target("feed_forward.down"),
            )
            .rename(
                &format!("{stack}.final_layer_norm"),
                &format!("{stack}.norm"),
            );
    }
    rules
}
