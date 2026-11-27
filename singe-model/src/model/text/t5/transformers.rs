//! Reads a `transformers` T5 config and imports its checkpoint, whose blocks number their parts `layer.0` to `layer.2` and whose first block holds its stack's position bias.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        RuleSet, join,
        transformers::{activation, default_true},
    },
    model::text::t5::{StackRole, SwitchSpec, T5Config, TransientGlobal},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct Fields {
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
    num_experts: Option<i64>,
    #[serde(default)]
    encoder_sparse_step: i64,
    #[serde(default)]
    decoder_sparse_step: i64,
    expert_capacity: Option<i64>,
    #[serde(default)]
    router_bias: bool,
    router_dtype: Option<String>,
    num_selected_experts: Option<i64>,
    encoder_attention_type: Option<String>,
    local_radius: Option<i64>,
    global_block_size: Option<i64>,
}

const fn max_distance() -> i64 {
    128
}

/// Returns `T5Config`'s default `feed_forward_proj`, which the original T5 checkpoints leave to it.
fn relu() -> String {
    "relu".to_owned()
}

pub(crate) fn config(source: &Source) -> Result<T5Config> {
    fields_config(&source.json("config.json")?)
}

/// Reads T5's config from its fields, as a composite model such as `MusicGen` nests them.
pub(crate) fn fields_config(fields: &Fields) -> Result<T5Config> {
    // `gated-gelu` names a gated feed-forward and its activation, as T5 v1.1 has it.
    let (gated, name) = fields
        .feed_forward_proj
        .strip_prefix("gated-")
        .map_or((false, fields.feed_forward_proj.as_str()), |name| {
            (true, name)
        });
    let activation = match name {
        // `transformers` maps T5's `gelu` to its tanh approximation.
        "gelu" if gated => Activation::Gelu,
        other => activation(other)?,
    };
    if fields.relative_attention_num_buckets < 2 {
        return Err(Error::unsupported("fewer than two position buckets"));
    }

    let experts = match fields.num_experts {
        Some(count) => {
            if fields.router_bias
                || fields
                    .router_dtype
                    .as_deref()
                    .is_some_and(|dtype| dtype != "float32")
                || fields
                    .num_selected_experts
                    .is_some_and(|selected| selected != 1)
            {
                return Err(Error::unsupported(
                    "Switch experts other than one per token, routed in F32 without bias",
                ));
            }

            Some(SwitchSpec {
                count,
                encoder_step: fields.encoder_sparse_step,
                decoder_step: fields.decoder_sparse_step,
                capacity: fields
                    .expert_capacity
                    .ok_or_else(|| Error::unsupported("Switch experts without a capacity"))?,
            })
        }
        None => None,
    };

    let transient_global = match fields.encoder_attention_type.as_deref() {
        None => None,
        Some("transient-global") => Some(TransientGlobal {
            radius: fields
                .local_radius
                .ok_or_else(|| Error::unsupported("LongT5 without a local radius"))?,
            block: fields
                .global_block_size
                .ok_or_else(|| Error::unsupported("LongT5 without a global block size"))?,
        }),
        Some(other) => {
            return Err(Error::unsupported(format!(
                "LongT5 encoder attention `{other}`"
            )));
        }
    };

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
        experts,
        transient_global,
    })
}

fn projection_rules(from: &str, to: &str) -> RuleSet {
    RuleSet::new().rename_each(
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

fn stack_rules(stack: &str, role: StackRole) -> RuleSet {
    let blocks = format!("{stack}.blocks.*");
    let part = |index: usize| format!("{stack}.block.*.layer.{index}");
    // A decoder block's cross-attention is its part 1, which moves its feed-forward layer to part 2.
    let (cross, feed_forward) = match role {
        StackRole::Encoder => (RuleSet::new(), part(1)),
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
    // LongT5's encoder names its attention `TransientGlobalSelfAttention`, with a global table beside the position table and a norm of the global tokens in each block.
    let global = join(&part(0), "TransientGlobalSelfAttention");
    RuleSet::new()
        .rename(
            &format!("{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            &format!("{stack}.position_bias"),
        )
        .rename(
            &format!("{stack}.block.0.layer.0.TransientGlobalSelfAttention.relative_attention_bias.weight"),
            &format!("{stack}.position_bias"),
        )
        .rename(
            &format!(
                "{stack}.block.0.layer.0.TransientGlobalSelfAttention.global_relative_attention_bias.weight"
            ),
            &format!("{stack}.global_bias"),
        )
        .rename(
            &join(&global, "global_input_layer_norm"),
            &join(&blocks, "global_norm"),
        )
        .then(projection_rules(&global, &join(&blocks, "attention")))
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
                // Switch Transformers' names, for its dense layers and its experts.
                ("mlp.wi", "feed_forward.up"),
                ("mlp.wo", "feed_forward.down"),
                ("mlp.router.classifier", "feed_forward.router"),
                ("mlp.experts", "feed_forward.experts"),
            ],
        )
        .rename(
            &format!("{stack}.final_layer_norm"),
            &format!("{stack}.norm"),
        )
}

/// Imports the encoder of a T5 stored under `prefix`, its embeddings `shared`, to [`T5Encoder`](crate::model::text::t5::T5Encoder)'s names under the same prefix.
pub(crate) fn encoder_rules(prefix: &str) -> RuleSet {
    RuleSet::new()
        .rename(&join(prefix, "shared"), &join(prefix, "tokens"))
        .skip(&join(prefix, "encoder.embed_tokens"))
        .then(stack_rules(&join(prefix, "encoder"), StackRole::Encoder))
}

pub(crate) fn rules(tied: bool) -> RuleSet {
    let rules = RuleSet::new()
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
