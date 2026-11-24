//! Reads a `transformers` `TimesFM` 2.5 snapshot: its config and its tensors' names.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules,
        transformers::{RotaryFields, activation_field},
    },
    model::time_series::timesfm2_5::TimesFmConfig,
    source::Source,
};

#[derive(Debug, Deserialize)]
struct Fields {
    #[serde(deserialize_with = "activation_field")]
    activation: Activation,
    context_length: usize,
    decode_index: usize,
    force_flip_invariance: bool,
    head_dim: i64,
    hidden_size: i64,
    horizon_length: usize,
    infer_is_positive: bool,
    intermediate_size: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    output_quantile_len: usize,
    patch_length: usize,
    quantiles: Vec<f64>,
    rms_norm_eps: f32,
    #[serde(flatten)]
    rotary: RotaryFields,
    use_continuous_quantile_head: bool,
}

pub(crate) fn config(source: &Source) -> Result<TimesFmConfig> {
    let fields: Fields = source.json("config.json")?;
    // The outputs are the mean, then each quantile.
    if fields.decode_index > fields.quantiles.len() {
        return Err(Error::unsupported("a decode index past the outputs"));
    }

    Ok(TimesFmConfig {
        patch: fields.patch_length,
        horizon: fields.horizon_length,
        quantile_horizon: fields.output_quantile_len,
        outputs: fields.quantiles.len() + 1,
        median: fields.decode_index,
        context: fields.context_length,
        width: fields.hidden_size,
        inner: fields.intermediate_size,
        layers: fields.num_hidden_layers,
        heads: fields.num_attention_heads,
        head_dim: fields.head_dim,
        eps: fields.rms_norm_eps,
        activation: fields.activation,
        frequencies: fields.rotary.frequencies()?,
        flip_invariant: fields.force_flip_invariance,
        positive: fields.infer_is_positive,
        continuous_quantiles: fields.use_continuous_quantile_head,
    })
}

/// Returns the rules that import the model, folding each attention's stored scaling into the scale its queries are multiplied by: its softplus times `1.442695041 / √D`.
pub(crate) fn rules(head_dim: i64) -> Rules {
    const RESIDUAL: &[(&str, &str)] = &[
        ("input_layer", "input"),
        ("output_layer", "output"),
        ("residual_layer", "residual"),
    ];
    #[expect(
        clippy::approx_constant,
        reason = "TimesFM scales by this truncated `log2(e)`"
    )]
    let factor = 1.442_695_041 / (head_dim as f64).sqrt();
    Rules::new()
        .rename_each("model.input_ff_layer", "tokenizer", RESIDUAL)
        .rename_each(
            "model.layers.*",
            "blocks.*",
            &[
                ("input_layernorm", "attention_norm"),
                ("post_attention_layernorm", "post_attention_norm"),
                ("pre_feedforward_layernorm", "feed_forward_norm"),
                ("mlp.ff0", "feed_forward.up"),
                ("mlp.ff1", "feed_forward.down"),
                ("post_feedforward_layernorm", "post_feed_forward_norm"),
            ],
        )
        .rename_each(
            "model.layers.*.self_attn",
            "blocks.*.attention",
            &[
                ("q_proj", "query"),
                ("k_proj", "key"),
                ("v_proj", "value"),
                ("o_proj", "output"),
                ("q_norm", "query_norm"),
                ("k_norm", "key_norm"),
            ],
        )
        .derive(
            "model.layers.*.self_attn.scaling",
            "blocks.*.attention.query_scale",
            move |scaling| scaling.softplus(1.0)?.mul_scalar(factor),
        )
        .rename_each("output_projection_point", "point", RESIDUAL)
        .rename_each("output_projection_quantiles", "quantiles", RESIDUAL)
}
