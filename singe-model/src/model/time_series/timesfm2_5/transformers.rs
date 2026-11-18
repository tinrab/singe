// Reads a `transformers` TimesFM 2.5 snapshot: its config and its tensors' names.

use serde::Deserialize;

use super::TimesFmConfig;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation},
    source::Source,
};

#[derive(Debug, Deserialize)]
struct RopeFields {
    rope_theta: f64,
}

#[derive(Debug, Deserialize)]
struct Fields {
    activation: String,
    context_length: usize,
    decode_index: usize,
    force_flip_invariance: bool,
    head_dim: i64,
    hidden_size: i64,
    horizon_length: i64,
    infer_is_positive: bool,
    intermediate_size: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    output_quantile_len: i64,
    patch_length: i64,
    quantiles: Vec<f64>,
    rms_norm_eps: f32,
    rope_parameters: RopeFields,
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
        outputs: i64::try_from(fields.quantiles.len()).unwrap_or(0) + 1,
        median: fields.decode_index,
        context: fields.context_length,
        width: fields.hidden_size,
        inner: fields.intermediate_size,
        layers: fields.num_hidden_layers,
        heads: fields.num_attention_heads,
        head_dim: fields.head_dim,
        eps: fields.rms_norm_eps,
        activation: activation(&fields.activation)?,
        theta: fields.rope_parameters.rope_theta,
        flip_invariant: fields.force_flip_invariance,
        positive: fields.infer_is_positive,
        continuous_quantiles: fields.use_continuous_quantile_head,
    })
}

// Imports the model, folding each attention's stored scaling into the scale its queries are multiplied by: its softplus times `1.442695041 / √D`.
pub(crate) fn rules(head_dim: i64) -> Rules {
    let layer = |name: &str| format!("model.layers.*.{name}");
    #[expect(clippy::cast_precision_loss, reason = "head widths are small")]
    #[expect(
        clippy::approx_constant,
        reason = "TimesFM scales by this truncated `log2(e)`"
    )]
    let factor = 1.442_695_041 / (head_dim as f64).sqrt();
    Rules::new()
        .rename("model.input_ff_layer.input_layer", "tokenizer.input")
        .rename("model.input_ff_layer.output_layer", "tokenizer.output")
        .rename("model.input_ff_layer.residual_layer", "tokenizer.residual")
        .rename(&layer("input_layernorm"), "blocks.*.attention_norm")
        .rename(&layer("self_attn.q_proj"), "blocks.*.attention.query")
        .rename(&layer("self_attn.k_proj"), "blocks.*.attention.key")
        .rename(&layer("self_attn.v_proj"), "blocks.*.attention.value")
        .rename(&layer("self_attn.o_proj"), "blocks.*.attention.output")
        .rename(&layer("self_attn.q_norm"), "blocks.*.attention.query_norm")
        .rename(&layer("self_attn.k_norm"), "blocks.*.attention.key_norm")
        .derive(
            &layer("self_attn.scaling"),
            "blocks.*.attention.query_scale",
            move |scaling| scaling.softplus(1.0)?.mul_scalar(factor),
        )
        .rename(
            &layer("post_attention_layernorm"),
            "blocks.*.post_attention_norm",
        )
        .rename(
            &layer("pre_feedforward_layernorm"),
            "blocks.*.feed_forward_norm",
        )
        .rename(&layer("mlp.ff0"), "blocks.*.feed_forward.up")
        .rename(&layer("mlp.ff1"), "blocks.*.feed_forward.down")
        .rename(
            &layer("post_feedforward_layernorm"),
            "blocks.*.post_feed_forward_norm",
        )
        .rename("output_projection_point.input_layer", "point.input")
        .rename("output_projection_point.output_layer", "point.output")
        .rename("output_projection_point.residual_layer", "point.residual")
        .rename("output_projection_quantiles.input_layer", "quantiles.input")
        .rename(
            "output_projection_quantiles.output_layer",
            "quantiles.output",
        )
        .rename(
            "output_projection_quantiles.residual_layer",
            "quantiles.residual",
        )
}
