//! Reads a `transformers` Granite-4.0-H config and imports its checkpoint, folding the residual multiplier into each branch's output projection.

use serde::Deserialize;
use singe::nn::Mamba2Spec;

use crate::{
    assembly::{llama_rules, rename_scaled},
    error::{Error, Result},
    format::{
        RuleSet,
        transformers::{DecoderDefaults, DecoderFields, LayerType, default_true},
    },
    model::text::granite_moe_hybrid::GraniteMoeHybridConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    shared_intermediate_size: i64,
    #[serde(default)]
    num_local_experts: i64,
    position_embedding_type: Option<String>,
    attention_multiplier: f32,
    embedding_multiplier: f32,
    logits_scaling: f32,
    residual_multiplier: f32,
    mamba_n_heads: i64,
    mamba_d_head: i64,
    mamba_n_groups: i64,
    mamba_d_state: i64,
    mamba_d_conv: i64,
    #[serde(default)]
    mamba_proj_bias: bool,
    #[serde(default = "default_true")]
    mamba_conv_bias: bool,
}

pub(crate) fn config(source: &Source) -> Result<GraniteMoeHybridConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.num_local_experts > 0 {
        return Err(Error::unsupported("granite-4.0-h experts"));
    }

    let mut decoder = fields.decoder.config(DecoderDefaults {
        tied: true,
        head_dim: None,
    })?;
    decoder.inner = Some(fields.shared_intermediate_size);
    let attention_layers = fields
        .decoder
        .layer_types(|_| LayerType::LinearAttention)?
        .into_iter()
        .enumerate()
        .filter_map(|(index, layer)| (layer == LayerType::FullAttention).then_some(index))
        .collect();
    Ok(GraniteMoeHybridConfig {
        decoder,
        mixer: Mamba2Spec {
            heads: fields.mamba_n_heads,
            head_dim: fields.mamba_d_head,
            groups: fields.mamba_n_groups,
            state: fields.mamba_d_state,
            kernel: fields.mamba_d_conv,
        },
        attention_layers,
        rotary: fields
            .position_embedding_type
            .is_some_and(|kind| kind == "rope"),
        mixer_bias: fields.mamba_proj_bias,
        convolution_bias: fields.mamba_conv_bias,
        attention_scale: fields.attention_multiplier,
        embedding_scale: fields.embedding_multiplier,
        logit_scale: fields.logits_scaling.recip(),
        residual_scale: fields.residual_multiplier,
    })
}

pub(crate) fn rules(residual_scale: f32) -> RuleSet {
    let layers = "model.layers.*";
    rename_scaled(
        &format!("{layers}.self_attn.o_proj.weight"),
        "blocks.*.attention.output.weight",
        residual_scale,
    )
    .then(rename_scaled(
        &format!("{layers}.mamba.out_proj.weight"),
        "blocks.*.mixer.output.weight",
        residual_scale,
    ))
    .then(rename_scaled(
        &format!("{layers}.shared_mlp.output_linear.weight"),
        "blocks.*.feed_forward.down.weight",
        residual_scale,
    ))
    .split(
        &format!("{layers}.shared_mlp.input_linear.weight"),
        0,
        &[
            "blocks.*.feed_forward.gate.weight",
            "blocks.*.feed_forward.up.weight",
        ],
    )
    .rename_each(
        &format!("{layers}.mamba"),
        "blocks.*.mixer",
        &[
            ("in_proj", "input"),
            ("conv1d", "convolution"),
            ("dt_bias", "step_bias"),
            ("A_log", "decay_log"),
            ("D", "skip"),
            ("norm", "norm"),
        ],
    )
    .then(llama_rules("model"))
}
