//! Reads a `transformers` `Mask2Former` config, whose backbone is `transformers`' Swin, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{SwinConfig, attention_rules, deformable_encoder_rules, swin_rules},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config},
    },
    model::vision::mask2former::Mask2FormerConfig,
    source::Source,
};

/// The backbone's stages, whose maps the pixel decoder reads.
const STAGES: usize = 4;

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct SwinFields {
    model_type: String,
    embed_dim: i64,
    depths: Vec<i64>,
    num_heads: Vec<i64>,
    window_size: i64,
    patch_size: i64,
    #[serde(default = "default_channels")]
    num_channels: i64,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    #[serde(default)]
    use_absolute_embeddings: bool,
    out_features: Vec<String>,
}

const fn default_channels() -> i64 {
    3
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone_config: SwinFields,
    feature_size: i64,
    mask_feature_size: i64,
    hidden_dim: i64,
    num_attention_heads: i64,
    encoder_layers: i64,
    encoder_feedforward_dim: i64,
    decoder_layers: i64,
    dim_feedforward: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    num_queries: i64,
    common_stride: i64,
    feature_strides: Vec<i64>,
    #[serde(default)]
    pre_norm: bool,
    #[serde(default)]
    enforce_input_projection: bool,
}

pub(crate) fn config(source: &Source) -> Result<Mask2FormerConfig> {
    let fields: Fields = source.json("config.json")?;
    let swin = &fields.backbone_config;
    let all_stages = (1..=swin.depths.len())
        .map(|stage| format!("stage{stage}"))
        .collect::<Vec<_>>();
    if swin.model_type != "swin"
        || swin.use_absolute_embeddings
        || swin.depths.len() != STAGES
        || swin.num_heads.len() != STAGES
        || swin.out_features != all_stages
    {
        return Err(Error::unsupported(
            "a Mask2Former backbone other than a Swin of four stages giving every stage",
        ));
    }

    // The pixel decoder adds one map at the finest stride, which the mask features keep, so the finest stride is the common one, half the next.
    if fields.feature_size != fields.hidden_dim
        || fields.mask_feature_size != fields.hidden_dim
        || fields.pre_norm
        || fields.enforce_input_projection
        || fields.feature_strides.first() != Some(&fields.common_stride)
        || fields.feature_strides.get(1) != Some(&(2 * fields.common_stride))
    {
        return Err(Error::unsupported(
            "a Mask2Former whose widths differ, whose decoder norms first, or whose mask features are not one map finer than the encoder's",
        ));
    }

    Ok(Mask2FormerConfig {
        backbone: SwinConfig {
            width: swin.embed_dim,
            stages: swin
                .depths
                .iter()
                .copied()
                .zip(swin.num_heads.iter().copied())
                .collect(),
            window: swin.window_size,
            patch_size: swin.patch_size,
            channels: swin.num_channels,
            mlp_ratio: swin.mlp_ratio,
            activation: swin.hidden_act,
            eps: swin.layer_norm_eps,
        },
        width: fields.hidden_dim,
        heads: fields.num_attention_heads,
        encoder_layers: fields.encoder_layers,
        encoder_inner: fields.encoder_feedforward_dim,
        // The first mask prediction reads the queries before any layer.
        decoder_layers: fields.decoder_layers - 1,
        decoder_inner: fields.dim_feedforward,
        activation: fields.activation_function,
        queries: fields.num_queries,
        stride: fields.common_stride,
        classifier: classifier_config(source)?,
    })
}

pub(crate) fn rules() -> Rules {
    let backbone = "model.pixel_level_module.encoder";
    let pixel = "model.pixel_level_module.decoder";
    let transformer = "model.transformer_module";
    let decoder = format!("{transformer}.decoder.layers.*");
    let mut rules = Rules::new()
        // The loss's weights of each class, which inference never reads.
        .skip("criterion.empty_weight")
        .then(swin_rules(
            &format!("{backbone}.encoder"),
            &format!("{backbone}.embeddings.patch_embeddings.projection"),
            &format!("{backbone}.embeddings.norm"),
            "backbone",
        ));
    for stage in 0..STAGES {
        rules = rules.rename(
            &format!("{backbone}.hidden_states_norms.stage{}", stage + 1),
            &format!("stage_norms.{stage}"),
        );
    }

    for (from, to) in [
        ("input_projections.*", "input_projections.*"),
        ("adapter_1", "lateral"),
        ("layer_1", "output"),
    ] {
        rules = rules.rename_each(
            &format!("{pixel}.{from}"),
            to,
            &[("0", "convolution"), ("1", "norm")],
        );
    }

    for part in ["weight", "bias"] {
        rules = rules.split(
            &format!("{decoder}.cross_attn.in_proj_{part}"),
            0,
            &[
                &format!("decoder.*.cross_attention.query.{part}"),
                &format!("decoder.*.cross_attention.key.{part}"),
                &format!("decoder.*.cross_attention.value.{part}"),
            ],
        );
    }
    rules
        .rename(&format!("{pixel}.level_embed"), "level_embeddings")
        .rename(&format!("{pixel}.mask_projection"), "mask_projection")
        .then(deformable_encoder_rules(
            &format!("{pixel}.encoder.layers.*"),
            "encoder.*",
        ))
        .rename(
            &format!("{transformer}.level_embed.weight"),
            "query_level_embeddings",
        )
        .rename(
            &format!("{transformer}.queries_embedder.weight"),
            "query_positions",
        )
        .rename(
            &format!("{transformer}.queries_features.weight"),
            "query_features",
        )
        .rename(&format!("{transformer}.decoder.layernorm"), "decoder_norm")
        .rename(
            &format!("{transformer}.decoder.mask_predictor.mask_embedder.*.0"),
            "mask_embedder.*",
        )
        .rename(
            &format!("{decoder}.cross_attn.out_proj"),
            "decoder.*.cross_attention.output",
        )
        .then(attention_rules(
            &format!("{decoder}.self_attn"),
            "decoder.*.attention",
        ))
        .rename_each(
            &decoder,
            "decoder.*",
            &[
                ("cross_attn_layer_norm", "cross_attention_norm"),
                ("self_attn_layer_norm", "attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
        .rename("class_predictor", "classes")
}
