//! Reads a `transformers` Grounding DINO config, whose backbone is `transformers`' Swin and whose text encoder is BERT, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{
        PostNormEncoderConfig, SwinConfig, deformable_attention_rules, deformable_encoder_rules,
        post_norm_encoder_rules, swin_rules,
    },
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    model::multimodal::grounding_dino::GroundingDinoConfig,
    source::Source,
};

/// The fields of Swin's config, each `transformers`' default when the config leaves it null.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct SwinFields {
    model_type: String,
    embed_dim: Option<i64>,
    depths: Vec<i64>,
    num_heads: Vec<i64>,
    window_size: Option<i64>,
    patch_size: Option<i64>,
    mlp_ratio: Option<f64>,
    layer_norm_eps: Option<f32>,
    #[serde(default)]
    use_absolute_embeddings: bool,
    out_indices: Vec<usize>,
}

/// The fields of BERT's config, each `transformers`' default when the config leaves it out.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TextFields {
    model_type: String,
    #[serde(default = "bert_vocabulary")]
    vocab_size: i64,
    #[serde(default = "bert_width")]
    hidden_size: i64,
    #[serde(default = "bert_layers")]
    num_hidden_layers: i64,
    #[serde(default = "bert_heads")]
    num_attention_heads: i64,
    #[serde(default = "bert_inner")]
    intermediate_size: i64,
    #[serde(default = "bert_activation", deserialize_with = "activation_field")]
    hidden_act: Activation,
    #[serde(default = "bert_positions")]
    max_position_embeddings: i64,
    #[serde(default = "bert_token_types")]
    type_vocab_size: i64,
    #[serde(default = "bert_eps")]
    layer_norm_eps: f32,
}

const fn bert_vocabulary() -> i64 {
    30_522
}

const fn bert_width() -> i64 {
    768
}

const fn bert_layers() -> i64 {
    12
}

const fn bert_heads() -> i64 {
    12
}

const fn bert_inner() -> i64 {
    3072
}

const fn bert_activation() -> Activation {
    Activation::GeluExact
}

const fn bert_positions() -> i64 {
    512
}

const fn bert_token_types() -> i64 {
    2
}

const fn bert_eps() -> f32 {
    1e-12
}

/// Whether the decoder's layers, and the two-stage proposals, share one box head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct BoxSharingFields {
    decoder_bbox_embed_share: bool,
    #[serde(default)]
    two_stage_bbox_embed_share: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone_config: SwinFields,
    text_config: TextFields,
    #[serde(default)]
    use_timm_backbone: bool,
    d_model: i64,
    num_feature_levels: i64,
    encoder_layers: i64,
    decoder_layers: i64,
    encoder_attention_heads: i64,
    decoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    decoder_ffn_dim: i64,
    encoder_n_points: i64,
    decoder_n_points: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    num_queries: i64,
    position_embedding_type: String,
    positional_embedding_temperature: f32,
    layer_norm_eps: f32,
    two_stage: bool,
    embedding_init_target: bool,
    #[serde(flatten)]
    box_sharing: BoxSharingFields,
    query_dim: i64,
}

pub(crate) fn config(source: &Source) -> Result<GroundingDinoConfig> {
    let fields: Fields = source.json("config.json")?;
    let swin = &fields.backbone_config;
    let stages = swin.depths.len();
    let backbone_levels = swin.out_indices.len();
    let last_stages = (stages + 1 - backbone_levels..=stages).collect::<Vec<_>>();
    if fields.use_timm_backbone
        || swin.model_type != "swin"
        || swin.use_absolute_embeddings
        || swin.depths.len() != swin.num_heads.len()
        || swin.out_indices != last_stages
    {
        return Err(Error::unsupported(
            "a Grounding DINO backbone other than a Swin giving its last stages",
        ));
    }

    if fields.text_config.model_type != "bert" {
        return Err(Error::unsupported(format!(
            "Grounding DINO text encoder `{}`",
            fields.text_config.model_type
        )));
    }

    if fields.position_embedding_type != "sine"
        || !fields.two_stage
        || !fields.embedding_init_target
        || !fields.box_sharing.decoder_bbox_embed_share
        || fields.box_sharing.two_stage_bbox_embed_share
        || fields.query_dim != 4
    {
        return Err(Error::unsupported(
            "a Grounding DINO other than a two-stage one with learned targets, one shared box head, a separate proposal box head, and sine positions",
        ));
    }

    let text = &fields.text_config;
    Ok(GroundingDinoConfig {
        backbone: SwinConfig {
            width: swin.embed_dim.unwrap_or(96),
            stages: swin
                .depths
                .iter()
                .copied()
                .zip(swin.num_heads.iter().copied())
                .collect(),
            window: swin.window_size.unwrap_or(7),
            patch_size: swin.patch_size.unwrap_or(4),
            channels: 3,
            mlp_ratio: swin.mlp_ratio.unwrap_or(4.0),
            activation: Activation::GeluExact,
            eps: swin.layer_norm_eps.unwrap_or(1e-5),
        },
        backbone_levels,
        text: PostNormEncoderConfig {
            vocab_size: text.vocab_size,
            width: text.hidden_size,
            layers: text.num_hidden_layers,
            heads: text.num_attention_heads,
            inner: text.intermediate_size,
            activation: text.hidden_act,
            max_positions: text.max_position_embeddings,
            first_position: 0,
            token_types: text.type_vocab_size,
            eps: text.layer_norm_eps,
            keys: None,
        },
        width: fields.d_model,
        levels: fields.num_feature_levels,
        encoder_layers: fields.encoder_layers,
        decoder_layers: fields.decoder_layers,
        encoder_heads: fields.encoder_attention_heads,
        decoder_heads: fields.decoder_attention_heads,
        encoder_inner: fields.encoder_ffn_dim,
        decoder_inner: fields.decoder_ffn_dim,
        encoder_points: fields.encoder_n_points,
        decoder_points: fields.decoder_n_points,
        activation: fields.activation_function,
        queries: fields.num_queries,
        temperature: fields.positional_embedding_temperature,
        norm_eps: fields.layer_norm_eps,
    })
}

/// Renames `GroundingDinoMultiheadAttention`'s projections under `from` to [`Attention`](crate::assembly::Attention)'s under `to`.
fn multihead_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("query", "query"),
            ("key", "key"),
            ("value", "value"),
            ("out_proj", "output"),
        ],
    )
}

/// Returns the rules that import a checkpoint whose backbone gives `config`'s last stages, whose norms `transformers` names after their stages from one.
pub(crate) fn rules(config: &GroundingDinoConfig) -> Rules {
    let backbone = "model.backbone.conv_encoder.model";
    let encoder = "model.encoder.layers.*";
    let decoder = "model.decoder.layers.*";
    let mut rules = swin_rules(
        &format!("{backbone}.encoder"),
        &format!("{backbone}.embeddings.patch_embeddings.projection"),
        &format!("{backbone}.embeddings.norm"),
        "backbone",
    )
    .then(post_norm_encoder_rules("model.text_backbone", "text"))
    .rename_each(
        "model.input_proj_vision.*",
        "projections.*",
        &[("0", "convolution"), ("1", "norm")],
    );
    let stages = config.backbone.stages.len();
    for (index, stage) in (stages + 1 - config.backbone_levels..=stages).enumerate() {
        rules = rules.rename(
            &format!("{backbone}.hidden_states_norms.stage{stage}"),
            &format!("stage_norms.{index}"),
        );
    }
    rules
        .rename("model.level_embed", "level_embeddings")
        .rename("model.text_projection", "text_projection")
        .rename_each(
            &format!("{encoder}.fusion_layer"),
            "encoder.*.fusion",
            &[
                ("layer_norm_vision", "vision_norm"),
                ("layer_norm_text", "text_norm"),
                ("vision_param", "vision_scale"),
                ("text_param", "text_scale"),
                ("attn.vision_proj", "vision"),
                ("attn.text_proj", "text"),
                ("attn.values_vision_proj", "vision_values"),
                ("attn.values_text_proj", "text_values"),
                ("attn.out_vision_proj", "vision_output"),
                ("attn.out_text_proj", "text_output"),
            ],
        )
        .then(multihead_rules(
            &format!("{encoder}.text_enhancer_layer.self_attn"),
            "encoder.*.text.attention",
        ))
        .rename_each(
            &format!("{encoder}.text_enhancer_layer"),
            "encoder.*.text",
            &[
                ("layer_norm_before", "attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("layer_norm_after", "feed_forward_norm"),
            ],
        )
        .then(deformable_encoder_rules(
            &format!("{encoder}.deformable_layer"),
            "encoder.*.deformable",
        ))
        .rename("model.enc_output_norm", "proposal_norm")
        .rename("model.enc_output", "proposal")
        .rename("model.encoder_output_bbox_embed.layers", "proposal_boxes")
        .rename("model.query_position_embeddings.weight", "targets")
        .rename(
            "model.decoder.reference_points_head.layers",
            "query_positions",
        )
        .rename("model.decoder.layer_norm", "decoder_norm")
        // Every decoder layer and the head share the first box head, which the checkpoint stores once.
        .rename("bbox_embed.0.layers", "boxes")
        .then(multihead_rules(
            &format!("{decoder}.self_attn"),
            "decoder.*.attention",
        ))
        .then(multihead_rules(
            &format!("{decoder}.encoder_attn_text"),
            "decoder.*.text_attention",
        ))
        .then(deformable_attention_rules(
            &format!("{decoder}.encoder_attn"),
            "decoder.*.cross_attention",
        ))
        .rename_each(
            decoder,
            "decoder.*",
            &[
                ("self_attn_layer_norm", "attention_norm"),
                ("encoder_attn_text_layer_norm", "text_attention_norm"),
                ("encoder_attn_layer_norm", "cross_attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
}
