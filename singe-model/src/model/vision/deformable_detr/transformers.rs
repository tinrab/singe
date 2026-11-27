//! Reads a `transformers` Deformable DETR config, whose backbone is timm's ResNet-50, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{deformable_decoder_rules, deformable_encoder_rules},
    error::{Error, Result},
    format::{
        RuleSet,
        transformers::{activation_field, classifier_config, detection_scoring},
    },
    model::vision::{
        deformable_detr::DeformableDetrConfig,
        detr::transformers::{backbone_rules, resnet50},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone: Option<String>,
    #[serde(default)]
    dilation: bool,
    d_model: i64,
    encoder_layers: i64,
    decoder_layers: i64,
    encoder_attention_heads: i64,
    decoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    decoder_ffn_dim: i64,
    encoder_n_points: i64,
    decoder_n_points: i64,
    num_feature_levels: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    num_queries: i64,
    position_embedding_type: String,
    #[serde(default)]
    two_stage: bool,
    #[serde(default)]
    with_box_refine: bool,
}

pub(crate) fn config(source: &Source) -> Result<DeformableDetrConfig> {
    let fields: Fields = source.json("config.json")?;
    let backbone = fields.backbone.as_deref().unwrap_or_default();
    if backbone != "resnet50" || fields.dilation {
        return Err(Error::unsupported(format!(
            "Deformable DETR backbone `{backbone}`, other than timm's ResNet-50 without dilation"
        )));
    }

    if fields.position_embedding_type != "sine" {
        return Err(Error::unsupported(format!(
            "Deformable DETR position embedding type `{}`",
            fields.position_embedding_type
        )));
    }

    if fields.two_stage || fields.with_box_refine {
        return Err(Error::unsupported(
            "Deformable DETR with box refinement or two stages",
        ));
    }

    // One level reads the backbone's last stage; more read its last three, as `out_indices` (2, 3, 4) does, and stride the rest from the last.
    let backbone_levels = if fields.num_feature_levels > 1 { 3 } else { 1 };
    Ok(DeformableDetrConfig {
        backbone: resnet50(),
        backbone_levels,
        levels: fields.num_feature_levels,
        width: fields.d_model,
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
        classifier: classifier_config(source)?,
        scoring: detection_scoring(source)?,
    })
}

pub(crate) fn rules() -> RuleSet {
    backbone_rules()
        .rename_each(
            "model.input_proj.*",
            "projections.*",
            &[("0", "convolution"), ("1", "norm")],
        )
        .rename("model.level_embed", "level_embeddings")
        .rename("model.query_position_embeddings.weight", "queries")
        .rename("model.reference_points", "references")
        // Without box refinement every decoder layer shares the first layer's heads, and the checkpoint stores only those.
        .rename("class_embed.0", "classes")
        .rename("bbox_embed.0.layers", "boxes")
        .then(deformable_encoder_rules(
            "model.encoder.layers.*",
            "encoder.*",
        ))
        .then(deformable_decoder_rules(
            "model.decoder.layers.*",
            "decoder.*",
        ))
}
