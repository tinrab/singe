//! Reads a `transformers` DETR config, whose backbone is timm's ResNet-50, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{ResNetConfig, attention_rules},
    error::{Error, Result},
    format::{
        RuleSet, join,
        transformers::{activation_field, classifier_config},
    },
    model::vision::detr::DetrConfig,
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
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    num_queries: i64,
    position_embedding_type: String,
}

pub(crate) fn config(source: &Source) -> Result<DetrConfig> {
    let fields: Fields = source.json("config.json")?;
    // timm's ResNet-50 without dilation, and sine positions, as `facebook/detr-resnet-50` has them.
    let backbone = fields.backbone.as_deref().unwrap_or_default();
    if backbone != "resnet50" {
        return Err(Error::unsupported(format!("DETR backbone `{backbone}`")));
    }

    if fields.dilation {
        return Err(Error::unsupported("a DETR backbone with dilation"));
    }

    if fields.position_embedding_type != "sine" {
        return Err(Error::unsupported(format!(
            "DETR position embedding type `{}`",
            fields.position_embedding_type
        )));
    }

    Ok(DetrConfig {
        backbone: resnet50(),
        width: fields.d_model,
        encoder_layers: fields.encoder_layers,
        decoder_layers: fields.decoder_layers,
        encoder_heads: fields.encoder_attention_heads,
        decoder_heads: fields.decoder_attention_heads,
        encoder_inner: fields.encoder_ffn_dim,
        decoder_inner: fields.decoder_ffn_dim,
        activation: fields.activation_function,
        queries: fields.num_queries,
        classifier: classifier_config(source)?,
    })
}

/// Returns timm's ResNet-50, the backbone of DETR's and Deformable DETR's checkpoints.
pub(crate) fn resnet50() -> ResNetConfig {
    ResNetConfig {
        depths: vec![3, 4, 6, 3],
        stem: 64,
    }
}

/// Imports timm's ResNet-50 as DETR's and Deformable DETR's checkpoints store it.
pub(crate) fn backbone_rules() -> RuleSet {
    RuleSet::new()
        // Counts training updates, which evaluation never reads.
        .skip("model.backbone.conv_encoder.model.*.*.downsample.1.num_batches_tracked")
        .rename("model.backbone.conv_encoder.model", "backbone")
}

pub(crate) fn rules() -> RuleSet {
    let mut rules = backbone_rules()
        .rename("model.input_projection", "projection")
        .rename("model.query_position_embeddings.weight", "queries")
        .rename("model.decoder.layernorm", "decoder_norm")
        .rename("class_labels_classifier", "classes")
        .rename("bbox_predictor.layers", "boxes");
    for part in ["encoder", "decoder"] {
        let layers = format!("model.{part}.layers.*");
        let blocks = format!("{part}.*");
        rules = rules
            .then(attention_rules(
                &join(&layers, "self_attn"),
                &join(&blocks, "attention"),
            ))
            .then(attention_rules(
                &join(&layers, "encoder_attn"),
                &join(&blocks, "cross_attention"),
            ))
            .rename_each(
                &layers,
                &blocks,
                &[
                    ("self_attn_layer_norm", "attention_norm"),
                    ("encoder_attn_layer_norm", "cross_attention_norm"),
                    ("fc1", "feed_forward.up"),
                    ("fc2", "feed_forward.down"),
                    ("final_layer_norm", "feed_forward_norm"),
                ],
            );
    }
    rules
}
