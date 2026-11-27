//! Reads a `transformers` YOLOS config and imports its checkpoint, which stores the `ViT` encoder under `vit.`.

use serde::Deserialize;
use singe::shape::spatial::Spatial2dI64;

use crate::{
    error::{Error, Result},
    format::{
        RuleSet,
        transformers::{classifier_config, spatial_field},
    },
    model::vision::{
        vit::transformers::{VitFields, encoder_rules},
        yolos::YolosConfig,
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    vit: VitFields,
    #[serde(deserialize_with = "spatial_field")]
    image_size: Spatial2dI64,
    num_detection_tokens: i64,
    #[serde(default)]
    use_mid_position_embeddings: bool,
}

pub(crate) fn config(source: &Source) -> Result<YolosConfig> {
    let fields: Fields = source.json("config.json")?;
    // Position embeddings added between the blocks, which larger YOLOS models use, would load and then be left out.
    if fields.use_mid_position_embeddings {
        return Err(Error::unsupported("YOLOS's mid-position embeddings"));
    }

    Ok(YolosConfig {
        encoder: fields.vit.encoder()?,
        layers: fields.vit.num_hidden_layers,
        image_size: fields.image_size,
        patch_size: fields.vit.patch_size,
        channels: fields.vit.num_channels,
        detection_tokens: fields.num_detection_tokens,
        classifier: classifier_config(source)?,
    })
}

pub(crate) fn rules() -> RuleSet {
    encoder_rules("vit", "")
        .rename("vit.embeddings.detection_tokens", "embeddings.detection")
        .rename("class_labels_classifier.layers", "classes")
        .rename("bbox_predictor.layers", "boxes")
}
