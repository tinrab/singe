//! Reads a `transformers` `ViT` config and imports its checkpoint, whose classifier stores the encoder under `vit.`.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{PreNormEncoderConfig, VisionTowerConfig, vit_layer_rules},
    error::{Error, Result},
    format::{
        RuleSet, join,
        transformers::{activation_field, classifier_config, default_channels, default_true},
    },
    model::vision::vit::VitConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct VitFields {
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    #[serde(deserialize_with = "activation_field")]
    pub hidden_act: Activation,
    pub layer_norm_eps: f32,
    pub patch_size: i64,
    #[serde(default = "default_channels")]
    pub num_channels: i64,
    #[serde(default = "default_true")]
    pub qkv_bias: bool,
}

impl VitFields {
    /// Returns the blocks' config, refusing a checkpoint without query, key, and value biases, since the blocks always have them.
    pub(crate) fn encoder(&self) -> Result<PreNormEncoderConfig> {
        if !self.qkv_bias {
            return Err(Error::unsupported(
                "a ViT without query, key, and value biases",
            ));
        }

        Ok(PreNormEncoderConfig {
            width: self.hidden_size,
            heads: self.num_attention_heads,
            inner: self.intermediate_size,
            activation: self.hidden_act,
            eps: self.layer_norm_eps,
            projection_biases: [true; 4],
            causal: false,
            layer_scale: false,
        })
    }

    pub(crate) fn tower(&self, image_size: i64) -> Result<VisionTowerConfig> {
        Ok(VisionTowerConfig {
            encoder: self.encoder()?,
            layers: self.num_hidden_layers,
            image_size,
            patch_size: self.patch_size,
            channels: self.num_channels,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    vit: VitFields,
    image_size: i64,
}

pub(crate) fn config(source: &Source) -> Result<VitConfig> {
    let fields: Fields = source.json("config.json")?;
    Ok(VitConfig {
        tower: fields.vit.tower(fields.image_size)?,
        classifier: classifier_config(source)?,
    })
}

/// Returns the rules that import a `ViT` encoder stored under `from`, its embeddings, `encoder.layer.N`, and a final `layernorm`, to the schema's names under `to`.
pub(crate) fn encoder_rules(from: &str, to: &str) -> RuleSet {
    RuleSet::new()
        .rename_each(
            &join(from, "embeddings"),
            &join(to, "embeddings"),
            &[
                ("cls_token", "class"),
                ("position_embeddings", "positions"),
                ("patch_embeddings.projection", "patches"),
            ],
        )
        .then(vit_layer_rules(
            &join(from, "encoder.layer"),
            &join(to, "blocks"),
        ))
        .rename(&join(from, "layernorm"), &join(to, "norm"))
}

pub(crate) fn rules() -> RuleSet {
    encoder_rules("vit", "")
        .rename("classifier", "classifier.output")
        // The pooler serves the base model's pooled output, which no head here reads.
        .skip("vit.pooler")
}
