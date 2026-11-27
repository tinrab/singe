//! Reads a `transformers` `SigLIP` config, whose towers take every field `SiglipVisionConfig` and `SiglipTextConfig` default, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{PreNormEncoderConfig, VisionTowerConfig, clip_layer_rules},
    error::Result,
    format::{RuleSet, transformers::activation_field},
    model::multimodal::siglip::{SiglipConfig, SiglipTextConfig},
    source::Source,
};

/// The fields both towers share, with `transformers`' defaults.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
struct TowerFields {
    hidden_size: i64,
    intermediate_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    num_channels: i64,
    image_size: i64,
    patch_size: i64,
    vocab_size: i64,
    max_position_embeddings: i64,
    num_patches: i64,
}

impl Default for TowerFields {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            hidden_act: Activation::Gelu,
            layer_norm_eps: 1e-6,
            num_channels: 3,
            image_size: 224,
            patch_size: 16,
            vocab_size: 32000,
            max_position_embeddings: 64,
            num_patches: 256,
        }
    }
}

impl TowerFields {
    const fn encoder(&self) -> PreNormEncoderConfig {
        PreNormEncoderConfig {
            width: self.hidden_size,
            heads: self.num_attention_heads,
            inner: self.intermediate_size,
            activation: self.hidden_act,
            eps: self.layer_norm_eps,
            projection_biases: [true; 4],
            causal: false,
            layer_scale: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    model_type: String,
    #[serde(default)]
    text_config: TowerFields,
    #[serde(default)]
    vision_config: TowerFields,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct PaddingFields {
    pad_id: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TokenizerFields {
    padding: Option<PaddingFields>,
}

pub(crate) fn config(source: &Source) -> Result<SiglipConfig> {
    let Fields {
        model_type,
        text_config: text,
        vision_config: vision,
    } = source.json("config.json")?;
    // NaFlex's positions are learned on a square grid of `num_patches`.
    let flexible_grid =
        (model_type == "siglip2").then(|| (vision.num_patches as f64).sqrt() as i64);
    // The id the tokenizer pads with, which the text tower reads at every padded position.
    let pad = source
        .optional_json::<TokenizerFields>("tokenizer.json")?
        .and_then(|tokenizer| tokenizer.padding)
        .map_or(0, |padding| padding.pad_id);
    Ok(SiglipConfig {
        text: SiglipTextConfig {
            encoder: text.encoder(),
            layers: text.num_hidden_layers,
            vocab_size: text.vocab_size,
            max_positions: text.max_position_embeddings,
            pad,
        },
        vision: VisionTowerConfig {
            encoder: vision.encoder(),
            layers: vision.num_hidden_layers,
            image_size: vision.image_size,
            patch_size: vision.patch_size,
            channels: vision.num_channels,
        },
        flexible_grid,
    })
}

pub(crate) fn rules(config: &SiglipConfig) -> RuleSet {
    let head = "vision_model.head";
    let positions = "vision_model.embeddings.position_embedding.weight";
    let positions = if config.flexible_grid.is_some() {
        RuleSet::new().rename(positions, "vision.embeddings.positions")
    } else {
        RuleSet::new().derive(positions, "vision.embeddings.positions", |positions| {
            positions.unsqueeze(0)
        })
    };
    positions
        .rename(
            "text_model.embeddings.token_embedding",
            "text.embeddings.tokens",
        )
        .rename(
            "text_model.embeddings.position_embedding",
            "text.embeddings.positions",
        )
        .then(clip_layer_rules("text_model.encoder.layers", "text.blocks"))
        .rename("text_model.final_layer_norm", "text.norm")
        .rename("text_model.head", "text.head")
        .rename(
            "vision_model.embeddings.patch_embedding",
            "vision.embeddings.patches",
        )
        .then(clip_layer_rules(
            "vision_model.encoder.layers",
            "vision.blocks",
        ))
        .rename("vision_model.post_layernorm", "vision.post_norm")
        .split(
            &format!("{head}.attention.in_proj_weight"),
            0,
            &[
                "vision.pooling.attention.query.weight",
                "vision.pooling.attention.key.weight",
                "vision.pooling.attention.value.weight",
            ],
        )
        .split(
            &format!("{head}.attention.in_proj_bias"),
            0,
            &[
                "vision.pooling.attention.query.bias",
                "vision.pooling.attention.key.bias",
                "vision.pooling.attention.value.bias",
            ],
        )
        .rename_each(
            head,
            "vision.pooling",
            &[
                ("probe", "probe"),
                ("attention.out_proj", "attention.output"),
                ("layernorm", "norm"),
                ("mlp.fc1", "feed_forward.up"),
                ("mlp.fc2", "feed_forward.down"),
            ],
        )
        .rename("logit_scale", "logit_scale")
        .rename("logit_bias", "logit_bias")
}
