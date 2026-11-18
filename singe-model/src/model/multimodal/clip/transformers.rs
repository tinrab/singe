// Reads a `transformers` CLIP config and imports its checkpoint; OWL-ViT's stores the same towers under `owlvit.`.

use serde::Deserialize;
use singe::nn::Activation;

use super::{ClipConfig, ClipTextConfig, ClipVisionConfig};
use crate::{
    assembly::{PreNormEncoderSpec, clip_layer_rules},
    error::Result,
    format::{Rules, transformers::activation_field},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct TowerFields {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    intermediate_size: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
}

impl TowerFields {
    fn encoder(&self, causal: bool) -> PreNormEncoderSpec {
        PreNormEncoderSpec {
            width: self.hidden_size,
            heads: self.num_attention_heads,
            inner: self.intermediate_size,
            activation: self.hidden_act,
            eps: self.layer_norm_eps,
            key_bias: true,
            causal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct TextFields {
    #[serde(flatten)]
    tower: TowerFields,
    vocab_size: i64,
    max_position_embeddings: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct VisionFields {
    #[serde(flatten)]
    tower: TowerFields,
    image_size: i64,
    patch_size: i64,
    #[serde(default = "three")]
    num_channels: i64,
}

const fn three() -> i64 {
    3
}

fn text_config(fields: &TextFields) -> ClipTextConfig {
    ClipTextConfig {
        encoder: fields.tower.encoder(true),
        layers: fields.tower.num_hidden_layers,
        vocab_size: fields.vocab_size,
        positions: fields.max_position_embeddings,
    }
}

fn vision_config(fields: &VisionFields) -> ClipVisionConfig {
    ClipVisionConfig {
        encoder: fields.tower.encoder(false),
        layers: fields.tower.num_hidden_layers,
        image_size: fields.image_size,
        patch_size: fields.patch_size,
        channels: fields.num_channels,
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    text_config: TextFields,
    vision_config: VisionFields,
    projection_dim: i64,
}

pub(crate) fn config(source: &Source) -> Result<ClipConfig> {
    let fields: Fields = source.json("config.json")?;
    Ok(ClipConfig {
        text: text_config(&fields.text_config),
        vision: vision_config(&fields.vision_config),
        projection: fields.projection_dim,
    })
}

// Imports CLIP's towers, projections, and logit scale stored after `prefix`, reshaping the class embedding and the vision positions to the layout `PatchTokens` reads.
pub(crate) fn clip_rules(prefix: &str) -> Rules {
    text_rules(&format!("{prefix}text_model"), "text.")
        .then(vision_rules(&format!("{prefix}vision_model"), "vision."))
        .rename(&format!("{prefix}visual_projection"), "image_projection")
        .rename(&format!("{prefix}text_projection"), "text_projection")
        .rename(&format!("{prefix}logit_scale"), "logit_scale")
}

// Imports a text tower stored under `from` (such as `text_model`) into the tower whose names start with `to` (such as `text.`, or nothing for the tower alone).
pub(crate) fn text_rules(from: &str, to: &str) -> Rules {
    let text = |name: &str| format!("{from}.{name}");
    Rules::new()
        .rename(
            &text("embeddings.token_embedding"),
            &format!("{to}embeddings.tokens"),
        )
        .rename(
            &text("embeddings.position_embedding"),
            &format!("{to}embeddings.positions"),
        )
        // A buffer of the positions an input leaves out, which the embeddings compute.
        .skip(&text("embeddings.position_ids"))
        .then(clip_layer_rules(
            &text("encoder.layers"),
            &format!("{to}blocks"),
        ))
        .rename(&text("final_layer_norm"), &format!("{to}norm"))
}

// A `CLIPTextModel`'s config, the text tower's fields at its top.
pub(crate) fn text_model_config(source: &Source) -> Result<ClipTextConfig> {
    Ok(text_config(&source.json::<TextFields>("config.json")?))
}

// The width a `CLIPTextModelWithProjection` projects into.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct ProjectionFields {
    pub projection_dim: i64,
}

// Imports a vision tower stored under `from` (such as `vision_model`) into the tower whose names start with `to`, reshaping the class embedding and the positions to the layout `PatchTokens` reads.
pub(crate) fn vision_rules(from: &str, to: &str) -> Rules {
    let vision = |name: &str| format!("{from}.{name}");
    let target = |name: &str| format!("{to}{name}");
    Rules::new()
        // A buffer of the positions an input leaves out, which the embeddings compute.
        .skip(&vision("embeddings.position_ids"))
        .rename(
            &vision("embeddings.patch_embedding"),
            &target("embeddings.patches"),
        )
        .derive(
            &vision("embeddings.class_embedding"),
            &target("embeddings.class"),
            |class| class.unsqueeze(0)?.unsqueeze(0),
        )
        .derive(
            &vision("embeddings.position_embedding.weight"),
            &target("embeddings.positions"),
            |positions| positions.unsqueeze(0),
        )
        // CLIP's checkpoints spell it `pre_layrnorm`, and OWL-ViT's `pre_layernorm`.
        .rename(&vision("pre_layrnorm"), &target("pre_norm"))
        .rename(&vision("pre_layernorm"), &target("pre_norm"))
        .then(clip_layer_rules(
            &vision("encoder.layers"),
            &target("blocks"),
        ))
        .rename(&vision("post_layernorm"), &target("post_norm"))
}

// A `CLIPVisionModel`'s config, the vision tower's fields at its top.
pub(crate) fn vision_model_config(source: &Source) -> Result<ClipVisionConfig> {
    Ok(vision_config(&source.json::<VisionFields>("config.json")?))
}
