// Reads a `transformers` ViT config and imports its checkpoint, whose classifier stores the encoder under `vit.`.

use serde::Deserialize;
use singe::nn::Activation;

use super::VitConfig;
use crate::{
    assembly::{PreNormEncoderSpec, vit_layer_rules},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config},
    },
    source::Source,
};

fn yes() -> bool {
    true
}

// `ViTConfig`'s default, which configs of models that read no images, such as AST's, leave out.
const fn three() -> i64 {
    3
}

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
    #[serde(default = "three")]
    pub num_channels: i64,
    #[serde(default = "yes")]
    pub qkv_bias: bool,
}

impl VitFields {
    // The blocks' spec; a checkpoint without query, key, and value biases is refused, since the blocks always have them.
    pub(crate) fn encoder(&self) -> Result<PreNormEncoderSpec> {
        if !self.qkv_bias {
            return Err(Error::unsupported(
                "a ViT without query, key, and value biases",
            ));
        }

        Ok(PreNormEncoderSpec {
            width: self.hidden_size,
            heads: self.num_attention_heads,
            inner: self.intermediate_size,
            activation: self.hidden_act,
            eps: self.layer_norm_eps,
            key_bias: true,
            causal: false,
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
        encoder: fields.vit.encoder()?,
        layers: fields.vit.num_hidden_layers,
        image_size: fields.image_size,
        patch_size: fields.vit.patch_size,
        channels: fields.vit.num_channels,
        classifier: classifier_config(source)?,
    })
}

// Imports a ViT encoder stored under `prefix`: its embeddings, `encoder.layer.N`, and a final `layernorm`, to the schema's names under `to`.
pub(crate) fn encoder_rules(prefix: &str, to: &str) -> Rules {
    let target = |name: &str| format!("{to}{name}");
    Rules::new()
        .rename(
            &format!("{prefix}embeddings.cls_token"),
            &target("embeddings.class"),
        )
        .rename(
            &format!("{prefix}embeddings.position_embeddings"),
            &target("embeddings.positions"),
        )
        .rename(
            &format!("{prefix}embeddings.patch_embeddings.projection"),
            &target("embeddings.patches"),
        )
        .then(vit_layer_rules(
            &format!("{prefix}encoder.layer"),
            &target("blocks"),
        ))
        .rename(&format!("{prefix}layernorm"), &target("norm"))
}

pub(crate) fn rules() -> Rules {
    encoder_rules("vit.", "")
        .rename("classifier", "classifier.output")
        // The pooler serves the base model's pooled output, which no head here reads.
        .skip("vit.pooler")
}
