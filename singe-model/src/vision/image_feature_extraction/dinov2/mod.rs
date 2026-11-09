use std::path::Path;

use serde::Deserialize;
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        Activation, Conv2dConfig, DenseAttentionInput, Layer, LayerNorm, LayerScale, Mlp, Model,
        ModelState, PatchEmbedding, PatchTokens, PreNormBlock, SelfAttention, StateBuilder,
    },
};

use crate::{
    assembly::load_weights,
    config::{Snapshot, unsupported},
    error::Result,
    schema::Schema,
};

const fn enabled() -> bool {
    true
}

// The fields of a Hugging Face DINOv2 `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Dinov2Config {
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub mlp_ratio: f64,
    pub hidden_act: Activation,
    pub layer_norm_eps: f32,
    pub image_size: i64,
    pub patch_size: i64,
    pub num_channels: i64,
    #[serde(default = "enabled")]
    pub qkv_bias: bool,
    #[serde(default)]
    pub use_swiglu_ffn: bool,
}

impl Dinov2Config {
    fn validate(&self) -> Result<()> {
        if self.use_swiglu_ffn {
            return Err(unsupported("a SwiGLU feed-forward"));
        }

        if self.patch_size <= 0 || self.image_size % self.patch_size != 0 {
            return Err(unsupported("an image size that is not whole patches"));
        }
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "`transformers` truncates the product to an integer the same way"
    )]
    fn inner(&self) -> i64 {
        (self.hidden_size as f64 * self.mlp_ratio) as i64
    }

    fn tokens(&self) -> i64 {
        let side = self.image_size / self.patch_size;
        side * side + 1
    }
}

type Block = PreNormBlock<SelfAttention, Mlp, LayerNorm>;

// DINOv2: a class token and patches at the trained image size, pre-norm blocks with layer scale, and a final norm.
// It returns every token's final hidden state, `[B, 1 + N, D]`, of which the class token's is the image's embedding.
#[derive(Debug, Clone)]
pub struct Dinov2Layers {
    embeddings: PatchTokens,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

pub type Dinov2 = Model<Dinov2Layers>;

impl Layer for Dinov2Layers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.apply(&self.embeddings, state)?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: None,
                },
                state,
            )?;
        }
        hidden.apply(&self.norm, state)
    }
}

// Each image's class token, `[B, D]`, which `transformers` returns as the pooled output.
pub fn class_embeddings(hidden: &Tensor) -> SingeResult<Tensor> {
    hidden.narrow(1, 0, 1)?.squeeze(1)
}

fn block(schema: &mut Schema<'_>, config: &Dinov2Config) -> SingeResult<Block> {
    let width = config.hidden_size;
    let eps = config.layer_norm_eps;
    let attention_norm = schema.layer_norm("norm1", width, eps)?;
    let attention = {
        let mut schema = schema.scope("attention");
        let [query, key, value] = {
            let mut schema = schema.scope("attention");
            [
                schema.linear("query", width, width, config.qkv_bias)?,
                schema.linear("key", width, width, config.qkv_bias)?,
                schema.linear("value", width, width, config.qkv_bias)?,
            ]
        };
        SelfAttention::create(
            query,
            key,
            value,
            schema.linear("output.dense", width, width, true)?,
            config.num_attention_heads,
            width,
        )?
    };
    let attention_scale = LayerScale::new(schema.parameter("layer_scale1.lambda1", [width])?);
    let feed_forward_norm = schema.layer_norm("norm2", width, eps)?;
    let feed_forward = Mlp::new(
        schema.linear("mlp.fc1", width, config.inner(), true)?,
        schema.linear("mlp.fc2", config.inner(), width, true)?,
        config.hidden_act,
    );
    let feed_forward_scale = LayerScale::new(schema.parameter("layer_scale2.lambda1", [width])?);
    Ok(
        PreNormBlock::new(attention_norm, attention, feed_forward_norm, feed_forward)
            .with_layer_scales(attention_scale, feed_forward_scale),
    )
}

fn schema(config: &Dinov2Config, data_type: DataType) -> Result<Dinov2> {
    let width = config.hidden_size;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut model = Schema::root(&mut builder, data_type);
        let embeddings = {
            let mut schema = model.scope("embeddings");
            let projection = schema.conv2d(
                "patch_embeddings.projection",
                config.num_channels,
                width,
                config.patch_size,
                Conv2dConfig {
                    stride: (config.patch_size, config.patch_size),
                    ..Conv2dConfig::default()
                },
                true,
            )?;
            PatchTokens::new(PatchEmbedding::new(projection))
                .with_class_token(schema.parameter("cls_token", [1, 1, width])?)
                .with_positions(
                    schema.parameter("position_embeddings", [1, config.tokens(), width])?,
                )
        };
        let blocks = (0..config.num_hidden_layers)
            .map(|layer| block(&mut model.scope(format!("encoder.layer.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("layernorm", width, config.layer_norm_eps)?;
        Dinov2Layers {
            embeddings,
            blocks,
            norm,
        }
    };
    Ok(Model::new(layers, builder.finish()?))
}

// Checkpoints of a classifier store the encoder under `dinov2.`, and the mask token serves only masked pretraining.
fn state_name(name: &str) -> Option<String> {
    let name = name.strip_prefix("dinov2.").unwrap_or(name);
    if name.starts_with("classifier.") || name == "embeddings.mask_token" {
        return None;
    }
    Some(name.to_owned())
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Dinov2> {
    let config: Dinov2Config = Snapshot::new(&snapshot).config()?;
    config.validate()?;
    load_weights(
        &schema(&config, data_type)?,
        snapshot,
        device,
        Some(&state_name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_names_map_to_the_encoder_and_pretraining_tensors_are_ignored() {
        assert_eq!(
            state_name("dinov2.encoder.layer.0.mlp.fc1.weight").unwrap(),
            "encoder.layer.0.mlp.fc1.weight"
        );
        assert_eq!(state_name("classifier.weight"), None);
        assert_eq!(state_name("embeddings.mask_token"), None);
    }
}
