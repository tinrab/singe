// Reads a `transformers` SegFormer config, whose stages are lists of one value per stage, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use super::{SegformerConfig, SegformerStageConfig};
use crate::{
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    hidden_sizes: Vec<i64>,
    depths: Vec<i64>,
    num_attention_heads: Vec<i64>,
    patch_sizes: Vec<i64>,
    strides: Vec<i64>,
    sr_ratios: Vec<i64>,
    mlp_ratios: Vec<i64>,
    num_channels: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    decoder_hidden_size: i64,
    #[serde(default = "enabled")]
    reshape_last_stage: bool,
}

const fn enabled() -> bool {
    true
}

pub(crate) fn config(source: &Source) -> Result<SegformerConfig> {
    let fields: Fields = source.json("config.json")?;
    let count = fields.hidden_sizes.len();
    if [
        fields.depths.len(),
        fields.num_attention_heads.len(),
        fields.patch_sizes.len(),
        fields.strides.len(),
        fields.sr_ratios.len(),
        fields.mlp_ratios.len(),
    ]
    .iter()
    .any(|&length| length != count)
        || !fields.reshape_last_stage
    {
        return Err(Error::unsupported("this SegFormer config"));
    }

    let stages = (0..count)
        .map(|index| SegformerStageConfig {
            width: fields.hidden_sizes[index],
            blocks: fields.depths[index],
            heads: fields.num_attention_heads[index],
            patch_size: fields.patch_sizes[index],
            stride: fields.strides[index],
            reduction: fields.sr_ratios[index],
            inner: fields.hidden_sizes[index] * fields.mlp_ratios[index],
        })
        .collect();
    Ok(SegformerConfig {
        stages,
        channels: fields.num_channels,
        activation: fields.hidden_act,
        decoder: fields.decoder_hidden_size,
        classifier: classifier_config(source)?,
    })
}

pub(crate) fn rules() -> Rules {
    let encoder = |name: &str| format!("segformer.encoder.{name}");
    let block = |name: &str| format!("segformer.encoder.block.*.*.{name}");
    Rules::new()
        .rename(&encoder("patch_embeddings.*.proj"), "stages.*.patches")
        .rename(
            &encoder("patch_embeddings.*.layer_norm"),
            "stages.*.patch_norm",
        )
        .rename(&block("layer_norm_1"), "stages.*.blocks.*.attention_norm")
        .rename(
            &block("attention.self.query"),
            "stages.*.blocks.*.attention.query",
        )
        .rename(
            &block("attention.self.key"),
            "stages.*.blocks.*.attention.key",
        )
        .rename(
            &block("attention.self.value"),
            "stages.*.blocks.*.attention.value",
        )
        .rename(
            &block("attention.self.sr"),
            "stages.*.blocks.*.attention.reduction",
        )
        .rename(
            &block("attention.self.layer_norm"),
            "stages.*.blocks.*.attention.reduction_norm",
        )
        .rename(
            &block("attention.output.dense"),
            "stages.*.blocks.*.attention.output",
        )
        .rename(
            &block("layer_norm_2"),
            "stages.*.blocks.*.feed_forward_norm",
        )
        .rename(&block("mlp.dense1"), "stages.*.blocks.*.feed_forward.up")
        .rename(
            &block("mlp.dwconv.dwconv"),
            "stages.*.blocks.*.feed_forward.depthwise",
        )
        .rename(&block("mlp.dense2"), "stages.*.blocks.*.feed_forward.down")
        .rename(&encoder("layer_norm.*"), "stages.*.norm")
        .rename("decode_head.linear_c.*.proj", "projections.*")
        .rename("decode_head.linear_fuse", "fuse")
        // A count training updates, which evaluation never reads.
        .skip("decode_head.batch_norm.num_batches_tracked")
        .rename("decode_head.batch_norm", "norm")
        .rename("decode_head.classifier", "classifier")
}
