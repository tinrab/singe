//! Reads a `transformers` `SegFormer` config, whose stage settings are lists of one value per stage, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config, default_true},
    },
    model::vision::segformer::{SegformerConfig, SegformerStageConfig},
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
    #[serde(default = "default_true")]
    reshape_last_stage: bool,
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
    {
        return Err(Error::unsupported(format!(
            "SegFormer stage lists of other than {count} entries, as `hidden_sizes` has"
        )));
    }

    if !fields.reshape_last_stage {
        return Err(Error::unsupported(
            "a SegFormer without `reshape_last_stage`",
        ));
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
    Rules::new()
        .rename_each(
            "segformer.encoder",
            "stages",
            &[
                ("patch_embeddings.*.proj", "*.patches"),
                ("patch_embeddings.*.layer_norm", "*.patch_norm"),
            ],
        )
        .rename_each(
            "segformer.encoder.block.*.*",
            "stages.*.blocks.*",
            &[
                ("layer_norm_1", "attention_norm"),
                ("attention.self.query", "attention.query"),
                ("attention.self.key", "attention.key"),
                ("attention.self.value", "attention.value"),
                ("attention.self.sr", "attention.reduction"),
                ("attention.self.layer_norm", "attention.reduction_norm"),
                ("attention.output.dense", "attention.output"),
                ("layer_norm_2", "feed_forward_norm"),
                ("mlp.dense1", "feed_forward.up"),
                ("mlp.dwconv.dwconv", "feed_forward.depthwise"),
                ("mlp.dense2", "feed_forward.down"),
            ],
        )
        .rename("segformer.encoder.layer_norm.*", "stages.*.norm")
        .rename("decode_head.linear_c.*.proj", "projections.*")
        .rename("decode_head.linear_fuse", "fuse")
        // A count training updates, which evaluation never reads.
        .skip("decode_head.batch_norm.num_batches_tracked")
        .rename("decode_head.batch_norm", "norm")
        .rename("decode_head.classifier", "classifier")
}
