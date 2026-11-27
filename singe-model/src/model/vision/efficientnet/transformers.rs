//! Reads a `transformers` `EfficientNet` config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config},
    },
    model::vision::{
        efficientnet::{BlockConfig, EfficientNetConfig},
        mobilenet_v2::make_divisible,
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    width_coefficient: f64,
    depth_coefficient: f64,
    depth_divisor: i64,
    kernel_sizes: Vec<i64>,
    in_channels: Vec<i64>,
    out_channels: Vec<i64>,
    strides: Vec<i64>,
    num_block_repeats: Vec<i64>,
    expand_ratios: Vec<i64>,
    squeeze_expansion_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    hidden_dim: i64,
    batch_norm_eps: f32,
    pooling_type: String,
    depthwise_padding: Vec<usize>,
    #[serde(default = "default_channels")]
    num_channels: i64,
}

const fn default_channels() -> i64 {
    3
}

pub(crate) fn config(source: &Source) -> Result<EfficientNetConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.pooling_type != "mean"
        || !fields.depthwise_padding.is_empty()
        || fields.num_channels != 3
    {
        return Err(Error::unsupported(
            "an EfficientNet pooling other than the mean, padding depthwise convolutions symmetrically, or of other than three input channels",
        ));
    }

    let stages = fields.in_channels.len();
    let stage_lists = [
        &fields.out_channels,
        &fields.kernel_sizes,
        &fields.strides,
        &fields.num_block_repeats,
        &fields.expand_ratios,
    ];
    if stage_lists.iter().any(|values| values.len() != stages) {
        return Err(Error::unsupported(
            "an EfficientNet whose per-stage lists differ in length",
        ));
    }

    // `round_filters`: channels scaled by the width coefficient, at least the divisor.
    let round = |channels: i64| {
        make_divisible(
            channels as f64 * fields.width_coefficient,
            fields.depth_divisor,
            fields.depth_divisor,
        )
    };
    let mut blocks = Vec::new();
    for stage in 0..stages {
        let output = round(fields.out_channels[stage]);
        let repeats =
            (fields.depth_coefficient * fields.num_block_repeats[stage] as f64).ceil() as i64;
        for repeat in 0..repeats {
            let first = repeat == 0;
            blocks.push(BlockConfig {
                input: if first {
                    round(fields.in_channels[stage])
                } else {
                    output
                },
                output,
                expansion: fields.expand_ratios[stage],
                kernel: fields.kernel_sizes[stage],
                stride: if first { fields.strides[stage] } else { 1 },
                residual: !first,
            });
        }
    }
    Ok(EfficientNetConfig {
        stem: round(32),
        blocks,
        last: round(fields.hidden_dim),
        squeeze_ratio: fields.squeeze_expansion_ratio,
        activation: fields.hidden_act,
        eps: fields.batch_norm_eps,
        classifier: classifier_config(source)?,
    })
}

pub(crate) fn rules() -> Rules {
    let blocks = "efficientnet.encoder.blocks.*";
    // Counts training updates, which evaluation never reads.
    let mut rules = Rules::new()
        .skip("efficientnet.embeddings.batchnorm.num_batches_tracked")
        .skip("efficientnet.encoder.top_bn.num_batches_tracked");
    for norm in [
        "depthwise_conv.depthwise_norm",
        "expansion.expand_bn",
        "projection.project_bn",
    ] {
        rules = rules.skip(&format!("{blocks}.{norm}.num_batches_tracked"));
    }

    for (from, to) in [
        ("efficientnet.embeddings.convolution", "stem.convolution"),
        ("efficientnet.embeddings.batchnorm", "stem.norm"),
        ("efficientnet.encoder.top_conv", "top.convolution"),
        ("efficientnet.encoder.top_bn", "top.norm"),
    ] {
        rules = rules.rename(from, to);
    }
    rules.rename("classifier", "classifier").rename_each(
        blocks,
        "blocks.*",
        &[
            ("expansion.expand_conv", "expand.convolution"),
            ("expansion.expand_bn", "expand.norm"),
            ("depthwise_conv.depthwise_conv", "depthwise.convolution"),
            ("depthwise_conv.depthwise_norm", "depthwise.norm"),
            ("squeeze_excite.reduce", "squeeze"),
            ("squeeze_excite.expand", "excite"),
            ("projection.project_conv", "project.convolution"),
            ("projection.project_bn", "project.norm"),
        ],
    )
}
