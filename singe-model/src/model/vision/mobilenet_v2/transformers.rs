//! Reads a `transformers` `MobileNetV2` config and imports its checkpoint, for classification or for `DeepLabV3`'s segmentation.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config},
    },
    model::vision::mobilenet_v2::{MobileNetV2Config, make_divisible},
    source::Source,
};

/// The stem's output channels and each inverted residual's, as `MobileNetV2Model` fixes them.
const CHANNELS: &[i64] = &[
    16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320,
];

/// The stride of each inverted residual after the stem's, as `MobileNetV2Model` fixes it.
const STRIDES: &[i64] = &[2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1];

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    depth_multiplier: f64,
    depth_divisible_by: i64,
    min_depth: i64,
    expand_ratio: f64,
    output_stride: i64,
    first_layer_is_expansion: bool,
    finegrained_output: bool,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    tf_padding: bool,
    #[serde(default = "default_channels")]
    num_channels: i64,
}

const fn default_channels() -> i64 {
    3
}

pub(crate) fn config(source: &Source) -> Result<MobileNetV2Config> {
    let fields: Fields = source.json("config.json")?;
    if !fields.tf_padding || fields.num_channels != 3 {
        return Err(Error::unsupported(
            "a MobileNetV2 without TensorFlow's padding, or of other than three input channels",
        ));
    }

    let scaled = |channels: i64| {
        make_divisible(
            (channels as f64 * fields.depth_multiplier).round(),
            fields.depth_divisible_by,
            fields.min_depth,
        )
    };
    let last = if fields.finegrained_output && fields.depth_multiplier < 1.0 {
        1280
    } else {
        scaled(1280)
    };
    Ok(MobileNetV2Config {
        channels: CHANNELS.iter().map(|&channels| scaled(channels)).collect(),
        strides: STRIDES.to_vec(),
        stem: scaled(32),
        first_expands: !fields.first_layer_is_expansion,
        last,
        expansion: fields.expand_ratio,
        divisor: fields.depth_divisible_by,
        min_depth: fields.min_depth,
        output_stride: fields.output_stride,
        activation: fields.hidden_act,
        eps: fields.layer_norm_eps,
        classifier: classifier_config(source)?,
    })
}

/// Renames a `MobileNetV2ConvLayer` under `from` to a convolution and norm under `to`.
fn conv_rules(rules: Rules, from: &str, to: &str) -> Rules {
    rules
        .skip(&format!("{from}.normalization.num_batches_tracked"))
        .rename_each(
            from,
            to,
            &[("convolution", "convolution"), ("normalization", "norm")],
        )
}

/// Imports the backbone, which `transformers` stores under `mobilenet_v2`.
fn backbone_rules() -> Rules {
    let mut rules = Rules::new();
    for (from, to) in [
        ("conv_stem.first_conv", "stem"),
        ("conv_stem.expand_1x1", "first.expand"),
        ("conv_stem.conv_3x3", "first.depthwise"),
        ("conv_stem.reduce_1x1", "first.reduce"),
        ("layer.*.expand_1x1", "blocks.*.expand"),
        ("layer.*.conv_3x3", "blocks.*.depthwise"),
        ("layer.*.reduce_1x1", "blocks.*.reduce"),
    ] {
        rules = conv_rules(rules, &format!("mobilenet_v2.{from}"), to);
    }
    rules
}

/// Imports the backbone and the classifier.
pub(crate) fn classifier_rules() -> Rules {
    conv_rules(backbone_rules(), "mobilenet_v2.conv_1x1", "output")
        .rename("classifier", "classifier")
}

/// Imports the backbone and `DeepLabV3`'s head.
pub(crate) fn segmenter_rules() -> Rules {
    let mut rules = backbone_rules();
    for (from, to) in [
        ("conv_pool", "pool"),
        ("conv_aspp", "branch"),
        ("conv_projection", "projection"),
        ("classifier", "classifier"),
    ] {
        rules = conv_rules(
            rules,
            &format!("segmentation_head.{from}"),
            &format!("deeplab.{to}"),
        );
    }
    // The checkpoints also store the classifier's last convolution, which DeepLabV3's head does not read.
    rules.skip("mobilenet_v2.conv_1x1")
}
