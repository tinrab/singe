//! Reads a `transformers` `SuperPoint` config and imports its checkpoint.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    format::Rules,
    head::KeypointConfig,
    model::vision::superpoint::SuperPointConfig,
    source::Source,
};

/// A `SuperPoint` config's fields, with `SuperPointConfig`'s defaults, which `SuperGlue`'s nested config leaves to them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub(crate) struct Fields {
    encoder_hidden_sizes: Vec<i64>,
    decoder_hidden_size: i64,
    keypoint_decoder_dim: i64,
    descriptor_decoder_dim: i64,
    keypoint_threshold: f32,
    max_keypoints: i64,
    nms_radius: usize,
    border_removal_distance: usize,
}

impl Default for Fields {
    fn default() -> Self {
        Self {
            encoder_hidden_sizes: vec![64, 64, 128, 128],
            decoder_hidden_size: 256,
            keypoint_decoder_dim: 65,
            descriptor_decoder_dim: 256,
            keypoint_threshold: 0.005,
            max_keypoints: -1,
            nms_radius: 4,
            border_removal_distance: 4,
        }
    }
}

pub(crate) fn config(source: &Source) -> Result<SuperPointConfig> {
    fields_config(&source.json("config.json")?)
}

pub(crate) fn fields_config(fields: &Fields) -> Result<SuperPointConfig> {
    // Three halvings and a cell of 8 × 8 pixels and one class for none.
    if fields.encoder_hidden_sizes.len() != 4 {
        return Err(Error::unsupported(format!(
            "a SuperPoint encoder of {} blocks",
            fields.encoder_hidden_sizes.len()
        )));
    }

    if fields.keypoint_decoder_dim != 65 {
        return Err(Error::unsupported(format!(
            "a SuperPoint keypoint decoder of {} classes",
            fields.keypoint_decoder_dim
        )));
    }

    Ok(SuperPointConfig {
        encoder: fields.encoder_hidden_sizes.clone(),
        decoder: fields.decoder_hidden_size,
        descriptor: fields.descriptor_decoder_dim,
        keypoints: KeypointConfig {
            threshold: fields.keypoint_threshold,
            max_keypoints: usize::try_from(fields.max_keypoints).ok(),
            radius: fields.nms_radius,
            border: fields.border_removal_distance,
        },
    })
}

/// Returns the rules that import the detector stored under `from`, as `LightGlue`'s checkpoints store it under `keypoint_detector`, to the schema's names under `to`.
pub(crate) fn detector_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("encoder.conv_blocks.*.conv_a", "blocks.*.first"),
            ("encoder.conv_blocks.*.conv_b", "blocks.*.second"),
            ("keypoint_decoder.conv_score_a", "score_hidden"),
            ("keypoint_decoder.conv_score_b", "score"),
            ("descriptor_decoder.conv_descriptor_a", "descriptor_hidden"),
            ("descriptor_decoder.conv_descriptor_b", "descriptor"),
        ],
    )
}

pub(crate) fn rules() -> Rules {
    detector_rules("", "")
}
