// Reads a `transformers` SuperPoint config and imports its checkpoint.

use serde::Deserialize;

use super::SuperPointConfig;
use crate::{
    error::{Error, Result},
    format::Rules,
    head::KeypointConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
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

pub(crate) fn config(source: &Source) -> Result<SuperPointConfig> {
    fields_config(source.json("config.json")?)
}

pub(crate) fn fields_config(fields: Fields) -> Result<SuperPointConfig> {
    // Three halvings and a cell of 8 × 8 pixels and one class for none.
    if fields.encoder_hidden_sizes.len() != 4 || fields.keypoint_decoder_dim != 65 {
        return Err(Error::unsupported("this SuperPoint config"));
    }

    Ok(SuperPointConfig {
        encoder: fields.encoder_hidden_sizes,
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

// Imports the detector stored under `prefix`, as LightGlue's checkpoints store it under `keypoint_detector.`, to the schema's names under `to`.
pub(crate) fn detector_rules(prefix: &str, to: &str) -> Rules {
    let name = |name: &str| format!("{prefix}{name}");
    let target = |name: &str| format!("{to}{name}");
    Rules::new()
        .rename(
            &name("encoder.conv_blocks.*.conv_a"),
            &target("blocks.*.first"),
        )
        .rename(
            &name("encoder.conv_blocks.*.conv_b"),
            &target("blocks.*.second"),
        )
        .rename(
            &name("keypoint_decoder.conv_score_a"),
            &target("score_hidden"),
        )
        .rename(&name("keypoint_decoder.conv_score_b"), &target("score"))
        .rename(
            &name("descriptor_decoder.conv_descriptor_a"),
            &target("descriptor_hidden"),
        )
        .rename(
            &name("descriptor_decoder.conv_descriptor_b"),
            &target("descriptor"),
        )
}

pub(crate) fn rules() -> Rules {
    detector_rules("", "")
}
