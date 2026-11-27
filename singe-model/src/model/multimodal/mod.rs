//! Models `transformers` documents among its multimodal models.

pub mod clip;
pub mod colmodernvbert;
pub mod florence2;
pub mod grounding_dino;
pub mod janus;
pub mod owlvit;
pub mod siglip;
pub mod smolvlm;
pub mod vision_encoder_decoder;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &clip::MODEL,
    &clip::TEXT_MODEL,
    &colmodernvbert::MODEL,
    &florence2::MODEL,
    &grounding_dino::MODEL,
    &janus::MODEL,
    &owlvit::MODEL,
    &siglip::MODEL,
    &smolvlm::MODEL,
    &vision_encoder_decoder::MODEL,
];
