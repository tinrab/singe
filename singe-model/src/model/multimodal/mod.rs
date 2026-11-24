//! Models `transformers` documents among its multimodal models.

pub mod clip;
pub mod colmodernvbert;
pub mod owlvit;
pub mod smolvlm;
pub mod vision_encoder_decoder;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &clip::MODEL,
    &clip::TEXT_MODEL,
    &colmodernvbert::MODEL,
    &owlvit::MODEL,
    &smolvlm::MODEL,
    &vision_encoder_decoder::MODEL,
];
