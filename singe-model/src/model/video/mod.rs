//! Models `transformers` documents among its video models.

pub mod timesformer;
pub mod videomae;
pub mod vjepa2;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&timesformer::MODEL, &videomae::MODEL, &vjepa2::MODEL];
