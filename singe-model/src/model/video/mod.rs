//! Models `transformers` documents among its video models.

pub mod videomae;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&videomae::MODEL];
