// Models `transformers` documents among its video models.

pub mod videomae;

use super::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&videomae::MODEL];
