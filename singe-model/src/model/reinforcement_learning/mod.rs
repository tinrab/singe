// Models `transformers` documents among its reinforcement learning models.

pub mod decision_transformer;

use super::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&decision_transformer::MODEL];
