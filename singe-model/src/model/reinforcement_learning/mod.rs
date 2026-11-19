// Models `transformers` documents among its reinforcement learning models.

pub mod decision_transformer;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&decision_transformer::MODEL];
