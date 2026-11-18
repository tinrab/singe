// Models `transformers` documents among its time series models.

pub mod timesfm2_5;

use super::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&timesfm2_5::MODEL];
