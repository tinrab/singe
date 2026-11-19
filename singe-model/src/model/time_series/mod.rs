// Models `transformers` documents among its time series models.

pub mod timesfm2_5;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&timesfm2_5::MODEL];
