//! Models `transformers` documents among its time series models.

pub mod autoformer;
mod lagged;
pub mod patchtst;
pub mod time_series_transformer;
pub mod timesfm2_5;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &autoformer::MODEL,
    &patchtst::MODEL,
    &time_series_transformer::MODEL,
    &timesfm2_5::MODEL,
];
