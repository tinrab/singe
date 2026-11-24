//! Models of tables, which `transformers` does not document; Mitra comes from `AutoGluon`.

pub mod mitra;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&mitra::MODEL];
