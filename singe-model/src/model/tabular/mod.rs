// Models of tables, which `transformers` does not document; Mitra comes from AutoGluon.

pub mod mitra;

use super::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[&mitra::MODEL];
