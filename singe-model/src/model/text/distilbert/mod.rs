//! `DistilBERT`: [`assembly`](crate::assembly)'s post-norm encoder without token types, serving every head over a text encoder.

mod transformers;

use crate::{head::encoder_heads, model::ModelEntry, source::Format};

#[derive(Debug, Clone, Copy)]
pub struct DistilBert;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "distilbert",
    formats: &[(Format::Transformers, &["distilbert"])],
    loaders: &encoder_heads::<DistilBert>(),
};
