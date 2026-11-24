//! BERT: [`assembly`](crate::assembly)'s post-norm encoder, serving every head over a text encoder.

mod transformers;

use crate::{head::encoder_heads, model::ModelEntry, source::Format};

#[derive(Debug, Clone, Copy)]
pub struct Bert;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "bert",
    formats: &[(Format::Transformers, &["bert"])],
    loaders: &encoder_heads::<Bert>(),
};
