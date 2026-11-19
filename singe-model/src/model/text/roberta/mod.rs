// RoBERTa: `assembly`'s post-norm encoder with positions numbered after the padding id, serving every head over a text encoder.

mod transformers;

use crate::{head::encoder_heads, model::ModelEntry, source::Format};

#[derive(Debug, Clone, Copy)]
pub struct Roberta;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "roberta",
    formats: &[(Format::Transformers, &["roberta", "xlm-roberta"])],
    loaders: &encoder_heads::<Roberta>(),
};
