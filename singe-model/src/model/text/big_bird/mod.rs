//! `BigBird`: BERT's post-norm encoder, which `transformers` runs with full attention up to `(5 + 2 · random blocks) · block size` tokens and block-sparse attention past them.

mod transformers;

use crate::{head::encoder_heads, model::ModelEntry, source::Format};

#[derive(Debug, Clone, Copy)]
pub struct BigBird;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "big_bird",
    formats: &[(Format::Transformers, &["big_bird"])],
    loaders: &encoder_heads::<BigBird>(),
};
