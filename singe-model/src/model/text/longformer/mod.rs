//! Longformer: `RoBERTa`'s post-norm encoder whose tokens read only the keys within half its window on either side.
//! Global tokens, which read and are read by every token through their own projections, serve the classification and question-answering heads; without them only the masked language model and the bare encoder load.

mod transformers;

use crate::{
    head::{HeadLoader, MaskedLm, TextEncoder, load_encoder_head},
    model::ModelEntry,
    source::Format,
};

#[derive(Debug, Clone, Copy)]
pub struct Longformer;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "longformer",
    formats: &[(Format::Transformers, &["longformer"])],
    loaders: &[
        HeadLoader::TextEncoder(load_encoder_head::<Longformer, TextEncoder>),
        HeadLoader::MaskedLm(load_encoder_head::<Longformer, MaskedLm>),
    ],
};
