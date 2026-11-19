// T5, summarizing and translating, as `verify.py` does with `transformers`' tokenizer and `generate`.

use singe_model::task::text::{
    summarization::SummarizationPipeline, translation::TranslationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<SummarizationPipeline>("summary");
    session.check::<TranslationPipeline>("german");
}
