// T5, summarizing and translating, as `verify.py` does with `transformers`' tokenizer and `generate`.

use singe::DataType;
use singe_model::task::text::{
    summarization::SummarizationPipeline, translation::TranslationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<SummarizationPipeline>("summary", DataType::F32);
    session.check::<TranslationPipeline>("german", DataType::F32);
}
