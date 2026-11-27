// Switch Transformers, summarizing an input long enough that its experts drop tokens at capacity, as `verify.py` does with `transformers`' tokenizer and `generate`.

use singe_model::task::text::summarization::SummarizationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<SummarizationPipeline>("summary");
}
