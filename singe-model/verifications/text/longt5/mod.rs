// LongT5, summarizing a text past its local band and one of about 15k tokens, as `verify.py` does with `transformers`' tokenizer and `generate`, both under the config's generation settings.

use singe_model::task::text::summarization::SummarizationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<SummarizationPipeline>("summary");
    session.check::<SummarizationPipeline>("book");
}
