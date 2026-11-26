// GPT-2 generating greedily after two prompts, and after one with a repetition penalty, as `verify.py` generates with `transformers`' `generate`.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt");
    session.check::<TextGenerationPipeline>("story");
    session.check::<TextGenerationPipeline>("repetition");
}
