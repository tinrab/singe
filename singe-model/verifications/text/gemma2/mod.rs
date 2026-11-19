// Gemma 2, through a tiny random checkpoint whose one layer slides its window by the default layout and caps its scores, generating greedily after two prompts, as `verify.py` generates with `transformers`' `generate`.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt");
    session.check::<TextGenerationPipeline>("story");
}
