// DeepSeek-V3, through a tiny random checkpoint of a dense layer and a mixture-of-experts layer, generating greedily after two prompts, as `verify.py` generates with `transformers`' `generate`.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt");
    session.check::<TextGenerationPipeline>("story");
}
