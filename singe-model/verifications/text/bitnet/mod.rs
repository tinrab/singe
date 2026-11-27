// BitNet b1.58 2B in BF16, generating greedily after a count and after a function header, as `verify.py` generates with `transformers`' `generate`.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("count");
    session.check::<TextGenerationPipeline>("code");
}
