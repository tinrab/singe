// RecurrentGemma 2B in BF16, generating greedily after a function header and after a count to one hundred, as `verify.py` generates with `transformers`' `generate`.
// It has no exported check: the exported loop's buffers do not fit beside the 5.4 GB of weights on an 8 GB GPU.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt");
    session.check::<TextGenerationPipeline>("long");
}
