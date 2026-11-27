// Mamba, through mamba-130m, generating greedily after two prompts, the first also as an exported CUDA C++ generation loop, as `verify.py` generates with `transformers`' `generate`.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt");
    session.check_exported("prompt");
    session.check::<TextGenerationPipeline>("story");
}
