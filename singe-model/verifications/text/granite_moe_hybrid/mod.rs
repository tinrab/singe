// Granite-4.0-H, through granite-4.0-h-350m, generating greedily after a short prompt and after one longer than the reference's scan chunk, the first also as an exported CUDA C++ generation loop, as `verify.py` generates with `transformers`' `generate`.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt");
    session.check_exported("prompt");
    session.check::<TextGenerationPipeline>("long");
}
