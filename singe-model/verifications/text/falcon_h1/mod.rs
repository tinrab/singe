// Falcon-H1, generating greedily through Falcon-H1-Tiny-90M after a short prompt, also as an exported CUDA C++ generation loop, and through Falcon-H1-0.5B after one longer than the reference's scan chunk, as `verify.py` generates with `transformers`' `generate`.

use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("tiny");
    session.check_exported("tiny");
    session.check::<TextGenerationPipeline>("long");
}
