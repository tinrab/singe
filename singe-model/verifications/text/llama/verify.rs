// Llama, through SmolLM2-135M, generating greedily after two prompts, the first also as an exported CUDA C++ generation loop, as `verify.py` generates with `transformers`' `generate`.

use singe::DataType;
use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt", DataType::F32);
    session.check_exported("prompt");
    session.check::<TextGenerationPipeline>("story", DataType::F32);
}
