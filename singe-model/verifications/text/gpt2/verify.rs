// GPT-2 generating greedily after two prompts, as `verify.py` generates with `transformers`' `generate`.

use singe::DataType;
use singe_model::task::text::text_generation::TextGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt", DataType::F32);
    session.check::<TextGenerationPipeline>("story", DataType::F32);
}
