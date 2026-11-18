// Qwen3, generating greedily after a prompt and after a chat and answering decision questions, as `verify.py` generates with `transformers`' `generate`.

use singe::DataType;
use singe_model::task::{
    multimodal::decision::DecisionPipeline, text::text_generation::TextGenerationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt", DataType::F32);
    session.check::<TextGenerationPipeline>("chat", DataType::F32);
    session.check::<TextGenerationPipeline>("prompt-bf16", DataType::Bf16);
    session.check::<DecisionPipeline>("decision", DataType::F32);
}
