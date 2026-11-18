// Qwen3.5, generating greedily after a prompt and about an image, as `verify.py` generates with `transformers`' `generate`.

use singe::DataType;
use singe_model::task::{
    multimodal::{any_to_any::AnyToAnyPipeline, image_text_to_text::ImageTextToTextPipeline},
    text::text_generation::TextGenerationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextGenerationPipeline>("prompt", DataType::F32);
    session.check::<ImageTextToTextPipeline>("image", DataType::F32);
    session.check::<AnyToAnyPipeline>("any", DataType::F32);
}
