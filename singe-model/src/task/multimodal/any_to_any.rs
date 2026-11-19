// `any-to-any`: messages that mix text and images to the text a multimodal language model writes after them, as `transformers`' any-to-any pipeline gives it for models that write text, here through the image-text-to-text pipeline's steps.

use crate::{
    error::Result,
    processor::chat::Message,
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        multimodal::image_text_to_text::{ImageTextToTextParameters, ImageTextToTextPipeline},
        outputs::GeneratedSequence,
    },
};

#[derive(Debug)]
pub struct AnyToAnyPipeline(ImageTextToTextPipeline);

impl Pipeline for AnyToAnyPipeline {
    type Input = Vec<Message>;

    type Parameters = ImageTextToTextParameters;

    type Output = GeneratedSequence;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToTextPipeline::load(source, options)?))
    }

    fn run_one(
        &self,
        messages: &Self::Input,
        parameters: &Self::Parameters,
    ) -> Result<Self::Output> {
        self.0.run_one(messages, parameters)
    }
}
