// `any-to-any`: messages that mix text and images to the text a multimodal language model writes after them, as `transformers`' any-to-any pipeline gives it for models that write text, here through the image-text-to-text pipeline's steps.

use crate::{
    error::Result,
    processor::chat::Message,
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        multimodal::image_text_to_text::{
            ImageTextToTextOutput, ImageTextToTextParameters, ImageTextToTextPipeline,
        },
    },
};

#[derive(Debug)]
pub struct AnyToAnyPipeline(ImageTextToTextPipeline);

impl Pipeline for AnyToAnyPipeline {
    const TASK: Task = Task::AnyToAny;

    type Input = Vec<Message>;

    type Parameters = ImageTextToTextParameters;

    type Output = ImageTextToTextOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToTextPipeline::load(source, options)?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        self.0.run(inputs, parameters)
    }
}
