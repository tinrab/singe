//! `visual-question-answering`: an image and a question to the answer a vision-language model writes, asked as one user message holding both, as `transformers` now answers it through image-text-to-text.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    processor::chat::{Content, Message, Part},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        multimodal::image_text_to_text::{ImageTextToTextParameters, ImageTextToTextPipeline},
    },
};

#[derive(Debug)]
pub struct VisualQuestionAnsweringPipeline(ImageTextToTextPipeline);

/// An image, as a `data:` URL or bare base64 of its file, and a question about it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct VisualQuestion {
    pub image: String,
    pub question: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VisualAnswer {
    pub answer: String,
}

impl Pipeline for VisualQuestionAnsweringPipeline {
    type Input = VisualQuestion;

    type Parameters = ImageTextToTextParameters;

    type Output = VisualAnswer;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToTextPipeline::load(source, options)?))
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let chat = vec![Message::new(
            "user",
            Content::Parts(vec![
                Part::Image {
                    image: input.image.clone(),
                },
                Part::Text {
                    text: input.question.clone(),
                },
            ]),
        )];
        Ok(VisualAnswer {
            answer: self.0.run_one(&chat, parameters)?.generated_text,
        })
    }
}
