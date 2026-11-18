// `visual-question-answering`: an image and a question to the answer a vision-language model writes, asked as one user message holding both, as `transformers` now answers it through image-text-to-text.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    processing::chat::{Content, Message, Part},
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        multimodal::image_text_to_text::{ImageTextToTextParameters, ImageTextToTextPipeline},
    },
};

#[derive(Debug)]
pub struct VisualQuestionAnsweringPipeline(ImageTextToTextPipeline);

// An image, named by its file, and a question about it.
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
    const TASK: Task = Task::VisualQuestionAnswering;

    type Input = VisualQuestion;

    type Parameters = ImageTextToTextParameters;

    type Output = VisualAnswer;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToTextPipeline::load(source, options)?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let chats = inputs
            .iter()
            .map(|input| {
                vec![Message::new(
                    "user",
                    Content::Parts(vec![
                        Part::Image {
                            image: input.image.clone(),
                        },
                        Part::Text {
                            text: input.question.clone(),
                        },
                    ]),
                )]
            })
            .collect::<Vec<_>>();
        Ok(self
            .0
            .run(&chats, parameters)?
            .into_iter()
            .map(|output| VisualAnswer {
                answer: output.generated_text,
            })
            .collect())
    }
}
