// `video-text-to-text`: a conversation whose messages hold a clip to the text a vision-language model writes after it, as `transformers`' processors render and expand such a conversation before `generate`, here through the image-text-to-text pipeline's steps.

use crate::{
    error::{Error, Result},
    processing::{
        chat::{Content, Message, Part},
        video::Video,
    },
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        multimodal::image_text_to_text::{
            ImageTextToTextOutput, ImageTextToTextParameters, ImageTextToTextPipeline,
        },
    },
};

#[derive(Debug)]
pub struct VideoTextToTextPipeline(ImageTextToTextPipeline);

impl Pipeline for VideoTextToTextPipeline {
    const TASK: Task = Task::VideoTextToText;

    type Input = Vec<Message>;

    type Parameters = ImageTextToTextParameters;

    type Output = ImageTextToTextOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToTextPipeline::load_for(
            source, options, true,
        )?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        self.0.generate_each(
            inputs,
            parameters,
            |model, tokenizer, messages, prompt, options| {
                let clips = messages
                    .iter()
                    .filter_map(|message| match &message.content {
                        Content::Parts(parts) => Some(parts),
                        Content::Text(_) => None,
                    })
                    .flatten()
                    .filter_map(|part| match part {
                        Part::Video { video, fps } => Some((video, *fps)),
                        Part::Text { .. } | Part::Image { .. } | Part::Audio { .. } => None,
                    })
                    .collect::<Vec<_>>();
                let [(video, fps)] = clips.as_slice() else {
                    return Err(Error::unsupported("a conversation of other than one clip"));
                };
                model.generate_video(tokenizer, prompt, &Video::open(video)?, *fps, options)
            },
        )
    }
}
