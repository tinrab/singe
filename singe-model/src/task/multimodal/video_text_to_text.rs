//! `video-text-to-text`: a conversation whose messages hold a clip to the text a vision-language model writes after it, as `transformers`' processors render and expand such a conversation before `generate`, here through the image-text-to-text pipeline's steps.

use crate::{
    error::{Error, Result},
    processor::{
        chat::{Message, Part},
        image::Image,
        video::Video,
    },
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        multimodal::image_text_to_text::{
            ImageTextToTextParameters, ImageTextToTextPipeline, Media, MediaKind,
        },
        outputs::GeneratedSequence,
    },
};

#[derive(Debug)]
pub struct VideoTextToTextPipeline(ImageTextToTextPipeline);

impl Pipeline for VideoTextToTextPipeline {
    type Input = Vec<Message>;

    type Parameters = ImageTextToTextParameters;

    type Output = GeneratedSequence;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToTextPipeline::load_for(
            source,
            options,
            MediaKind::Clip,
        )?))
    }

    fn run_one(
        &self,
        messages: &Self::Input,
        parameters: &Self::Parameters,
    ) -> Result<Self::Output> {
        let clips = messages
            .iter()
            .flat_map(|message| message.content.parts())
            .filter_map(|part| match part {
                Part::Video { video, fps } => Some((video, *fps)),
                Part::Text { .. } | Part::Image { .. } | Part::Audio { .. } => None,
            })
            .collect::<Vec<_>>();
        let [(video, fps)] = clips.as_slice() else {
            return Err(Error::invalid("a conversation of other than one clip"));
        };
        if video.is_empty() {
            return Err(Error::Media("a clip without frames".to_owned()));
        }

        let media = Media::Clip {
            video: Video {
                frames: video
                    .iter()
                    .map(|frame| Image::from_data(frame))
                    .collect::<Result<_>>()?,
            },
            fps: *fps,
        };
        self.0.respond(messages, parameters, &media)
    }
}
