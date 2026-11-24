//! `image-text-to-text`: a conversation whose messages hold images to the text a vision-language model writes after it.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    BatchOptions,
    error::{Error, Result},
    head::ImageTextToText,
    model,
    processor::{
        chat::{ChatTemplate, Message, Part},
        image::Image,
        text::TextTokenizer,
        video::Video,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, outputs::GeneratedSequence},
};

#[derive(Debug)]
pub struct ImageTextToTextPipeline {
    model: Box<dyn ImageTextToText>,
    tokenizer: TextTokenizer,
    template: ChatTemplate,
    stop_tokens: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediaKind {
    Images,
    Clip,
}

#[derive(Debug, Clone)]
pub(crate) enum Media {
    Images(Vec<Image>),
    Clip { video: Video, fps: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ImageTextToTextParameters {
    pub max_new_tokens: usize,
    pub compile: bool,
    /// Arguments the chat template reads, such as `enable_thinking`.
    pub chat_template_kwargs: Map<String, Value>,
}

impl Default for ImageTextToTextParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            compile: true,
            chat_template_kwargs: Map::new(),
        }
    }
}

impl ImageTextToTextPipeline {
    pub(crate) fn load_for(
        source: &Source,
        options: LoadOptions,
        media: MediaKind,
    ) -> Result<Self> {
        let model: Box<dyn ImageTextToText> = model::load(source, options)?;
        let template = ChatTemplate::load(source)?
            .ok_or_else(|| Error::unsupported("a vision-language model without a chat template"))?;
        let template = match model.video_template().filter(|_| media == MediaKind::Clip) {
            Some(video) => template.with_template(video),
            None => template,
        };
        Ok(Self {
            model,
            tokenizer: TextTokenizer::load(source)?,
            template,
            stop_tokens: source.stop_tokens()?,
        })
    }

    pub(crate) fn respond(
        &self,
        messages: &[Message],
        parameters: &ImageTextToTextParameters,
        media: &Media,
    ) -> Result<GeneratedSequence> {
        let mut options = BatchOptions::new(parameters.max_new_tokens, parameters.compile);
        options.generation.stop_tokens.clone_from(&self.stop_tokens);
        let prompt = self
            .template
            .render(messages, true, &parameters.chat_template_kwargs)?;
        let tokens = match media {
            Media::Images(images) => {
                self.model
                    .generate(&self.tokenizer, &prompt, images, &options)?
            }
            Media::Clip { video, fps } => {
                self.model
                    .generate_video(&self.tokenizer, &prompt, video, *fps, &options)?
            }
        };
        Ok(GeneratedSequence {
            generated_text: self.tokenizer.decode(&tokens, true)?,
            generated_token_ids: tokens,
        })
    }
}

impl Pipeline for ImageTextToTextPipeline {
    type Input = Vec<Message>;

    type Parameters = ImageTextToTextParameters;

    type Output = GeneratedSequence;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Self::load_for(source, options, MediaKind::Images)
    }

    fn run_one(
        &self,
        messages: &Self::Input,
        parameters: &Self::Parameters,
    ) -> Result<Self::Output> {
        let images = messages
            .iter()
            .flat_map(|message| message.content.parts())
            .filter_map(|part| match part {
                Part::Image { image } => Some(image),
                Part::Text { .. } | Part::Audio { .. } | Part::Video { .. } => None,
            })
            .map(|image| Image::from_data(image))
            .collect::<Result<Vec<_>>>()?;
        self.respond(messages, parameters, &Media::Images(images))
    }
}
