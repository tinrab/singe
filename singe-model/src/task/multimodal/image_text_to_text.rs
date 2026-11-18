// `image-text-to-text`: a conversation whose messages hold images to the text a vision-language model writes after it.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use singe::generation::GenerationOptions;

use crate::{
    BatchOptions,
    error::{Error, Result},
    head::ImageTextToText,
    model,
    processing::{
        chat::{ChatTemplate, Message, Part},
        image::Image,
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct ImageTextToTextPipeline {
    model: Box<dyn ImageTextToText>,
    tokenizer: TextTokenizer,
    template: ChatTemplate,
    stop_tokens: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ImageTextToTextParameters {
    pub max_new_tokens: usize,
    pub compile: bool,
    // Arguments the chat template reads, such as `enable_thinking`.
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageTextToTextOutput {
    pub generated_text: String,
    pub generated_token_ids: Vec<i64>,
}

impl ImageTextToTextPipeline {
    // Loads the pipeline, whose conversations render through the template the model keeps for clips when `clips` is set and the model keeps one, as its processor renders a conversation about a clip.
    pub(crate) fn load_for(source: &Source, options: LoadOptions, clips: bool) -> Result<Self> {
        let model: Box<dyn ImageTextToText> = model::load(source, options)?;
        let template = ChatTemplate::load(source)?
            .ok_or_else(|| Error::unsupported("a vision-language model without a chat template"))?;
        let template = match model.video_template().filter(|_| clips) {
            Some(video) => template.with_template(video),
            None => template,
        };
        Ok(Self {
            model,
            tokenizer: TextTokenizer::load(source.directory())?,
            template,
            stop_tokens: source.stop_tokens()?,
        })
    }

    // Generates after each conversation of `inputs`, rendered through the template, by `generate`, which reads the conversation's media and is given the model, the tokenizer, the conversation, its prompt, and the options `parameters` ask for.
    pub(crate) fn generate_each(
        &self,
        inputs: &[Vec<Message>],
        parameters: &ImageTextToTextParameters,
        generate: impl Fn(
            &dyn ImageTextToText,
            &TextTokenizer,
            &[Message],
            &str,
            &BatchOptions,
        ) -> Result<Vec<i64>>,
    ) -> Result<Vec<ImageTextToTextOutput>> {
        let options = BatchOptions::new(GenerationOptions {
            max_new_tokens: parameters.max_new_tokens,
            stop_tokens: self.stop_tokens.clone(),
            ..GenerationOptions::default()
        })
        .with_compile(parameters.compile);
        inputs
            .iter()
            .map(|messages| {
                let prompt =
                    self.template
                        .render(messages, true, &parameters.chat_template_kwargs)?;
                let tokens = generate(
                    self.model.as_ref(),
                    &self.tokenizer,
                    messages,
                    &prompt,
                    &options,
                )?;
                Ok(ImageTextToTextOutput {
                    generated_text: self.tokenizer.decode(&tokens, true)?,
                    generated_token_ids: tokens,
                })
            })
            .collect()
    }
}

impl Pipeline for ImageTextToTextPipeline {
    const TASK: Task = Task::ImageTextToText;

    type Input = Vec<Message>;

    type Parameters = ImageTextToTextParameters;

    type Output = ImageTextToTextOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Self::load_for(source, options, false)
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        self.generate_each(
            inputs,
            parameters,
            |model, tokenizer, messages, prompt, options| {
                let images = messages
                    .iter()
                    .flat_map(|message| {
                        message.content.files(|part| match part {
                            Part::Image { image } => Some(image),
                            Part::Text { .. } | Part::Audio { .. } | Part::Video { .. } => None,
                        })
                    })
                    .map(Image::open)
                    .collect::<Result<Vec<_>>>()?;
                model.generate(tokenizer, prompt, &images, options)
            },
        )
    }
}
