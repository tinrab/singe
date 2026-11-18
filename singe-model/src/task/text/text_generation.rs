// `text-generation`: a prompt or a conversation to the text a causal language model generates after it.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use singe::{
    generation::{DecoderOptions, GenerationOptions, generate_batch},
    random::Generator,
    tensor::SamplingOptions,
};

use crate::{
    error::{Error, Result},
    head::CausalLm,
    model,
    processing::{
        chat::{ChatTemplate, Message},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct TextGenerationPipeline {
    model: Box<dyn CausalLm>,
    tokenizer: TextTokenizer,
    template: Option<ChatTemplate>,
    stop_tokens: Vec<i64>,
}

// A prompt, read as it is, or a conversation, which the snapshot's chat template turns into one.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum TextGenerationInput {
    Text(String),
    Chat(Vec<Message>),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TextGenerationParameters {
    pub max_new_tokens: usize,
    // Samples each token with these settings rather than taking the likeliest.
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub seed: u64,
    // Runs compiled steps rather than eager ones.
    pub compile: bool,
    // Arguments the chat template reads, such as `enable_thinking`.
    pub chat_template_kwargs: Map<String, Value>,
}

impl Default for TextGenerationParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            do_sample: false,
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            seed: 0,
            compile: true,
            chat_template_kwargs: Map::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TextGenerationOutput {
    // The generated text alone, as `return_full_text=False` gives it.
    pub generated_text: String,
    pub generated_token_ids: Vec<i64>,
}

impl TextGenerationPipeline {
    // Returns an input's prompt tokens: a text as it is, without special tokens, or a conversation through the chat template, ending where the assistant's turn starts.
    fn prompt(
        &self,
        input: &TextGenerationInput,
        template_arguments: &Map<String, Value>,
    ) -> Result<Vec<i64>> {
        let text = match input {
            TextGenerationInput::Text(text) => text.clone(),
            TextGenerationInput::Chat(messages) => self
                .template
                .as_ref()
                .ok_or_else(|| {
                    Error::unsupported("a conversation for a model without a chat template")
                })?
                .render(messages, true, template_arguments)?,
        };
        self.tokenizer.encode(&text, false)
    }
}

impl Pipeline for TextGenerationPipeline {
    const TASK: Task = Task::TextGeneration;

    type Input = TextGenerationInput;

    type Parameters = TextGenerationParameters;

    type Output = TextGenerationOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
            template: ChatTemplate::load(source)?,
            stop_tokens: source.stop_tokens()?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let prompts = inputs
            .iter()
            .map(|input| self.prompt(input, &parameters.chat_template_kwargs))
            .collect::<Result<Vec<_>>>()?;
        let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let capacity = i64::try_from(longest + parameters.max_new_tokens)
            .map_err(|_| Error::unsupported("a sequence this long"))?;
        let mut decoder = self.model.decoder(
            DecoderOptions::new(prompts.len(), capacity),
            parameters.compile,
        )?;
        let options = GenerationOptions {
            max_new_tokens: parameters.max_new_tokens,
            stop_tokens: self.stop_tokens.clone(),
            sampling: parameters.do_sample.then(|| {
                SamplingOptions::new()
                    .with_temperature(parameters.temperature)
                    .with_top_k(
                        parameters
                            .top_k
                            .map(i64::try_from)
                            .transpose()
                            .ok()
                            .flatten(),
                    )
                    .with_top_p(parameters.top_p)
            }),
            ..GenerationOptions::default()
        };
        let generator = Generator::new(parameters.seed);
        let rows = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let tokens = generate_batch(
            decoder.as_mut(),
            &rows,
            &options,
            Some(&generator),
            |_, _, _| Ok(()),
            |_, _| Ok(()),
        )?;
        tokens
            .into_iter()
            .map(|tokens| {
                Ok(TextGenerationOutput {
                    generated_text: self.tokenizer.decode(&tokens, true)?,
                    generated_token_ids: tokens,
                })
            })
            .collect()
    }
}
