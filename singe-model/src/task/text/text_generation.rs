//! `text-generation`: a prompt or a conversation to the text a causal language model generates after it.

use serde::Deserialize;
use serde_json::{Map, Value};
use singe::{
    generation::{Decoder, DecoderOptions, GenerationOptions, PenaltySet, generate_batch},
    random::Generator,
    tensor::SamplingOptions,
};

use crate::{
    decoder_cache::{DecoderCache, DecoderKey, bucketed_capacity},
    error::{Error, Result},
    head::CausalLm,
    model,
    processor::{
        chat::{ChatTemplate, Message},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, outputs::GeneratedSequence, run_alone},
};

#[derive(Debug)]
pub struct TextGenerationPipeline {
    model: Box<dyn CausalLm>,
    tokenizer: TextTokenizer,
    template: Option<ChatTemplate>,
    stop_tokens: Vec<i64>,
    defaults: GenerationDefaults,
    context_length: Option<usize>,
    decoders: DecoderCache<DecoderKey, Box<dyn Decoder>>,
}

/// The sampling `generation_config.json` asks for, which a server applies where a request names none.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct GenerationDefaults {
    pub do_sample: bool,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

/// A prompt, read as it is, or a conversation, which the snapshot's chat template turns into one.
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
    /// Whether to sample each token with `temperature`, `top_k`, and `top_p` rather than take the likeliest.
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    /// The penalty on tokens in the prompt or already generated, as `transformers`' `repetition_penalty`: it divides their positive logits and multiplies their negative ones.
    pub repetition_penalty: f32,
    pub seed: u64,
    /// Whether the decoder runs compiled steps rather than eager ones, `true` by default.
    pub compile: bool,
    /// Arguments the chat template reads, such as `enable_thinking`.
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
            repetition_penalty: 1.0,
            seed: 0,
            compile: true,
            chat_template_kwargs: Map::new(),
        }
    }
}

impl TextGenerationParameters {
    /// Returns the penalties these parameters ask for: `repetition_penalty`, and no frequency or presence penalty.
    #[must_use]
    pub fn penalties(&self) -> PenaltySet {
        PenaltySet {
            repetition: self.repetition_penalty,
            ..PenaltySet::default()
        }
    }
}

impl TextGenerationPipeline {
    /// Returns a decoder over the model of `options`' rows and capacity, compiled if `compile` is set, for a caller that runs its own generation loop.
    pub fn decoder(&self, options: DecoderOptions, compile: bool) -> Result<Box<dyn Decoder>> {
        self.model.decoder(options, compile)
    }

    #[must_use]
    pub const fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub const fn template(&self) -> Option<&ChatTemplate> {
        self.template.as_ref()
    }

    #[must_use]
    pub fn stop_tokens(&self) -> &[i64] {
        &self.stop_tokens
    }

    #[must_use]
    pub const fn defaults(&self) -> GenerationDefaults {
        self.defaults
    }

    /// Returns the most tokens a sequence holds, prompt included, as the model's config limits its positions, or `None` when the config sets no limit.
    #[must_use]
    pub const fn context_length(&self) -> Option<usize> {
        self.context_length
    }

    /// Returns an input's prompt tokens: a text as it is, without special tokens, or a conversation through the chat template, ending where the assistant's turn starts.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is a conversation and the snapshot has no chat template, or if rendering or encoding fails.
    pub fn prompt(
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
    type Input = TextGenerationInput;

    type Parameters = TextGenerationParameters;

    type Output = GeneratedSequence;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source)?,
            template: ChatTemplate::load(source)?,
            stop_tokens: source.stop_tokens()?,
            defaults: source
                .optional_json("generation_config.json")?
                .unwrap_or_default(),
            context_length: source.context_length()?,
            decoders: DecoderCache::default(),
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        run_alone(self, input, parameters)
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
            penalties: parameters.penalties(),
            ..GenerationOptions::default()
        };
        let generator = Generator::new(parameters.seed);
        let rows = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let capacity = bucketed_capacity(longest + parameters.max_new_tokens);
        let tokens = self.decoders.run(
            (prompts.len(), capacity, parameters.compile),
            || {
                self.model.decoder(
                    DecoderOptions::new(prompts.len(), capacity),
                    parameters.compile,
                )
            },
            |decoder| {
                Ok(generate_batch(
                    decoder,
                    &rows,
                    &options,
                    Some(&generator),
                    |_, _, _| Ok(()),
                    |_, _| Ok(()),
                )?)
            },
        )?;
        tokens
            .into_iter()
            .map(|tokens| {
                Ok(GeneratedSequence {
                    generated_text: self.tokenizer.decode(&tokens, true)?,
                    generated_token_ids: tokens,
                })
            })
            .collect()
    }
}
