//! What the HTTP side of text generation needs of a model: its tokenizer, chat template, stop tokens, and sampling defaults.
//! Rendering and tokenizing run here, off the model's thread.

use std::sync::Arc;

use serde_json::{Map, Value};
use singe::{
    generation::{PenaltySet, SequenceOptions},
    tensor::SamplingOptions,
};
use singe_model::{
    Source,
    processor::{
        chat::{ChatTemplate, Message},
        text::TextTokenizer,
    },
    task::text::text_generation::GenerationDefaults,
};

use crate::{
    api::{
        chat::{ChatLogprobs, ChatTokenLogprob, TopLogprob},
        completion::CompletionLogprobs,
    },
    error::{ApiError, ApiResult},
    generation::output::LoggedToken,
};

/// A text model's tokenizer, chat template, stop tokens, and sampling defaults, read from its snapshot.
#[derive(Debug)]
pub struct TextFormat {
    /// The tokenizer prompts are encoded and output decoded with.
    pub tokenizer: Arc<TextTokenizer>,
    /// The chat template, which chat requests need.
    pub template: Option<ChatTemplate>,
    /// The tokens that end a sequence unless a request sets `ignore_eos`.
    pub stop_tokens: Vec<i64>,
    /// The sampling settings of `generation_config.json`, used where a request leaves one out.
    pub defaults: GenerationDefaults,
    /// The model's own context, before the server's limit.
    pub context_length: Option<usize>,
}

/// The sampling a request asks for; what it leaves out comes from the model's defaults.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sampling {
    /// In [0, 2]; at or below 1e-5 it picks the likeliest token.
    pub temperature: Option<f32>,
    /// In (0, 1].
    pub top_p: Option<f32>,
    /// Zero or negative keeps every token.
    pub top_k: Option<i64>,
    /// In [0, 1].
    pub min_p: Option<f32>,
    /// The seed of the sequence's random stream; unset, the engine draws one.
    pub seed: Option<u64>,
    /// In [-2, 2].
    pub presence_penalty: Option<f32>,
    /// In [-2, 2].
    pub frequency_penalty: Option<f32>,
    /// Above 0.
    pub repetition_penalty: Option<f32>,
    /// Token ids in the vocabulary, with biases in [-100, 100].
    pub logit_bias: Vec<(i64, f32)>,
    /// At most the sequence's token limit.
    pub min_tokens: usize,
    /// How many alternatives to return with each token's log probability, or `None` for no log probabilities.
    pub logprobs: Option<usize>,
}

/// The least log probability a response reports: one that underflows to negative infinity, which JSON cannot hold, reports this, as vLLM's does.
const LEAST_LOGPROB: f32 = -9999.0;

/// Temperatures at or below this choose the likeliest token.
const GREEDY_TEMPERATURE: f32 = 1.0e-5;

impl TextFormat {
    /// Loads the format from the snapshot `source`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer, chat template, stop tokens, generation config, or context length cannot be read.
    pub fn load(source: &Source) -> singe_model::Result<Self> {
        Ok(Self {
            tokenizer: Arc::new(TextTokenizer::load(source)?),
            template: ChatTemplate::load(source)?,
            stop_tokens: source.stop_tokens()?,
            defaults: source
                .optional_json("generation_config.json")?
                .unwrap_or_default(),
            context_length: source.context_length()?,
        })
    }

    /// Renders a conversation through the chat template, ending where the assistant's turn starts when `generation_prompt` asks.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if the model has no chat template or the template fails to render.
    pub fn render(
        &self,
        messages: &[Message],
        generation_prompt: bool,
        arguments: &Map<String, Value>,
    ) -> ApiResult<String> {
        let template = self.template.as_ref().ok_or_else(|| {
            ApiError::invalid("a conversation for a model without a chat template")
        })?;
        Ok(template.render(messages, generation_prompt, arguments)?)
    }

    /// Encodes a prompt, adding the tokenizer's special tokens when `special_tokens` is set; a rendered conversation holds its special tokens already.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn encode(&self, text: &str, special_tokens: bool) -> ApiResult<Vec<i64>> {
        Ok(self.tokenizer.encode(text, special_tokens)?)
    }

    /// Returns what a sequence generates: up to `max_tokens`, stopping at the model's stop tokens unless `ignore_eos`, and at the non-negative `stop_token_ids`.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if a sampling setting, or the model's default for it, is out of range, a biased token is outside the vocabulary, or `min_tokens` exceeds `max_tokens`.
    pub fn sequence(
        &self,
        max_tokens: usize,
        sampling: &Sampling,
        stop_token_ids: &[i64],
        ignore_eos: bool,
    ) -> ApiResult<SequenceOptions> {
        if sampling.min_tokens > max_tokens {
            return Err(ApiError::invalid_field(
                "min_tokens",
                format!(
                    "{} tokens at least, more than the limit of {max_tokens}",
                    sampling.min_tokens
                ),
            ));
        }

        let vocabulary = self.tokenizer.vocabulary_size();
        if let Some((token, _)) = sampling
            .logit_bias
            .iter()
            .find(|(token, _)| !(0..vocabulary).contains(token))
        {
            return Err(ApiError::invalid_field(
                "logit_bias",
                format!("token {token} outside a vocabulary of {vocabulary}"),
            ));
        }

        let mut stop_tokens = if ignore_eos {
            Vec::new()
        } else {
            self.stop_tokens.clone()
        };
        stop_tokens.extend(stop_token_ids.iter().filter(|&&token| token >= 0));
        Ok(SequenceOptions {
            max_new_tokens: max_tokens,
            stop_tokens,
            sampling: self.sampling(sampling)?,
            seed: sampling.seed,
            logit_bias: sampling.logit_bias.clone(),
            min_tokens: sampling.min_tokens,
            penalties: self.penalties(sampling)?,
            logprobs: sampling.logprobs,
        })
    }

    /// Returns the log probabilities of `tokens` as a legacy completion lists them, their text starting `offset` characters into the choice's text; `offset` moves past them.
    ///
    /// # Errors
    ///
    /// Returns an error if a token fails to decode.
    pub fn completion_logprobs(
        &self,
        tokens: &[LoggedToken],
        offset: &mut usize,
    ) -> ApiResult<CompletionLogprobs> {
        let mut logprobs = CompletionLogprobs::default();
        for logged in tokens {
            let text = self.token_text(logged.token)?;
            let mut top = Map::new();
            top.insert(text.clone(), Value::from(reported(logged.logprobs.logprob)));
            for &(token, logprob) in &logged.logprobs.alternatives {
                top.insert(self.token_text(token)?, Value::from(reported(logprob)));
            }

            logprobs.text_offset.push(*offset);
            *offset += text.chars().count();
            logprobs
                .token_logprobs
                .push(reported(logged.logprobs.logprob));
            logprobs.top_logprobs.push(top);
            logprobs.tokens.push(text);
        }
        Ok(logprobs)
    }

    /// Returns the log probabilities of `tokens` as a chat completion lists them.
    ///
    /// # Errors
    ///
    /// Returns an error if a token fails to decode.
    pub fn chat_logprobs(&self, tokens: &[LoggedToken]) -> ApiResult<ChatLogprobs> {
        let content = tokens
            .iter()
            .map(|logged| {
                let top_logprobs = logged
                    .logprobs
                    .alternatives
                    .iter()
                    .map(|&(token, logprob)| {
                        let token = self.token_text(token)?;
                        Ok(TopLogprob {
                            bytes: token.as_bytes().to_vec(),
                            token,
                            logprob: reported(logprob),
                        })
                    })
                    .collect::<ApiResult<Vec<_>>>()?;
                let token = self.token_text(logged.token)?;
                Ok(ChatTokenLogprob {
                    bytes: token.as_bytes().to_vec(),
                    token,
                    logprob: reported(logged.logprobs.logprob),
                    top_logprobs,
                })
            })
            .collect::<ApiResult<Vec<_>>>()?;
        Ok(ChatLogprobs { content })
    }

    fn token_text(&self, token: i64) -> ApiResult<String> {
        Ok(self.tokenizer.decode(&[token], false)?)
    }

    fn penalties(&self, sampling: &Sampling) -> ApiResult<PenaltySet> {
        let defaults = self.defaults;
        let penalties = PenaltySet {
            repetition: sampling
                .repetition_penalty
                .or(defaults.repetition_penalty)
                .unwrap_or(1.0),
            frequency: sampling
                .frequency_penalty
                .or(defaults.frequency_penalty)
                .unwrap_or(0.0),
            presence: sampling
                .presence_penalty
                .or(defaults.presence_penalty)
                .unwrap_or(0.0),
        };
        if !(penalties.repetition.is_finite() && penalties.repetition > 0.0) {
            return Err(ApiError::invalid_field(
                "repetition_penalty",
                "a repetition_penalty at or below 0",
            ));
        }

        for (name, value) in [
            ("frequency_penalty", penalties.frequency),
            ("presence_penalty", penalties.presence),
        ] {
            if !(-2.0..=2.0).contains(&value) {
                return Err(ApiError::invalid_field(
                    name,
                    format!("a {name} outside [-2, 2]"),
                ));
            }
        }
        Ok(penalties)
    }

    fn sampling(&self, sampling: &Sampling) -> ApiResult<Option<SamplingOptions>> {
        let defaults = self.defaults;
        let temperature = sampling.temperature.or(defaults.temperature).unwrap_or(1.0);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::invalid_field(
                "temperature",
                "a temperature outside [0, 2]",
            ));
        }

        if temperature <= GREEDY_TEMPERATURE {
            return Ok(None);
        }

        let top_p = sampling.top_p.or(defaults.top_p).unwrap_or(1.0);
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(ApiError::invalid_field("top_p", "a top_p outside (0, 1]"));
        }

        let min_p = sampling.min_p.or(defaults.min_p).unwrap_or(0.0);
        if !(0.0..=1.0).contains(&min_p) {
            return Err(ApiError::invalid_field("min_p", "a min_p outside [0, 1]"));
        }

        // Zero or a negative top-k keeps every token, as vLLM reads it.
        let top_k = sampling
            .top_k
            .or_else(|| defaults.top_k.and_then(|top_k| i64::try_from(top_k).ok()))
            .filter(|&top_k| top_k > 0);
        Ok(Some(
            SamplingOptions::new()
                .with_temperature(temperature)
                .with_top_p(top_p)
                .with_top_k(top_k)
                .with_min_p(min_p),
        ))
    }
}

fn reported(logprob: f32) -> f32 {
    logprob.max(LEAST_LOGPROB)
}
