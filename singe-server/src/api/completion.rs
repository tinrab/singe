//! `POST /v1/completions`: the request, its prompts, and the response, and the generation fields chat completions share.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    api::{Stop, Usage},
    error::{ApiError, ApiResult},
    generation::format::Sampling,
};

/// What chat and text completions both read about generating.
/// A missing field takes its default; fields this server does not do yet are refused by [`GenerationFields::check_supported`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct GenerationFields {
    /// The most tokens to generate for each prompt; 0 is refused with 400.
    /// Text completions default to 16, and chat completions to what is left of the model's context.
    pub max_tokens: Option<usize>,
    /// The sampling temperature in [0, 2]; at or near 0 generation takes the likeliest token.
    /// Defaults to the model's generation config, else 1.
    pub temperature: Option<f32>,
    /// The nucleus sampling probability mass in (0, 1]; defaults to the model's generation config, else 1.
    pub top_p: Option<f32>,
    /// How many likeliest tokens to sample from; 0 or a negative value keeps every token.
    /// Defaults to the model's generation config.
    pub top_k: Option<i64>,
    /// The least probability, relative to the likeliest token's, a token needs to be sampled, in [0, 1]; defaults to the model's generation config, else 0.
    pub min_p: Option<f32>,
    /// Text that ends generation where it appears.
    pub stop: Option<Stop>,
    /// Token ids that end generation, in addition to the model's stop tokens; negative ids are dropped.
    pub stop_token_ids: Vec<i64>,
    /// Whether generation runs past the model's own stop tokens, ending only at `stop_token_ids`, `stop`, or the token limit.
    pub ignore_eos: bool,
    /// Whether special tokens are left out of the generated text; defaults to `true`.
    pub skip_special_tokens: Option<bool>,
    /// Whether the response streams as server-sent events instead of one JSON body.
    pub stream: bool,
    /// Options for a streamed response; setting them without `stream` is refused with 400.
    pub stream_options: Option<StreamOptions>,
    /// A seed for sampling; any value is refused with 400.
    pub seed: Option<u64>,
    /// How many choices to generate for each prompt; anything but 1 is refused with 400.
    pub n: Option<usize>,
    /// An id for the end user, which the server ignores.
    pub user: Option<String>,
    /// OpenAI's penalty on tokens already present; anything but 0 is refused with 400.
    pub presence_penalty: Option<f32>,
    /// OpenAI's penalty on tokens by how often they appeared; anything but 0 is refused with 400.
    pub frequency_penalty: Option<f32>,
    /// vLLM's multiplicative penalty on repeated tokens; anything but 1 is refused with 400.
    pub repetition_penalty: Option<f32>,
    /// Biases added to token ids' logits; a non-empty map is refused with 400.
    pub logit_bias: Option<Map<String, Value>>,
    /// The fewest tokens to generate before stopping; anything but 0 is refused with 400.
    pub min_tokens: Option<usize>,
}

/// The `stream_options` of a streamed request.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(default)]
pub struct StreamOptions {
    /// Whether the stream ends with a chunk that holds the token counts and no choices.
    pub include_usage: bool,
}

impl GenerationFields {
    /// Returns the sampling fields, unset where the request leaves them out.
    #[must_use]
    pub const fn sampling(&self) -> Sampling {
        Sampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
        }
    }

    /// Returns whether a streamed response ends with a usage chunk.
    #[must_use]
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .is_some_and(|options| options.include_usage)
    }

    /// Refuses what the server reads but does not do yet, unless it is set to what does nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if `n` is not 1, a penalty is not neutral, `logit_bias` is not empty, `seed` is set, `min_tokens` is not 0, or `stream_options` is set without `stream`.
    pub fn check_supported(&self) -> ApiResult<()> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::unsupported_field("n"));
        }

        let penalties = [
            ("presence_penalty", self.presence_penalty, 0.0),
            ("frequency_penalty", self.frequency_penalty, 0.0),
            ("repetition_penalty", self.repetition_penalty, 1.0),
        ];
        for (name, value, neutral) in penalties {
            if value.is_some_and(|value| (value - neutral).abs() > f32::EPSILON) {
                return Err(ApiError::unsupported_field(name));
            }
        }

        if self
            .logit_bias
            .as_ref()
            .is_some_and(|bias| !bias.is_empty())
        {
            return Err(ApiError::unsupported_field("logit_bias"));
        }

        if self.seed.is_some() {
            return Err(ApiError::unsupported_field("seed"));
        }

        if self.min_tokens.is_some_and(|tokens| tokens > 0) {
            return Err(ApiError::unsupported_field("min_tokens"));
        }

        if self.stream_options.is_some() && !self.stream {
            return Err(ApiError::invalid_field(
                "stream_options",
                "`stream_options` without `stream`",
            ));
        }
        Ok(())
    }
}

/// A text completion request: prompts the model continues as they are, without a chat template.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// The prompts, each generated as its own choice.
    pub prompt: Prompt,
    /// The sampling, stopping, and streaming fields chat completions share.
    #[serde(flatten)]
    pub generation: GenerationFields,
    /// Whether the prompt is repeated before the completion; `true` is refused with 400.
    #[serde(default)]
    pub echo: bool,
    /// Text that follows the completion; a non-empty suffix is refused with 400.
    pub suffix: Option<String>,
    /// How many completions to generate to return the best; anything but 1 is refused with 400.
    pub best_of: Option<usize>,
    /// How many likeliest tokens to return log probabilities for; any value is refused with 400.
    pub logprobs: Option<u32>,
    /// The fields the server does not read, logged by name and otherwise ignored.
    #[serde(flatten)]
    pub unknown: Map<String, Value>,
}

/// A prompt as text or as tokens, or several of either.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    /// One text prompt.
    Text(String),
    /// Several text prompts.
    Texts(Vec<String>),
    /// One prompt as token ids.
    Tokens(Vec<i64>),
    /// Several prompts as token ids.
    TokenLists(Vec<Vec<i64>>),
}

/// One prompt of a [`Prompt`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptInput {
    /// Text the tokenizer encodes, adding the model's special tokens.
    Text(String),
    /// Token ids, each of which must lie in the model's vocabulary.
    Tokens(Vec<i64>),
}

impl Prompt {
    /// Returns each prompt on its own, in request order.
    #[must_use]
    pub fn inputs(self) -> Vec<PromptInput> {
        match self {
            Self::Text(text) => vec![PromptInput::Text(text)],
            Self::Texts(texts) => texts.into_iter().map(PromptInput::Text).collect(),
            Self::Tokens(tokens) => vec![PromptInput::Tokens(tokens)],
            Self::TokenLists(lists) => lists.into_iter().map(PromptInput::Tokens).collect(),
        }
    }
}

/// A text completion: the whole response when the request does not stream, or one server-sent event of a streamed one.
/// A streamed response sends a chunk per piece of text, one per choice with its finish reason, and, when `stream_options.include_usage` is set, a last one with the usage and no choices.
#[derive(Debug, Clone, Serialize)]
pub struct Completion {
    /// `cmpl-` followed by the request's `X-Request-Id`, or a new id without one.
    pub id: String,
    /// Always `text_completion`.
    pub object: &'static str,
    /// When the response was created, in seconds since the Unix epoch.
    pub created: u64,
    /// The served model's name.
    pub model: String,
    /// One choice per prompt, in prompt order; a streamed chunk holds one choice, or none in the usage chunk.
    pub choices: Vec<CompletionChoice>,
    /// The token counts over every prompt, set in a whole response and in the usage chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One prompt's generated text in a [`Completion`].
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    /// The prompt's position in the request.
    pub index: usize,
    /// The generated text, or in a streamed chunk the text it adds.
    pub text: String,
    /// Why generation ended: `stop` at a stop token or string, `length` at the token limit; unset in streamed chunks before the choice ends.
    pub finish_reason: Option<&'static str>,
    /// Token log probabilities, always `null` because the server refuses requests for them.
    pub logprobs: Option<Value>,
}
