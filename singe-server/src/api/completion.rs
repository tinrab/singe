//! `POST /v1/completions`: the request, its prompts, and the response, and the generation fields chat completions share.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    api::{Stop, Usage},
    error::{ApiError, ApiResult},
    generation::format::Sampling,
};

/// The most alternatives a request may ask log probabilities for, as OpenAI allows.
pub const MAX_LOGPROBS: usize = 20;

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
    /// The seed of the request's random stream: its `k`-th sampled token draws from the stream's `k`-th block, whatever else runs beside it; unset, the engine draws one.
    pub seed: Option<u64>,
    /// How many choices to generate for each prompt; anything but 1 is refused with 400.
    pub n: Option<usize>,
    /// An id for the end user, which the server ignores.
    pub user: Option<String>,
    /// OpenAI's penalty on tokens already generated, subtracted once from the logit of each, in [-2, 2]; defaults to the model's generation config, else 0.
    pub presence_penalty: Option<f32>,
    /// OpenAI's penalty on tokens already generated, subtracted from the logit of each once per time it was generated, in [-2, 2]; defaults to the model's generation config, else 0.
    pub frequency_penalty: Option<f32>,
    /// vLLM's penalty on tokens in the prompt or already generated, dividing positive logits by it and multiplying negative ones, above 0; defaults to the model's generation config, else 1.
    pub repetition_penalty: Option<f32>,
    /// Biases added to token ids' logits, keyed by the id as a string, each in [-100, 100].
    pub logit_bias: Option<Map<String, Value>>,
    /// The fewest tokens to generate before a stop token or stop string may end generation, at most the token limit.
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
    /// Returns the sampling fields, unset where the request leaves them out, with `logprobs` alternatives to each token's log probability, which each endpoint asks for in its own field.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if a `logit_bias` key is not a token id or a bias is not a number in [-100, 100].
    pub fn sampling(&self, logprobs: Option<usize>) -> ApiResult<Sampling> {
        let logit_bias = self
            .logit_bias
            .iter()
            .flatten()
            .map(|(token, bias)| {
                let token = token.parse::<i64>().map_err(|_| {
                    ApiError::invalid_field(
                        "logit_bias",
                        format!("a key {token} that is not a token id"),
                    )
                })?;
                let bias = bias
                    .as_f64()
                    .filter(|bias| (-100.0..=100.0).contains(bias))
                    .ok_or_else(|| {
                        ApiError::invalid_field(
                            "logit_bias",
                            format!("a bias for token {token} that is not a number in [-100, 100]"),
                        )
                    })?;
                Ok((token, bias as f32))
            })
            .collect::<ApiResult<Vec<_>>>()?;
        Ok(Sampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed: self.seed,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
            logit_bias,
            min_tokens: self.min_tokens.unwrap_or(0),
            logprobs,
        })
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
    /// Returns an error if `n` is not 1, or `stream_options` is set without `stream`.
    pub fn check_supported(&self) -> ApiResult<()> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::unsupported_field("n"));
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
    /// Returns each generated token's log probability, with this many likeliest alternatives, in [0, 20].
    pub logprobs: Option<usize>,
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
    /// The log probabilities of the tokens `text` holds, when the request asks for them.
    pub logprobs: Option<CompletionLogprobs>,
}

/// The log probabilities of a completion's tokens, in the shape of OpenAI's legacy completions.
/// A stop token, which the text leaves out, is left out here too.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CompletionLogprobs {
    /// Each token's text.
    pub tokens: Vec<String>,
    /// Each token's log probability under the model's distribution, before penalties, bias, and temperature.
    pub token_logprobs: Vec<f32>,
    /// Each token's likeliest alternatives, by their text, the chosen token among them.
    pub top_logprobs: Vec<Map<String, Value>>,
    /// Where each token's text starts in the choice's text, in characters.
    pub text_offset: Vec<usize>,
}
