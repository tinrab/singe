//! `POST /v1/chat/completions`: the request, its messages, and the response, whole or streamed in chunks.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::{Stop, Usage, completion::GenerationFields};

/// A chat completion request: a conversation the model's chat template renders into a prompt.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// The conversation, oldest message first; an empty list is refused with 400.
    pub messages: Vec<ChatMessage>,
    /// The most tokens to generate; it takes precedence over `max_tokens`, and without either the completion may fill the model's context.
    pub max_completion_tokens: Option<usize>,
    /// The sampling, stopping, and streaming fields text completions share.
    #[serde(flatten)]
    pub generation: GenerationFields,
    /// Whether to return token log probabilities; `true` is refused with 400.
    #[serde(default)]
    pub logprobs: bool,
    /// How many alternative tokens to return log probabilities for; any value is refused with 400.
    pub top_logprobs: Option<u32>,
    /// Tools the model may call; a non-empty list is refused with 400.
    pub tools: Option<Vec<Value>>,
    /// Which tool the model must call; anything but `"none"` or `null` is refused with 400.
    pub tool_choice: Option<Value>,
    /// The output's format; anything but `{"type": "text"}` is refused with 400.
    pub response_format: Option<Value>,
    /// Arguments the chat template reads, such as Qwen3's `enable_thinking`.
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
    /// Whether the rendered prompt ends with the template's opening of an assistant turn; defaults to `true`.
    pub add_generation_prompt: Option<bool>,
    /// The fields the server does not read, logged by name and otherwise ignored.
    #[serde(flatten)]
    pub unknown: Map<String, Value>,
}

/// One message of a conversation.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant`, or `tool`, or `developer`, which the template reads as `system`; other roles are refused with 400.
    pub role: String,
    /// The message's text; a missing content is empty text.
    pub content: Option<MessageContent>,
    /// A name for the message's author, which the server ignores.
    pub name: Option<String>,
    /// Tool calls an assistant message made; a non-empty list is refused with 400.
    pub tool_calls: Option<Vec<Value>>,
    /// The tool call a `tool` message answers, which the server ignores.
    pub tool_call_id: Option<String>,
}

/// A message's content, as a string or as a list of parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// The whole content as one string.
    Text(String),
    /// Content parts, whose texts join with newlines.
    Parts(Vec<ContentPart>),
}

/// One part of a message's content, tagged by its `type`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A `text` part.
    Text {
        /// The part's text.
        text: String,
    },
    /// An `image_url` part, refused with 400 because text-generation models do not read images.
    ImageUrl {
        /// The image the part refers to.
        image_url: ImageUrl,
    },
    /// An `input_audio` part, refused with 400 because text-generation models do not read audio.
    InputAudio {
        /// The audio the part carries.
        input_audio: InputAudio,
    },
}

/// The image of an `image_url` content part.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageUrl {
    /// The image's URL, or a `data:` URL holding the image.
    pub url: String,
    /// OpenAI's level of detail to read the image at, such as `low` or `high`.
    pub detail: Option<String>,
}

/// The audio of an `input_audio` content part.
#[derive(Debug, Clone, Deserialize)]
pub struct InputAudio {
    /// The audio file's bytes, base64-encoded.
    pub data: String,
    /// The audio file's format, such as `wav` or `mp3`.
    pub format: String,
}

/// A whole chat completion, the response when the request does not stream.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletion {
    /// `chatcmpl-` followed by the request's `X-Request-Id`, or a new id without one.
    pub id: String,
    /// Always `chat.completion`.
    pub object: &'static str,
    /// When the response was created, in seconds since the Unix epoch.
    pub created: u64,
    /// The served model's name.
    pub model: String,
    /// The generated messages; this server returns one.
    pub choices: Vec<ChatChoice>,
    /// The prompt's and the completion's token counts.
    pub usage: Usage,
}

/// One generated message of a [`ChatCompletion`].
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    /// The choice's position in `choices`.
    pub index: usize,
    /// The generated message.
    pub message: AssistantMessage,
    /// Why generation ended: `stop` at a stop token or string, `length` at the token limit.
    pub finish_reason: &'static str,
    /// Token log probabilities, always `null` because the server refuses requests for them.
    pub logprobs: Option<Value>,
}

/// The message a model generated.
#[derive(Debug, Clone, Serialize)]
pub struct AssistantMessage {
    /// Always `assistant`.
    pub role: &'static str,
    /// The generated text.
    pub content: String,
}

/// One server-sent event of a streamed chat completion.
/// The first chunk carries the role, later ones the text, then one the finish reason, and, when `stream_options.include_usage` is set, a last one the usage with no choices.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    /// The same id in every chunk of a response, as [`ChatCompletion::id`] forms it.
    pub id: String,
    /// Always `chat.completion.chunk`.
    pub object: &'static str,
    /// When the response was created, in seconds since the Unix epoch, the same in every chunk.
    pub created: u64,
    /// The served model's name.
    pub model: String,
    /// The chunk's changes to the message, empty in the usage chunk.
    pub choices: Vec<ChunkChoice>,
    /// The token counts, set only in the usage chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One choice's change in a [`ChatCompletionChunk`].
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    /// The choice's position, always 0.
    pub index: usize,
    /// What the chunk adds to the message.
    pub delta: Delta,
    /// Why generation ended, set only in the chunk that ends the choice; see [`ChatChoice::finish_reason`].
    pub finish_reason: Option<&'static str>,
    /// Token log probabilities, always `null`.
    pub logprobs: Option<Value>,
}

/// The part of the message a chunk adds; fields a chunk does not change are left out.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Delta {
    /// `assistant` in the first chunk only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Text to append to the message's content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionRequest {
    /// Returns the request's stop strings, empty when it gives none.
    #[must_use]
    pub fn stops(&self) -> Vec<String> {
        Stop::strings(self.generation.stop.as_ref())
    }
}
