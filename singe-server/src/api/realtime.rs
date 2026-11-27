//! `/v1/realtime/{model}`: the messages of the sessions the server serves for its own models, each a JSON object tagged by `type`; audio travels as binary frames of 16-bit little-endian mono PCM.
//! Every session answers a message it refuses, or work that fails, with `{"type": "error", "id", "error"}`, the error in the shape an HTTP error's body has.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use singe_model::{
    model::audio::silero_vad::TurnSettings, task::audio::text_to_audio::TextToAudioParameters,
};

use crate::api::{Usage, chat::ChatMessage, completion::GenerationFields};

/// What a client sends a text generator's session.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextRequest {
    /// Generates a completion, as `/v1/chat/completions` or `/v1/completions` would, streamed as `delta` messages and ended by `done`.
    Generate(Box<Generate>),
    /// Cancels the generation `id`, which ends with `done` and the reason `cancelled`.
    Cancel {
        /// The generation's id.
        id: String,
    },
}

/// A generation: chat `messages` through the model's chat template, or a raw `prompt`.
#[derive(Debug, Clone, Deserialize)]
pub struct Generate {
    /// The id the generation's messages carry, which no generation in progress may share.
    pub id: String,
    /// A conversation to complete, as chat completions take it.
    pub messages: Option<Vec<ChatMessage>>,
    /// Text to continue, as text completions take it.
    pub prompt: Option<String>,
    /// Arguments the chat template reads, such as `enable_thinking`.
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
    /// The fields chat and text completions share: `max_tokens`, sampling, stops, and penalties.
    #[serde(flatten)]
    pub generation: GenerationFields,
}

/// What a text generator's session sends.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextEvent {
    /// More of the generation's text.
    Delta {
        /// The generation's id.
        id: String,
        /// The text.
        text: String,
    },
    /// The generation ended.
    Done {
        /// The generation's id.
        id: String,
        /// Why: `stop`, `length`, or `cancelled`.
        finish_reason: &'static str,
        /// Its token counts, unless it was cancelled.
        usage: Option<Usage>,
    },
}

/// What a client sends a speech recognizer's session, besides its audio.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecognitionRequest {
    /// Replaces the session's settings and starts its stream over.
    Configure(RecognitionSettings),
    /// Ends the turn with the audio since the last one.
    Commit,
    /// Drops the audio since the last turn.
    Clear,
}

/// How a speech recognizer's session reads its audio and finds its turns.
#[derive(Debug, Clone, Deserialize)]
pub struct RecognitionSettings {
    /// The audio's sample rate, 24 kHz by default.
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// The spoken language, for recognizers that take one.
    #[serde(default)]
    pub language: Option<String>,
    /// How turns end: by voice activity detection, as the model's detector allows by default, or, when `null`, only by `commit`.
    #[serde(default, deserialize_with = "present")]
    pub turn_detection: Option<Option<TurnDetection>>,
}

/// The sample rate audio takes when a session sets none.
pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

const fn default_sample_rate() -> u32 {
    DEFAULT_SAMPLE_RATE
}

/// Reads a field that was given, `null` included, as `Some`, so that a missing field stays `None`.
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

/// When speech starts, how long a silence ends its turn, and how much audio before it the turn keeps.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct TurnDetection {
    /// The probability of speech, in [0, 1], at which a window counts as speech; 0.5 by default.
    pub threshold: f32,
    /// The silence that ends a turn, 500 ms by default.
    pub silence_ms: u64,
    /// The audio before speech starts that its turn keeps, 300 ms by default.
    pub prefix_ms: u64,
}

impl Default for TurnDetection {
    fn default() -> Self {
        let turns = TurnSettings::default();
        Self {
            threshold: turns.threshold,
            silence_ms: turns.silence_ms,
            prefix_ms: 300,
        }
    }
}

/// What a speech recognizer's session sends.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecognitionEvent {
    /// Speech started, in milliseconds from the stream's start.
    SpeechStarted {
        /// Where.
        at_ms: u64,
    },
    /// The silence that ended speech started, in milliseconds from the stream's start.
    SpeechStopped {
        /// Where.
        at_ms: u64,
    },
    /// A turn's transcript.
    Transcript {
        /// The turn, counted from 1.
        turn: u64,
        /// What was said.
        text: String,
    },
}

/// What a client sends a speech synthesizer's session.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SynthesisRequest {
    /// Speaks `text`, after the requests before it.
    Speak(Speak),
}

/// Text to speak, with the speech endpoint's parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct Speak {
    /// The id the speech's messages carry.
    pub id: String,
    /// The text, which must not be blank.
    pub text: String,
    /// The seed and noise of the voice, and a generator's sampling.
    #[serde(flatten)]
    pub parameters: TextToAudioParameters,
}

/// What a speech synthesizer's session sends around each request's audio, which comes between them as binary frames.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SynthesisEvent {
    /// The request's audio follows, at `sample_rate`.
    AudioStart {
        /// The request's id.
        id: String,
        /// The audio's sample rate.
        sample_rate: u32,
    },
    /// The request's audio ended.
    AudioEnd {
        /// The request's id.
        id: String,
    },
}
