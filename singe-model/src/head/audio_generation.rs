//! Text to audio generated as frames of codec codes and decoded, as CSM speaks.

use std::fmt::Debug;

use serde::Deserialize;

use crate::{error::Result, head::SamplingRequest, processor::audio::Audio};

/// What a speech generator speaks: text, or a conversation whose earlier turns give their speakers' voices by their text and audio, as CSM's chat template reads it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum SpeechPrompt {
    Text(String),
    Conversation(Vec<SpokenTurn>),
}

/// A turn of a conversation: its speaker, a stringified integer, and its text and, before the last turn, its audio.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpokenTurn {
    pub role: String,
    pub content: Vec<TurnContent>,
}

/// A part of a turn's content: its text or its clip.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TurnContent {
    Text { text: String },
    Audio { audio: Audio },
}

/// A model that generates a clip's codec codes from text and decodes them.
pub trait AudioGenerator: Debug {
    /// Generates a clip for `prompt` over at most `max_new_tokens` frames, or the generation config's when `None`, choosing codes as `sampling` asks over the generation config's sampling, and returns its samples.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt does not tokenize or is not a conversation the model reads, or if a decoder step or the codec fails.
    fn generate_audio(
        &self,
        prompt: &SpeechPrompt,
        max_new_tokens: Option<usize>,
        sampling: &SamplingRequest,
    ) -> Result<Vec<f32>>;

    /// Returns the samples per second of the clips the model generates.
    fn sample_rate(&self) -> i64;
}
