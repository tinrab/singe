//! `POST /v1/audio/transcriptions`, whose request is a multipart form, and `POST /v1/audio/speech`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The body a transcription returns, as its form's `response_format` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TranscriptionFormat {
    /// `json`: a [`Transcription`].
    #[default]
    Json,
    /// `text`: the transcribed text alone, as plain text.
    Text,
    /// `verbose_json`: a [`VerboseTranscription`].
    VerboseJson,
    /// `srt`: SubRip subtitles with one cue spanning the clip.
    Srt,
    /// `vtt`: WebVTT subtitles with one cue spanning the clip, as `text/vtt`.
    Vtt,
}

impl TranscriptionFormat {
    /// Returns the format a `response_format` value names, or `None` for a name other than `json`, `text`, `verbose_json`, `srt`, or `vtt`.
    #[must_use]
    pub fn parse(format: &str) -> Option<Self> {
        Some(match format {
            "json" => Self::Json,
            "text" => Self::Text,
            "verbose_json" => Self::VerboseJson,
            "srt" => Self::Srt,
            "vtt" => Self::Vtt,
            _ => return None,
        })
    }
}

/// A transcription in the `json` format.
#[derive(Debug, Clone, Serialize)]
pub struct Transcription {
    /// The transcribed text, trimmed of surrounding whitespace.
    pub text: String,
}

/// A transcription in the `verbose_json` format.
#[derive(Debug, Clone, Serialize)]
pub struct VerboseTranscription {
    /// Always `transcribe`.
    pub task: &'static str,
    /// The language the recognizer reports, if it reports one.
    pub language: Option<String>,
    /// The clip's length, in seconds.
    pub duration: f64,
    /// The transcribed text, trimmed of surrounding whitespace.
    pub text: String,
    /// One segment holding the whole text and spanning the whole clip.
    pub segments: Vec<TranscriptionSegment>,
}

/// A span of a clip and the text spoken in it.
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionSegment {
    /// The segment's position among the transcription's segments, from 0.
    pub id: usize,
    /// When the segment starts, in seconds from the clip's start.
    pub start: f64,
    /// When the segment ends, in seconds from the clip's start.
    pub end: f64,
    /// The text spoken in the segment.
    pub text: String,
}

/// A request to `POST /v1/audio/speech`.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// The text to speak, which must not be blank.
    pub input: String,
    /// Accepted and ignored: a model of one voice speaks in it whatever this names.
    pub voice: Option<Value>,
    /// `wav`, the default, for a WAV file, or `pcm` for raw 16-bit little-endian samples with the rate in an `X-Sample-Rate` header.
    pub response_format: Option<String>,
    /// Refused unless it is 1.
    pub speed: Option<f32>,
    /// Refused unless it is blank.
    pub instructions: Option<String>,
    /// Varies the voice's noise, 0 by default; the same seed speaks the same way.
    pub seed: Option<u64>,
    /// How much noise varies the voice, or the model's own amount when unset.
    pub noise_scale: Option<f32>,
    /// How much noise varies the durations of the spoken sounds, or the model's own amount when unset.
    pub noise_scale_duration: Option<f32>,
    /// Fields the server does not read, which it logs and ignores.
    #[serde(flatten)]
    pub unknown: Map<String, Value>,
}
